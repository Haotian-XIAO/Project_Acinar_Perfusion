import numpy as np
import pickle
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay
import gmsh
import random
import pandas as pd
import networkx as nx
import math

import matplotlib.pyplot as plt
##########


def compute_strahler_orders(parents):
    """
    输入:
        parents[i] = 节点 i 的 parent index(root 的 parent = -1)

    输出:
        order[i] = 节点 i 的 Strahler 级别
    """

    N = len(parents)
    children = [[] for _ in range(N)]

    # build children list
    root = None
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
        else:
            root = i

    # 1. 叶子节点初始级别 = 1
    order = [0] * N
    leaves = [i for i in range(N) if len(children[i]) == 0]
    for leaf in leaves:
        order[leaf] = 1

    # 2. 后序遍历，从叶子向 root 计算级数
    # 建议先构建拓扑序（root → leaves）
    topo = []
    stack = [root]
    while stack:
        u = stack.pop()
        topo.append(u)
        for v in children[u]:
            stack.append(v)

    # 倒序计算 Strahler number
    for u in reversed(topo):
        if order[u] == 0:  # 不是叶子
            child_orders = [order[v] for v in children[u]]
            max_order = max(child_orders)
            count_max = child_orders.count(max_order)
            if count_max >= 2:
                order[u] = max_order + 1
            else:
                order[u] = max_order

    return order

def generate_hexagon_coords(domain_x, domain_y, scale=0.45):
    """
    返回一个尖头朝上的规则六边形 6x2 坐标矩阵。
    scale: 六边形大小的相对比例（0~0.5），默认 0.45
    """

    # 六边形中心
    cx = domain_x * 0.5
    cy = domain_y * 0.5

    # 半径（从中心到顶点的距离）
    R = min(domain_x, domain_y) * scale

    # 尖朝上的六边形顶点角度
    # 顶部点是 90°，逆时针方向排列
    angles = np.deg2rad([90, 30, -30, -90, -150, 150])

    hex_coords = []
    for ang in angles:
        x = cx + R * np.cos(ang)
        y = cy + R * np.sin(ang)
        hex_coords.append((x, y))

    return np.array(hex_coords)

def get_voronoi_ridge_segments(vor):
    """
    返回字典 ridx -> (p1, p2)
    其中 p1,p2 为 ridge 的两个端点坐标（仅限 bounded ridge）
    """
    ridge_segments = {}
    for ridx, verts in enumerate(vor.ridge_vertices):
        if -1 in verts:
            continue  # unbounded ridge, 不用于厚壁
        v1 = vor.vertices[verts[0]]
        v2 = vor.vertices[verts[1]]
        ridge_segments[ridx] = (v1, v2)
    return ridge_segments

import numpy as np

def segments_intersect(a1, a2, b1, b2, tol=1e-12):
    """
    判断线段 a1--a2 和 b1--b2 是否相交
    """
    def cross(u, v):
        return u[0]*v[1] - u[1]*v[0]

    a1 = np.array(a1); a2 = np.array(a2)
    b1 = np.array(b1); b2 = np.array(b2)

    r = a2 - a1
    s = b2 - b1
    rxs = cross(r, s)
    q_p = b1 - a1

    # 平行或共线
    if abs(rxs) < tol:
        return False

    t = cross(q_p, s) / rxs
    u = cross(q_p, r) / rxs

    return (0 <= t <= 1) and (0 <= u <= 1)

def extract_physical_seeds(
    seeds, domain_x, domain_y,
    tol=1e-8,
    boundary_offset=0.2,
    root_global_index=None     # ⭐ NEW: airway 入口 seed 的 global index，必须保留
):
    """
    提取主域内部 seeds（退后一格），但保留入口 seed。
    返回:
      phys_seeds, local_to_global, global_to_local
    """
    seeds = np.asarray(seeds)

    # ---- Step 1: points in domain ----
    in_dom = (
        (seeds[:, 0] >= -tol) & (seeds[:, 0] <= domain_x + tol) &
        (seeds[:, 1] >= -tol) & (seeds[:, 1] <= domain_y + tol)
    )

    # ---- Step 2: interior (退后一格) ----
    interior = (
        (seeds[:, 0] > boundary_offset) &
        (seeds[:, 0] < domain_x - boundary_offset) &
        (seeds[:, 1] > boundary_offset) &
        (seeds[:, 1] < domain_y - boundary_offset)
    )

    # ---- 默认过滤条件 ----
    keep = in_dom & interior

    # ---- Step 3: 强制保留入口 seed ----
    
    # 入口 seed 可能不在 interior，但必须保留！
    if root_global_index is not None:
        keep[root_global_index] = True

    # ---- 提取 phys_seeds ----
    global_indices = np.where(keep)[0]
    phys_seeds = seeds[global_indices, :2]

    local_to_global = list(global_indices)
    global_to_local = {g: i for i, g in enumerate(local_to_global)}

    return phys_seeds, local_to_global, global_to_local

def build_delaunay_graph(points):
    """
    在给定点集上构建 Delaunay 图的邻接关系。
    返回 networkx.Graph, 节点 0..N-1
    """
    tri = Delaunay(points)
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a+1, 3):
                i = simplex[a]
                j = simplex[b]
                if i != j:
                    G.add_edge(i, j)
    return G


def choose_entrance_seed_top_middle(points, domain_x, domain_y, top_band_ratio=0.1):
    """
    在 y 方向 top_band 内的 seeds 中找 x 最接近中点的，作为入口 seed。
    """
    x = points[:, 0]
    y = points[:, 1]

    y_threshold = (domain_y-0.3) * (1.0 - top_band_ratio)
    candidates = np.where(y >= y_threshold)[0]
    if len(candidates) == 0:
        # 如果 top_band 里没点，就直接选 y 最大的那些
        max_y = np.max(y)
        candidates = np.where(np.abs(y - max_y) < 1e-6)[0]

    mid_x = 0.5 * domain_x
    best = min(candidates, key=lambda i: abs(x[i] - mid_x))
    return best
    
def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        # 射线法
        if ((y1 <= y < y2) or (y2 <= y < y1)):
            if x < ((x2-x1)*(y-y1)/(y2-y1) + x1):
                inside = not inside
    return inside


def build_initial_tree(G, root, max_children=2):
    """
    构建一棵合法的初始树：
      - 无环
      - 每个节点最多 max_children 个孩子
      - 所有节点都连通到 root
    策略：从 root 做 BFS，优先连接近邻。
    """
    N = G.number_of_nodes()
    parents = [-1] * N
    children_count = [0] * N

    visited = set([root])
    queue = [root]

    while queue:
        u = queue.pop(0)
        # 邻居按距离排序更自然一点
        neighbors = list(G.neighbors(u))
        random.shuffle(neighbors)
        for v in neighbors:
            if v in visited:
                continue
            # u 还能再带孩子么？
            if children_count[u] >= max_children:
                continue
            parents[v] = u
            children_count[u] += 1
            visited.add(v)
            queue.append(v)

    # 有可能还有没连上的节点（比如局部度太小造成“卡死”）
    # 这种情况我们再全局扫描一遍，把剩余节点连到任意已有节点（满足 max_children）
    unvisited = [i for i in range(N) if i not in visited]
    for v in unvisited:
        # 在所有已有节点中找还没满孩子数的，按几何距离排序
        candidates = [u for u in range(N) if u in visited and children_count[u] < max_children and G.has_edge(u, v)]
        if not candidates:
            # 万一真的没有，就放宽一点，不管 max_children 先连上，后续退火再修
            candidates = [u for u in range(N) if u in visited and G.has_edge(u, v)]
            if not candidates:
                continue
        u = min(candidates, key=lambda u_: np.linalg.norm(points[u_] - points[v]))
        parents[v] = u
        children_count[u] += 1
        visited.add(v)

    return parents


def compute_tree_cost(points, parents, omega, domain_x, domain_y):
    """
    计算 C1', C2', C。
      C1' = (Nt - Na) / Nt
      C2' = mean(l_i) / Lref,  Lref 取域对角线长度
    """
    N = len(parents)
    Nt = N

    # children 列表 & terminal
    children = [[] for _ in range(N)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    terminal_nodes = [i for i in range(N) if len(children[i]) == 0]
    Na = len(terminal_nodes)

    # C1
    C1 = Nt - Na
    C1p = C1 / max(Nt, 1)

    # 各节点到 root 的路径长度
    # 先找 root
    roots = [i for i, p in enumerate(parents) if p < 0]
    if len(roots) != 1:
        # 非法树，给大惩罚
        return 1e9, 1.0, 1.0
    root = roots[0]

    # 动态规划距离
    dist = [0.0] * N
    # 拓扑顺序：从 root 往下走
    order = [root]
    for u in order:
        for v in children[u]:
            order.append(v)
            seg_len = np.linalg.norm(points[u] - points[v])
            dist[v] = dist[u] + seg_len

    if Na > 0:
        mean_li = float(np.mean([dist[i] for i in terminal_nodes]))
    else:
        mean_li = 0.0

    Lref = math.sqrt(domain_x**2 + domain_y**2)
    C2p = mean_li / max(Lref, 1e-8)

    C = omega * C1p + (1.0 - omega) * C2p
    return C, C1p, C2p


def get_subtree_nodes(parents, node):
    """
    返回以 node 为根的子树所有节点（包括自身）。
    用于避免新 parent 落在自己子树里 → 产生环。
    """
    N = len(parents)
    children = [[] for _ in range(N)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    stack = [node]
    subtree = set()
    while stack:
        u = stack.pop()
        if u in subtree:
            continue
        subtree.add(u)
        stack.extend(children[u])
    return subtree


def simulated_annealing_optimize(points, G, parents_init,
                                 omega, domain_x, domain_y,
                                 max_children=2,
                                 n_steps=5000,
                                 T0=1.0, alpha=0.995):
    """
    在 Delaunay 邻接图 G 上，对给定初始树 parents_init 做模拟退火优化。
    """
    parents = list(parents_init)
    N = len(parents)
    # 初始 children 计数
    children = [[] for _ in range(N)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    children_count = [len(ch) for ch in children]

    C, _, _ = compute_tree_cost(points, parents, omega, domain_x, domain_y)
    best_parents = list(parents)
    best_C = C

    T = T0
    for step in range(n_steps):
        # 选一个非 root 的节点
        candidates_nodes = [i for i, p in enumerate(parents) if p >= 0]
        if not candidates_nodes:
            break
        v = random.choice(candidates_nodes)
        old_parent = parents[v]

        # 可能的新 parent：v 的 Delaunay 邻居
        neighs = list(G.neighbors(v))
        if old_parent in neighs and len(neighs) > 1:
            neighs.remove(old_parent)

        if not neighs:
            T *= alpha
            continue

        # 避免成环 & 超过 max_children
        subtree = get_subtree_nodes(parents, v)
        valid_new_parents = [
            u for u in neighs
            if (u not in subtree) and (children_count[u] < max_children)
        ]
        if not valid_new_parents:
            T *= alpha
            continue

        new_parent = random.choice(valid_new_parents)

        # 提案：修改 parent[v]
        parents_trial = list(parents)
        parents_trial[v] = new_parent

        C_new, _, _ = compute_tree_cost(points, parents_trial, omega, domain_x, domain_y)
        dC = C_new - C

        accept = False
        if dC <= 0:
            accept = True
        else:
            if random.random() < math.exp(-dC / max(T, 1e-8)):
                accept = True

        if accept:
            # 更新
            parents = parents_trial
            # 更新 children_count
            children_count[old_parent] -= 1
            children_count[new_parent] += 1
            C = C_new
            if C_new < best_C:
                best_C = C_new
                best_parents = list(parents)

        T *= alpha

    return best_parents

def choose_global_entrance_seed_top_middle(seeds_all, domain_x, domain_y, y_ratio=0.85):
    """
    在 global seeds 中寻找位于 top-middle 的入口 seed。
    同时考虑 y 接近顶部、x 接近中间两个因素。
    """
    seeds_all = np.asarray(seeds_all)

    # ------------ Step 1: 找到 top 区域 seeds ------------
    candidates = np.where(seeds_all[:, 1] > domain_y * y_ratio)[0]
    if len(candidates) == 0:
        raise RuntimeError("No seed found near top boundary!")

    # ------------ Step 2: 计算评分函数 ------------
    center_x = domain_x * 0.5

    def score(idx):
        x, y = seeds_all[idx]
        dist_y = abs(domain_y - y)        # 越靠近顶部越好
        dist_x = abs(center_x - x)        # 越靠近中心越好
        return (dist_y, dist_x)

    # ------------ Step 3: 选出 best seed ------------
    best = min(candidates, key=score)

    return best



def generate_airway_tree_hex(
    seeds_all,
    hex_coords,             # 🔥 六边形顶点坐标 (6×2)
    omega=0.5,
    max_children=2,
    n_steps=5000
):
    """
    根据六边形 domain 生成 airway tree，
    保证入口 seed 在六边形顶部并且所有分支不会延伸到六边形外。
    """

    hex_coords = np.asarray(hex_coords)

    # -------------------------------
    # 1. Physical seeds = seeds inside hexagon
    # -------------------------------
    phys_indices = []
    for idx, p in enumerate(seeds_all):
        if point_in_polygon(p, hex_coords):
            phys_indices.append(idx)

    phys_seeds = seeds_all[phys_indices]
    print(f"[airway] physical seeds in hex = {len(phys_seeds)}")

    # local-global mapping
    local_to_global = phys_indices
    global_to_local = {g: i for i, g in enumerate(phys_indices)}

    # -------------------------------
    # 2. Entrance seed = top hex vertex
    # -------------------------------
    root = choose_hex_top_seed(phys_seeds, hex_coords)
    print(f"[airway] entrance seed (local index) = {root}")

    # -------------------------------
    # 3. Build Delaunay graph (inside hex)
    # -------------------------------
    G = build_delaunay_graph(phys_seeds)

    # -------------------------------
    # 4. Build initial airway tree
    # -------------------------------
    parents0 = build_initial_tree(G, root, max_children=max_children)

    # -------------------------------
    # 5. Optimize using simulated annealing
    # -------------------------------
    parents_opt = simulated_annealing_optimize(
        phys_seeds, G, parents0,
        omega=omega,
        domain_x=1.0, domain_y=1.0,  # unused now
        max_children=max_children,
        n_steps=n_steps
    )

    # -------------------------------
    # 6. Build airway edges in local index
    # -------------------------------
    airway_edges_local = []
    for i, p in enumerate(parents_opt):
        if p >= 0:
            airway_edges_local.append((i, p))

    return parents_opt, airway_edges_local, local_to_global, phys_seeds

def export_airway_with_strahler(points, parents, strahler, filename="airway_strahler.csv"):
    import pandas as pd
    data = []
    for i, (x,y) in enumerate(points):
        data.append({
            "local_index": i,
            "x": x,
            "y": y,
            "parent": parents[i],
            "strahler": strahler[i]
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"[airway] Saved airway Strahler table to {filename}")

def plot_airway_tree(points, parents, airway_edges_local,
                      title="Airway Tree"):
    """
    用 matplotlib 在 2D 平面上画出种子点与 airway 树。
    airway_edges_local 是 (i_local, j_local) 的列表。
    """
    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=10, color="black", alpha=0.6, label="Seeds")

    # 画 airway 边
    for (i, j) in airway_edges_local:
        xi, yi = points[i]
        xj, yj = points[j]
        plt.plot([xi, xj], [yi, yj], color="red", linewidth=1.5)

    # 画入口 seed
    root = [i for i,p in enumerate(parents) if p < 0][0]
    plt.scatter([points[root,0]], [points[root,1]],
                color="blue", s=60, label="Entrance")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.show()




#######################################################################



def convert_vtk_to_xdmf(mesh_filename, dim=2):
    import meshio
    mesh = meshio.read(mesh_filename + ".vtk")
    

    if dim == 2:
        # Keep only triangle or triangle6 cells
        cells_2d = []
        cell_data_2d = {}

        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type in ["triangle", "triangle6"]:
                cells_2d.append(cell_block)
                if mesh.cell_data:
                    for key, data_list in mesh.cell_data.items():
                        if key not in cell_data_2d:
                            cell_data_2d[key] = []
                        cell_data_2d[key].append(data_list[i])
        
        mesh.cells = cells_2d
        if cell_data_2d:
            mesh.cell_data = cell_data_2d

        # Optional: Drop z-coordinate if present
        if mesh.points.shape[1] == 3:
            mesh.points = mesh.points[:, :2]
    elif dim == 3:
        # For 3D, filter cells: keep only tetrahedral cells.
        cells_3d = []
        cell_data_3d = {}
        
        # mesh.cells is a list of CellBlock objects.
        # We'll keep only those with type "tetra" or "tetra10".
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type in ["tetra", "tetra10"]:
                cells_3d.append(cell_block)
                # If cell_data is available, collect the corresponding data.
                if mesh.cell_data:
                    for key, data_list in mesh.cell_data.items():
                        # Initialize the list for this key if not already done.
                        if key not in cell_data_3d:
                            cell_data_3d[key] = []
                        cell_data_3d[key].append(data_list[i])
        # Update the mesh with the filtered cells and cell data.
        mesh.cells = cells_3d
        if cell_data_3d:
            mesh.cell_data = cell_data_3d

    # Write out the filtered mesh to an XDMF file.
    meshio.write(mesh_filename + ".xdmf", mesh)


def map_airway_to_ridges(vor, local_to_global, airway_edges_local):
    """
    输入:
      vor: scipy.spatial.Voronoi 对象
      local_to_global: local seeds -> global seeds
      airway_edges_local: [(i_local, j_local), ...]

    输出:
      airway_ridge_indices: set of ridge indices in vor.ridge_points
    """
    airway_ridges = set()

    # 把 local index 转 global
    airway_edges_global = set()
    for (i, j) in airway_edges_local:
        gi = local_to_global[i]
        gj = local_to_global[j]
        airway_edges_global.add((gi, gj))
        airway_edges_global.add((gj, gi))

    # 在 Voronoi 的 ridge_points 里查找对应边
    for ridx, (p, q) in enumerate(vor.ridge_points):
        if (p, q) in airway_edges_global:
            airway_ridges.add(ridx)

    return airway_ridges



def build_region_ridge_map(vor):
    """
    为每个 bounded region 构建映射:
      region_ridge_map[(region_idx, edge_i)] = ridge_idx

    这里 edge_i 是 region 多边形中第 i 条边（顶点 i -> i+1）。
    """
    region_ridge_map = {}

    # 每条 ridge 的两个端点坐标
    ridge_verts_xy = []
    for verts in vor.ridge_vertices:
        if -1 in verts:
            ridge_verts_xy.append(None)
        else:
            ridge_verts_xy.append([vor.vertices[verts[0]], vor.vertices[verts[1]]])

    # 遍历每个 region
    for region_idx, region in enumerate(vor.regions):
        if not region or -1 in region or len(region) < 3:
            continue

        for i in range(len(region)):
            v1 = region[i]
            v2 = region[(i + 1) % len(region)]
            p1 = vor.vertices[v1]
            p2 = vor.vertices[v2]

            # 在所有有限 ridge 中寻找端点匹配的那条
            for ridx, rxy in enumerate(ridge_verts_xy):
                if rxy is None:
                    continue
                rv1, rv2 = rxy
                if (np.allclose(rv1, p1) and np.allclose(rv2, p2)) or \
                   (np.allclose(rv1, p2) and np.allclose(rv2, p1)):
                    region_ridge_map[(region_idx, i)] = ridx
                    break

    return region_ridge_map

def choose_hex_top_seed(points, hex_coords):
    top = hex_coords[0]   # 六边形顶点顺序: 第0个是最顶部
    d = np.sum((points - top)**2, axis=1)
    return np.argmin(d)

def generate_2D_voronoi_with_thickness_hex(
    mesh_filename,
    seeds_filename,
    domain_x,
    domain_y,
    lcar=0.01,
    offset_distance=0.01,
    N_inner_density=80,     # ★ 六边形内边界每条边撒点数
    airway_segments=None,   # [(a1,a2), ...] in global coords
):
    """
    生成 2D Voronoi 厚壁结构，被尖头朝上的六边形裁剪，并在外侧再扩一圈生成六边形外墙。
    - 内外墙厚度 = offset_distance
    - Voronoi 厚壁也用 offset_distance
    - airway_segments 用于不生成气道上的厚壁
    - 内六边形边界使用手动撒点控制密度，再与 Voronoi 厚壁 fragment 保证连通
    """

    import gmsh
    import numpy as np
    import pickle
    from scipy.spatial import Voronoi

    # ------------------ 初始化 Gmsh ------------------
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("Voronoi_Hex_Acinar")
    occ = gmsh.model.occ
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmin)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)

    # ------------------ 读取 seeds & Voronoi ------------------
    with open(seeds_filename, "rb") as f:
        seeds = pickle.load(f)
    seeds = np.asarray(seeds)[:, :2]
    print(f"[Voronoi] Loaded {len(seeds)} seeds from {seeds_filename}")

    vor = Voronoi(seeds)

    # ------------------ 辅助函数：对一个 region 做内缩 ------------------
    def inward_polygon(coords):
        centroid = coords.mean(axis=0)
        in_coords = []
        for p in coords:
            dv = centroid - p
            nrm = np.linalg.norm(dv)
            if nrm == 0:
                in_coords.append(p)
            else:
                in_coords.append(p + offset_distance * dv / nrm)
        return np.asarray(in_coords)

    # ------------------ 1. 生成 Voronoi 厚壁（不裁剪） ------------------
    all_voro_walls = []
    print("[Voronoi] Building thickened Voronoi walls (uncut)...")

    for region_idx, region in enumerate(vor.regions):
        if not region or -1 in region or len(region) < 3:
            continue

        coords = np.array([vor.vertices[v] for v in region])
        coords_in = inward_polygon(coords)

        # 添加点
        out_pts = [occ.addPoint(p[0], p[1], 0.0, lcar) for p in coords]
        in_pts  = [occ.addPoint(p[0], p[1], 0.0, lcar) for p in coords_in]

        # 外 / 内多边形边
        n = len(coords)
        out_lines = [occ.addLine(out_pts[i], out_pts[(i+1) % n]) for i in range(n)]
        in_lines  = [occ.addLine(in_pts[i],  in_pts[(i+1) % n]) for i in range(n)]

        # 针对每条边生成厚壁
        for i in range(n):
            q1 = coords[i]
            q2 = coords[(i+1) % n]

            # 如果这条 Voronoi 边被 airway 穿过，则跳过
            if airway_segments is not None:
                is_airway = False
                for (a1, a2) in airway_segments:
                    if segments_intersect(q1, q2, a1, a2):
                        is_airway = True
                        break
                if is_airway:
                    continue

            p1 = out_pts[i]
            p2 = out_pts[(i+1) % n]
            p3 = in_pts[(i+1) % n]
            p4 = in_pts[i]

            l1 = occ.addLine(p1, p4)
            l2 = occ.addLine(p2, p3)

            wire = occ.addWire([out_lines[i], l2, in_lines[i], l1])
            sf = occ.addPlaneSurface([wire])
            all_voro_walls.append((2, sf))

    occ.synchronize()
    print(f"[Voronoi] Built {len(all_voro_walls)} thick wall surfaces before cutting.")

    # ------------------ 2. 构造尖头朝上的六边形 (作为裁剪区域) ------------------
    print("[HEX] Building pointy-top hexagon for cutting...")

    cx = 0.5 * domain_x
    cy = 0.5 * domain_y
    R = 0.5 * min(domain_x, domain_y)   # R = min(domain_x,domain_y)/2

    # 尖头朝上的正六边形顶点（从上顶点开始逆时针）
    hex_coords = np.array([
        (cx,                     cy + R),           # top
        (cx + np.sqrt(3)/2 * R, cy + R/2),
        (cx + np.sqrt(3)/2 * R, cy - R/2),
        (cx,                     cy - R),           # bottom
        (cx - np.sqrt(3)/2 * R, cy - R/2),
        (cx - np.sqrt(3)/2 * R, cy + R/2),
    ])

    # 为裁剪六边形创建点/线/面（先用于 extrude）
    hex_pts_cut = [occ.addPoint(x, y, 0.0, lcar) for (x, y) in hex_coords]
    hex_lines_cut = [occ.addLine(hex_pts_cut[i], hex_pts_cut[(i+1) % 6]) for i in range(6)]
    hex_wire_cut = occ.addWire(hex_lines_cut)
    hex_surface_cut = occ.addPlaneSurface([hex_wire_cut])
    occ.synchronize()

    # 用很薄的挤出体积做 boolean
    tiny_h = 0.01 * offset_distance if offset_distance > 0 else 1e-3
    extruded = occ.extrude([(2, hex_surface_cut)], 0, 0, tiny_h)
    hex_vols = [ent for ent in extruded if ent[0] == 3]
    if not hex_vols:
        gmsh.finalize()
        raise RuntimeError("Hex volume extrusion failed; cannot cut Voronoi.")
    hex_vol = hex_vols[0]  # (3, tag)
    occ.synchronize()

    # ------------------ 3. 用六边形体积裁剪 Voronoi 厚壁 ------------------
    print("[CUT] Cutting Voronoi walls with hex volume...")

    cut_objs, _ = occ.intersect(
        all_voro_walls,
        [hex_vol],
        removeObject=True,
        removeTool=True
    )
    occ.synchronize()

    # 裁剪后仅保留 2D 面
    cut_voro_walls = [ent for ent in cut_objs if ent[0] == 2]
    print(f"[CUT] Voronoi walls after cutting: {len(cut_voro_walls)} surfaces.")

    # ------------------ 4. 内六边形边界均匀撒点 ------------------
    print("[DENSE] Adding dense boundary points on inner hexagon...")

    dense_inner_pts = []   # list of list: 6 边，每边一串点 tag

    for i in range(6):
        x1, y1 = hex_coords[i]
        x2, y2 = hex_coords[(i+1) % 6]

        pts_on_edge = []
        for t in np.linspace(0.0, 1.0, N_inner_density):
            xx = x1 + t*(x2 - x1)
            yy = y1 + t*(y2 - y1)
            tag = occ.addPoint(xx, yy, 0.0, lcar)
            pts_on_edge.append(tag)
        dense_inner_pts.append(pts_on_edge)

    occ.synchronize()
    total_dense = sum(len(edge) for edge in dense_inner_pts)
    print(f"[DENSE] Added {total_dense} dense points on inner hexagon boundary.")

    # ------------------ 5. 按“外扩”生成外六边形，并用 dense inner 边构造外墙 ------------------
    print("[HEX] Building offset outer hex wall...")

    # 质心
    hex_center = hex_coords.mean(axis=0)

    # 向外扩 offset_distance（略放大一些）
    outer_hex_coords = []
    for (x, y) in hex_coords:
        v = np.array([x, y]) - hex_center
        nrm = np.linalg.norm(v)
        if nrm == 0:
            outer_hex_coords.append((x, y))
        else:
            v_unit = v / nrm
            outer_hex_coords.append((x + 2*offset_distance * v_unit[0],
                                     y + 2*offset_distance * v_unit[1]))
    outer_hex_coords = np.array(outer_hex_coords)

    # 外六边形点 & 线
    hex_pts_outer = [occ.addPoint(x, y, 0.0, lcar) for (x, y) in outer_hex_coords]
    outer_lines = [occ.addLine(hex_pts_outer[i], hex_pts_outer[(i+1) % 6]) for i in range(6)]

    hex_wall_surfaces = []

    for i in range(6):
        inner_pt_list = dense_inner_pts[i]
        inner_line = occ.addSpline(inner_pt_list)

        # 外边
        outer_line = outer_lines[i]

        # 两条“径向”线：内边起点 → 外点 i， 内边终点 → 外点 i+1
        p_inner_start = inner_pt_list[0]
        p_inner_end   = inner_pt_list[-1]
        p_outer_i     = hex_pts_outer[i]
        p_outer_ip1   = hex_pts_outer[(i+1) % 6]

        l1 = occ.addLine(p_inner_start, p_outer_i)
        l2 = occ.addLine(p_outer_ip1,   p_inner_end)

        wire = occ.addWire([inner_line, l1, outer_line, l2])
        sf   = occ.addPlaneSurface([wire])
        hex_wall_surfaces.append((2, sf))

    occ.synchronize()
    print(f"[HEX] Built {len(hex_wall_surfaces)} hex wall surfaces.")

    # ------------------ 6. fragment unify topology ------------------
    print("[FRAG] Fragmenting Voronoi + hex walls...")

    all_surfs = cut_voro_walls + hex_wall_surfaces

    frag, _ = occ.fragment(all_surfs, [])
    occ.synchronize()

    occ.removeAllDuplicates()
    occ.synchronize()

    print(f"[FRAG] Fragment complete. Total entities: {len(frag)}")

    # ------------------ 7. 生成网格 ------------------
    print("[Mesh] Generating 2D mesh...")
    gmsh.model.mesh.generate(2)

    # 可选多次 refine
    n_refine = 2
    for _ in range(n_refine):
        print("[Mesh] Refining mesh...")
        gmsh.model.mesh.refine()
    
    gmsh.write(mesh_filename + ".msh")
    gmsh.write(mesh_filename + ".vtk")
    gmsh.finalize()

    convert_vtk_to_xdmf(mesh_filename)
    print(f"[DONE] Mesh saved as {mesh_filename}.msh/.vtk/.xdmf")


def generate_2D_voronoi_with_thickness_rect(
    mesh_filename,
    seeds_filename,
    domain_x,
    domain_y,
    lcar=0.01,
    offset_distance=0.01,
    airway_segments=None,
):
    """
    构造 Voronoi 厚壁结构 + 外矩形厚壁，
    并确保所有墙壁之间在拓扑上连通（通过 fragment ）。
    """

    import gmsh
    import numpy as np
    import pickle
    from scipy.spatial import Voronoi

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("Voronoi_Rect_Wall")
    occ = gmsh.model.occ

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)

    # ------------------ Load seeds ------------------
    with open(seeds_filename, "rb") as f:
        seeds = pickle.load(f)
    seeds = np.asarray(seeds)[:, :2]
    print(f"[Voronoi] Loaded {len(seeds)} seeds")

    vor = Voronoi(seeds)

    def inward_polygon(coords):
        centroid = coords.mean(axis=0)
        in_coords = []
        for p in coords:
            dv = centroid - p
            nrm = np.linalg.norm(dv)
            if nrm == 0:
                in_coords.append(p)
            else:
                in_coords.append(p + offset_distance * dv / nrm)
        return np.asarray(in_coords)

    # ------------------ Voronoi walls ------------------
    all_voro_surfs = []

    print("[Voronoi] Building uncut Voronoi walls...")

    for region in vor.regions:
        if not region or -1 in region or len(region) < 3:
            continue

        coords = np.array([vor.vertices[v] for v in region])
        coords_in = inward_polygon(coords)
        n = len(coords)

        out_pts = [occ.addPoint(p[0], p[1], 0, lcar) for p in coords]
        in_pts  = [occ.addPoint(p[0], p[1], 0, lcar) for p in coords_in]

        out_lines = [occ.addLine(out_pts[i], out_pts[(i+1)%n]) for i in range(n)]
        in_lines  = [occ.addLine(in_pts[i],  in_pts[(i+1)%n]) for i in range(n)]

        for i in range(n):

            q1 = coords[i]
            q2 = coords[(i+1)%n]

            # airway filter
            is_airway = False
            if airway_segments:
                for (a1, a2) in airway_segments:
                    if segments_intersect(q1, q2, a1, a2):
                        is_airway = True
                        break
            if is_airway:
                continue

            p1 = out_pts[i]
            p2 = out_pts[(i+1)%n]
            p3 = in_pts[(i+1)%n]
            p4 = in_pts[i]

            l1 = occ.addLine(p1, p4)
            l2 = occ.addLine(p2, p3)

            wire = occ.addWire([out_lines[i], l2, in_lines[i], l1])
            s    = occ.addPlaneSurface([wire])
            all_voro_surfs.append((2, s))

    occ.synchronize()
    print(f"[Voronoi] Built {len(all_voro_surfs)} Voronoi wall surfaces.")

    # ------------------ Rectangle clipping ------------------
    xmin, xmax = 0, domain_x
    ymin, ymax = 0, domain_y
    rect_tag = occ.addRectangle(xmin, ymin, 0, xmax-xmin, ymax-ymin)

    occ.synchronize()

    cut_objs, _ = occ.intersect(
        all_voro_surfs,
        [(2, rect_tag)],
        removeObject=True,
        removeTool=True
    )

    occ.synchronize()

    cut_voro_surfs = [obj for obj in cut_objs if obj[0] == 2]
    print(f"[RECT] After cut: {len(cut_voro_surfs)} Voronoi surfaces remain.")

    # ------------------ Outer rectangle wall ------------------

    inner_coords = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    outer_coords = [
        (xmin-offset_distance, ymin-offset_distance),
        (xmax+offset_distance, ymin-offset_distance),
        (xmax+offset_distance, ymax+offset_distance),
        (xmin-offset_distance, ymax+offset_distance)
    ]

    inner_pts = [occ.addPoint(x,y,0,lcar) for (x,y) in inner_coords]
    outer_pts = [occ.addPoint(x,y,0,lcar) for (x,y) in outer_coords]

    inner_lines = [occ.addLine(inner_pts[i], inner_pts[(i+1)%4]) for i in range(4)]
    outer_lines = [occ.addLine(outer_pts[i], outer_pts[(i+1)%4]) for i in range(4)]

    rect_wall_surfs = []
    for i in range(4):
        p1 = inner_pts[i]
        p2 = inner_pts[(i+1)%4]
        p3 = outer_pts[(i+1)%4]
        p4 = outer_pts[i]

        l1 = occ.addLine(p1,p4)
        l2 = occ.addLine(p2,p3)

        s = occ.addPlaneSurface([occ.addWire([inner_lines[i], l2, outer_lines[i], l1])])
        rect_wall_surfs.append((2,s))

    occ.synchronize()
    print(f"[RECT] Built {len(rect_wall_surfs)} outer-wall surfaces.")

    # ----------------------------------------------------------
    # 🔴 NEW: Fragment ALL surfaces together
    # 让 Voronoi 墙 & 外墙 在拓扑上真正 “连起来”
    # ----------------------------------------------------------

    print("[FRAG] Fragmenting all walls to ensure connectivity...")

    # fragment 接受格式 [(dim,tag)]
    all_surfs = cut_voro_surfs + rect_wall_surfs

    # fragment 保证所有 surfaces 共享节点、边界一致
    frag, _ = occ.fragment(all_surfs, [])
    occ.synchronize()

    # 清理重复拓扑
    occ.removeAllDuplicates()
    occ.synchronize()

    print(f"[FRAG] Fragment done. Total surfaces = {len(frag)}")

    # ------------------ mesh ------------------
    print("[Mesh] Generating mesh...")
    gmsh.model.mesh.generate(2)
    n_refine=2
    for _ in range(n_refine):
        print("[Mesh] Refining mesh...")
        gmsh.model.mesh.refine()

    gmsh.write(mesh_filename + ".msh")
    gmsh.write(mesh_filename + ".vtk")

    gmsh.finalize()
    convert_vtk_to_xdmf(mesh_filename)
    print(f"[DONE] Mesh saved as {mesh_filename}.msh/.vtk/.xdmf")


def generate_hexagonal_seeds_with_periodicity_2D(grid_x, grid_y, domain_x, domain_y, DoI, seeds_filename="hexagonal_seeds-2D-periodic.dat"):
    """
    Generate 2D hexagonal lattice seeds with periodic neighbors and disorder.
    """

    # Hexagonal lattice spacing
    cell_x = domain_x / grid_x
    cell_y = domain_y / (grid_y * np.sqrt(3) / 2)  # Adjust for vertical spacing in hex packing

    seeds = []

    for i in range(grid_x):
        for j in range(grid_y):
            # Stagger every other row (hexagonal packing)
            offset_x = (cell_x / 2) if j % 2 == 1 else 0

            x = i * cell_x + offset_x + (np.random.random() - 0.5) * cell_x * DoI
            y = j * cell_y * np.sqrt(3) / 2 + (np.random.random() - 0.5) * cell_y * DoI

            if 0 <= x <= domain_x and 0 <= y <= domain_y:
                seeds.append([x, y])

    # Add periodic neighbors (8 neighbors: surrounding tiles)
    periodic_neighbors = []
    shifts = [
        (sx * domain_x, sy * domain_y)
        for sx in [-1, 0, 1]
        for sy in [-1, 0, 1]
        if not (sx == 0 and sy == 0)
    ]

    for dx, dy in shifts:
        for s in seeds:
            periodic_neighbors.append([s[0] + dx, s[1] + dy])

    all_seeds = np.array(seeds + periodic_neighbors)

    # Save seeds
    with open(seeds_filename, "wb") as f:
        pickle.dump(all_seeds, f)

    print(f"Generated {len(all_seeds)} seeds (including periodic neighbors) and saved to {seeds_filename}")
    return all_seeds


def plot_airway_tree_with_strahler(points, parents, airway_edges_local, strahler,
                                   title="Airway Tree with Strahler Orders"):
    """
    绘制 airway tree，并在每个节点标注 Strahler 级别。
    points: (N,2) local coordinates
    parents: list of parent indexes
    airway_edges_local: [(i,j)], edges
    strahler: list of Strahler order per node
    """

    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(7,7))

    # ---- 画点 ----
    plt.scatter(x, y, s=20, c="black", alpha=0.7)

    # ---- 画 airway edges ----
    for (i, j) in airway_edges_local:
        xi, yi = points[i]
        xj, yj = points[j]
        plt.plot([xi, xj], [yi, yj], color="gray", linewidth=1.0, alpha=0.6)

    # ---- 标注 Strahler 级别 ----
    for i in range(len(points)):
        plt.text(points[i,0], points[i,1],
                 str(strahler[i]),
                 fontsize=10, color="red", ha="center", va="center")

    # ---- 标注 root ----
    root = [i for i,p in enumerate(parents) if p < 0][0]
    plt.scatter([points[root,0]], [points[root,1]], s=80, color="blue", label="Root")

    plt.title(title)
    plt.axis("equal")
    plt.show()

def plot_airway_segments(seeds, airway_segments):
    plt.figure(figsize=(6,6))

    # 画 seeds（灰色）
    seeds = np.array(seeds)
    plt.scatter(seeds[:,0], seeds[:,1], s=5, color="gray", label="Seeds")

    # 画 airway line segments（红色）
    for (a1, a2) in airway_segments:
        x = [a1[0], a2[0]]
        y = [a1[1], a2[1]]
        plt.plot(x, y, color="red", linewidth=1.2)

    plt.axis("equal")
    plt.title("Airway Tree Segments")
    plt.show()


# Parameters
epsilon = 0.0
domain_x, domain_y = 1.0 + epsilon, 1.0 + epsilon
grid_x, grid_y = 6, 6
DoI = 0.0  # Degree of Irregularity (0 = perfect lattice)
thickness = 0.02
lcar = 0.01

seeds_filename = "seeds-2D-periodic.dat"
mesh_filename = "voronoi_2D_thick"

crop_window = (
    0.142857142857,
    0.857142857143,
    (0.880670889045+0.785995777621)/2,
    (0.214004222379+0.119329110955)/2
)

# Step 1: Generate 2D Seeds with Periodicity
seeds = generate_hexagonal_seeds_with_periodicity_2D(
    grid_x, grid_y, domain_x, domain_y, DoI, seeds_filename
)

vor = Voronoi(seeds)




#plot_airway_segments(seeds, airway_segments)
hex_coords = generate_hexagon_coords(domain_x=1.0, domain_y=1.0)
parents, edges, ltg, phys = generate_airway_tree_hex(seeds, hex_coords)
#plot_airway_tree(phys_seeds, parents, airway_edges_local)
strahler = compute_strahler_orders(parents)
export_airway_with_strahler(phys, parents, strahler)

# plot_airway_tree_with_strahler(
#      phys,
#      parents,
#      edges,
#      strahler,
#      title="Hex Acinar Airway Tree with Strahler Number"
#  )
# 这里 airway_ridges 是你用 Voronoi(seeds) + airway tree 生成的 ridge index 集合

# airway_ridges = map_airway_to_ridges(vor, local_to_global, airway_edges_local)
# region_ridge_map = build_region_ridge_map(vor)

generate_2D_voronoi_with_thickness_rect(
    mesh_filename="with_airway",
    seeds_filename="seeds-2D-periodic.dat",
    domain_x=1.0, domain_y=1.0,
    lcar=0.01,
    offset_distance=0.01,
    airway_segments=None,   # 最核心参数！
)

