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
generate_2D_voronoi_with_thickness_rect(
    mesh_filename="with_airway",
    seeds_filename="seeds-2D-periodic.dat",
    domain_x=1.0, domain_y=1.0,
    lcar=0.01,
    offset_distance=0.01,
    airway_segments=None,   # 最核心参数！
)

