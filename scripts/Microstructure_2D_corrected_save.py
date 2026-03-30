import numpy as np
import pickle
from scipy.spatial import Voronoi
import gmsh
import random
import pandas as pd
import os

def setPeriodic(dim,coord, xmin, ymin, zmin, xmax, ymax, zmax, e):
    # From https://gitlab.onelab.info/gmsh/gmsh/-/issues/744

    smin = gmsh.model.getEntitiesInBoundingBox(
        xmin - e,
        ymin - e,
        zmin - e,
        (xmin + e) if (coord == 0) else (xmax + e),
        (ymin + e) if (coord == 1) else (ymax + e),
        (zmin + e) if (coord == 2) else (zmax + e),
        2)
    dx = (xmax - xmin) if (coord == 0) else 0
    dy = (ymax - ymin) if (coord == 1) else 0
    dz = (zmax - zmin) if (coord == 2) else 0

    for i in smin:
        bb = gmsh.model.getBoundingBox(i[0], i[1])
        bbe = [bb[0] - e + dx, bb[1] - e + dy, bb[2] - e + dz,
               bb[3] + e + dx, bb[4] + e + dy, bb[5] + e + dz]
        smax = gmsh.model.getEntitiesInBoundingBox(bbe[0], bbe[1], bbe[2],
                                                   bbe[3], bbe[4], bbe[5])
        for j in smax:
            bb2 = list(gmsh.model.getBoundingBox(j[0], j[1]))
            bb2[0] -= dx; bb2[1] -= dy; bb2[2] -= dz
            bb2[3] -= dx; bb2[4] -= dy; bb2[5] -= dz
            if ((abs(bb2[0] - bb[0]) < e) and (abs(bb2[1] - bb[1]) < e) and
                (abs(bb2[2] - bb[2]) < e) and (abs(bb2[3] - bb[3]) < e) and
                (abs(bb2[4] - bb[4]) < e) and (abs(bb2[5] - bb[5]) < e)):
                print("Found periodic pair:", i, j)
                gmsh.model.mesh.setPeriodic(dim, [j[1]], [i[1]], [1, 0, 0, dx,\
                                                                0, 1, 0, dy,\
                                                                0, 0, 1, dz,\
                                                                0, 0, 0, 1 ])


def convert_msh_to_xml(mesh_filename):

    os.system("gmsh -2 -o " + mesh_filename + ".msh -format msh22 " + mesh_filename + ".msh")
    os.system("dolfin-convert " + mesh_filename + ".msh " + mesh_filename + ".xml")

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

def generate_hexagonal_seeds_with_periodicity_2D(grid_x, grid_y, domain_x, domain_y, DoI, seeds_filename="hexagonal_seeds-2D-periodic.dat"):
    """
    Generate 2D hexagonal lattice seeds with periodic neighbors and disorder.
    """
    cell_x = domain_x / grid_x
    cell_y = domain_y / (grid_y * np.sqrt(3) / 2)  

    seeds = []

    for i in range(grid_x):
        for j in range(grid_y):
            offset_x = (cell_x / 2) if j % 2 == 1 else 0

            x = i * cell_x + offset_x + (np.random.random() - 0.5) * cell_x * DoI
            y = j * cell_y * np.sqrt(3) / 2 + (np.random.random() - 0.5) * cell_y * DoI

            if 0 <= x <= domain_x and 0 <= y <= domain_y:
                seeds.append([x, y])

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


def generate_2D_voronoi_with_thickness(mesh_filename, seeds_filename, domain_x, domain_y, lcar, offset_distance, pore_shape="hex"):  # or "circle"
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("Voronoi_2D_Thickened")
    occ = gmsh.model.occ
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 80) 

    with open(seeds_filename, "rb") as f:
        seeds = pickle.load(f)
    seeds = np.array(seeds)[:, :2]

    vor = Voronoi(seeds)
    all_surface_tags = []

    for region in vor.regions:
        if not region or -1 in region or len(region) < 3:
            continue
        try:
            coords = np.array([vor.vertices[v] for v in region])
        except IndexError:
            continue
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] > domain_x) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] > domain_y):
            continue

        centroid = coords.mean(axis=0)
        inward_coords = []
        
        for v in coords:
            dir_vec = centroid - v
            dir_vec /= np.linalg.norm(dir_vec)
            inward_coords.append(v + offset_distance * dir_vec)

        if pore_shape == "hex":
            outer_tags = [occ.addPoint(p[0], p[1], 0, lcar) for p in coords]
            inner_tags = [occ.addPoint(p[0], p[1], 0, lcar) for p in inward_coords]

            outer_lines = [occ.addLine(outer_tags[i], outer_tags[(i + 1) % len(outer_tags)])
                        for i in range(len(outer_tags))]
            inner_lines = [occ.addLine(inner_tags[i], inner_tags[(i + 1) % len(inner_tags)])
                        for i in range(len(inner_tags))]

            for i in range(len(outer_tags)):
                p1 = outer_tags[i]
                p2 = outer_tags[(i + 1) % len(outer_tags)]
                p3 = inner_tags[(i + 1) % len(inner_tags)]
                p4 = inner_tags[i]
                l1 = occ.addLine(p1, p4)
                l2 = occ.addLine(p2, p3)
                l3 = inner_lines[i]
                l4 = outer_lines[i]
                wire = occ.addWire([l4, l2, l3, l1])
                surface = occ.addPlaneSurface([wire])
                all_surface_tags.append((2, surface))
        elif pore_shape == "circle":
            # outer surface
            outer_tags = [occ.addPoint(p[0], p[1], 0, lcar) for p in coords]
            outer_lines = [occ.addLine(outer_tags[i], outer_tags[(i + 1) % len(outer_tags)])
                        for i in range(len(outer_tags))]
            outer_loop = occ.addCurveLoop(outer_lines)
            outer_surf = occ.addPlaneSurface([outer_loop])
            #gmsh.fltk.run()

            # circle radius from inward points (circumcircle-like)
            r = float(np.mean(np.linalg.norm(inward_coords - centroid, axis=1)))

            disk = occ.addDisk(float(centroid[0]), float(centroid[1]), 0.0, r, r)
            occ.synchronize()
            cut_out, _ = occ.cut([(2, outer_surf)], [(2, disk)],
                                removeObject=True, removeTool=True)
            occ.synchronize()
            #gmsh.fltk.run() 

            all_surface_tags.extend(cut_out)

        else:
            raise ValueError("pore_shape must be 'hex' or 'circle'")
    gmsh.fltk.run()

    occ.synchronize()
    if all_surface_tags:
        fused, _ = occ.fuse(all_surface_tags, all_surface_tags)
        occ.synchronize()
    gmsh.fltk.run()
    occ.removeAllDuplicates()
    occ.synchronize()
    #gmsh.fltk.run()

    cell_x = domain_x / grid_x
    rve_width = cell_x
    rve_height = np.sqrt(3) * cell_x 
    center_x = domain_x / 2
    center_y = domain_y / 2
    rve_xmin = center_x - rve_width / 2
    rve_ymin = center_y - rve_height / 2
    box_tag = occ.addRectangle(rve_xmin, rve_ymin, 0, rve_width, rve_height)
    occ.synchronize()
    gmsh.fltk.run()
    
    intersected, _ = occ.intersect(fused, [(2, box_tag)], removeObject=True, removeTool=True)

    occ.synchronize()
    gmsh.fltk.run()
    A_solid = 0.0
    for dim, tag in intersected:
        A_solid += gmsh.model.occ.getMass(dim, tag)
    #print(f"Solid area fraction in RVE: {A_solid/(rve_width*rve_height):.4f}")
    # filename suffix (avoid "." in filenames if you like)
    A_rve = rve_width * rve_height
    porosity = 1.0 - A_solid / A_rve
    phi_str = f"{porosity:.4f}"             # e.g. 0.5123
    phi_str_safe = phi_str.replace(".", "p") # e.g. 0p5123

    mesh_base = f"{mesh_filename}_phi{phi_str_safe}"
    edges = gmsh.model.getEntities(1)
    for dim, tag in edges:
        pts = gmsh.model.getBoundary([(dim, tag)], oriented=False)
        gmsh.model.mesh.setSize(pts, lcar)
    occ.synchronize()
    zmin = 0; zmax = 0

    dx = rve_width
    dy = rve_height
    Lref = max(dx, dy)
    e = 1e-6 * Lref   # 先用这个，别用 1e-10

    print("Using e =", e)
    print("RVE xmin/xmax/ymin/ymax =",
        rve_xmin, rve_xmin + rve_width, rve_ymin, rve_ymin + rve_height)

    # 看看 xmin 侧边界能找到多少条曲线
    smin_test = gmsh.model.getEntitiesInBoundingBox(
        rve_xmin - e, rve_ymin - e, -e,
        rve_xmin + e, rve_ymin + rve_height + e, e,
        1
    )
    print("Curves near xmin:", smin_test)

    smax_test = gmsh.model.getEntitiesInBoundingBox(
        (rve_xmin + rve_width) - e, rve_ymin - e, -e,
        (rve_xmin + rve_width) + e, rve_ymin + rve_height + e, e,
        1
    )
    print("Curves near xmax:", smax_test)
    
   
    setPeriodic(dim=1,coord=0, xmin=rve_xmin, ymin=rve_ymin, zmin=zmin, xmax=rve_xmin + rve_width, ymax=rve_ymin + rve_height, zmax=zmax, e=1e-10)
    setPeriodic(dim=1,coord=1, xmin=rve_xmin, ymin=rve_ymin, zmin=zmin, xmax=rve_xmin + rve_width, ymax=rve_ymin + rve_height, zmax=zmax, e=1e-10)
    gmsh.model.mesh.generate(2)
    #gmsh.fltk.run()
    gmsh.write(mesh_base + "_RVE.msh")
    gmsh.write(mesh_base + "_RVE.vtk")

    def has_periodic_block(msh_path: str) -> bool:
        with open(msh_path, "r") as f:
            return "$Periodic" in f.read()

    print("Has $Periodic:", has_periodic_block(mesh_base + "_RVE.msh"))

    gmsh.finalize()
    convert_vtk_to_xdmf(mesh_base + "_RVE")

# Min

# offset_values = np.array([
# 0.13, 0.1229, 0.120, 0.1159, 0.1120, 0.1090, 0.1022,
# 0.0955, 0.0889, 0.0824, 0.0761, 0.0698,
# 0.0637, 0.05, 0.0577, 0.0518, 0.0460, 0.04, 0.03, 0.022, 0.025, 0.022, 0.02,
# ])

offset_values = np.array([
0.05
])


epsilon = 0.0
domain_x, domain_y = 1.0 + epsilon, np.sqrt(3)/2 + epsilon
grid_x, grid_y = 4, 4
DoI = 0.0
lcar = 0.005

seeds_filename = "seeds-2D-periodic.dat"

a = domain_x / grid_x

#output_folder = "voronoi_2D_batch_hex"
output_folder = "voronoi_2D_batch_circle"
os.makedirs(output_folder, exist_ok=True)

generate_hexagonal_seeds_with_periodicity_2D(
    grid_x, grid_y, domain_x, domain_y, DoI, seeds_filename
)

for offset in offset_values:


    mesh_basename = os.path.join(
        output_folder,
        f"mesh"
    )


    generate_2D_voronoi_with_thickness(
        mesh_basename,
        seeds_filename,
        domain_x,
        domain_y,
        lcar,
        offset_distance=offset,
        pore_shape="circle"
    )