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

def convert_msh_to_xdmf_with_domains(mesh_filename, dim=2):
    import meshio
    import numpy as np

    msh = meshio.read(mesh_filename + ".msh")

    if dim == 2:
        cell_type = "triangle"
    elif dim == 3:
        cell_type = "tetra"
    else:
        raise ValueError("dim must be 2 or 3")

    # points
    points = msh.points[:, :2] if dim == 2 else msh.points

    # cells
    cells = msh.get_cells_type(cell_type)
    if len(cells) == 0:
        raise ValueError(f"No cells of type '{cell_type}' found in {mesh_filename}.msh")

    # main mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[(cell_type, cells)]
    )
    meshio.write(mesh_filename + ".xdmf", mesh)

    # physical group data for domains
    physical = None
    if "gmsh:physical" in msh.cell_data_dict:
        if cell_type in msh.cell_data_dict["gmsh:physical"]:
            physical = msh.cell_data_dict["gmsh:physical"][cell_type]

    if physical is None:
        print("[Warning] No gmsh:physical data found for cell domains.")
        return

    domains = meshio.Mesh(
        points=points,
        cells=[(cell_type, cells)],
        cell_data={"domains": [np.array(physical, dtype=np.int32)]}
    )
    meshio.write(mesh_filename + "_domains.xdmf", domains)

    print(f"[OK] Wrote {mesh_filename}.xdmf")
    print(f"[OK] Wrote {mesh_filename}_domains.xdmf")

def build_row_col_map_for_selected_cells(cell_metadata, selected_indices, y_tol=None):
    """
    Build (row, col) -> cell index using only selected cells.

    Row numbering:
        1 = top row
    Col numbering:
        left to right within each row
    """
    if len(selected_indices) == 0:
        raise RuntimeError("No selected cells provided.")

    centers = np.array([cell_metadata[idx]["center"] for idx in selected_indices], dtype=float)
    y_vals = centers[:, 1]

    if y_tol is None:
        y_sorted = np.sort(y_vals)
        if len(y_sorted) > 1:
            dy = np.diff(y_sorted)
            dy = dy[dy > 1e-12]
            if len(dy) > 0:
                y_tol = 0.5 * np.median(dy)
            else:
                y_tol = 1e-6
        else:
            y_tol = 1e-6

    # sort selected cells by descending y
    local_order = np.argsort(-y_vals)
    rows = []
    current_row = [local_order[0]]

    for loc in local_order[1:]:
        y_ref = y_vals[current_row[0]]
        if abs(y_vals[loc] - y_ref) < y_tol:
            current_row.append(loc)
        else:
            rows.append(current_row)
            current_row = [loc]
    rows.append(current_row)

    cell_rc_map = {}
    for i_row, row_local_ids in enumerate(rows, start=1):
        row_local_ids = sorted(row_local_ids, key=lambda k: centers[k, 0])
        for i_col, loc in enumerate(row_local_ids, start=1):
            global_idx = selected_indices[loc]
            cell_rc_map[(i_row, i_col)] = global_idx

    return cell_rc_map

def filter_full_cells_in_rect(cell_metadata, xmin, xmax, ymin, ymax, tol=1e-9):
    """
    Return indices of cells whose original Voronoi polygon is fully inside the rectangle.
    """
    kept = []

    for idx, cell in enumerate(cell_metadata):
        verts = np.array(cell["vertices"], dtype=float)

        inside = (
            np.all(verts[:, 0] >= xmin - tol) and
            np.all(verts[:, 0] <= xmax + tol) and
            np.all(verts[:, 1] >= ymin - tol) and
            np.all(verts[:, 1] <= ymax + tol)
        )

        if inside:
            kept.append(idx)

    return kept

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

import gmsh
import numpy as np
import pickle
import json
import math
from scipy.spatial import Voronoi


def angle_diff_deg(a, b):
    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return abs(d)


def classify_edge_direction(theta):
    """
    Classification for pointy-top hexagons / Voronoi-like cells.

    Input
    -----
    theta : float
        angle in radians from atan2(dy, dx)

    Returns
    -------
    side_name : str
        one of:
        right, top_right, top_left, left, bottom_left, bottom_right
    side_id : int
        1 -> right
        2 -> top_right
        3 -> top_left
        4 -> left
        5 -> bottom_left
        6 -> bottom_right
    """
    deg = math.degrees(theta)

    while deg >= 180:
        deg -= 360
    while deg < -180:
        deg += 360

    directions = {
        "right":        0.0,
        "top_right":    60.0,
        "top_left":    120.0,
        "left":       180.0,
        "bottom_left": -120.0,
        "bottom_right": -60.0,
    }

    side_ids = {
        "right": 1,
        "top_right": 2,
        "top_left": 3,
        "left": 4,
        "bottom_left": 5,
        "bottom_right": 6,
    }

    best_label = None
    best_diff = 1e30
    for label, ref_deg in directions.items():
        d = angle_diff_deg(deg, ref_deg)
        if d < best_diff:
            best_diff = d
            best_label = label

    return best_label, side_ids[best_label]

def extract_polygon_wall_edges_with_labels(coords, coords_in):
    center = coords.mean(axis=0)
    xc, yc = center

    edges = []
    n = len(coords)

    for i in range(n):
        p1_out = coords[i]
        p2_out = coords[(i + 1) % n]

        p1_in = coords_in[i]
        p2_in = coords_in[(i + 1) % n]

        xm_out = 0.5 * (p1_out[0] + p2_out[0])
        ym_out = 0.5 * (p1_out[1] + p2_out[1])

        xm_in = 0.5 * (p1_in[0] + p2_in[0])
        ym_in = 0.5 * (p1_in[1] + p2_in[1])

        xm_wall = 0.5 * (xm_out + xm_in)
        ym_wall = 0.5 * (ym_out + ym_in)

        theta = math.atan2(ym_out - yc, xm_out - xc)
        side, side_id = classify_edge_direction(theta)

        edges.append({
            "local_edge_index": int(i),

            "p1_out": [float(p1_out[0]), float(p1_out[1])],
            "p2_out": [float(p2_out[0]), float(p2_out[1])],
            "midpoint_out": [float(xm_out), float(ym_out)],

            "p1_in": [float(p1_in[0]), float(p1_in[1])],
            "p2_in": [float(p2_in[0]), float(p2_in[1])],
            "midpoint_in": [float(xm_in), float(ym_in)],

            "midpoint_wall": [float(xm_wall), float(ym_wall)],

            "angle": float(theta),
            "side": side,
            "side_id": int(side_id),
        })

    return [float(xc), float(yc)], edges

def generate_2D_voronoi_with_thickness_rect(
    mesh_filename,
    seeds_filename,
    domain_x,
    domain_y,
    lcar=0.01,
    offset_distance=0.01,
    patch_specs=None,
    patch_offset=0.02,
    patch_radius=0.01,
    n_refine=2,
    save_metadata=True,
):
    import json
    import pickle
    import numpy as np
    import gmsh
    from scipy.spatial import Voronoi

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("Voronoi_Rect_Wall")
    occ = gmsh.model.occ

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

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
                in_coords.append(p.copy())
            else:
                in_coords.append(p + offset_distance * dv / nrm)
        return np.asarray(in_coords)

    # ------------------ Build Voronoi wall surfaces + metadata ------------------
    all_voro_surfs = []
    cell_metadata = []

    print("[Voronoi] Building uncut Voronoi walls...")

    for i_seed, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]

        # skip open / invalid regions
        if not region or (-1 in region) or (len(region) < 3):
            continue

        coords = np.array([vor.vertices[v] for v in region], dtype=float)
        coords_in = inward_polygon(coords)
        n = len(coords)

        # metadata from outer polygon
        center, edge_info = extract_polygon_wall_edges_with_labels(coords, coords_in)

        cell_id = len(cell_metadata)
        cell_metadata.append({
            "cell_id": int(cell_id),
            "seed_id": int(i_seed),
            "seed": [float(seeds[i_seed, 0]), float(seeds[i_seed, 1])],
            "center": center,
            "vertices": [[float(x), float(y)] for x, y in coords],
            "edges": edge_info,
        })

        # build thick wall strips between outer and inner polygon
        out_pts = [occ.addPoint(float(p[0]), float(p[1]), 0, lcar) for p in coords]
        in_pts  = [occ.addPoint(float(p[0]), float(p[1]), 0, lcar) for p in coords_in]

        out_lines = [occ.addLine(out_pts[i], out_pts[(i + 1) % n]) for i in range(n)]
        in_lines  = [occ.addLine(in_pts[i],  in_pts[(i + 1) % n]) for i in range(n)]

        for i in range(n):
            p1 = out_pts[i]
            p2 = out_pts[(i + 1) % n]
            p3 = in_pts[(i + 1) % n]
            p4 = in_pts[i]

            l1 = occ.addLine(p1, p4)
            l2 = occ.addLine(p2, p3)

            wire = occ.addWire([out_lines[i], l2, in_lines[i], l1])
            s = occ.addPlaneSurface([wire])
            all_voro_surfs.append((2, s))

    occ.synchronize()
    print(f"[Voronoi] Built {len(all_voro_surfs)} Voronoi wall surfaces.")
    print(f"[Voronoi] Saved metadata for {len(cell_metadata)} cells before clipping.")

    # ------------------ Rectangle clipping ------------------
    xmin, xmax = 0.0, float(domain_x)
    ymin, ymax = 0.0, float(domain_y)
    rect_tag = occ.addRectangle(xmin, ymin, 0, xmax - xmin, ymax - ymin)

    occ.synchronize()

    print("[RECT] Intersecting Voronoi walls with rectangle...")
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
    inner_coords = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]
    outer_coords = [
        (xmin - offset_distance, ymin - offset_distance),
        (xmax + offset_distance, ymin - offset_distance),
        (xmax + offset_distance, ymax + offset_distance),
        (xmin - offset_distance, ymax + offset_distance),
    ]

    inner_pts = [occ.addPoint(float(x), float(y), 0, lcar) for (x, y) in inner_coords]
    outer_pts = [occ.addPoint(float(x), float(y), 0, lcar) for (x, y) in outer_coords]

    inner_lines = [occ.addLine(inner_pts[i], inner_pts[(i + 1) % 4]) for i in range(4)]
    outer_lines = [occ.addLine(outer_pts[i], outer_pts[(i + 1) % 4]) for i in range(4)]

    rect_wall_surfs = []
    for i in range(4):
        p1 = inner_pts[i]
        p2 = inner_pts[(i + 1) % 4]
        p3 = outer_pts[(i + 1) % 4]
        p4 = outer_pts[i]

        l1 = occ.addLine(p1, p4)
        l2 = occ.addLine(p2, p3)

        wire = occ.addWire([inner_lines[i], l2, outer_lines[i], l1])
        s = occ.addPlaneSurface([wire])
        rect_wall_surfs.append((2, s))

    occ.synchronize()
    print(f"[RECT] Built {len(rect_wall_surfs)} outer-wall surfaces.")

    # ------------------ Fragment all wall surfaces ------------------
    print("[FRAG] Fragmenting all walls to ensure connectivity...")
    all_surfs = cut_voro_surfs + rect_wall_surfs

    frag, _ = occ.fragment(all_surfs, [])
    occ.synchronize()

    occ.removeAllDuplicates()
    occ.synchronize()

    all_wall_surfs = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
    print(f"[FRAG] Fragment done. Total surfaces = {len(all_wall_surfs)}")

    # ------------------ Optional patch step ------------------
    inlet_surfs = []
    outlet_surfs = []

    if patch_specs is not None and len(patch_specs) > 0:
        print("[PATCH] Adding inlet/outlet patches...")

        # build row/col map from cell metadata
        full_cell_indices = filter_full_cells_in_rect(
            cell_metadata,
            xmin=0.0,
            xmax=domain_x,
            ymin=0.0,
            ymax=domain_y
        )

        cell_rc_map = build_row_col_map_for_selected_cells(
            cell_metadata,
            selected_indices=full_cell_indices,
            y_tol=0.03
        )

        # compute patch centers
        patch_data = build_patch_centers_from_specs(
            cell_metadata=cell_metadata,
            cell_rc_map=cell_rc_map,
            patch_specs=patch_specs,
            offset=patch_offset
        )

        # create disk surfaces
        disk_entities = []
        for p in patch_data:
            x, y = p["patch_center"]
            dtag = occ.addDisk(float(x), float(y), 0.0, patch_radius, patch_radius)
            disk_entities.append((2, dtag))

        occ.synchronize()

        # fragment all current surfaces with disks
        all_current_surfs = gmsh.model.getEntities(2)
        frag2, _ = occ.fragment(all_current_surfs, disk_entities)
        occ.synchronize()

        occ.removeAllDuplicates()
        occ.synchronize()

        print("[PATCH] Fragment with patches done.")

        # identify which resulting surfaces correspond to inlet / outlet disks
        all_surfs_after_patch = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]

        for p in patch_data:
            s_tag, dist = find_surface_closest_to_point(
                all_surfs_after_patch,
                p["patch_center"]
            )

            if s_tag is None:
                raise RuntimeError(f"Could not identify patch surface for {p}")

            print(f"[PATCH] {p['kind']} patch at {p['patch_center']} -> surface {s_tag} (dist={dist:.4e})")

            if p["kind"] == "inlet":
                inlet_surfs.append(s_tag)
            elif p["kind"] == "outlet":
                outlet_surfs.append(s_tag)
            else:
                raise RuntimeError(f"Unknown patch kind: {p['kind']}")

    # ------------------ Bulk surfaces ------------------
    all_final_surfs = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
    patch_surfs_set = set(inlet_surfs + outlet_surfs)
    bulk_surfs = [s for s in all_final_surfs if s not in patch_surfs_set]

    print(f"[PG] bulk_surfs   = {len(bulk_surfs)}")
    print(f"[PG] inlet_surfs  = {len(inlet_surfs)}")
    print(f"[PG] outlet_surfs = {len(outlet_surfs)}")

    # ------------------ Physical groups ------------------
    if len(bulk_surfs) > 0:
        pg_bulk = gmsh.model.addPhysicalGroup(2, bulk_surfs)
        gmsh.model.setPhysicalName(2, pg_bulk, "bulk_region")

    if len(inlet_surfs) > 0:
        pg_in = gmsh.model.addPhysicalGroup(2, inlet_surfs)
        gmsh.model.setPhysicalName(2, pg_in, "inlet_region")

    if len(outlet_surfs) > 0:
        pg_out = gmsh.model.addPhysicalGroup(2, outlet_surfs)
        gmsh.model.setPhysicalName(2, pg_out, "outlet_region")

    # ------------------ Save metadata ------------------
    if save_metadata:
        kept_indices = filter_full_cells_in_rect(
            cell_metadata,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax
        )

        cell_metadata_rect = [cell_metadata[i] for i in kept_indices]

        metadata_pkl = mesh_filename + "_cell_metadata.pkl"
        metadata_json = mesh_filename + "_cell_metadata.json"

        with open(metadata_pkl, "wb") as f:
            pickle.dump(cell_metadata_rect, f)

        with open(metadata_json, "w") as f:
            json.dump(cell_metadata_rect, f, indent=2)

        print(f"[SAVE] Saved {len(cell_metadata_rect)} cells inside rectangle.")
        print(f"[SAVE] Cell metadata saved to {metadata_pkl}")
        print(f"[SAVE] Cell metadata saved to {metadata_json}")

    # ------------------ Mesh ------------------
    print("[Mesh] Generating mesh...")
    gmsh.model.mesh.generate(2)

    for _ in range(n_refine):
        print("[Mesh] Refining mesh...")
        gmsh.model.mesh.refine()

    gmsh.write(mesh_filename + ".msh")
    gmsh.finalize()

    convert_msh_to_xdmf_with_domains(mesh_filename, dim=2)
    print(f"[DONE] Mesh saved as {mesh_filename}.msh/.xdmf/.xdmf_domains")

def find_surface_closest_to_point(surface_tags, point):
    x0, y0 = point
    best_tag = None
    best_dist = 1e30

    for s in surface_tags:
        xg, yg, zg = gmsh.model.occ.getCenterOfMass(2, s)
        d = ((xg - x0)**2 + (yg - y0)**2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_tag = s

    return best_tag, best_dist

def side_to_ref_angle_deg(side):
    directions = {
        "right":         0.0,
        "top_right":    60.0,
        "top_left":    120.0,
        "left":        180.0,
        "bottom_left": -120.0,
        "bottom_right": -60.0,
    }
    if side not in directions:
        raise ValueError(f"Unknown side: {side}")
    return directions[side]

def get_edge_by_side(cell, side):
    cx, cy = cell["center"]
    ref_deg = side_to_ref_angle_deg(side)

    best_edge = None
    best_diff = 1e30

    for e in cell["edges"]:
        mx, my = e["midpoint_wall"]
        theta = math.atan2(my - cy, mx - cx)
        deg = math.degrees(theta)

        while deg >= 180.0:
            deg -= 360.0
        while deg < -180.0:
            deg += 360.0

        diff = angle_diff_deg(deg, ref_deg)

        if diff < best_diff:
            best_diff = diff
            best_edge = e

    if best_edge is None:
        raise RuntimeError(
            f"No edge found for side={side} in cell_id={cell['cell_id']}"
        )

    return best_edge

def compute_patch_center_from_cell_edge(cell, side, offset_along_wall=0.0):
    """
    Place the patch center inside the wall strip associated with one Voronoi edge.

    Parameters
    ----------
    cell : dict
    side : str or int
    offset_along_wall : float
        optional small shift along the edge tangent direction

    Returns
    -------
    patch_center : tuple
    edge : dict
    """
    edge = get_edge_by_side(cell, side)

    xw, yw = edge["midpoint_wall"]

    # tangent direction from outer edge
    x1, y1 = edge["p1_out"]
    x2, y2 = edge["p2_out"]

    tx = x2 - x1
    ty = y2 - y1
    tn = (tx**2 + ty**2)**0.5

    if tn < 1e-14:
        raise RuntimeError(f"Degenerate edge for cell_id={cell['cell_id']}")

    tx /= tn
    ty /= tn

    xc_patch = xw + offset_along_wall * tx
    yc_patch = yw + offset_along_wall * ty

    return (xc_patch, yc_patch), edge

def build_patch_centers_from_specs(cell_metadata, cell_rc_map, patch_specs, offset):
    """
    patch_specs example:
    [
        {"row": 3, "col": 3, "side": "top_left", "kind": "inlet"},
        {"row": 3, "col": 3, "side": "bottom_left", "kind": "outlet"},
        {"row": 3, "col": 3, "side": "bottom_right", "kind": "outlet"},
    ]
    """
    patch_data = []

    for spec in patch_specs:
        row = spec["row"]
        col = spec["col"]
        side = spec["side"]
        kind = spec["kind"]

        idx = cell_rc_map[(row, col)]
        cell = cell_metadata[idx]

        center, edge = compute_patch_center_from_cell_edge(cell, side, offset)

        patch_data.append({
            "row": row,
            "col": col,
            "cell_index": idx,
            "cell_id": cell["cell_id"],
            "side": side,
            "kind": kind,
            "edge_midpoint": edge["midpoint_wall"],
            "patch_center": center,
        })

    return patch_data

def generate_hexagonal_seeds_with_periodicity_2D(
    grid_x, grid_y, domain_x, domain_y, DoI,
    seeds_filename="hexagonal_seeds-2D-periodic.dat"
):
    import numpy as np
    import pickle

    cell_x = domain_x / grid_x
    y_spacing = domain_y / grid_y

    seeds = []

    for j in range(grid_y):
        for i in range(grid_x):
            offset_x = 0.5 * cell_x if (j % 2 == 1) else 0.0

            x0 = i * cell_x + offset_x
            y0 = j * y_spacing

            x = x0 + (np.random.random() - 0.5) * cell_x * DoI
            y = y0 + (np.random.random() - 0.5) * y_spacing * DoI

            # periodic wrap instead of discard
            x = x % domain_x
            y = y % domain_y

            seeds.append([x, y])

    seeds = np.array(seeds, dtype=float)

    periodic_neighbors = []
    shifts = [
        (sx * domain_x, sy * domain_y)
        for sx in [-1, 0, 1]
        for sy in [-1, 0, 1]
        if not (sx == 0 and sy == 0)
    ]

    for dx, dy in shifts:
        periodic_neighbors.append(seeds + np.array([dx, dy]))

    all_seeds = np.vstack([seeds] + periodic_neighbors)

    with open(seeds_filename, "wb") as f:
        pickle.dump(all_seeds, f)

    print(f"Generated {len(seeds)} base seeds")
    print(f"Generated {len(all_seeds)} seeds including periodic neighbors")
    print(f"Saved to {seeds_filename}")

    return all_seeds
# Parameters
epsilon = 0.0
domain_x, domain_y = 1.0 + epsilon, 1.0 + epsilon
#grid_x, grid_y = 6, 6
grid_x, grid_y = 12, 12
DoI = 0.3  # Degree of Irregularity (0 = perfect lattice)
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

# patch_specs =     [
#         {"row": 1, "col": 2, "side": "top_left", "kind": "inlet"},
#         {"row": 2, "col": 4, "side": "top_left", "kind": "outlet"},
#         {"row": 4, "col": 1, "side": "bottom_left", "kind": "outlet"},
#         {"row": 5, "col": 4, "side": "bottom_right", "kind": "inlet"},
#     ]

#top/bottom rows
patch_specs = [
    {"row": 1,  "col": 2,  "side": "top_left",     "kind": "inlet"},
    {"row": 1,  "col": 4,  "side": "top_left",     "kind": "inlet"},
    {"row": 1,  "col": 6,  "side": "top_left",     "kind": "inlet"},
    {"row": 1,  "col": 8,  "side": "top_left",     "kind": "inlet"},
    {"row": 1,  "col": 10, "side": "top_left",     "kind": "inlet"},

    {"row": 11, "col": 2,  "side": "bottom_right", "kind": "outlet"},
    {"row": 11, "col": 4,  "side": "bottom_right", "kind": "outlet"},
    {"row": 11, "col": 6,  "side": "bottom_right", "kind": "outlet"},
    {"row": 11, "col": 8,  "side": "bottom_right", "kind": "outlet"},
    {"row": 11, "col": 10, "side": "bottom_right", "kind": "outlet"},
]


## center
# patch_specs = [
#     {"row": 5, "col": 6, "side": "top_left",     "kind": "inlet"},
#     {"row": 5, "col": 7, "side": "top_right",    "kind": "inlet"},
#     {"row": 6, "col": 6, "side": "bottom_left",  "kind": "inlet"},
#     {"row": 6, "col": 7, "side": "bottom_right", "kind": "inlet"},

#     {"row": 1,  "col": 3,  "side": "top_left",     "kind": "outlet"},
#     {"row": 1,  "col": 9,  "side": "top_left",     "kind": "outlet"},
#     {"row": 3,  "col": 1,  "side": "bottom_left",  "kind": "outlet"},
#     {"row": 8,  "col": 1,  "side": "bottom_left",  "kind": "outlet"},
#     {"row": 3,  "col": 10, "side": "top_right",    "kind": "outlet"},
#     {"row": 8,  "col": 10, "side": "top_right",    "kind": "outlet"},
#     {"row": 10, "col": 3,  "side": "bottom_right", "kind": "outlet"},
#     {"row": 10, "col": 9,  "side": "bottom_right", "kind": "outlet"},
# ]


# random
# patch_specs = [
#     {"row": 1,  "col": 2,  "side": "top_left",     "kind": "inlet"},
#     {"row": 1,  "col": 10, "side": "top_left",     "kind": "outlet"},
#     {"row": 3,  "col": 5,  "side": "bottom_left",  "kind": "outlet"},
#     {"row": 3,  "col": 8, "side": "top_right",    "kind": "inlet"},
#     {"row": 6,  "col": 1,  "side": "bottom_left",  "kind": "inlet"},
#     {"row": 6,  "col": 10, "side": "top_right",    "kind": "outlet"},
#     {"row": 9,  "col": 3,  "side": "bottom_left",  "kind": "outlet"},
#     {"row": 9,  "col": 7, "side": "top_right",    "kind": "inlet"},
#     {"row": 11, "col": 3,  "side": "bottom_right", "kind": "inlet"},
#     {"row": 11, "col": 10, "side": "bottom_right", "kind": "outlet"},
# ]
import time

t0 = time.perf_counter()

generate_2D_voronoi_with_thickness_rect(
    mesh_filename="mesh/Mesh_Acinar_Perfusion_2D",
    seeds_filename="seeds-2D-periodic.dat",
    domain_x=1.0,
    domain_y=1.0,
    lcar=0.01,
    offset_distance=0.01,
    patch_specs=patch_specs,
    patch_offset=0.02,
    patch_radius=0.003,
    n_refine=1,
)

t1 = time.perf_counter()
print(f"Mesh generation time: {t1 - t0:.6f} s")