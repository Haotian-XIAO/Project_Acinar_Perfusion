#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Haotian XIAO, 2024-2027                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import gmsh
import meshio
import numpy
import math
import json

import dolfin_mech as dmech

import gmsh
import meshio
import dolfin
import numpy

import gmsh
import meshio
import dolfin
import numpy as np
import math

################################################################################

def setPeriodic(dim, coord, xmin, ymin, zmin, xmax, ymax, zmax, e=1e-6):
    # From https://gitlab.onelab.info/gmsh/gmsh/-/issues/744

    dx = (xmax - xmin) if (coord == 0) else 0.
    dy = (ymax - ymin) if (coord == 1) else 0.
    dz = (zmax - zmin) if (coord == 2) else 0.
    d = max(dx, dy, dz)
    e *= d

    smin = gmsh.model.getEntitiesInBoundingBox(
        xmin      - e, ymin      - e, zmin      - e,
        xmax - dx + e, ymax - dy + e, zmax - dz + e,
        dim-1)

    if len(smin) == 0:
        raise RuntimeError(
            f"No master boundary curves found for periodic direction {coord}."
        )

    periodic_pairs = []
    used_slave_tags = set()

    for i in smin:
        bb = gmsh.model.getBoundingBox(*i)
        bbe = [bb[0] + dx, bb[1] + dy, bb[2] + dz,
               bb[3] + dx, bb[4] + dy, bb[5] + dz]
        smax = gmsh.model.getEntitiesInBoundingBox(
            bbe[0] - e, bbe[1] - e, bbe[2] - e,
            bbe[3] + e, bbe[4] + e, bbe[5] + e,
            dim-1)

        matching_slaves = []
        for j in smax:
            bb2 = gmsh.model.getBoundingBox(*j)
            bb2e = [bb2[0] - dx, bb2[1] - dy, bb2[2] - dz,
                    bb2[3] - dx, bb2[4] - dy, bb2[5] - dz]
            if (numpy.linalg.norm(numpy.asarray(bb2e) - numpy.asarray(bb)) < e):
                matching_slaves.append(j)

        if len(matching_slaves) != 1:
            raise RuntimeError(
                "Periodic boundary curve matching failed: "
                f"master curve {i[1]} in direction {coord} has "
                f"{len(matching_slaves)} translated counterparts; expected exactly one."
            )

        j = matching_slaves[0]
        if j[1] in used_slave_tags:
            raise RuntimeError(
                "Periodic boundary curve matching failed: "
                f"slave curve {j[1]} is matched by more than one master curve."
            )

        used_slave_tags.add(j[1])
        periodic_pairs.append((i, j))

    for i, j in periodic_pairs:
        gmsh.model.mesh.setPeriodic(
            dim-1,
            [j[1]], [i[1]],
            [1, 0, 0, dx,\
             0, 1, 0, dy,\
             0, 0, 1, dz,\
             0, 0, 0, 1 ])

################################################################################

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


def point_in_regular_hex(px, py, cx, cy, R, angle0=np.pi / 2.0):
    verts = []
    for k in range(6):
        th = angle0 + 2.0 * np.pi * k / 6.0
        verts.append((cx + R * math.cos(th), cy + R * math.sin(th)))

    inside = False
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            xint = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-16) + x1
            if px < xint:
                inside = not inside
    return inside

def find_surface_closest_to_point_in_center_hex(surface_tags, point, cx, cy, R, angle0):
    x0, y0 = point
    best_tag = None
    best_dist = 1e30

    for s in surface_tags:
        xg, yg, zg = gmsh.model.occ.getCenterOfMass(2, s)
        if not point_in_regular_hex(xg, yg, cx, cy, R, angle0):
            continue
        d = ((xg - x0)**2 + (yg - y0)**2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_tag = s

    return best_tag, best_dist

def add_hex_ring_surfaces(occ, cx, cy, Rin, Rout, angle0, lc):
    inner_vertices = []
    outer_vertices = []
    for k in range(6):
        th = angle0 + k * math.pi / 3.0
        inner_vertices.append((cx + Rin * math.cos(th), cy + Rin * math.sin(th)))
        outer_vertices.append((cx + Rout * math.cos(th), cy + Rout * math.sin(th)))

    inner_pts = [occ.addPoint(float(x), float(y), 0.0, lc) for x, y in inner_vertices]
    outer_pts = [occ.addPoint(float(x), float(y), 0.0, lc) for x, y in outer_vertices]

    inner_lines = [occ.addLine(inner_pts[i], inner_pts[(i + 1) % 6]) for i in range(6)]
    outer_lines = [occ.addLine(outer_pts[i], outer_pts[(i + 1) % 6]) for i in range(6)]

    surfs = []
    for i in range(6):
        l1 = occ.addLine(inner_pts[i], outer_pts[i])
        l2 = occ.addLine(inner_pts[(i + 1) % 6], outer_pts[(i + 1) % 6])
        wire = occ.addWire([inner_lines[i], l2, outer_lines[i], l1])
        s = occ.addPlaneSurface([wire])
        surfs.append(s)

    return surfs

import gmsh
import meshio
import dolfin
import numpy as np
import math
import pickle

def add_regular_hex_surface(occ, cx, cy, R, angle0=0., lc=0.05):
    pts = []
    for k in range(6):
        th = angle0 + k * math.pi / 3.0
        pts.append(occ.addPoint(cx + R * math.cos(th), cy + R * math.sin(th), 0.0, lc))
    lines = [occ.addLine(pts[i], pts[(i + 1) % 6]) for i in range(6)]
    loop = occ.addCurveLoop(lines)
    return occ.addPlaneSurface([loop])

def add_rectangle_surface(occ, xmin, ymin, xmax, ymax):
    return occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)

def inward_polygon(coords, offset_distance):
    centroid = coords.mean(axis=0)
    out = []
    for p in coords:
        dv = centroid - p
        nrm = np.linalg.norm(dv)
        if nrm == 0:
            out.append(p.copy())
        else:
            out.append(p + offset_distance * dv / nrm)
    return np.asarray(out)


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

def add_hex_ring_surfaces(occ, cx, cy, Rin, Rout, angle0, lc):
    inner_pts = []
    outer_pts = []
    for k in range(6):
        th = angle0 + k * math.pi / 3.0
        inner_pts.append(occ.addPoint(cx + Rin * math.cos(th), cy + Rin * math.sin(th), 0.0, lc))
        outer_pts.append(occ.addPoint(cx + Rout * math.cos(th), cy + Rout * math.sin(th), 0.0, lc))
    inner_lines = [occ.addLine(inner_pts[i], inner_pts[(i + 1) % 6]) for i in range(6)]
    outer_lines = [occ.addLine(outer_pts[i], outer_pts[(i + 1) % 6]) for i in range(6)]
    surfs = []
    for i in range(6):
        l1 = occ.addLine(inner_pts[i], outer_pts[i])
        l2 = occ.addLine(inner_pts[(i + 1) % 6], outer_pts[(i + 1) % 6])
        wire = occ.addWire([inner_lines[i], l2, outer_lines[i], l1])
        surfs.append(occ.addPlaneSurface([wire]))
    return surfs

def build_raw_voronoi_wall_surfaces(occ, seeds, lc, offset_distance):
    from scipy.spatial import Voronoi
    vor = Voronoi(seeds)
    all_voro_surfs = []
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not region or (-1 in region) or (len(region) < 3):
            continue
        coords = np.array([vor.vertices[v] for v in region], dtype=float)
        coords_in = inward_polygon(coords, offset_distance)
        n = len(coords)
        out_pts = [occ.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in coords]
        in_pts = [occ.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in coords_in]
        out_lines = [occ.addLine(out_pts[i], out_pts[(i + 1) % n]) for i in range(n)]
        in_lines = [occ.addLine(in_pts[i], in_pts[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            l1 = occ.addLine(out_pts[i], in_pts[i])
            l2 = occ.addLine(out_pts[(i + 1) % n], in_pts[(i + 1) % n])
            wire = occ.addWire([out_lines[i], l2, in_lines[i], l1])
            s = occ.addPlaneSurface([wire])
            all_voro_surfs.append((2, s))
    return all_voro_surfs


def generate_rectangular_base_seeds(
    xmin,
    ymin,
    xmax,
    ymax,
    grid_x,
    grid_y,
    irregularity=0.0,
    rng_seed=None,
):
    """Generate one deterministic seed population in a half-open unit cell."""
    if grid_x < 2 or grid_y < 2:
        raise ValueError("grid_x and grid_y must both be at least 2.")
    if not 0.0 <= irregularity < 1.0:
        raise ValueError("irregularity must satisfy 0 <= irregularity < 1.")

    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        raise ValueError("The periodic unit cell must have positive width and height.")

    dx = width / grid_x
    dy = height / grid_y
    rng = np.random.default_rng(rng_seed)
    seeds = []

    for j in range(grid_y):
        row_shift = 0.5 * dx if (j % 2) else 0.0
        for i in range(grid_x):
            x = xmin + (i + 0.5) * dx + row_shift
            y = ymin + (j + 0.5) * dy

            x += (rng.random() - 0.5) * irregularity * dx
            y += (rng.random() - 0.5) * irregularity * dy

            # Wrapping preserves one canonical seed population in the
            # half-open cell even for the shifted final seed in odd rows.
            x = xmin + ((x - xmin) % width)
            y = ymin + ((y - ymin) % height)
            seeds.append((x, y))

    return np.asarray(seeds, dtype=float)


def make_periodic_seed_cloud(base_seeds, a1, a2):
    """Return the central seeds and their eight nearest lattice copies."""
    base_seeds = np.asarray(base_seeds, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    if base_seeds.ndim != 2 or base_seeds.shape[1] < 2:
        raise ValueError("base_seeds must have shape (n, 2) or (n, 3).")
    base_seeds = base_seeds[:, :2]
    if len(base_seeds) < 4:
        raise ValueError("At least four base seeds are required.")
    if a1.shape != (2,) or a2.shape != (2,):
        raise ValueError("a1 and a2 must be two-dimensional lattice vectors.")
    if abs(np.linalg.det(np.column_stack((a1, a2)))) < 1e-14:
        raise ValueError("a1 and a2 must be linearly independent.")

    seed_sets = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            seed_sets.append(base_seeds + i * a1 + j * a2)

    return np.vstack(seed_sets)


def build_periodic_voronoi_walls_in_rectangle(
    occ,
    base_seeds,
    xmin,
    ymin,
    xmax,
    ymax,
    lc,
    offset_distance,
):
    """Build the extended thick-wall tessellation, then clip the central cell."""
    a1 = np.array([xmax - xmin, 0.0], dtype=float)
    a2 = np.array([0.0, ymax - ymin], dtype=float)
    periodic_seeds = make_periodic_seed_cloud(base_seeds, a1, a2)

    raw_walls = build_raw_voronoi_wall_surfaces(
        occ=occ,
        seeds=periodic_seeds,
        lc=lc,
        offset_distance=offset_distance,
    )
    occ.synchronize()

    clip_surface = add_rectangle_surface(occ, xmin, ymin, xmax, ymax)
    occ.synchronize()

    clipped, _ = occ.intersect(
        objectDimTags=raw_walls,
        toolDimTags=[(2, clip_surface)],
        removeObject=True,
        removeTool=True,
    )
    occ.synchronize()

    clipped = [dimtag for dimtag in clipped if dimtag[0] == 2]
    if len(clipped) == 0:
        raise RuntimeError("Periodic Voronoi clipping produced no wall surfaces.")

    occ.fragment(clipped, [])
    occ.synchronize()
    occ.removeAllDuplicates()
    occ.synchronize()

    wall_tags = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
    if len(wall_tags) == 0:
        raise RuntimeError("Periodic Voronoi construction produced no final surfaces.")

    return sorted(set(wall_tags)), periodic_seeds


def load_vascular_patch_sites(metadata_json, number_of_venous_sinks=2):
    """Load canonical arterial and selected venous patch sites from metadata."""
    with open(metadata_json, "r") as stream:
        metadata = json.load(stream)

    arterial = metadata.get("arterial_sources", [])
    if len(arterial) != 4:
        raise ValueError("Vascular metadata must contain Pa0--Pa3 arterial sources.")

    solutions = metadata.get("fixed_number_solutions", [])
    solution = next(
        (item for item in solutions
         if int(item["number_of_venous_sinks"]) == int(number_of_venous_sinks)),
        None,
    )
    if solution is None:
        raise ValueError(
            f"No fixed-Nv={number_of_venous_sinks} solution in vascular metadata."
        )

    sites = []
    for item in sorted(arterial, key=lambda value: value["arterial_id"]):
        sites.append({
            "vascular_id": str(item["arterial_id"]),
            "vascular_type": "arterial",
            "coordinate": [float(v) for v in item["arterial_coordinate"]],
            "terminal_id": int(item["terminal_id"]),
            "acinus_id": int(item["acinus_id"]),
            "terminal_coordinate": [float(v) for v in item["terminal_coordinate"]],
        })
    for item in sorted(solution["selected_sinks"], key=lambda value: value["candidate_id"]):
        sites.append({
            "vascular_id": str(item["candidate_id"]),
            "vascular_type": "venous",
            "coordinate": [float(v) for v in item["coordinate"]],
            "adjacent_acini": [int(v) for v in item["adjacent_acini"]],
            "candidate_type": str(item["candidate_type"]),
        })

    expected = {"Pa0", "Pa1", "Pa2", "Pa3", "V002", "V006"}
    found = {item["vascular_id"] for item in sites}
    if found != expected:
        raise ValueError(f"Expected vascular IDs {sorted(expected)}, found {sorted(found)}.")
    return sites


def _periodic_patch_centers(point, radius, xmin, ymin, xmax, ymax):
    """Return translated copies whose disks can intersect the canonical cell."""
    point = np.asarray(point, dtype=float)
    centers = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            center = point + np.array([i * (xmax - xmin), j * (ymax - ymin)])
            if (center[0] + radius >= xmin and center[0] - radius <= xmax and
                    center[1] + radius >= ymin and center[1] - radius <= ymax):
                centers.append(center)
    return centers


def fragment_wall_with_vascular_patches(
    occ,
    wall_tags,
    sites,
    patch_radius,
    xmin,
    ymin,
    xmax,
    ymax,
):
    """Subdivide existing wall surfaces and return per-site surface tags.

    Only descendants of the original wall surfaces are retained.  Disk-only
    Boolean fragments are removed, preventing a patch from adding geometry in
    an alveolar void or outside the canonical wall domain.
    """
    if patch_radius <= 0.0:
        raise ValueError("patch_radius must be positive.")

    wall_inputs = [(2, int(tag)) for tag in sorted(set(wall_tags))]
    disk_inputs = []
    for site in sites:
        for center in _periodic_patch_centers(
            site["coordinate"], patch_radius, xmin, ymin, xmax, ymax
        ):
            disk_tag = occ.addDisk(
                float(center[0]), float(center[1]), 0.0,
                float(patch_radius), float(patch_radius),
            )
            disk_inputs.append((2, disk_tag))
    occ.synchronize()

    _, fragment_map = occ.fragment(wall_inputs, disk_inputs)
    occ.synchronize()

    wall_descendants = set()
    for mapped in fragment_map[:len(wall_inputs)]:
        wall_descendants.update(
            int(tag) for dim, tag in mapped if int(dim) == 2
        )

    all_surface_tags = {int(tag) for dim, tag in occ.getEntities(2)}
    foreign_tags = sorted(all_surface_tags - wall_descendants)
    if foreign_tags:
        occ.remove([(2, tag) for tag in foreign_tags], recursive=True)
        occ.synchronize()

    retained = {int(tag) for dim, tag in occ.getEntities(2)}
    wall_descendants &= retained
    if not wall_descendants:
        raise RuntimeError("Vascular patch fragmentation removed all wall surfaces.")

    patch_tags = {site["vascular_id"]: [] for site in sites}
    tol = max(1e-10, 1e-6 * patch_radius)
    for tag in sorted(wall_descendants):
        center = np.asarray(occ.getCenterOfMass(2, tag)[:2], dtype=float)
        matches = []
        for site in sites:
            for image_center in _periodic_patch_centers(
                site["coordinate"], patch_radius, xmin, ymin, xmax, ymax
            ):
                if np.linalg.norm(center - image_center) <= patch_radius + tol:
                    matches.append(site["vascular_id"])
                    break
        matches = sorted(set(matches))
        if len(matches) > 1:
            raise RuntimeError(
                f"Vascular patch overlap detected for surface {tag}: {matches}."
            )
        if len(matches) == 1:
            patch_tags[matches[0]].append(tag)

    for vascular_id, tags in patch_tags.items():
        if not tags:
            raise RuntimeError(
                f"Vascular patch {vascular_id} produced no wall surface fragments."
            )

    return sorted(wall_descendants), patch_tags

def intersect_surfaces(occ, obj_dimtags, tool_dimtags):
    out, _ = occ.intersect(
        objectDimTags=obj_dimtags,
        toolDimTags=tool_dimtags,
        removeObject=True,
        removeTool=False
    )
    occ.synchronize()
    return [dt for dt in out if dt[0] == 2]

def transform_points_local(base_pts, ref_cx, ref_cy, new_cx, new_cy, mirror_x=False, mirror_y=False):
    pts = np.asarray(base_pts, dtype=float).copy()
    pts[:, 0] -= ref_cx
    pts[:, 1] -= ref_cy
    if mirror_x:
        pts[:, 0] *= -1.0
    if mirror_y:
        pts[:, 1] *= -1.0
    pts[:, 0] += new_cx
    pts[:, 1] += new_cy
    return pts

def build_voronoi_with_outer_ring_in_clip(
    occ,
    seeds,
    cx,
    cy,
    rin,
    rout,
    angle0,
    lc,
    offset_distance,
    clip_dimtags
):
    raw_voro = build_raw_voronoi_wall_surfaces(occ, seeds, lc, offset_distance)
    occ.synchronize()

    inner_hex = add_regular_hex_surface(occ, cx, cy, rin, angle0, lc)
    occ.synchronize()

    voro_in_hex = intersect_surfaces(
        occ,
        raw_voro,
        [(2, inner_hex)]
    )

    voro_clipped = intersect_surfaces(
        occ,
        voro_in_hex,
        clip_dimtags
    )

    ring_surfs = add_hex_ring_surfaces(occ, cx, cy, rin, rout, angle0, lc)
    occ.synchronize()

    ring_clipped = intersect_surfaces(
        occ,
        [(2, s) for s in ring_surfs],
        clip_dimtags
    )

    return [tag for dim, tag in voro_clipped + ring_clipped if dim == 2]


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


def angle_diff_deg(a, b):
    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return abs(d)


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


def build_center_voronoi_with_metadata(
    occ,
    seeds,
    cx,
    cy,
    rin,
    rout,
    angle0,
    lc,
    offset_distance,
    clip_dimtags
):
    from scipy.spatial import Voronoi

    vor = Voronoi(seeds)
    raw_voro = []
    cell_metadata = []

    for i_seed, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region or (-1 in region) or (len(region) < 3):
            continue

        coords = np.array([vor.vertices[v] for v in region], dtype=float)
        coords_in = inward_polygon(coords, offset_distance)
        n = len(coords)

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

        out_pts = [occ.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in coords]
        in_pts = [occ.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in coords_in]

        out_lines = [occ.addLine(out_pts[i], out_pts[(i + 1) % n]) for i in range(n)]
        in_lines = [occ.addLine(in_pts[i], in_pts[(i + 1) % n]) for i in range(n)]

        for i in range(n):
            l1 = occ.addLine(out_pts[i], in_pts[i])
            l2 = occ.addLine(out_pts[(i + 1) % n], in_pts[(i + 1) % n])
            wire = occ.addWire([out_lines[i], l2, in_lines[i], l1])
            s = occ.addPlaneSurface([wire])
            raw_voro.append((2, s))

    occ.synchronize()

    inner_hex = add_regular_hex_surface(occ, cx, cy, rin, angle0, lc)
    occ.synchronize()

    voro_in_hex = intersect_surfaces(
        occ,
        raw_voro,
        [(2, inner_hex)]
    )

    voro_clipped = intersect_surfaces(
        occ,
        voro_in_hex,
        clip_dimtags
    )

    ring_surfs = add_hex_ring_surfaces(occ, cx, cy, rin, rout, angle0, lc)
    occ.synchronize()

    ring_clipped = intersect_surfaces(
        occ,
        [(2, s) for s in ring_surfs],
        clip_dimtags
    )

    kept_indices = []
    for i, cell in enumerate(cell_metadata):
        verts = np.array(cell["vertices"], dtype=float)
        inside = True
        for vx, vy in verts:
            if not point_in_regular_hex(vx, vy, cx, cy, rin + 1e-9, angle0):
                inside = False
                break
        if inside:
            kept_indices.append(i)

    center_cell_metadata = [cell_metadata[i] for i in kept_indices]
    center_tags = [tag for dim, tag in voro_clipped + ring_clipped if dim == 2]

    return center_tags, center_cell_metadata




def find_surface_closest_to_point_filtered(surface_tags, point, angle0, center=None, radius=None):
    x0, y0 = point
    best_tag = None
    best_dist = 1e30

    for s in surface_tags:
        xg, yg, zg = gmsh.model.occ.getCenterOfMass(2, s)

        if center is not None and radius is not None:
            if not point_in_regular_hex(xg, yg, center[0], center[1], radius, angle0):
                continue

        d = ((xg - x0) ** 2 + (yg - y0) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_tag = s

    return best_tag, best_dist

def rebuild_center_hex_structure(occ, center_tags):
    center_dimtags = [(2, s) for s in center_tags]

    frag_out, frag_map = occ.fragment(center_dimtags, [])
    occ.synchronize()
    occ.removeAllDuplicates()
    occ.synchronize()

    rebuilt_tags = []
    for children in frag_map:
        for dim, tag in children:
            if dim == 2:
                rebuilt_tags.append(tag)

    return sorted(set(rebuilt_tags))


def run_PeriodicVoronoi_Mesh(params=None):
    """Generate a strictly periodic rectangular Voronoi thick-wall patch."""
    if params is None:
        params = {}

    xmin = float(params.get("xmin", 0.0))
    ymin = float(params.get("ymin", 0.0))
    xmax = float(params.get("xmax", 1.0))
    ymax = float(params.get("ymax", 1.0))
    lc = float(params.get("l", 0.01))
    offset_distance = float(params.get("voronoi_offset", 0.01))
    mesh_filebasename = params.get(
        "mesh_filebasename",
        "mesh/Mesh_Acinar_Perfusion_Periodic",
    )
    show_gui = bool(params.get("show_gui", False))
    gmsh_verbose = bool(params.get("gmsh_verbose", True))
    vascular_metadata_json = params.get("vascular_metadata_json")
    vascular_patch_radius = float(params.get("patch_radius", 0.003))
    vascular_number_of_venous_sinks = int(params.get("number_of_venous_sinks", 2))

    if offset_distance <= 0.0:
        raise ValueError("voronoi_offset must be positive.")
    if lc <= 0.0:
        raise ValueError("l must be positive.")

    base_seeds = params.get("base_seeds")
    if base_seeds is None:
        base_seeds = generate_rectangular_base_seeds(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            grid_x=int(params.get("grid_x", 12)),
            grid_y=int(params.get("grid_y", 12)),
            irregularity=float(params.get("DoI", 0.3)),
            rng_seed=params.get("rng_seed", 1),
        )
    else:
        base_seeds = np.asarray(base_seeds, dtype=float)

    if base_seeds.ndim != 2 or base_seeds.shape[1] < 2:
        raise ValueError("base_seeds must have shape (n, 2) or (n, 3).")
    base_seeds = base_seeds[:, :2]

    tol = 1e-12 * max(xmax - xmin, ymax - ymin)
    inside = (
        (base_seeds[:, 0] >= xmin - tol)
        & (base_seeds[:, 0] < xmax - tol)
        & (base_seeds[:, 1] >= ymin - tol)
        & (base_seeds[:, 1] < ymax - tol)
    )
    if not np.all(inside):
        bad_ids = np.where(~inside)[0].tolist()
        raise ValueError(
            "Every base seed must belong to the half-open central unit cell; "
            f"invalid seed indices: {bad_ids}."
        )

    gmsh.initialize()
    try:
        gmsh.model.add("PeriodicVoronoiPatch")
        occ = gmsh.model.occ
        gmsh.option.setNumber("General.Terminal", 1 if gmsh_verbose else 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        wall_tags, periodic_seeds = build_periodic_voronoi_walls_in_rectangle(
            occ=occ,
            base_seeds=base_seeds,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            lc=lc,
            offset_distance=offset_distance,
        )

        vascular_sites = None
        vascular_patch_tags = {}
        if vascular_metadata_json is not None:
            vascular_sites = load_vascular_patch_sites(
                vascular_metadata_json,
                number_of_venous_sinks=vascular_number_of_venous_sinks,
            )
            wall_tags, vascular_patch_tags = fragment_wall_with_vascular_patches(
                occ=occ,
                wall_tags=wall_tags,
                sites=vascular_sites,
                patch_radius=vascular_patch_radius,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )

        ordinary_wall_tags = list(wall_tags)
        if vascular_sites is not None:
            patch_union = {
                int(tag) for tags in vascular_patch_tags.values() for tag in tags
            }
            ordinary_wall_tags = [tag for tag in wall_tags if int(tag) not in patch_union]
        wall_group = gmsh.model.addPhysicalGroup(2, ordinary_wall_tags, tag=2)
        gmsh.model.setPhysicalName(2, wall_group, "voronoi_wall")
        marker_mapping = {}
        if vascular_sites is not None:
            next_tag = 3
            site_by_id = {item["vascular_id"]: item for item in vascular_sites}
            for vascular_id in sorted(vascular_patch_tags):
                physical_tag = next_tag
                next_tag += 1
                gmsh.model.addPhysicalGroup(
                    2, sorted(set(vascular_patch_tags[vascular_id])), tag=physical_tag
                )
                gmsh.model.setPhysicalName(2, physical_tag, vascular_id)
                marker_mapping[vascular_id] = {
                    "physical_tag": physical_tag,
                    "vascular_type": site_by_id[vascular_id]["vascular_type"],
                    "coordinate": site_by_id[vascular_id]["coordinate"],
                    "terminal_id": site_by_id[vascular_id].get("terminal_id"),
                    "acinus_id": site_by_id[vascular_id].get("acinus_id"),
                    "candidate_type": site_by_id[vascular_id].get("candidate_type"),
                }
        occ.synchronize()

        # The geometry is complete before the accepted Gmsh periodic meshing
        # mechanism is invoked.
        setPeriodic(
            dim=2,
            coord=0,
            xmin=xmin,
            ymin=ymin,
            zmin=0.0,
            xmax=xmax,
            ymax=ymax,
            zmax=0.0,
        )
        setPeriodic(
            dim=2,
            coord=1,
            xmin=xmin,
            ymin=ymin,
            zmin=0.0,
            xmax=xmax,
            ymax=ymax,
            zmax=0.0,
        )

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
        gmsh.model.mesh.generate(2)

        if show_gui:
            gmsh.fltk.run()

        gmsh.write(mesh_filebasename + ".msh")
    finally:
        gmsh.finalize()

    convert_msh_to_xdmf_with_domains(mesh_filebasename, dim=2)
    if vascular_sites is not None:
        with open(mesh_filebasename + "_vascular_markers.json", "w") as stream:
            json.dump({
                "metadata_json": str(vascular_metadata_json),
                "patch_radius": vascular_patch_radius,
                "number_of_venous_sinks": vascular_number_of_venous_sinks,
                "markers": marker_mapping,
            }, stream, indent=2)
    print(f"[DONE] Periodic mesh saved as {mesh_filebasename}.msh/.xdmf/_domains.xdmf")

    return {
        "base_seed_count": int(len(base_seeds)),
        "periodic_seed_count": int(len(periodic_seeds)),
        "a1": np.array([xmax - xmin, 0.0], dtype=float),
        "a2": np.array([0.0, ymax - ymin], dtype=float),
        "wall_surface_count": int(len(wall_tags)),
        "vascular_markers": marker_mapping,
    }


def run_HollowBox_Mesh_Simple(params={}):
    xmin = params.get("xmin", 0.0)
    ymin = params.get("ymin", 0.0)
    xmax = params.get("xmax", 1.0)
    ymax = params.get("ymax", 1.0)

    xshift = params.get("xshift", 0.0)
    yshift = params.get("yshift", 0.0)

    r_corner = params.get("r_corner", 0.55)
    r_center = params.get("r_center", 0.55)

    l = params.get("l", 0.05)
    angle0 = params.get("angle0", np.pi / 2.0)

    use_hex_aspect_ratio = params.get("use_hex_aspect_ratio", False)

    use_center_voronoi = params.get("use_center_voronoi", True)
    use_corner_voronoi = params.get("use_corner_voronoi", True)
    use_periodic = params.get("use_periodic", True)

    center_seeds_filename = params.get("center_seeds_filename", "seeds-2D-periodic.dat")
    corner_seeds_filename = params.get("corner_seeds_filename", "seeds-2D-periodic.dat")

    voronoi_offset = params.get("voronoi_offset", 0.01)
    voronoi_lcar = params.get("voronoi_lcar", l)

    patch_radius = params.get("patch_radius", 0.002)
    patch_offset = params.get("patch_offset", 0.0)
    center_row_tol = params.get("center_row_tol", None)

    center_patch_specs = params.get(
        "center_patch_specs",
        [
            {"row": 2, "col": 2, "side": "top_left", "kind": "inlet"},
            {"row": 3, "col": 1, "side": "bottom_left", "kind": "outlet"},
            {"row": 3, "col": 3, "side": "bottom_right", "kind": "outlet"},
        ]
    )

    mesh_filebasename = params.get("mesh_filebasename", "mesh_simple")
    show_gui = params.get("show_gui", False)

    if use_hex_aspect_ratio:
        width = xmax - xmin
        ycenter_tmp = 0.5 * (ymin + ymax)
        height = np.sqrt(3.0) * width
        ymin = ycenter_tmp - 0.5 * height
        ymax = ycenter_tmp + 0.5 * height

    xmin_s = xmin + xshift
    xmax_s = xmax + xshift
    ymin_s = ymin + yshift
    ymax_s = ymax + yshift

    xcenter = 0.5 * (xmin_s + xmax_s)
    ycenter = 0.5 * (ymin_s + ymax_s)

    ul = (xmin_s, ymax_s)
    ur = (xmax_s, ymax_s)
    ll = (xmin_s, ymin_s)
    lr = (xmax_s, ymin_s)

    gmsh.initialize()
    gmsh.model.add("HollowBoxSimple")
    occ = gmsh.model.occ

    box_tag = occ.addRectangle(xmin_s, ymin_s, 0.0, xmax_s - xmin_s, ymax_s - ymin_s)

    hole_tags = []
    for cx, cy in [ul, ur, ll, lr]:
        hole_tags.append((2, add_regular_hex_surface(occ, cx, cy, r_corner + 2 * voronoi_offset, angle0, l)))

    hole_tags.append((2, add_regular_hex_surface(occ, xcenter, ycenter, r_center + 2 * voronoi_offset, angle0, l)))

    occ.synchronize()

    outer_cut, _ = occ.cut(
        objectDimTags=[(2, box_tag)],
        toolDimTags=hole_tags,
        removeObject=True,
        removeTool=True
    )
    occ.synchronize()

    outer_tags = [tag for dim, tag in outer_cut if dim == 2]

    corner_tags = []
    center_tags = []
    center_cell_metadata = []
    inlet_surfs = []
    outlet_surfs = []

    if use_center_voronoi:
        with open(center_seeds_filename, "rb") as f:
            center_seeds = pickle.load(f)
        center_seeds = np.asarray(center_seeds)[:, :2]

        center_clip_hex = add_regular_hex_surface(
            occ,
            xcenter,
            ycenter,
            r_center + 2 * voronoi_offset,
            angle0,
            voronoi_lcar
        )
        occ.synchronize()

        center_tags, center_cell_metadata = build_center_voronoi_with_metadata(
            occ=occ,
            seeds=center_seeds,
            cx=xcenter,
            cy=ycenter,
            rin=r_center,
            rout=r_center + 2 * voronoi_offset,
            angle0=angle0,
            lc=voronoi_lcar,
            offset_distance=voronoi_offset,
            clip_dimtags=[(2, center_clip_hex)]
        )

    if use_corner_voronoi:
        with open(corner_seeds_filename, "rb") as f:
            corner_seeds_ul = pickle.load(f)
        corner_seeds_ul = np.asarray(corner_seeds_ul)[:, :2]

        clip_rect = add_rectangle_surface(occ, xmin_s, ymin_s, xmax_s, ymax_s)
        occ.synchronize()

        ul_outer_hex = add_regular_hex_surface(occ, ul[0], ul[1], r_corner + 2 * voronoi_offset, angle0, voronoi_lcar)
        ur_outer_hex = add_regular_hex_surface(occ, ur[0], ur[1], r_corner + 2 * voronoi_offset, angle0, voronoi_lcar)
        ll_outer_hex = add_regular_hex_surface(occ, ll[0], ll[1], r_corner + 2 * voronoi_offset, angle0, voronoi_lcar)
        lr_outer_hex = add_regular_hex_surface(occ, lr[0], lr[1], r_corner + 2 * voronoi_offset, angle0, voronoi_lcar)
        occ.synchronize()

        ul_clip, _ = occ.intersect([(2, ul_outer_hex)], [(2, clip_rect)], removeObject=True, removeTool=False)
        ur_clip, _ = occ.intersect([(2, ur_outer_hex)], [(2, clip_rect)], removeObject=True, removeTool=False)
        ll_clip, _ = occ.intersect([(2, ll_outer_hex)], [(2, clip_rect)], removeObject=True, removeTool=False)
        lr_clip, _ = occ.intersect([(2, lr_outer_hex)], [(2, clip_rect)], removeObject=True, removeTool=False)
        occ.synchronize()

        ul_clip = [dt for dt in ul_clip if dt[0] == 2]
        ur_clip = [dt for dt in ur_clip if dt[0] == 2]
        ll_clip = [dt for dt in ll_clip if dt[0] == 2]
        lr_clip = [dt for dt in lr_clip if dt[0] == 2]

        seeds_ur = transform_points_local(corner_seeds_ul, ul[0], ul[1], ur[0], ur[1], mirror_x=True, mirror_y=False)
        seeds_ll = transform_points_local(corner_seeds_ul, ul[0], ul[1], ll[0], ll[1], mirror_x=False, mirror_y=True)
        seeds_lr = transform_points_local(corner_seeds_ul, ul[0], ul[1], lr[0], lr[1], mirror_x=True, mirror_y=True)

        tags_ul = build_voronoi_with_outer_ring_in_clip(
            occ=occ,
            seeds=corner_seeds_ul,
            cx=ul[0],
            cy=ul[1],
            rin=r_corner,
            rout=r_corner + 2 * voronoi_offset,
            angle0=angle0,
            lc=voronoi_lcar,
            offset_distance=voronoi_offset,
            clip_dimtags=ul_clip
        )

        tags_ur = build_voronoi_with_outer_ring_in_clip(
            occ=occ,
            seeds=seeds_ur,
            cx=ur[0],
            cy=ur[1],
            rin=r_corner,
            rout=r_corner + 2 * voronoi_offset,
            angle0=angle0,
            lc=voronoi_lcar,
            offset_distance=voronoi_offset,
            clip_dimtags=ur_clip
        )

        tags_ll = build_voronoi_with_outer_ring_in_clip(
            occ=occ,
            seeds=seeds_ll,
            cx=ll[0],
            cy=ll[1],
            rin=r_corner,
            rout=r_corner + 2 * voronoi_offset,
            angle0=angle0,
            lc=voronoi_lcar,
            offset_distance=voronoi_offset,
            clip_dimtags=ll_clip
        )

        tags_lr = build_voronoi_with_outer_ring_in_clip(
            occ=occ,
            seeds=seeds_lr,
            cx=lr[0],
            cy=lr[1],
            rin=r_corner,
            rout=r_corner + 2 * voronoi_offset,
            angle0=angle0,
            lc=voronoi_lcar,
            offset_distance=voronoi_offset,
            clip_dimtags=lr_clip
        )

        corner_tags = sorted(set(tags_ul + tags_ur + tags_ll + tags_lr))

    occ.synchronize()

    if use_center_voronoi and len(center_patch_specs) > 0 and len(center_cell_metadata) > 0:
        center_tags = rebuild_center_hex_structure(
            occ=occ,
            center_tags=center_tags
        )

        selected_indices = list(range(len(center_cell_metadata)))

        center_cell_rc_map = build_row_col_map_for_selected_cells(
            center_cell_metadata,
            selected_indices=selected_indices,
            y_tol=0.1
        )

        center_patch_data = build_patch_centers_from_specs(
            cell_metadata=center_cell_metadata,
            cell_rc_map=center_cell_rc_map,
            patch_specs=center_patch_specs,
            offset=patch_offset
        )

        disk_entities = []
        for p in center_patch_data:
            x, y = p["patch_center"]
            dtag = occ.addDisk(float(x), float(y), 0.0, patch_radius, patch_radius)
            disk_entities.append((2, dtag))

        occ.synchronize()

        target_patch_surfs = [(2, s) for s in center_tags]
        frag_out, frag_map = occ.fragment(target_patch_surfs, disk_entities)
        occ.synchronize()
        occ.removeAllDuplicates()
        occ.synchronize()

        new_center_tags = []
        for children in frag_map[:len(target_patch_surfs)]:
            for dim, tag in children:
                if dim == 2:
                    new_center_tags.append(tag)

        center_tags = sorted(set(new_center_tags))

        all_surfs_after_patch = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]

        for p in center_patch_data:
            s_tag, dist = find_surface_closest_to_point_filtered(
                all_surfs_after_patch,
                p["patch_center"],
                angle0=angle0,
                center=(xcenter, ycenter),
                radius=r_center + 2 * voronoi_offset
            )

            if s_tag is None:
                raise RuntimeError(f"Could not identify center patch surface for {p}")

            if p["kind"] == "inlet":
                inlet_surfs.append(s_tag)
            elif p["kind"] == "outlet":
                outlet_surfs.append(s_tag)
            else:
                raise RuntimeError(f"Unknown patch kind: {p['kind']}")

        patch_surfs_set = set(inlet_surfs + outlet_surfs)
        center_tags = [s for s in center_tags if s not in patch_surfs_set]

    inlet_surfs = sorted(set(inlet_surfs))
    outlet_surfs = sorted(set(outlet_surfs))
    center_tags = sorted(set(center_tags))
    corner_tags = sorted(set(corner_tags))

    if outer_tags:
        pg_outer = gmsh.model.addPhysicalGroup(2, outer_tags)
        gmsh.model.setPhysicalName(2, pg_outer, "barrier")

    voro_tags = sorted(set(center_tags + corner_tags))
    if voro_tags:
        pg_voro = gmsh.model.addPhysicalGroup(2, voro_tags)
        gmsh.model.setPhysicalName(2, pg_voro, "voronoi_wall")

    if inlet_surfs:
        pg_in = gmsh.model.addPhysicalGroup(2, inlet_surfs)
        gmsh.model.setPhysicalName(2, pg_in, "inlet_region")

    if outlet_surfs:
        pg_out = gmsh.model.addPhysicalGroup(2, outlet_surfs)
        gmsh.model.setPhysicalName(2, pg_out, "outlet_region")

    occ.synchronize()

    if use_periodic:
        setPeriodic(
            dim=2,
            coord=0,
            xmin=xmin_s,
            ymin=ymin_s,
            zmin=0.0,
            xmax=xmax_s,
            ymax=ymax_s,
            zmax=0.0
        )

        setPeriodic(
            dim=2,
            coord=1,
            xmin=xmin_s,
            ymin=ymin_s,
            zmin=0.0,
            xmax=xmax_s,
            ymax=ymax_s,
            zmax=0.0
        )

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), l)
    gmsh.model.mesh.generate(2)

    gmsh.write(mesh_filebasename + ".msh")
    gmsh.finalize()

    convert_msh_to_xdmf_with_domains(mesh_filebasename, dim=2)
    print(f"[DONE] Mesh saved as {mesh_filebasename}.msh/.xdmf/.xdmf_domains")

def find_surface_closest_to_point_filtered(surface_tags, point, angle0, center=None, radius=None):
    x0, y0 = point
    best_tag = None
    best_dist = 1e30

    for s in surface_tags:
        xg, yg, zg = gmsh.model.occ.getCenterOfMass(2, s)

        if center is not None and radius is not None:
            if not point_in_regular_hex(xg, yg, center[0], center[1], radius, angle0):
                continue

        d = ((xg - x0) ** 2 + (yg - y0) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_tag = s

    return best_tag, best_dist

if __name__ == "__main__":
    import time

    t0 = time.time()

    run_PeriodicVoronoi_Mesh({
        "xmin": 0.0,
        "ymin": 0.0,
        "xmax": 1.0,
        "ymax": 1.0,
        "grid_x": 12,
        "grid_y": 12,
        "DoI": 0.3,
        "rng_seed": 1,
        "voronoi_offset": 0.013,
        "l": 0.01,
        "mesh_filebasename": "mesh/Mesh_Acinar_Perfusion_Periodic",
    })

    t1 = time.time()
    print("Mesh generation time:", t1 - t0, "seconds")
