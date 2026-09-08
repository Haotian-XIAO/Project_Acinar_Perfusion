import os
import json
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi


def angle_diff_deg(a, b):
    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return abs(d)


def classify_edge_direction(theta):
    deg = math.degrees(theta)
    while deg >= 180:
        deg -= 360
    while deg < -180:
        deg += 360

    directions = {
        "right": 0.0,
        "top_right": 60.0,
        "top_left": 120.0,
        "left": 180.0,
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


def inward_polygon(coords, offset_distance):
    centroid = coords.mean(axis=0)
    in_coords = []

    for p in coords:
        dv = centroid - p
        nrm = np.linalg.norm(dv)
        if nrm < 1e-14:
            in_coords.append(p.copy())
        else:
            in_coords.append(p + offset_distance * dv / nrm)

    return np.asarray(in_coords)


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


def filter_full_cells_in_hex(cell_metadata, cx, cy, R, angle0=np.pi / 2.0, tol=1e-9):
    kept = []

    for idx, cell in enumerate(cell_metadata):
        verts = np.array(cell["vertices"], dtype=float)
        inside = True

        for vx, vy in verts:
            if not point_in_regular_hex(vx, vy, cx, cy, R + tol, angle0):
                inside = False
                break

        if inside:
            kept.append(idx)

    return kept


def build_voronoi_cell_metadata_from_seeds(seeds, offset_distance, active_seed_count=None):
    seeds = np.asarray(seeds, dtype=float)[:, :2]
    vor = Voronoi(seeds)
    cell_metadata = []

    if active_seed_count is None:
        active_seed_count = len(seeds)

    for i_seed, region_index in enumerate(vor.point_region):
        if i_seed >= active_seed_count:
            continue

        region = vor.regions[region_index]

        if not region or (-1 in region) or (len(region) < 3):
            continue

        coords = np.array([vor.vertices[v] for v in region], dtype=float)
        coords_in = inward_polygon(coords, offset_distance)

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

    return cell_metadata


def save_cell_metadata_for_hex(cell_metadata, out_prefix, cx, cy, R, angle0=np.pi / 2.0):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    kept_indices = filter_full_cells_in_hex(
        cell_metadata=cell_metadata,
        cx=cx,
        cy=cy,
        R=R,
        angle0=angle0,
    )

    kept = [cell_metadata[i] for i in kept_indices]

    pkl_path = out_prefix + "_cell_metadata.pkl"
    json_path = out_prefix + "_cell_metadata.json"

    with open(pkl_path, "wb") as f:
        pickle.dump(kept, f)

    with open(json_path, "w") as f:
        json.dump(kept, f, indent=2)

    return kept, pkl_path, json_path


def axial_to_cartesian_pointy(q, r, a, center=(0.0, 0.0)):
    cx, cy = center
    x = cx + 1.5 * a * q
    y = cy + math.sqrt(3.0) * a * (r + 0.5 * q)
    return np.array([x, y], dtype=float)


def axial_ring_id(q, r):
    return max(abs(q), abs(r), abs(-q - r))


def hex_vertices(cx, cy, R, angle0=np.pi / 2.0):
    return np.array([
        [
            cx + R * math.cos(angle0 + 2.0 * np.pi * k / 6.0),
            cy + R * math.sin(angle0 + 2.0 * np.pi * k / 6.0)
        ]
        for k in range(6)
    ], dtype=float)


def generate_hex_lattice_seeds_with_rings(
    n_total,
    a,
    center=(0.0, 0.0),
    n_real=6,
    DoI=0.0,
    boundary_DoI=0.05,
    ghost_DoI=0.05,
    rng_seed=None
):
    rng = np.random.default_rng(rng_seed)

    seeds = []
    axial_ids = []
    ring_ids = []

    for q in range(-n_total, n_total + 1):
        r_min = max(-n_total, -q - n_total)
        r_max = min(n_total, -q + n_total)

        for r in range(r_min, r_max + 1):
            p = axial_to_cartesian_pointy(q, r, a, center=center)
            rid = axial_ring_id(q, r)

            if rid < n_real:
                doi = DoI
            elif rid == n_real:
                doi = boundary_DoI
            else:
                doi = ghost_DoI

            if doi > 0.0:
                p = p + (rng.random(2) - 0.5) * 2.0 * doi * a

            seeds.append(p)
            axial_ids.append((q, r))
            ring_ids.append(rid)

    return np.asarray(seeds, dtype=float), axial_ids, np.asarray(ring_ids, dtype=int)


def point_line_distance_signed(p, a, b):
    e = b - a
    n = np.array([e[1], -e[0]], dtype=float)
    nn = np.linalg.norm(n)

    if nn < 1e-14:
        raise RuntimeError("Degenerate edge")

    n /= nn
    return float(np.dot(p - a, n)), n


def orient_edge_normal_outward(edge_a, edge_b, center):
    d, n = point_line_distance_signed(center, edge_a, edge_b)

    if d > 0:
        n = -n

    return n


def nearest_hex_edge_for_point(p, verts, center):
    best_i = None
    best_dist = 1e30

    for i in range(6):
        a = verts[i]
        b = verts[(i + 1) % 6]
        n = orient_edge_normal_outward(a, b, center)
        dist = abs(np.dot(p - a, n))

        if dist < best_dist:
            best_dist = dist
            best_i = i

    return best_i, best_dist


def mirror_point_across_line(p, a, b):
    e = b - a
    ee = np.dot(e, e)

    if ee < 1e-14:
        raise RuntimeError("Degenerate mirror line")

    t = np.dot(p - a, e) / ee
    proj = a + t * e
    return 2.0 * proj - p


def enforce_outer_ring_distance_to_hex_boundary(
    seeds,
    ring_ids,
    n_real,
    center,
    R_outer,
    target_distance,
    angle0=np.pi / 2.0
):
    out = np.asarray(seeds, dtype=float).copy()
    center = np.asarray(center, dtype=float)
    verts = hex_vertices(center[0], center[1], R_outer, angle0)

    for i, p in enumerate(out):
        if ring_ids[i] != n_real:
            continue

        edge_id, _ = nearest_hex_edge_for_point(p, verts, center)
        a = verts[edge_id]
        b = verts[(edge_id + 1) % 6]
        n = orient_edge_normal_outward(a, b, center)
        s = float(np.dot(p - a, n))
        delta = -target_distance - s
        out[i] = p + delta * n

    return out


def mirror_outer_real_ring_to_hex_boundary(
    real_seeds,
    real_ring_ids,
    n_real,
    center,
    R_outer,
    angle0=np.pi / 2.0
):
    center = np.asarray(center, dtype=float)
    verts = hex_vertices(center[0], center[1], R_outer, angle0)

    ghosts = []
    ghost_parent_ids = []
    ghost_edge_ids = []

    for i, p in enumerate(real_seeds):
        if real_ring_ids[i] != n_real:
            continue

        edge_id, _ = nearest_hex_edge_for_point(p, verts, center)
        a = verts[edge_id]
        b = verts[(edge_id + 1) % 6]
        q = mirror_point_across_line(p, a, b)

        ghosts.append(q)
        ghost_parent_ids.append(i)
        ghost_edge_ids.append(edge_id)

    if len(ghosts) == 0:
        ghosts = np.zeros((0, 2), dtype=float)
    else:
        ghosts = np.asarray(ghosts, dtype=float)

    return ghosts, ghost_parent_ids, ghost_edge_ids


def build_fewer_real_cells_with_outer_ghost_rings(
    n_real,
    n_ghost,
    R_outer,
    center=(0.0, 0.0),
    DoI=0.18,
    boundary_DoI=0.05,
    ghost_DoI=0.05,
    boundary_fraction=0.5,
    rng_seed=None,
    angle0=np.pi / 2.0,
    use_mirror_ghosts=True,
    enforce_boundary_distance=True
):
    n_total = n_real + n_ghost
    a = R_outer / (math.sqrt(3.0) * (n_real + boundary_fraction))
    d_seed = math.sqrt(3.0) * a
    target_boundary_distance = boundary_fraction * d_seed

    seeds_all_lattice, axial_ids_all, ring_ids_all = generate_hex_lattice_seeds_with_rings(
        n_total=n_total,
        a=a,
        center=center,
        n_real=n_real,
        DoI=DoI,
        boundary_DoI=boundary_DoI,
        ghost_DoI=ghost_DoI,
        rng_seed=rng_seed
    )

    if enforce_boundary_distance:
        seeds_all_lattice = enforce_outer_ring_distance_to_hex_boundary(
            seeds=seeds_all_lattice,
            ring_ids=ring_ids_all,
            n_real=n_real,
            center=center,
            R_outer=R_outer,
            target_distance=target_boundary_distance,
            angle0=angle0
        )

    real_mask = ring_ids_all <= n_real
    ghost_lattice_mask = ring_ids_all > n_real

    real_seeds = seeds_all_lattice[real_mask].copy()
    real_ring_ids = ring_ids_all[real_mask].copy()
    real_axial_ids = [axial_ids_all[i] for i in np.where(real_mask)[0]]

    lattice_ghost_seeds = seeds_all_lattice[ghost_lattice_mask].copy()
    lattice_ghost_ring_ids = ring_ids_all[ghost_lattice_mask].copy()
    lattice_ghost_axial_ids = [axial_ids_all[i] for i in np.where(ghost_lattice_mask)[0]]

    if use_mirror_ghosts:
        mirror_ghost_seeds, ghost_parent_ids, ghost_edge_ids = mirror_outer_real_ring_to_hex_boundary(
            real_seeds=real_seeds,
            real_ring_ids=real_ring_ids,
            n_real=n_real,
            center=center,
            R_outer=R_outer,
            angle0=angle0
        )
    else:
        mirror_ghost_seeds = np.zeros((0, 2), dtype=float)
        ghost_parent_ids = []
        ghost_edge_ids = []

    all_seed_sets = [real_seeds]

    if len(mirror_ghost_seeds) > 0:
        all_seed_sets.append(mirror_ghost_seeds)

    if len(lattice_ghost_seeds) > 0:
        all_seed_sets.append(lattice_ghost_seeds)

    all_seeds = np.vstack(all_seed_sets)
    active_seed_count = len(real_seeds)

    return {
        "real_seeds": real_seeds,
        "mirror_ghost_seeds": mirror_ghost_seeds,
        "lattice_ghost_seeds": lattice_ghost_seeds,
        "all_seeds": all_seeds,
        "active_seed_count": active_seed_count,
        "real_ring_ids": real_ring_ids,
        "lattice_ghost_ring_ids": lattice_ghost_ring_ids,
        "real_axial_ids": real_axial_ids,
        "lattice_ghost_axial_ids": lattice_ghost_axial_ids,
        "ghost_parent_ids": ghost_parent_ids,
        "ghost_edge_ids": ghost_edge_ids,
        "a": a,
        "d_seed": d_seed,
        "R_outer": R_outer,
        "R_real": math.sqrt(3.0) * a * n_real,
        "target_boundary_distance": target_boundary_distance,
        "n_real": n_real,
        "n_ghost": n_ghost,
        "n_total": n_total,
    }


def save_fewer_real_cells_seed_sets(out_prefix, seed_data):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    real_path = out_prefix + "_seeds_real.pkl"
    mirror_ghost_path = out_prefix + "_seeds_mirror_ghost.pkl"
    lattice_ghost_path = out_prefix + "_seeds_lattice_ghost.pkl"
    all_path = out_prefix + "_seeds_all.pkl"
    base_path = out_prefix + "_seeds_base.pkl"
    periodic_path = out_prefix + "_seeds_periodic.pkl"
    info_path = out_prefix + "_seed_info.json"

    with open(real_path, "wb") as f:
        pickle.dump(seed_data["real_seeds"], f)

    with open(mirror_ghost_path, "wb") as f:
        pickle.dump(seed_data["mirror_ghost_seeds"], f)

    with open(lattice_ghost_path, "wb") as f:
        pickle.dump(seed_data["lattice_ghost_seeds"], f)

    with open(all_path, "wb") as f:
        pickle.dump(seed_data["all_seeds"], f)

    with open(base_path, "wb") as f:
        pickle.dump(seed_data["real_seeds"], f)

    with open(periodic_path, "wb") as f:
        pickle.dump(seed_data["all_seeds"], f)

    info = {
        "active_seed_count": int(seed_data["active_seed_count"]),
        "a": float(seed_data["a"]),
        "d_seed": float(seed_data["d_seed"]),
        "R_outer": float(seed_data["R_outer"]),
        "R_real": float(seed_data["R_real"]),
        "target_boundary_distance": float(seed_data["target_boundary_distance"]),
        "n_real": int(seed_data["n_real"]),
        "n_ghost": int(seed_data["n_ghost"]),
        "n_total": int(seed_data["n_total"]),
        "n_real_seeds": int(len(seed_data["real_seeds"])),
        "n_mirror_ghost_seeds": int(len(seed_data["mirror_ghost_seeds"])),
        "n_lattice_ghost_seeds": int(len(seed_data["lattice_ghost_seeds"])),
        "n_all_seeds": int(len(seed_data["all_seeds"])),
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return {
        "real_path": real_path,
        "mirror_ghost_path": mirror_ghost_path,
        "lattice_ghost_path": lattice_ghost_path,
        "all_path": all_path,
        "base_path": base_path,
        "periodic_path": periodic_path,
        "info_path": info_path,
    }


def plot_fewer_real_cells_seed_sets(seed_data, center, R_outer, angle0=np.pi / 2.0, title="", save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    real_seeds = seed_data["real_seeds"]
    mirror_ghost_seeds = seed_data["mirror_ghost_seeds"]
    lattice_ghost_seeds = seed_data["lattice_ghost_seeds"]

    ax.scatter(real_seeds[:, 0], real_seeds[:, 1], s=36, label="real seeds")

    if len(mirror_ghost_seeds) > 0:
        ax.scatter(mirror_ghost_seeds[:, 0], mirror_ghost_seeds[:, 1], s=28, marker="x", label="mirror ghost seeds")

    if len(lattice_ghost_seeds) > 0:
        ax.scatter(lattice_ghost_seeds[:, 0], lattice_ghost_seeds[:, 1], s=18, alpha=0.5, label="lattice ghost seeds")

    verts = hex_vertices(center[0], center[1], R_outer, angle0)
    poly = np.vstack([verts, verts[0]])

    ax.plot(poly[:, 0], poly[:, 1], "-", linewidth=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def prepare_fewer_real_cells_hexcenter_seeds_and_metadata(
    out_prefix,
    n_real,
    n_ghost,
    R_outer,
    center=(0.0, 0.0),
    DoI=0.18,
    boundary_DoI=0.05,
    ghost_DoI=0.05,
    boundary_fraction=0.5,
    rng_seed=None,
    offset_distance=0.01,
    angle0=np.pi / 2.0,
    use_mirror_ghosts=True,
    enforce_boundary_distance=True,
    plot=True
):
    seed_data = build_fewer_real_cells_with_outer_ghost_rings(
        n_real=n_real,
        n_ghost=n_ghost,
        R_outer=R_outer,
        center=center,
        DoI=DoI,
        boundary_DoI=boundary_DoI,
        ghost_DoI=ghost_DoI,
        boundary_fraction=boundary_fraction,
        rng_seed=rng_seed,
        angle0=angle0,
        use_mirror_ghosts=use_mirror_ghosts,
        enforce_boundary_distance=enforce_boundary_distance
    )

    paths = save_fewer_real_cells_seed_sets(
        out_prefix=out_prefix,
        seed_data=seed_data
    )

    cell_metadata = build_voronoi_cell_metadata_from_seeds(
        seeds=seed_data["all_seeds"],
        offset_distance=offset_distance,
        active_seed_count=seed_data["active_seed_count"]
    )

    kept_metadata, metadata_pkl, metadata_json = save_cell_metadata_for_hex(
        cell_metadata=cell_metadata,
        out_prefix=out_prefix,
        cx=center[0],
        cy=center[1],
        R=R_outer,
        angle0=angle0
    )

    fig_path = out_prefix + "_seeds.png"

    if plot:
        plot_fewer_real_cells_seed_sets(
            seed_data=seed_data,
            center=center,
            R_outer=R_outer,
            angle0=angle0,
            title=f"real={n_real}, ghost={n_ghost}, R={R_outer}, DoI={DoI}",
            save_path=fig_path,
            show=True
        )

    return {
        "base_seeds": seed_data["real_seeds"],
        "real_seeds": seed_data["real_seeds"],
        "mirror_ghost_seeds": seed_data["mirror_ghost_seeds"],
        "lattice_ghost_seeds": seed_data["lattice_ghost_seeds"],
        "all_seeds": seed_data["all_seeds"],
        "active_seed_count": seed_data["active_seed_count"],
        "hex_radius": R_outer,
        "a": seed_data["a"],
        "d_seed": seed_data["d_seed"],
        "R_real": seed_data["R_real"],
        "target_boundary_distance": seed_data["target_boundary_distance"],
        "paths": paths,
        "metadata_pkl": metadata_pkl,
        "metadata_json": metadata_json,
        "cell_metadata": kept_metadata,
        "figure_path": fig_path,
    }


def select_corner_template_from_center_base(base_seeds, center, mode="bottom_right"):
    seeds = np.asarray(base_seeds, dtype=float)[:, :2]
    cx, cy = center

    if mode == "bottom_right":
        mask = (seeds[:, 0] >= cx) & (seeds[:, 1] <= cy)
    elif mode == "bottom_left":
        mask = (seeds[:, 0] <= cx) & (seeds[:, 1] <= cy)
    elif mode == "top_right":
        mask = (seeds[:, 0] >= cx) & (seeds[:, 1] >= cy)
    elif mode == "top_left":
        mask = (seeds[:, 0] <= cx) & (seeds[:, 1] >= cy)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    corner_base = seeds[mask].copy()

    if len(corner_base) < 4:
        raise RuntimeError(f"Selected corner template has only {len(corner_base)} seeds")

    return corner_base


def axial_patch_shift_pointy(q, r, R_patch):
    x = math.sqrt(3.0) * R_patch * (q + 0.5 * r)
    y = 1.5 * R_patch * r
    return np.array([x, y], dtype=float)


def generate_hex_periodic_seeds_pointy(base_seeds, R_patch, copy_ring=1):
    all_sets = []
    shifts = []

    for q in range(-copy_ring, copy_ring + 1):
        r_min = max(-copy_ring, -q - copy_ring)
        r_max = min(copy_ring, -q + copy_ring)

        for r in range(r_min, r_max + 1):
            shift = axial_patch_shift_pointy(q, r, R_patch)
            all_sets.append(base_seeds + shift)
            shifts.append((q, r, shift.copy()))

    return np.vstack(all_sets), shifts


def save_corner_template_with_neighbors(corner_base_seeds, out_filename, template_radius, copy_ring=1):
    corner_all, _ = generate_hex_periodic_seeds_pointy(
        base_seeds=np.asarray(corner_base_seeds, dtype=float)[:, :2],
        R_patch=template_radius,
        copy_ring=copy_ring
    )

    os.makedirs(os.path.dirname(out_filename) or ".", exist_ok=True)

    with open(out_filename, "wb") as f:
        pickle.dump(corner_all, f)

    return corner_all


def plot_center_and_corner_seeds(center_file, corner_file, save_path=None, show=True):
    with open(center_file, "rb") as f:
        center_seeds = np.asarray(pickle.load(f), dtype=float)[:, :2]

    with open(corner_file, "rb") as f:
        corner_seeds = np.asarray(pickle.load(f), dtype=float)[:, :2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(center_seeds[:, 0], center_seeds[:, 1], s=18, label="center seeds")
    ax.scatter(corner_seeds[:, 0], corner_seeds[:, 1], s=40, marker="x", label="corner seeds")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Center and corner seeds")
    ax.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    n_real = 6
    n_ghost = 0
    R_outer = 0.52
    center = (0.5, 0.5)
    DoI = 0.3
    boundary_DoI = 0.04
    ghost_DoI = 0.04
    boundary_fraction = 0.5

    prep = prepare_fewer_real_cells_hexcenter_seeds_and_metadata(
        out_prefix="mesh/Mesh_Acinar_Perfusion_HexCenter",
        n_real=n_real,
        n_ghost=n_ghost,
        R_outer=R_outer,
        center=center,
        DoI=DoI,
        boundary_DoI=boundary_DoI,
        ghost_DoI=ghost_DoI,
        boundary_fraction=boundary_fraction,
        rng_seed=1,
        offset_distance=0.013,
        angle0=np.pi / 2.0,
        use_mirror_ghosts=True,
        enforce_boundary_distance=True,
        plot=True
    )

    corner_base = select_corner_template_from_center_base(
        base_seeds=prep["base_seeds"],
        center=center,
        mode="bottom_right"
    )

    template_radius = 0.5 * prep["hex_radius"]

    corner_all = save_corner_template_with_neighbors(
        corner_base_seeds=corner_base,
        out_filename="mesh/Mesh_Acinar_Perfusion_CornerTemplate_seeds.pkl",
        template_radius=template_radius,
        copy_ring=1
    )

    plot_center_and_corner_seeds(
        center_file="mesh/Mesh_Acinar_Perfusion_HexCenter_seeds_periodic.pkl",
        corner_file="mesh/Mesh_Acinar_Perfusion_CornerTemplate_seeds.pkl",
        save_path="mesh/Mesh_Acinar_Perfusion_CenterCornerSeeds.png",
        show=True
    )

    print("a =", prep["a"])
    print("d_seed =", prep["d_seed"])
    print("R_real =", prep["R_real"])
    print("R_outer =", prep["hex_radius"])
    print("target_boundary_distance =", prep["target_boundary_distance"])
    print("active_seed_count =", prep["active_seed_count"])
    print("real seed file =", prep["paths"]["base_path"])
    print("all seed file =", prep["paths"]["periodic_path"])
    print("seed info file =", prep["paths"]["info_path"])