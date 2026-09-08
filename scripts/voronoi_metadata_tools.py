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


def build_voronoi_cell_metadata_from_seeds(seeds, offset_distance):
    seeds = np.asarray(seeds)[:, :2]
    vor = Voronoi(seeds)
    cell_metadata = []

    for i_seed, region_index in enumerate(vor.point_region):
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

def generate_hex_lattice_base_seeds_pointy(n_ring, a, center=(0.0, 0.0), DoI=0.0, rng_seed=None):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    seeds = []
    axial_ids = []

    for q in range(-n_ring, n_ring + 1):
        r_min = max(-n_ring, -q - n_ring)
        r_max = min(n_ring, -q + n_ring)
        for r in range(r_min, r_max + 1):
            p = axial_to_cartesian_pointy(q, r, a, center=center)
            if DoI > 0.0:
                p = p + (np.random.rand(2) - 0.5) * 2.0 * DoI * a
            seeds.append(p)
            axial_ids.append((q, r))

    return np.asarray(seeds, dtype=float), axial_ids


def hex_patch_radius_from_lattice(n_ring, a):
    return math.sqrt(3.0) * a * n_ring


def axial_patch_shift_pointy(q, r, R_patch):
    x = math.sqrt(3.0) * R_patch * (q + 0.5 * r)
    y = 1.5 * R_patch * r
    return np.array([x, y], dtype=float)


def generate_hex_periodic_seeds_pointy(
    base_seeds,
    R_patch,
    copy_ring=1,
    buffer_width=None,
    center=None,
    angle0=np.pi / 2.0
):
    base_seeds = np.asarray(base_seeds, dtype=float)[:, :2]

    if center is None:
        center = np.mean(base_seeds, axis=0)
    else:
        center = np.asarray(center, dtype=float)

    if buffer_width is None:
        buffer_width = 0.25 * R_patch

    all_sets = []
    shifts = []

    for q in range(-copy_ring, copy_ring + 1):
        r_min = max(-copy_ring, -q - copy_ring)
        r_max = min(copy_ring, -q + copy_ring)

        for r in range(r_min, r_max + 1):
            shift = axial_patch_shift_pointy(q, r, R_patch)
            pts = base_seeds + shift

            if q == 0 and r == 0:
                all_sets.append(pts)
                shifts.append((q, r, shift.copy()))
                continue

            kept = []
            for x, y in pts:
                inside_outer = point_in_regular_hex(
                    x,
                    y,
                    center[0],
                    center[1],
                    R_patch + buffer_width,
                    angle0
                )
                inside_inner = point_in_regular_hex(
                    x,
                    y,
                    center[0],
                    center[1],
                    R_patch,
                    angle0
                )

                if inside_outer and not inside_inner:
                    kept.append([x, y])

            if len(kept) > 0:
                all_sets.append(np.asarray(kept, dtype=float))
                shifts.append((q, r, shift.copy()))

    return np.vstack(all_sets), shifts


def hex_vertices(cx, cy, R, angle0=np.pi / 2.0):
    return np.array([
        [cx + R * math.cos(angle0 + 2.0 * np.pi * k / 6.0),
         cy + R * math.sin(angle0 + 2.0 * np.pi * k / 6.0)]
        for k in range(6)
    ])


def plot_hex_seed_sets(base_seeds, periodic_seeds, center, R, angle0=np.pi / 2.0, title="", save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    if periodic_seeds is not None:
        ax.scatter(periodic_seeds[:, 0], periodic_seeds[:, 1], s=18, alpha=0.35, label="periodic seeds")

    ax.scatter(base_seeds[:, 0], base_seeds[:, 1], s=42, label="base seeds")

    verts = hex_vertices(center[0], center[1], R, angle0)
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


def save_seed_sets(out_prefix, base_seeds, periodic_seeds):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    base_path = out_prefix + "_seeds_base.pkl"
    periodic_path = out_prefix + "_seeds_periodic.pkl"

    with open(base_path, "wb") as f:
        pickle.dump(base_seeds, f)

    with open(periodic_path, "wb") as f:
        pickle.dump(periodic_seeds, f)

    return base_path, periodic_path


def prepare_hexcenter_seeds_and_metadata(
    out_prefix,
    n_ring,
    a,
    center=(0.0, 0.0),
    DoI=0.0,
    rng_seed=None,
    copy_ring=1,
    offset_distance=0.01,
    angle0=np.pi / 2.0,
    plot=True,
):
    base_seeds, axial_ids = generate_hex_lattice_base_seeds_pointy(
        n_ring=n_ring,
        a=a,
        center=center,
        DoI=DoI,
        rng_seed=rng_seed,
    )

    R_patch = hex_patch_radius_from_lattice(n_ring, a)

    periodic_seeds, shifts = generate_hex_periodic_seeds_pointy(
        base_seeds=base_seeds,
        R_patch=R_patch,
        copy_ring=copy_ring,
        buffer_width=2.0 * a,
        center=center,
        angle0=angle0,
    )

    base_path, periodic_path = save_seed_sets(
        out_prefix=out_prefix,
        base_seeds=base_seeds,
        periodic_seeds=periodic_seeds,
    )

    cell_metadata = build_voronoi_cell_metadata_from_seeds(
        seeds=periodic_seeds,
        offset_distance=offset_distance,
    )

    kept_metadata, metadata_pkl, metadata_json = save_cell_metadata_for_hex(
        cell_metadata=cell_metadata,
        out_prefix=out_prefix,
        cx=center[0],
        cy=center[1],
        R=R_patch,
        angle0=angle0,
    )

    fig_path = out_prefix + "_seeds.png"
    if plot:
        plot_hex_seed_sets(
            base_seeds=base_seeds,
            periodic_seeds=periodic_seeds,
            center=center,
            R=R_patch,
            angle0=angle0,
            title=f"Pointy-top hex seeds: n_ring={n_ring}, a={a}, DoI={DoI}",
            save_path=fig_path,
            show=True,
        )

    return {
        "base_seeds": base_seeds,
        "periodic_seeds": periodic_seeds,
        "axial_ids": axial_ids,
        "shifts": shifts,
        "hex_radius": R_patch,
        "base_path": base_path,
        "periodic_path": periodic_path,
        "metadata_pkl": metadata_pkl,
        "metadata_json": metadata_json,
        "cell_metadata": kept_metadata,
        "figure_path": fig_path,
    }


def select_corner_template_from_center_base(base_seeds, center, mode="bottom_right"):
    seeds = np.asarray(base_seeds)[:, :2]
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


def save_corner_template_with_neighbors(corner_base_seeds, out_filename, template_radius, copy_ring=1):
    corner_all, _ = generate_hex_periodic_seeds_pointy(
        base_seeds=np.asarray(corner_base_seeds)[:, :2],
        R_patch=template_radius,
        copy_ring=copy_ring,
    )

    with open(out_filename, "wb") as f:
        pickle.dump(corner_all, f)

    return corner_all


def plot_center_and_corner_seeds(center_file, corner_file, save_path=None, show=True):
    with open(center_file, "rb") as f:
        center_seeds = np.asarray(pickle.load(f))[:, :2]

    with open(corner_file, "rb") as f:
        corner_seeds = np.asarray(pickle.load(f))[:, :2]

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
    n_ring = 8
    a = 0.08
    center = (0.5, 0.5)
    DoI = 0.3

    prep = prepare_hexcenter_seeds_and_metadata(
        out_prefix="mesh/Mesh_Acinar_Perfusion_HexCenter",
        n_ring=n_ring,
        a=a,
        center=center,
        DoI=DoI,
        rng_seed=1,
        copy_ring=1,
        offset_distance=0.01,
        angle0=np.pi / 2.0,
        plot=True,
    )

    corner_base = select_corner_template_from_center_base(
        base_seeds=prep["base_seeds"],
        center=center,
        mode="bottom_right",
    )

    template_radius = 0.5 * prep["hex_radius"]

    corner_all = save_corner_template_with_neighbors(
        corner_base_seeds=corner_base,
        out_filename="mesh/Mesh_Acinar_Perfusion_CornerTemplate_seeds.pkl",
        template_radius=template_radius,
        copy_ring=1,
    )

    plot_center_and_corner_seeds(
        center_file="mesh/Mesh_Acinar_Perfusion_HexCenter_seeds_periodic.pkl",
        corner_file="mesh/Mesh_Acinar_Perfusion_CornerTemplate_seeds.pkl",
        save_path="mesh/Mesh_Acinar_Perfusion_CenterCornerSeeds.png",
        show=True,
    )