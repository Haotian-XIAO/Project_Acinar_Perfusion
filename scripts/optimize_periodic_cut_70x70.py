#!/usr/bin/env python3
"""Diagnose and cheaply optimize the artificial cut of the 70x70 torus.

This script never runs the FE solver and never changes the physical Voronoi
realization.  It scans cut locations using the exact paired-ridge wall bands,
then performs local strip-clipping predictions only for the best candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import meshio
import numpy as np

import generate_directional_70x70_mesh as gm


UNIT_UM = gm.MODEL_UNIT_MM * 1000.0


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {}
    return {
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def periodic_delta(values, origin, length):
    return (np.asarray(values) - origin + 0.5 * length) % length - 0.5 * length


def triangle_metrics(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
    signed_area2 = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
    area = 0.5 * abs(signed_area2)
    cosine = np.empty((len(triangles), 3))
    for i in range(3):
        a, b, c = edge[:, (i + 2) % 3], edge[:, i], edge[:, (i + 1) % 3]
        cosine[:, i] = np.clip((a*a + b*b - c*c) / (2*a*b), -1.0, 1.0)
    minimum_angle = np.degrees(np.arccos(cosine)).min(axis=1)
    aspect = edge.max(axis=1) / (2.0 * area / edge.max(axis=1))
    return minimum_angle, aspect, edge.min(axis=1) * UNIT_UM


def geometry_data():
    length = gm.DOMAIN_L
    base, cells = gm.build_cells(length, gm.GRID_N)
    paired = gm.pair_ridges(cells, length)
    groups = sorted(paired["groups"].values(), key=lambda records: records[0]["ridge"])
    ridges = []
    sides = []
    for records in groups:
        primary, mate = records
        a, b = primary["a"], primary["b"]
        tangent = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-tangent[1], tangent[0]])
        one_sided = []
        for rec in records:
            one_sided.extend((abs(np.dot(rec["ai"] - rec["a"], normal)),
                              abs(np.dot(rec["bi"] - rec["b"], normal))))
        width_model = 0.5 * sum(one_sided)
        normal_h_um = width_model * UNIT_UM / (2 * gm.N_HALF_LAYERS)
        ridge = {
            "id": int(primary["ridge"]),
            "tangent": tangent,
            "width_um": width_model * UNIT_UM,
            "normal_h_um": normal_h_um,
        }
        ridges.append(ridge)
        for rec in records:
            q = np.asarray([rec["a"], rec["b"], rec["bi"], rec["ai"]])
            side_tangent = (rec["b"] - rec["a"]) / np.linalg.norm(rec["b"] - rec["a"])
            sides.append({
                "ridge": ridge,
                "record": rec,
                "quad": q,
                "mid": np.mod(q.mean(axis=0), length),
                "halfspan": 0.5 * (q.max(axis=0) - q.min(axis=0)),
                "angle_vertical_deg": math.degrees(math.atan2(abs(side_tangent[0]),
                                                               abs(side_tangent[1]))),
                "angle_horizontal_deg": math.degrees(math.atan2(abs(side_tangent[1]),
                                                                 abs(side_tangent[0]))),
            })
    junctions = np.unique(np.round(np.mod(np.vstack([c["outer"] for c in cells]), length), 10),
                          axis=0)
    return base, cells, paired, ridges, sides, junctions


def axis_arrays(sides: list[dict], axis: int):
    return (
        np.asarray([s["mid"][axis] for s in sides]),
        np.asarray([s["halfspan"][axis] for s in sides]),
        np.asarray([s["angle_vertical_deg" if axis == 0 else "angle_horizontal_deg"]
                    for s in sides]),
    )


def basic_axis_score(origin: float, axis: int, sides: list[dict], junctions: np.ndarray,
                     arrays=None) -> dict:
    length = gm.DOMAIN_L
    mids, halfspans, crossing_angles = arrays or axis_arrays(sides, axis)
    delta = periodic_delta(mids, origin, length)
    mask = abs(delta) <= halfspans + 2e-14
    penetration_um = (halfspans[mask] - abs(delta[mask])) * UNIT_UM
    angles = crossing_angles[mask]
    junction_distance_um = float(np.min(abs(periodic_delta(junctions[:, axis], origin, length)))
                                 * UNIT_UM)
    minimum_angle = float(angles.min())
    p05_angle = float(np.percentile(angles, 5))
    minimum_penetration = float(penetration_um.min())
    p05_penetration = float(np.percentile(penetration_um, 5))
    # Worst-tail score: a candidate cannot hide one dangerous mechanism behind
    # a good mean.  The smaller normalized safeguard controls the main score.
    safeguards = (minimum_angle / 8.0,
                  minimum_penetration / 2.0,
                  junction_distance_um / 2.0)
    score = min(safeguards) + 0.02 * min(p05_angle / 12.0,
                                         p05_penetration / 4.0)
    return {
        "origin_model": float(origin),
        "origin_physical_mm": float(origin * gm.MODEL_UNIT_MM),
        "intersected_one_sided_bands": int(mask.sum()),
        "crossing_angle_deg": stats(angles),
        "band_penetration_um": stats(penetration_um),
        "distance_to_nearest_junction_line_um": junction_distance_um,
        "worst_tail_score": float(score),
    }


def clean_polygon(poly):
    clean = []
    for point in poly:
        if not clean or np.linalg.norm(point - clean[-1]) > 1e-13:
            clean.append(point)
    if len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) <= 1e-13:
        clean.pop()
    return clean


def snap_axis_parameters(ts, a, b, axis, clearance_um=4.0):
    out = ts.copy()
    denominator = b[axis] - a[axis]
    if abs(denominator) < 1e-15:
        return out
    crossing = -a[axis] / denominator
    if not (1e-12 < crossing < 1.0 - 1e-12) or len(out) <= 2:
        return out
    length_um = np.linalg.norm(b - a) * UNIT_UM
    index = 1 + int(np.argmin(abs(out[1:-1] - crossing)))
    if abs(out[index] - crossing) * length_um <= clearance_um:
        if out[index - 1] + 1e-10 < crossing < out[index + 1] - 1e-10:
            out[index] = crossing
    return out


def shifted_record_arrays(rec: dict, origin: float, axis: int):
    arrays = [np.asarray(rec[key], dtype=float).copy() for key in ("a", "b", "ai", "bi")]
    for array in arrays:
        array[axis] -= origin
    mean = np.mean([array[axis] for array in arrays])
    shift = round(mean / gm.DOMAIN_L) * gm.DOMAIN_L
    for array in arrays:
        array[axis] -= shift
    return arrays


def exact_axis_prediction(origin: float, axis: int, sides: list[dict], junctions: np.ndarray,
                          arrays=None) -> dict:
    mids, halfspans, _ = arrays or axis_arrays(sides, axis)
    delta = periodic_delta(mids, origin, gm.DOMAIN_L)
    selected = np.where(abs(delta) <= halfspans + 2e-14)[0]
    predicted = []
    for side_index in selected:
        side = sides[int(side_index)]
        a, b, ai, bi = shifted_record_arrays(side["record"], origin, axis)
        ts = gm.tangential_parameters(np.linalg.norm(b - a),
                                      side["ridge"]["normal_h_um"])
        grid = np.empty((gm.N_HALF_LAYERS + 1, len(ts), 2), dtype=float)
        for layer in range(gm.N_HALF_LAYERS + 1):
            fraction = layer / gm.N_HALF_LAYERS
            row_a = (1.0 - fraction) * a + fraction * ai
            row_b = (1.0 - fraction) * b + fraction * bi
            row_ts = snap_axis_parameters(ts, row_a, row_b, axis)
            grid[layer] = ((1.0 - row_ts[:, None]) * row_a[None, :] +
                           row_ts[:, None] * row_b[None, :])
        for layer in range(gm.N_HALF_LAYERS):
            for interval in range(len(ts) - 1):
                quad = np.asarray([grid[layer, interval], grid[layer, interval + 1],
                                   grid[layer + 1, interval + 1], grid[layer + 1, interval]])
                if quad[:, axis].min() > 1e-14 or quad[:, axis].max() < -1e-14:
                    continue
                for keep_greater in (False, True):
                    poly = clean_polygon(gm.clip_polygon([p.copy() for p in quad], axis, 0.0,
                                                         keep_greater))
                    if len(poly) < 3:
                        continue
                    for tri in gm.best_convex_triangulation(poly):
                        tri = np.asarray(tri)
                        if abs(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-18:
                            predicted.append(tri)
    triangles = np.asarray(predicted)
    minimum_angle, aspect, shortest = triangle_metrics(triangles)
    basic = basic_axis_score(origin, axis, sides, junctions, arrays)
    basic["local_clipping_prediction"] = {
        "triangles": int(len(triangles)),
        "minimum_angle_deg": stats(minimum_angle),
        "aspect_ratio": stats(aspect),
        "shortest_edge_um": stats(shortest),
        "angle_lt_5_deg": int(np.count_nonzero(minimum_angle < 5.0)),
        "angle_lt_8_deg": int(np.count_nonzero(minimum_angle < 8.0)),
        "aspect_ratio_gt_10": int(np.count_nonzero(aspect > 10.0)),
        "aspect_ratio_gt_15": int(np.count_nonzero(aspect > 15.0)),
    }
    return basic


def prediction_rank(result: dict):
    q = result["local_clipping_prediction"]
    return (q["angle_lt_5_deg"], q["aspect_ratio_gt_15"],
            q["angle_lt_8_deg"], q["aspect_ratio_gt_10"],
            -q["minimum_angle_deg"]["min"], q["aspect_ratio"]["max"],
            -result["worst_tail_score"])


def scan_axis(axis: int, sides: list[dict], junctions: np.ndarray,
              scan_points: int, top_candidates: int):
    arrays = axis_arrays(sides, axis)
    origins = np.linspace(0.0, gm.DOMAIN_L, scan_points, endpoint=False)
    cheap = [basic_axis_score(float(origin), axis, sides, junctions, arrays)
             for origin in origins]
    cheap_sorted = sorted(cheap, key=lambda item: item["worst_tail_score"], reverse=True)
    coarse_step = gm.DOMAIN_L / scan_points
    refine_origins = {item["origin_model"] for item in cheap_sorted[:top_candidates]}
    for item in cheap_sorted[:min(8, top_candidates)]:
        center = item["origin_model"]
        for offset in np.linspace(-coarse_step, coarse_step, 17):
            refine_origins.add(float((center + offset) % gm.DOMAIN_L))
    exact = [exact_axis_prediction(origin, axis, sides, junctions, arrays)
             for origin in sorted(refine_origins)]
    exact.sort(key=prediction_rank)
    return {
        "axis": "vertical_x" if axis == 0 else "horizontal_y",
        "scan_points": scan_points,
        "current_cut": exact_axis_prediction(0.0, axis, sides, junctions, arrays),
        "best_candidate": exact[0],
        "top_candidates": exact[:min(12, len(exact))],
    }


def nearest_junction_distance(point, junctions):
    delta = periodic_delta(junctions, np.asarray(point), gm.DOMAIN_L)
    return float(np.linalg.norm(delta, axis=1).min() * UNIT_UM)


def select_pair(x_scan, y_scan, junctions):
    candidates = []
    for x_rank, x in enumerate(x_scan["top_candidates"]):
        for y_rank, y in enumerate(y_scan["top_candidates"]):
            point = (x["origin_model"], y["origin_model"])
            corner_distance = nearest_junction_distance(point, junctions)
            rank = (max(prediction_rank(x), prediction_rank(y)),
                    x_rank + y_rank, -corner_distance)
            candidates.append((rank, point, corner_distance, x, y))
    _, point, corner_distance, x, y = min(candidates, key=lambda item: item[0])
    return {
        "x0_model": float(point[0]), "y0_model": float(point[1]),
        "x0_physical_mm": float(point[0] * gm.MODEL_UNIT_MM),
        "y0_physical_mm": float(point[1] * gm.MODEL_UNIT_MM),
        "nearest_junction_to_periodic_corner_um": corner_distance,
        "coordinate_transform": "translated = mod(original - [x0, y0], L)",
        "vertical_cut_prediction": x,
        "horizontal_cut_prediction": y,
    }


def diagnose_current_mesh(mesh_path: Path, metadata_path: Path,
                          ridges: list[dict], junctions: np.ndarray) -> dict:
    mesh = meshio.read(mesh_path)
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    metadata = np.load(metadata_path)
    xyz = points[triangles]
    minimum_angle, aspect, shortest = triangle_metrics(xyz)
    seam = metadata["periodic_cut"].astype(bool)
    severe = seam & ((minimum_angle < 5.0) | (aspect > 15.0))
    ids = np.where(severe)[0]
    selected = xyz[ids]
    tol = 2e-11
    on_lr = np.any((abs(selected[:, :, 0]) < tol) |
                   (abs(selected[:, :, 0] - gm.DOMAIN_L) < tol), axis=1)
    on_bt = np.any((abs(selected[:, :, 1]) < tol) |
                   (abs(selected[:, :, 1] - gm.DOMAIN_L) < tol), axis=1)
    corner = on_lr & on_bt
    ridge_ids = metadata["ridge_id"][ids].astype(int)
    tangents = np.asarray([ridges[rid]["tangent"] for rid in ridge_ids])
    vertical_angle = np.degrees(np.arctan2(abs(tangents[:, 0]), abs(tangents[:, 1])))
    horizontal_angle = np.degrees(np.arctan2(abs(tangents[:, 1]), abs(tangents[:, 0])))
    crossing_angle = np.where(on_lr & ~on_bt, vertical_angle,
                              np.where(on_bt & ~on_lr, horizontal_angle,
                                       np.minimum(vertical_angle, horizontal_angle)))
    centroids = selected.mean(axis=1)
    nearest = np.empty(len(ids))
    for start in range(0, len(ids), 256):
        block = centroids[start:start + 256]
        delta = periodic_delta(junctions[None, :, :], block[:, None, :], gm.DOMAIN_L)
        nearest[start:start + len(block)] = np.linalg.norm(delta, axis=2).min(axis=1) * UNIT_UM
    near_junction = nearest < 15.0
    tangential = crossing_angle < 8.0
    shallow = (crossing_angle >= 8.0) & (crossing_angle < 15.0)
    short = shortest[ids] < 1.0
    junction_flag = metadata["junction"][ids].astype(bool)
    primary = np.full(len(ids), "other", dtype=object)
    primary[short] = "short_clipped_fragment"
    primary[shallow] = "shallow_grazing_intersection_8_to_15deg"
    primary[tangential] = "nearly_tangential_intersection"
    primary[near_junction | junction_flag] = "near_triple_junction"
    primary[corner] = "periodic_corner_interaction"
    labels, counts = np.unique(primary, return_counts=True)
    cell_ids = metadata["canonical_cell_id"][ids].astype(int)
    half_layers = metadata["half_layer"][ids].astype(int)
    records = []
    for local, triangle_index in enumerate(ids):
        ridge = ridges[ridge_ids[local]]
        if corner[local]:
            cut_side = "periodic_corner"
        elif on_lr[local]:
            cut_side = "left_right"
        elif on_bt[local]:
            cut_side = "bottom_top"
        else:
            cut_side = "seam_neighborhood"
        records.append({
            "triangle_index": int(triangle_index),
            "ridge_id": int(ridge_ids[local]),
            "canonical_cell_id": int(cell_ids[local]),
            "half_layer": int(half_layers[local]),
            "cut_side": cut_side,
            "junction_transition_flag": bool(junction_flag[local]),
            "minimum_angle_deg": float(minimum_angle[triangle_index]),
            "aspect_ratio": float(aspect[triangle_index]),
            "shortest_edge_um": float(shortest[triangle_index]),
            "wall_cut_crossing_angle_deg": float(crossing_angle[local]),
            "nearest_triple_junction_um": float(nearest[local]),
            "ridge_tangent_original_frame": [float(value) for value in ridge["tangent"]],
            "full_wall_width_um": float(ridge["width_um"]),
            "primary_cause": str(primary[local]),
        })
    def count(mask):
        return int(np.count_nonzero(mask))
    return {
        "seam_triangles": int(seam.sum()),
        "severe_definition": "periodic-cut triangle with minimum angle <5 deg or aspect ratio >15",
        "severe_triangles": int(len(ids)),
        "severe_unique_ridges": int(len(np.unique(ridge_ids))),
        "minimum_angle_deg": stats(minimum_angle[ids]),
        "aspect_ratio": stats(aspect[ids]),
        "shortest_edge_um": stats(shortest[ids]),
        "crossing_angle_deg": stats(crossing_angle),
        "nearest_junction_distance_um": stats(nearest),
        "overlapping_cause_counts": {
            "cut_near_triple_junction_lt_15um": count(near_junction | junction_flag),
            "nearly_tangential_crossing_lt_8deg": count(tangential),
            "shallow_crossing_8_to_15deg": count(shallow),
            "very_short_edge_lt_1um": count(short),
            "periodic_corner": count(corner),
            "left_right_cut": count(on_lr),
            "bottom_top_cut": count(on_bt),
        },
        "exclusive_primary_cause_counts": {str(label): int(value)
                                           for label, value in zip(labels, counts)},
        "severe_triangle_records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-points", type=int, default=4096)
    parser.add_argument("--top-candidates", type=int, default=32)
    parser.add_argument("--output", default="results/physiological_geometry_70x70_cut_scan.json")
    parser.add_argument("--mesh", default="mesh/Mesh_Acinar_Perfusion_Physiological_70x70_Directional.msh")
    parser.add_argument("--metadata", default="mesh/Mesh_Acinar_Perfusion_Physiological_70x70_Directional_directional_metadata.npz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    _, _, _, ridges, sides, junctions = geometry_data()
    diagnosis = diagnose_current_mesh(root / args.mesh, root / args.metadata, ridges, junctions)
    x_scan = scan_axis(0, sides, junctions, args.scan_points, args.top_candidates)
    y_scan = scan_axis(1, sides, junctions, args.scan_points, args.top_candidates)
    selected = select_pair(x_scan, y_scan, junctions)
    report = {
        "geometry_unchanged": {
            "canonical_cells": gm.GRID_N * gm.GRID_N,
            "domain_length_model": gm.DOMAIN_L,
            "physical_domain_mm": [14.0, 14.0],
            "DoI": gm.DOI,
            "rng_seed": gm.RNG_SEED,
            "wall_offset_model": gm.OFFSET_MODEL,
            "full_wall_layers": 2 * gm.N_HALF_LAYERS,
        },
        "current_cut_diagnosis": diagnosis,
        "vertical_scan": x_scan,
        "horizontal_scan": y_scan,
        "selected_periodic_window": selected,
        "full_mesh_regenerated": False,
        "fe_solver_run": False,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(output),
        "current_severe_triangles": diagnosis["severe_triangles"],
        "selected_periodic_window": {
            "x0_model": selected["x0_model"],
            "y0_model": selected["y0_model"],
            "corner_clearance_um": selected["nearest_junction_to_periodic_corner_um"],
        },
        "vertical_prediction": selected["vertical_cut_prediction"]["local_clipping_prediction"],
        "horizontal_prediction": selected["horizontal_cut_prediction"]["local_clipping_prediction"],
    }, indent=2))


if __name__ == "__main__":
    main()
