#!/usr/bin/env python3
"""Directional thin-wall meshing feasibility study for the 70x70 RVE.

The mesh follows the exact piecewise-linear wall construction used by
Mesh_Acinar_RVE.inward_polygon.  Each one-sided Voronoi wall strip has two
normal layers, hence every paired internal septum has four layers across its
full width.  Tangential intervals are graded from 12 um near Voronoi vertices
to at most 28 um in regular strip interiors.

This script is geometry/mesh only: it neither imports dolfin_mech nor runs an
FE problem.  It deliberately does not construct acinar or vascular metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import meshio
import numpy as np
from scipy.spatial import Voronoi


DOMAIN_L = 5.0 / 3.0
GRID_N = 70
DOI = 0.3
RNG_SEED = 1
MODEL_UNIT_MM = 8.4
OFFSET_MODEL = 0.0010329600874476761
N_HALF_LAYERS = 2
ENDPOINT_SPACING_UM = 12.0
MAX_INTERIOR_SPACING_UM = 28.0
ROUND_DIGITS = 12

# Read-only FEniCS validation on the deterministic mesh written by this script.
# This is the dimension of the current mixed space; no FE solve was run.
MEASURED_MIXED_SPACE_REFERENCES = {
    (0.0, 0.0, 555_825, 895_320): {
        "macroscopic_stretch_R": 4,
        "displacement_CG2_vector": 4_017_080,
        "surface_area_R": 1,
        "liquid_pressure_CG1": 554_189,
        "pressure_zero_mean_multiplier_R": 1,
    },
    (0.196533203125, 1.06201171875, 555_033, 894_768): {
        "macroscopic_stretch_R": 4,
        "displacement_CG2_vector": 4_015_526,
        "surface_area_R": 1,
        "liquid_pressure_CG1": 554_099,
        "pressure_zero_mean_multiplier_R": 1,
    },
}


def generate_base_seeds(L: float, n: int, irregularity: float, seed: int) -> np.ndarray:
    dx = L / n
    rng = np.random.default_rng(seed)
    points = []
    for j in range(n):
        row_shift = 0.5 * dx if j % 2 else 0.0
        for i in range(n):
            x = (i + 0.5) * dx + row_shift
            y = (j + 0.5) * dx
            x += (rng.random() - 0.5) * irregularity * dx
            y += (rng.random() - 0.5) * irregularity * dx
            points.append((x % L, y % L))
    return np.asarray(points)


def periodic_cloud(base: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    cloud, owners = [], []
    # Central copy first so canonical regions have a stable, simple index.
    shifts = [(0, 0)] + [(i, j) for j in (-1, 0, 1) for i in (-1, 0, 1)
                           if (i, j) != (0, 0)]
    for i, j in shifts:
        cloud.append(base + np.array([i * L, j * L]))
        owners.extend(range(len(base)))
    return np.vstack(cloud), np.asarray(owners, dtype=np.int32)


def inward_polygon(coords: np.ndarray, offset: float) -> np.ndarray:
    centroid = coords.mean(axis=0)
    dv = centroid - coords
    return coords + offset * dv / np.linalg.norm(dv, axis=1)[:, None]


def polygon_area(poly: np.ndarray) -> float:
    return 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) -
                     np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))


def build_cells(L: float, n: int,
                cut_origin: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, list[dict]]:
    """Build the unchanged toroidal realization in a translated cut window.

    ``cut_origin`` is expressed in the original periodic coordinates.  Moving
    it changes only where the torus is cut open; wrapping the translated seeds
    back into ``[0, L)^2`` preserves all physical distances and adjacency.
    """
    origin = np.asarray(cut_origin, dtype=float)
    base = np.mod(generate_base_seeds(L, n, DOI, RNG_SEED) - origin, L)
    cloud, _ = periodic_cloud(base, L)
    vor = Voronoi(cloud)
    cells = []
    for cid in range(len(base)):
        region = vor.regions[vor.point_region[cid]]
        if not region or -1 in region:
            raise RuntimeError(f"Canonical cell {cid} has an unbounded region")
        outer = np.asarray([vor.vertices[v] for v in region], dtype=float)
        # Scipy returns cyclic polygons but does not promise orientation.
        signed = 0.5 * (np.dot(outer[:, 0], np.roll(outer[:, 1], -1)) -
                        np.dot(outer[:, 1], np.roll(outer[:, 0], -1)))
        if signed < 0:
            outer = outer[::-1].copy()
        inner = inward_polygon(outer, OFFSET_MODEL)
        cells.append({"id": cid, "outer": outer, "inner": inner})
    return base, cells


def torus_edge_key(a: np.ndarray, b: np.ndarray, L: float) -> tuple:
    midpoint = 0.5 * (a + b)
    wrapped_mid = np.mod(midpoint, L)
    shift = midpoint - wrapped_mid
    aa, bb = a - shift, b - shift
    ends = sorted((tuple(np.round(aa, 10)), tuple(np.round(bb, 10))))
    return (tuple(np.round(wrapped_mid, 10)), ends[0], ends[1])


def pair_ridges(cells: list[dict], L: float) -> dict:
    groups = defaultdict(list)
    for cell in cells:
        o, inn = cell["outer"], cell["inner"]
        for e in range(len(o)):
            a, b = o[e], o[(e + 1) % len(o)]
            groups[torus_edge_key(a, b, L)].append({
                "cell": cell["id"], "edge": e, "a": a, "b": b,
                "ai": inn[e], "bi": inn[(e + 1) % len(o)],
            })
    bad = {str(k): len(v) for k, v in groups.items() if len(v) != 2}
    if bad:
        sample = list(bad.items())[:5]
        raise RuntimeError(f"Could not pair {len(bad)} periodic ridges; sample={sample}")
    ordered = {}
    for rid, (key, records) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        for rec in records:
            rec["ridge"] = rid
            ordered[(rec["cell"], rec["edge"])] = rec
        records[0]["mate"] = records[1]
        records[1]["mate"] = records[0]
    return {"groups": groups, "records": ordered}


def tangential_parameters(length_model: float, normal_h_um: float) -> np.ndarray:
    length_um = length_model * MODEL_UNIT_MM * 1000.0
    # Couple the longitudinal spacing to the actual local width.  This keeps
    # the intended directional ratio near 3--6 even at the thinnest ridges.
    endpoint_target = min(ENDPOINT_SPACING_UM, 3.5 * normal_h_um)
    interior_target = min(MAX_INTERIOR_SPACING_UM, 5.5 * normal_h_um)
    endpoint = min(endpoint_target, 0.5 * length_um)
    remaining = max(0.0, length_um - 2.0 * endpoint)
    n_interior = int(math.ceil(remaining / interior_target)) if remaining else 0
    if n_interior == 0:
        widths = np.array([0.5 * length_um, 0.5 * length_um])
    else:
        widths = np.r_[endpoint, np.full(n_interior, remaining / n_interior), endpoint]
    return np.r_[0.0, np.cumsum(widths) / length_um]


def line_cut_parameters(a: np.ndarray, b: np.ndarray, L: float) -> list[float]:
    values = []
    for axis in (0, 1):
        lo, hi = sorted((a[axis], b[axis]))
        for m in range(math.floor(lo / L), math.ceil(hi / L) + 1):
            cut = m * L
            if lo + 1e-14 < cut < hi - 1e-14:
                values.append((cut - a[axis]) / (b[axis] - a[axis]))
    return values


def record_cut_parameters(rec: dict, L: float) -> list[float]:
    values = []
    for k in range(N_HALF_LAYERS + 1):
        s = k / N_HALF_LAYERS
        a = (1-s)*rec["a"] + s*rec["ai"]
        b = (1-s)*rec["b"] + s*rec["bi"]
        values.extend(line_cut_parameters(a, b, L))
    return values


def cut_aware_parameters(ts: np.ndarray, rec: dict, L: float) -> np.ndarray:
    """Move mesh-only longitudinal nodes away from nearly coincident cuts.

    Exact gas/ridge geometry is untouched: only parameter locations of
    interior subdivision lines move.  Both sides of a paired ridge contribute
    cut crossings, so the shared ridge discretization remains conforming.
    """
    length_um = np.linalg.norm(rec["b"] - rec["a"]) * MODEL_UNIT_MM * 1000
    cuts = record_cut_parameters(rec, L)
    mate = rec["mate"]
    mate_cuts = record_cut_parameters(mate, L)
    # Map mate orientation to rec orientation on the torus.
    da = np.mod(rec["a"] - mate["a"] + .5*L, L) - .5*L
    db = np.mod(rec["a"] - mate["b"] + .5*L, L) - .5*L
    same = np.linalg.norm(da) < np.linalg.norm(db)
    cuts.extend(mate_cuts if same else [1-c for c in mate_cuts])
    cuts = np.asarray([c for c in cuts if 1e-12 < c < 1-1e-12])
    if len(cuts) == 0 or len(ts) <= 2:
        return ts
    out = ts.copy()
    clearance = min(3.0 / length_um, 0.22 * np.median(np.diff(out)))
    minsep = min(4.0 / length_um, 0.25 * np.median(np.diff(out)))
    for _ in range(3):
        for i in range(1, len(out)-1):
            nearest = cuts[np.argmin(abs(cuts-out[i]))]
            if abs(nearest-out[i]) >= clearance:
                continue
            candidates = [nearest-clearance, nearest+clearance]
            candidates = [x for x in candidates if out[i-1]+minsep <= x <= out[i+1]-minsep]
            if candidates:
                out[i] = min(candidates, key=lambda x: abs(x-ts[i]))
    return out


def clip_polygon(poly: list[np.ndarray], axis: int, value: float, keep_greater: bool) -> list[np.ndarray]:
    if not poly:
        return []
    out = []
    for p, q in zip(poly, poly[1:] + poly[:1]):
        pin = p[axis] >= value - 2e-14 if keep_greater else p[axis] <= value + 2e-14
        qin = q[axis] >= value - 2e-14 if keep_greater else q[axis] <= value + 2e-14
        if pin:
            out.append(p)
        if pin != qin:
            den = q[axis] - p[axis]
            t = 0.0 if abs(den) < 1e-30 else (value - p[axis]) / den
            r = p + t * (q - p)
            r[axis] = value
            out.append(r)
    return out


def clip_to_box(tri: np.ndarray, L: float) -> list[np.ndarray]:
    poly = [p.copy() for p in tri]
    for axis, value, greater in ((0, 0.0, True), (0, L, False),
                                 (1, 0.0, True), (1, L, False)):
        poly = clip_polygon(poly, axis, value, greater)
    # Remove sequential duplicates introduced exactly at corners.
    clean = []
    for p in poly:
        if not clean or np.linalg.norm(p - clean[-1]) > 1e-13:
            clean.append(p)
    if len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) <= 1e-13:
        clean.pop()
    return clean


def best_quad_diagonal(p00, p10, p11, p01):
    options = [((p00, p10, p11), (p00, p11, p01)),
               ((p00, p10, p01), (p10, p11, p01))]
    def min_quality(pair):
        vals = []
        for tri in pair:
            t = np.asarray(tri)
            lens = np.linalg.norm(t - np.roll(t, -1, axis=0), axis=1)
            area = 0.5 * abs(np.cross(t[1] - t[0], t[2] - t[0]))
            vals.append(4.0 * math.sqrt(3.0) * area / np.dot(lens, lens))
        return min(vals)
    return max(options, key=min_quality)


def best_convex_triangulation(poly: list[np.ndarray]):
    """Maximize the worst triangle quality in a small convex clipped polygon."""
    p = np.asarray(poly)
    n = len(p)
    if n == 3:
        return [(p[0], p[1], p[2])]
    score = [[-1.0] * n for _ in range(n)]
    tris = [[None] * n for _ in range(n)]
    for i in range(n - 1):
        score[i][i + 1] = 1.0
        tris[i][i + 1] = []
    for gap in range(2, n):
        for i in range(n - gap):
            j = i + gap
            best_score, best_tris = -1.0, None
            for k in range(i + 1, j):
                tri = p[[i, k, j]]
                lens = np.linalg.norm(tri - np.roll(tri, -1, axis=0), axis=1)
                area = 0.5 * abs(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
                q = 4.0 * math.sqrt(3.0) * area / np.dot(lens, lens)
                candidate = min(score[i][k], score[k][j], q)
                if candidate > best_score:
                    best_score = candidate
                    best_tris = tris[i][k] + tris[k][j] + [(p[i], p[k], p[j])]
            score[i][j], tris[i][j] = best_score, best_tris
    return tris[0][n - 1]


def triangle_min_angle(tri: np.ndarray) -> float:
    lens = np.linalg.norm(tri - np.roll(tri, -1, axis=0), axis=1)
    values = []
    for i in range(3):
        a, b, c = lens[(i+2) % 3], lens[i], lens[(i+1) % 3]
        values.append(math.degrees(math.acos(np.clip((a*a+b*b-c*c)/(2*a*b), -1, 1))))
    return min(values)


def refine_periodic_cut_intervals(ts: np.ndarray, rec: dict, L: float,
                                  target_angle: float = 6.0,
                                  max_rounds: int = 10) -> np.ndarray:
    """Refine only longitudinal intervals whose clipped cells form slivers."""
    primary = rec
    mate = rec["mate"]
    da = np.mod(primary["a"] - mate["a"] + .5*L, L) - .5*L
    db = np.mod(primary["a"] - mate["b"] + .5*L, L) - .5*L
    mate_same = np.linalg.norm(da) < np.linalg.norm(db)

    def side_interval(side, t0, t1):
        if side is mate and not mate_same:
            return 1-t1, 1-t0
        return t0, t1

    for _ in range(max_rounds):
        split = set()
        for j, (t0, t1) in enumerate(zip(ts[:-1], ts[1:])):
            worst = 90.0
            saw_cut = False
            for side in (primary, mate):
                u0, u1 = side_interval(side, t0, t1)
                for k in range(N_HALF_LAYERS):
                    s0, s1 = k/N_HALF_LAYERS, (k+1)/N_HALF_LAYERS
                    def point(s, u):
                        po = (1-u)*side["a"] + u*side["b"]
                        pi = (1-u)*side["ai"] + u*side["bi"]
                        return (1-s)*po + s*pi
                    quad = np.asarray([point(s0,u0), point(s0,u1),
                                       point(s1,u1), point(s1,u0)])
                    lo, hi = quad.min(0), quad.max(0)
                    for sx in (-L, 0.0, L):
                        if hi[0]+sx < -1e-14 or lo[0]+sx > L+1e-14: continue
                        for sy in (-L, 0.0, L):
                            if hi[1]+sy < -1e-14 or lo[1]+sy > L+1e-14: continue
                            poly = clip_to_box(quad+np.array([sx,sy]), L)
                            if len(poly) < 3: continue
                            on_cut = any(abs(p[0])<1e-13 or abs(p[0]-L)<1e-13 or
                                         abs(p[1])<1e-13 or abs(p[1]-L)<1e-13 for p in poly)
                            if not on_cut: continue
                            saw_cut = True
                            for tri in best_convex_triangulation(poly):
                                worst = min(worst, triangle_min_angle(np.asarray(tri)))
            if saw_cut and worst < target_angle:
                split.add(j)
        if not split:
            break
        new_ts = [ts[0]]
        for j in range(len(ts)-1):
            if j in split: new_ts.append(.5*(ts[j]+ts[j+1]))
            new_ts.append(ts[j+1])
        ts = np.asarray(new_ts)
    return ts


def snap_row_nodes_to_cut(ts: np.ndarray, a: np.ndarray, b: np.ndarray,
                          L: float, clearance_um: float = 4.0) -> np.ndarray:
    """Move a nearby row node to its exact line/cut intersection.

    Movement is along the exact straight ridge/gas/layer line, so it changes
    only mesh subdivision, never a geometric boundary.  Row-wise snapping
    lets a structured strip terminate conformingly at a Cartesian cut.
    """
    out = ts.copy()
    length_um = np.linalg.norm(b-a) * MODEL_UNIT_MM * 1000
    if length_um == 0: return out
    for cut in line_cut_parameters(a, b, L):
        if len(out) <= 2: continue
        j = 1 + int(np.argmin(abs(out[1:-1]-cut)))
        if abs(out[j]-cut)*length_um <= clearance_um:
            if out[j-1] + 1e-10 < cut < out[j+1] - 1e-10:
                out[j] = cut
    return out


def construct_mesh(L: float, n: int, label: str,
                   cut_origin: tuple[float, float] = (0.0, 0.0)) -> dict:
    start = time.perf_counter()
    base, cells = build_cells(L, n, cut_origin=cut_origin)
    paired = pair_ridges(cells, L)
    raw = []
    ridge_samples = {}
    ridge_parameter_cache = {}
    gas_area = sum(polygon_area(c["inner"]) for c in cells)

    for cell in cells:
        outer, inner = cell["outer"], cell["inner"]
        for e in range(len(outer)):
            rec = paired["records"][(cell["id"], e)]
            a, b, ai, bi = rec["a"], rec["b"], rec["ai"], rec["bi"]
            mate = rec["mate"]
            tangent = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-tangent[1], tangent[0]])
            one_a = abs(np.dot(ai - a, normal))
            one_b = abs(np.dot(bi - b, normal))
            ma, mb = mate["a"], mate["b"]
            mai, mbi = mate["ai"], mate["bi"]
            one_m_a = abs(np.dot(mai - ma, normal))
            one_m_b = abs(np.dot(mbi - mb, normal))
            width = 0.5 * (one_a + one_b + one_m_a + one_m_b)
            normal_h_um = width * MODEL_UNIT_MM * 1000 / (2 * N_HALF_LAYERS)
            if rec["ridge"] not in ridge_parameter_cache:
                base_ts = tangential_parameters(np.linalg.norm(b - a), normal_h_um)
                base_ts = cut_aware_parameters(base_ts, rec, L)
                ridge_parameter_cache[rec["ridge"]] = (a.copy(), b.copy(), base_ts)
            ca, cb, cts = ridge_parameter_cache[rec["ridge"]]
            da = np.mod(a-ca+.5*L, L)-.5*L
            db = np.mod(a-cb+.5*L, L)-.5*L
            ts = cts if np.linalg.norm(da) < np.linalg.norm(db) else 1-cts[::-1]
            if rec["ridge"] not in ridge_samples:
                intervals_um = np.diff(ts) * np.linalg.norm(b - a) * MODEL_UNIT_MM * 1000
                ridge_samples[rec["ridge"]] = {
                    "width_um": width * MODEL_UNIT_MM * 1000,
                    "normal_h_um": normal_h_um,
                    "n_through": 2 * N_HALF_LAYERS,
                    "tangential_um": intervals_um.tolist(),
                    "aspect_design": (intervals_um / (width * MODEL_UNIT_MM * 1000 /
                                                         (2 * N_HALF_LAYERS))).tolist(),
                    "midpoint": np.mod(0.5 * (a + b), L).tolist(),
                    "tangent": tangent.tolist(),
                }

            grid = np.empty((N_HALF_LAYERS + 1, len(ts), 2), dtype=float)
            for k in range(N_HALF_LAYERS + 1):
                s = k / N_HALF_LAYERS
                row_a = (1-s)*a + s*ai
                row_b = (1-s)*b + s*bi
                row_ts = snap_row_nodes_to_cut(ts, row_a, row_b, L)
                for j, t in enumerate(row_ts):
                    pout = (1 - t) * a + t * b
                    pin = (1 - t) * ai + t * bi
                    grid[k, j] = (1 - s) * pout + s * pin
            for k in range(N_HALF_LAYERS):
                for j in range(len(ts) - 1):
                    p00, p10 = grid[k, j], grid[k, j + 1]
                    p01, p11 = grid[k + 1, j], grid[k + 1, j + 1]
                    junction = j == 0 or j == len(ts) - 2
                    raw.append((np.asarray([p00, p10, p11, p01]), junction,
                                k, rec["ridge"], cell["id"]))

    points, triangles = [], []
    junction_flags, seam_flags, layers, ridge_ids, cell_ids = [], [], [], [], []
    point_map, triangle_set = {}, set()

    def point_id(p):
        p = np.clip(p, 0.0, L)
        key = tuple(np.round(p, ROUND_DIGITS))
        if key not in point_map:
            point_map[key] = len(points)
            points.append(np.asarray(key))
        return point_map[key]

    # Canonical polygons may cross a cut.  Periodic copies are clipped to the
    # central cell, making paired boundary nodes algebraically identical.
    for quad, junction, layer, rid, cid in raw:
        lo, hi = quad.min(axis=0), quad.max(axis=0)
        for sx in (-L, 0.0, L):
            if hi[0] + sx < -1e-14 or lo[0] + sx > L + 1e-14:
                continue
            for sy in (-L, 0.0, L):
                if hi[1] + sy < -1e-14 or lo[1] + sy > L + 1e-14:
                    continue
                poly = clip_to_box(quad + np.array([sx, sy]), L)
                if len(poly) < 3:
                    continue
                on_seam = any(abs(q[0]) < 1e-13 or abs(q[0]-L) < 1e-13 or
                              abs(q[1]) < 1e-13 or abs(q[1]-L) < 1e-13 for q in poly)
                for tri in best_convex_triangulation(poly):
                    verts = np.asarray(tri)
                    area2 = np.cross(verts[1] - verts[0], verts[2] - verts[0])
                    if abs(area2) <= 1e-18:
                        continue
                    if area2 < 0:
                        verts[[1, 2]] = verts[[2, 1]]
                    ids = tuple(point_id(p) for p in verts)
                    identity = tuple(sorted(ids))
                    if identity in triangle_set:
                        continue
                    triangle_set.add(identity)
                    triangles.append(ids)
                    junction_flags.append(junction)
                    seam_flags.append(on_seam)
                    layers.append(layer)
                    ridge_ids.append(rid)
                    cell_ids.append(cid)

    return {
        "label": label, "L": L, "n": n, "base": base, "cells": cells,
        "cut_origin": np.mod(np.asarray(cut_origin, dtype=float), L),
        "ridge_count": len(paired["groups"]), "ridge_samples": ridge_samples,
        "gas_area": gas_area, "points": np.asarray(points),
        "triangles": np.asarray(triangles, dtype=np.int32),
        "junction": np.asarray(junction_flags, dtype=bool),
        "seam": np.asarray(seam_flags, dtype=bool),
        "layer": np.asarray(layers, dtype=np.int8),
        "ridge": np.asarray(ridge_ids, dtype=np.int32),
        "cell": np.asarray(cell_ids, dtype=np.int32),
        "generation_seconds": time.perf_counter() - start,
    }


def mesh_metrics(mesh: dict) -> dict:
    p, t = mesh["points"], mesh["triangles"]
    xy = p[t]
    vec = xy - np.roll(xy, -1, axis=1)
    lens = np.linalg.norm(vec, axis=2)
    signed2 = np.cross(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0])
    area = 0.5 * signed2
    quality = 4.0 * np.sqrt(3.0) * area / np.sum(lens ** 2, axis=1)
    cosang = np.empty((len(t), 3))
    for i in range(3):
        a = lens[:, (i + 2) % 3]
        b = lens[:, i]
        c = lens[:, (i + 1) % 3]
        cosang[:, i] = np.clip((a*a + b*b - c*c) / (2*a*b), -1, 1)
    angles = np.degrees(np.arccos(cosang))
    min_angles = angles.min(axis=1)
    altitude = 2 * area / lens.max(axis=1)
    aspect = lens.max(axis=1) / altitude

    edge_map = defaultdict(list)
    for ti, tri in enumerate(t):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_map[tuple(sorted((int(a), int(b))))].append(ti)
    unique_edges = np.asarray(list(edge_map), dtype=np.int32)
    edge_lengths = np.linalg.norm(p[unique_edges[:, 0]] - p[unique_edges[:, 1]], axis=1)
    boundary_edges = [e for e, own in edge_map.items() if len(own) == 1]

    adjacency = [[] for _ in range(len(t))]
    for own in edge_map.values():
        if len(own) == 2:
            a, b = own
            adjacency[a].append(b); adjacency[b].append(a)
        elif len(own) > 2:
            raise RuntimeError(f"Non-manifold edge with {len(own)} incident triangles")
    visited = np.zeros(len(t), dtype=bool)
    components = []
    for root in range(len(t)):
        if visited[root]:
            continue
        q = deque([root]); visited[root] = True; count = 0
        while q:
            u = q.popleft(); count += 1
            for v in adjacency[u]:
                if not visited[v]: visited[v] = True; q.append(v)
        components.append(count)

    L = mesh["L"]
    tol = 2e-11
    sides = {
        "left": np.where(abs(p[:, 0]) < tol)[0],
        "right": np.where(abs(p[:, 0] - L) < tol)[0],
        "bottom": np.where(abs(p[:, 1]) < tol)[0],
        "top": np.where(abs(p[:, 1] - L) < tol)[0],
    }
    lr_y = [np.round(np.sort(p[sides[s], 1]), ROUND_DIGITS) for s in ("left", "right")]
    bt_x = [np.round(np.sort(p[sides[s], 0]), ROUND_DIGITS) for s in ("bottom", "top")]

    def boundary_intervals(side, axis_fixed, value, free_axis):
        out = []
        for a, b in boundary_edges:
            if abs(p[a, axis_fixed] - value) < tol and abs(p[b, axis_fixed] - value) < tol:
                out.append(tuple(np.round(sorted((p[a, free_axis], p[b, free_axis])), ROUND_DIGITS)))
        return sorted(out)
    lr_edges = [boundary_intervals("left", 0, 0.0, 1),
                boundary_intervals("right", 0, L, 1)]
    bt_edges = [boundary_intervals("bottom", 1, 0.0, 0),
                boundary_intervals("top", 1, L, 0)]

    # Count components on the physical torus as well as in the cut-open square.
    parent = np.arange(len(t), dtype=np.int32)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return int(a)
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for own in edge_map.values():
        if len(own) == 2: union(own[0], own[1])
    incident = defaultdict(list)
    for ti, tri in enumerate(t):
        for v in tri: incident[int(v)].append(ti)
    for master, slave, free_axis in ((sides["left"], sides["right"], 1),
                                     (sides["bottom"], sides["top"], 0)):
        slave_by_coord = {round(float(p[v, free_axis]), ROUND_DIGITS): int(v) for v in slave}
        for u in master:
            v = slave_by_coord[round(float(p[u, free_axis]), ROUND_DIGITS)]
            if incident[int(u)] and incident[v]: union(incident[int(u)][0], incident[v][0])
    periodic_component_sizes = defaultdict(int)
    for ti in range(len(t)): periodic_component_sizes[find(ti)] += 1

    ridge_data = list(mesh["ridge_samples"].values())
    widths = np.asarray([r["width_um"] for r in ridge_data])
    normal_h = np.asarray([r["normal_h_um"] for r in ridge_data])
    n_through = np.asarray([r["n_through"] for r in ridge_data])
    tangential = np.concatenate([r["tangential_um"] for r in ridge_data])
    design_aspect = np.concatenate([r["aspect_design"] for r in ridge_data])

    def stats(x, include_max=True):
        x = np.asarray(x, dtype=float)
        result = {"min": float(x.min()), "p05": float(np.percentile(x, 5)),
                  "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
                  "p99": float(np.percentile(x, 99))}
        if include_max: result["max"] = float(x.max())
        return result
    def region_stats(mask):
        count = int(mask.sum())
        if count == 0:
            return {"triangles": 0}
        return {
            "triangles": count,
            "quality": stats(quality[mask]),
            "minimum_angle_deg": stats(min_angles[mask]),
            "aspect_ratio_longest_edge_over_shortest_altitude": stats(aspect[mask]),
            "angle_lt_5_deg": int(np.count_nonzero(min_angles[mask] < 5.0)),
            "angle_lt_8_deg": int(np.count_nonzero(min_angles[mask] < 8.0)),
            "aspect_ratio_gt_10": int(np.count_nonzero(aspect[mask] > 10.0)),
            "aspect_ratio_gt_15": int(np.count_nonzero(aspect[mask] > 15.0)),
        }

    tri_on_lr = np.any((abs(xy[:, :, 0]) < tol) | (abs(xy[:, :, 0] - L) < tol), axis=1)
    tri_on_bt = np.any((abs(xy[:, :, 1]) < tol) | (abs(xy[:, :, 1] - L) < tol), axis=1)
    tri_on_corner = tri_on_lr & tri_on_bt

    return {
        "vertices": len(p), "triangles": len(t), "boundary_facets": len(boundary_edges),
        "area_model2": float(area.sum()), "expected_wall_area_model2": float(L*L-mesh["gas_area"]),
        "relative_area_error": float(abs(area.sum() - (L*L-mesh["gas_area"])) / (L*L-mesh["gas_area"])),
        "edge_length_model": stats(edge_lengths),
        "edge_length_um": stats(edge_lengths * MODEL_UNIT_MM * 1000),
        "triangle_quality_4sqrt3A_over_sum_l2": stats(quality),
        "minimum_angle_deg": stats(min_angles),
        "aspect_ratio_longest_edge_over_shortest_altitude": stats(aspect),
        "degenerate_or_inverted": int(np.count_nonzero(area <= 1e-16)),
        "cut_open_connected_components": len(components),
        "cut_open_component_triangle_counts": sorted(components, reverse=True)[:10],
        "periodic_connected_components": len(periodic_component_sizes),
        "periodic_component_triangle_counts": sorted(periodic_component_sizes.values(), reverse=True)[:10],
        "periodic": {
            "left_nodes": len(sides["left"]), "right_nodes": len(sides["right"]),
            "bottom_nodes": len(sides["bottom"]), "top_nodes": len(sides["top"]),
            "left_right_node_match": bool(np.array_equal(lr_y[0], lr_y[1])),
            "bottom_top_node_match": bool(np.array_equal(bt_x[0], bt_x[1])),
            "left_right_edge_match": lr_edges[0] == lr_edges[1],
            "bottom_top_edge_match": bt_edges[0] == bt_edges[1],
            "left_edges": len(lr_edges[0]), "right_edges": len(lr_edges[1]),
            "bottom_edges": len(bt_edges[0]), "top_edges": len(bt_edges[1]),
        },
        "through_wall": {
            "sampled_paired_septa": len(ridge_data),
            "wall_width_um": stats(widths),
            "elements_through_full_wall": stats(n_through),
            "normal_element_size_um": stats(normal_h),
            "tangential_interval_size_um": stats(tangential),
            "tangential_over_normal_design_ratio": stats(design_aspect),
        },
        "regular_strips_excluding_periodic_cut": region_stats(~mesh["junction"] & ~mesh["seam"]),
        "junction_regions_excluding_periodic_cut": region_stats(mesh["junction"] & ~mesh["seam"]),
        "periodic_cut_regions": region_stats(mesh["seam"]),
        "validation_regions": {
            "all": region_stats(np.ones(len(t), dtype=bool)),
            "interior_strips": region_stats(~mesh["junction"] & ~mesh["seam"]),
            "junctions": region_stats(mesh["junction"] & ~mesh["seam"]),
            "left_right_periodic_cut_excluding_corners": region_stats(tri_on_lr & ~tri_on_corner),
            "bottom_top_periodic_cut_excluding_corners": region_stats(tri_on_bt & ~tri_on_corner),
            "periodic_corners": region_stats(tri_on_corner),
        },
        "generation_seconds": mesh["generation_seconds"],
    }


def mesh_segments(mesh: dict, mask=None):
    tri = mesh["triangles"] if mask is None else mesh["triangles"][mask]
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return mesh["points"][edges]


def plot_prototype(mesh, path):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.add_collection(LineCollection(mesh_segments(mesh, ~mesh["junction"]), colors="#285f8f", linewidths=.35))
    ax.add_collection(LineCollection(mesh_segments(mesh, mesh["junction"]), colors="#d1495b", linewidths=.55))
    ax.set(xlim=(0, mesh["L"]), ylim=(0, mesh["L"]), aspect="equal",
           title="Same-scale 8×8 periodic prototype\nblue: regular strip; red: junction transition")
    ax.set_xlabel("model x"); ax.set_ylabel("model y")
    fig.savefig(path, dpi=260); plt.close(fig)


def plot_full(mesh, path):
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    ax.add_collection(LineCollection(mesh_segments(mesh), colors="#263238", linewidths=.025, rasterized=True))
    ax.set(xlim=(0, mesh["L"]), ylim=(0, mesh["L"]), aspect="equal",
           title=f"70×70 directional FE wall mesh ({len(mesh['triangles']):,} triangles)")
    ax.set_xlabel("model x"); ax.set_ylabel("model y")
    fig.savefig(path, dpi=300); plt.close(fig)


def central_window_mask(mesh, center, radius):
    cent = mesh["points"][mesh["triangles"]].mean(axis=1)
    return ((abs(cent[:, 0] - center[0]) < radius) &
            (abs(cent[:, 1] - center[1]) < radius))


def plot_zoom(mesh, path):
    center = np.array([0.5 * mesh["L"], 0.5 * mesh["L"]])
    radius = 0.055
    mask = central_window_mask(mesh, center, radius)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.add_collection(LineCollection(mesh_segments(mesh, mask & ~mesh["junction"]), colors="#1f77b4", linewidths=.55))
    ax.add_collection(LineCollection(mesh_segments(mesh, mask & mesh["junction"]), colors="#d62728", linewidths=.8))
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           aspect="equal", title="Directional mesh zoom: long tangential / short normal edges")
    ax.set_xlabel("model x"); ax.set_ylabel("model y")
    fig.savefig(path, dpi=300); plt.close(fig)


def choose_ridge(mesh, target=None):
    if target is None: target = np.array([0.5*mesh["L"], 0.5*mesh["L"]])
    records = mesh["ridge_samples"]
    return min(records, key=lambda r: np.linalg.norm(np.asarray(records[r]["midpoint"])-target))


def plot_layers(mesh, path):
    rid = choose_ridge(mesh)
    mask = mesh["ridge"] == rid
    xy = mesh["points"][mesh["triangles"][mask]].reshape(-1, 2)
    lo, hi = xy.min(axis=0), xy.max(axis=0); pad = .004
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = np.where(mesh["layer"][mask] == 0, "#4c78a8", "#f58518")
    coll = PolyCollection(mesh["points"][mesh["triangles"][mask]], facecolors=colors,
                          edgecolors="black", linewidths=.45)
    ax.add_collection(coll)
    ax.set(xlim=(lo[0]-pad, hi[0]+pad), ylim=(lo[1]-pad, hi[1]+pad), aspect="equal",
           title="Four full-septum layers: two layers on each gas-cell side")
    ax.set_xlabel("model x"); ax.set_ylabel("model y")
    fig.savefig(path, dpi=320); plt.close(fig)


def plot_junction(mesh, path):
    # A Voronoi vertex nearest the centre; all incident endpoint-transition
    # triangles are shown, including the three cell sectors.
    verts = np.vstack([c["outer"] for c in mesh["cells"]])
    target = np.array([0.5*mesh["L"], 0.5*mesh["L"]])
    v = verts[np.argmin(np.linalg.norm(verts-target, axis=1))]
    v = np.mod(v, mesh["L"])
    radius = .012
    mask = central_window_mask(mesh, v, radius)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    faces = mesh["points"][mesh["triangles"][mask]]
    cols = np.where(mesh["junction"][mask], "#e45756", "#72b7b2")
    ax.add_collection(PolyCollection(faces, facecolors=cols, edgecolors="black", linewidths=.45))
    ax.plot(v[0], v[1], "ko", ms=3)
    ax.set(xlim=(v[0]-radius, v[0]+radius), ylim=(v[1]-radius, v[1]+radius),
           aspect="equal", title="Hybrid triple-junction transition (red) and regular strips (green)")
    ax.set_xlabel("model x"); ax.set_ylabel("model y")
    fig.savefig(path, dpi=320); plt.close(fig)


def plot_quality(mesh, path):
    p, t = mesh["points"], mesh["triangles"]
    xy = p[t]; lens = np.linalg.norm(xy - np.roll(xy, -1, axis=1), axis=2)
    area = .5*np.cross(xy[:,1]-xy[:,0], xy[:,2]-xy[:,0])
    quality = 4*np.sqrt(3)*area/np.sum(lens*lens, axis=1)
    altitude = 2*area/lens.max(axis=1); aspect=lens.max(axis=1)/altitude
    cosang=np.empty((len(t),3))
    for i in range(3):
        a=lens[:,(i+2)%3]; b=lens[:,i]; c=lens[:,(i+1)%3]
        cosang[:,i]=np.clip((a*a+b*b-c*c)/(2*a*b),-1,1)
    amin=np.degrees(np.arccos(cosang)).min(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].hist(quality, bins=80, color="#4c78a8"); axes[0].set(xlabel="triangle quality", ylabel="count")
    axes[1].hist(amin, bins=80, color="#59a14f"); axes[1].set(xlabel="minimum angle (deg)")
    axes[2].hist(np.clip(aspect,0,20), bins=80, color="#f28e2b"); axes[2].set(xlabel="aspect ratio (clipped at 20)")
    fig.suptitle("70×70 directional mesh quality distributions")
    fig.savefig(path, dpi=260); plt.close(fig)


def write_mesh(mesh, basename: Path):
    points3 = np.c_[mesh["points"], np.zeros(len(mesh["points"]))]
    tags = np.full(len(mesh["triangles"]), 2, dtype=np.int32)
    msh = meshio.Mesh(points3, [("triangle", mesh["triangles"])],
                      cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
                      field_data={"voronoi_wall": np.array([2, 2])})
    meshio.write(str(basename)+".msh", msh, file_format="gmsh22", binary=True)
    meshio.write(str(basename)+".xdmf", meshio.Mesh(mesh["points"], [("triangle", mesh["triangles"])]))
    meshio.write(str(basename)+"_domains.xdmf",
                 meshio.Mesh(mesh["points"], [("triangle", mesh["triangles"])],
                             cell_data={"domains": [tags]}))
    np.savez_compressed(str(basename)+"_directional_metadata.npz",
                        junction=mesh["junction"], periodic_cut=mesh["seam"], half_layer=mesh["layer"],
                        ridge_id=mesh["ridge"], canonical_cell_id=mesh["cell"],
                        periodic_cut_origin=np.asarray(mesh["cut_origin"], dtype=float),
                        domain_length=np.asarray(mesh["L"], dtype=float),
                        coordinate_transform=np.asarray(
                            "translated = mod(original - periodic_cut_origin, domain_length)"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/physiological_geometry_70x70_directional")
    parser.add_argument("--mesh-prefix", default="mesh/Mesh_Acinar_Perfusion_Physiological_70x70_Directional")
    parser.add_argument("--cut-origin-x", type=float, default=0.0,
                        help="Periodic-window x origin in the original torus coordinates")
    parser.add_argument("--cut-origin-y", type=float, default=0.0,
                        help="Periodic-window y origin in the original torus coordinates")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.output_dir; out.mkdir(parents=True, exist_ok=True)

    prototype_L = DOMAIN_L * 8 / GRID_N
    prototype = construct_mesh(prototype_L, 8, "8x8_same_scale")
    prototype_metrics = mesh_metrics(prototype)
    prototype_ok = (
        prototype_metrics["degenerate_or_inverted"] == 0 and
        prototype_metrics["periodic_connected_components"] == 1 and
        prototype_metrics["periodic"]["left_right_node_match"] and
        prototype_metrics["periodic"]["bottom_top_node_match"] and
        prototype_metrics["periodic"]["left_right_edge_match"] and
        prototype_metrics["periodic"]["bottom_top_edge_match"] and
        prototype_metrics["through_wall"]["elements_through_full_wall"]["min"] >= 4 and
        # Exact fixed cuts can create a few high-aspect cut-open fragments when
        # a septum is almost parallel to a cut.  They are audited separately.
        prototype_metrics["minimum_angle_deg"]["min"] > 1.0
    )
    plot_prototype(prototype, out / "anisotropic_mesh_prototype.png")
    with open(out / "prototype_validation.json", "w") as f:
        json.dump({"passed": prototype_ok, "metrics": prototype_metrics}, f, indent=2)
    if not prototype_ok:
        raise RuntimeError("Directional prototype failed its validation gate; full mesh was not generated")

    cut_origin = (args.cut_origin_x % DOMAIN_L, args.cut_origin_y % DOMAIN_L)
    full = construct_mesh(DOMAIN_L, GRID_N, "70x70", cut_origin=cut_origin)
    full_metrics = mesh_metrics(full)
    basename = root / args.mesh_prefix
    write_mesh(full, basename)

    plot_full(full, out / "physiological_70x70_directional_mesh.png")
    plot_zoom(full, out / "physiological_70x70_directional_mesh_zoom.png")
    plot_layers(full, out / "physiological_70x70_through_wall_layers.png")
    plot_junction(full, out / "physiological_70x70_junction_mesh.png")
    plot_quality(full, out / "physiological_70x70_mesh_quality.png")

    original_base = generate_base_seeds(DOMAIN_L, GRID_N, DOI, RNG_SEED)
    seeds_hash = hashlib.sha256(np.ascontiguousarray(original_base).tobytes()).hexdigest()
    shifted_seeds_hash = hashlib.sha256(np.ascontiguousarray(full["base"]).tobytes()).hexdigest()
    # Topological estimate for the current vector-CG2/scalar-CG1 mixed space.
    # The exact reference value below was also measured by constructing the
    # constrained FEniCS FunctionSpace on this output mesh (without solving).
    edge_count = (3 * full_metrics["triangles"] + full_metrics["boundary_facets"]) // 2
    unconstrained_dofs = 3 * full_metrics["vertices"] + 2 * edge_count + 6
    vertex_pairs = (full_metrics["periodic"]["right_nodes"] +
                    full_metrics["periodic"]["top_nodes"] - 1)
    edge_pairs = (full_metrics["periodic"]["right_edges"] +
                  full_metrics["periodic"]["top_edges"])
    topological_dofs = unconstrained_dofs - 3 * vertex_pairs - 2 * edge_pairs
    reference_key = (round(cut_origin[0], 12), round(cut_origin[1], 12),
                     full_metrics["vertices"], full_metrics["triangles"])
    measured_subspaces = MEASURED_MIXED_SPACE_REFERENCES.get(reference_key)
    dofs = sum(measured_subspaces.values()) if measured_subspaces else topological_dofs
    ref_dofs, ref_mem, ref_time = 241592, 4.12, 275.0
    estimates = {
        "mixed_space": "2D vector CG2 displacement + scalar CG1 liquid pressure + 2x2 real macroscopic stretch + real surface area + real pressure zero-mean multiplier",
        "dof_count_method": ("Direct read-only FEniCS FunctionSpace dimension on this exact mesh; no FE solve"
                             if measured_subspaces else
                             "Topological periodic estimate (3V + 2E + 6 with paired boundary vertex/edge reductions)"),
        "mesh_edges": int(edge_count),
        "unconstrained_mixed_dofs": int(unconstrained_dofs),
        "periodic_vertex_pairs": int(vertex_pairs),
        "periodic_edge_pairs": int(edge_pairs),
        "topological_periodic_estimate": int(topological_dofs),
        "measured_subspace_dimensions": measured_subspaces,
        "estimated_mixed_dofs": int(dofs),
        "linear_memory_GB": dofs / ref_dofs * ref_mem,
        "linear_runtime_seconds": dofs / ref_dofs * ref_time,
        "linear_runtime_hours": dofs / ref_dofs * ref_time / 3600,
        "warning": "Extrapolations only. The current direct nonlinear solve can scale superlinearly in memory and time.",
    }
    periodic_ok = all(full_metrics["periodic"][key] for key in (
        "left_right_node_match", "bottom_top_node_match",
        "left_right_edge_match", "bottom_top_edge_match"))
    layers_ok = full_metrics["through_wall"]["elements_through_full_wall"]["min"] >= 4
    minimum_angle = full_metrics["minimum_angle_deg"]["min"]
    maximum_aspect = full_metrics["aspect_ratio_longest_edge_over_shortest_altitude"]["max"]
    solver_ready = (periodic_ok and layers_ok and
                    full_metrics["degenerate_or_inverted"] == 0 and
                    minimum_angle > 5.0 and maximum_aspect < 15.0)
    preferred_quality = solver_ready and minimum_angle > 8.0 and maximum_aspect < 10.0
    if preferred_quality:
        status = "directional_mesh_solver_ready_preferred_quality"
        acceptance_reason = "The translated cut removes the severe seam population and meets the preferred global angle/aspect targets while retaining periodic correspondence and four wall-normal layers."
    elif solver_ready:
        status = "directional_mesh_solver_ready"
        acceptance_reason = "The translated cut meets the required global angle/aspect targets while retaining periodic correspondence and four wall-normal layers."
    else:
        status = "directional_mesh_generated_with_periodic_cut_quality_failure"
        acceptance_reason = "At least one global quality, periodic-correspondence, degeneracy, or through-wall-layer requirement failed; a periodic seam-band remesh is required."
    report = {
        "status": status,
        "method": {
            "description": "Conforming paired-ridge strip mesh with two layers per one-sided cell ring (four across each full shared septum), endpoint-graded tangential intervals, and conforming endpoint-sector transitions at triple junctions.",
            "normal_layers_per_cell_side": N_HALF_LAYERS,
            "normal_layers_per_full_septum": 2*N_HALF_LAYERS,
            "endpoint_target_um": ENDPOINT_SPACING_UM,
            "maximum_regular_interior_target_um": MAX_INTERIOR_SPACING_UM,
            "periodic_cut_method": "Translate exact periodic strip copies, clip to the fundamental square, and coordinate-deduplicate; opposite cut node and edge sets are checked exactly after rounding to 1e-12 model units.",
        },
        "geometry": {
            "canonical_cells": GRID_N*GRID_N, "domain_model": [0, DOMAIN_L, 0, DOMAIN_L],
            "domain_physical_mm": [14.0, 14.0], "model_unit_mm": MODEL_UNIT_MM,
            "DoI": DOI, "rng_seed": RNG_SEED, "seed_sha256": seeds_hash,
            "translated_seed_sha256": shifted_seeds_hash,
            "periodic_cut_origin_model": [float(cut_origin[0]), float(cut_origin[1])],
            "periodic_cut_origin_physical_mm": [float(cut_origin[0] * MODEL_UNIT_MM),
                                                 float(cut_origin[1] * MODEL_UNIT_MM)],
            "coordinate_transform": "translated = mod(original - periodic_cut_origin, L)",
            "offset_distance_model": OFFSET_MODEL, "gas_area_model2": full["gas_area"],
            "gas_fraction": full["gas_area"]/(DOMAIN_L*DOMAIN_L),
            "wall_fraction": 1-full["gas_area"]/(DOMAIN_L*DOMAIN_L),
            "paired_periodic_ridges": full["ridge_count"],
        },
        "prototype": {"passed": prototype_ok, "metrics": prototype_metrics},
        "full_mesh": full_metrics,
        "acceptance": {
            "solver_ready": solver_ready,
            "preferred_quality": preferred_quality,
            "required_targets": {"minimum_angle_gt_deg": 5.0,
                                 "maximum_aspect_ratio_lt": 15.0},
            "preferred_targets": {"minimum_angle_gt_deg": 8.0,
                                  "maximum_aspect_ratio_lt": 10.0},
            "reason": acceptance_reason,
        },
        "computational_estimates": estimates,
        "files": {
            "msh": str(basename)+".msh", "xdmf": str(basename)+".xdmf",
            "h5": str(basename)+".h5", "domains_xdmf": str(basename)+"_domains.xdmf",
            "domains_h5": str(basename)+"_domains.h5",
            "directional_metadata": str(basename)+"_directional_metadata.npz",
        },
        "fe_solver_run": False, "acinar_partition_generated": False,
        "vascular_markers_generated": False,
    }
    with open(out / "directional_mesh_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
