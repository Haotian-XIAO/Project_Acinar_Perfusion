#!/usr/bin/env python3
"""Partition the accepted periodic Voronoi patch into labelled acinar territories.

The partition is metadata only.  It does not create mesh entities, physical
groups, interfaces, boundary conditions, or solver coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import Voronoi


DEFAULT_TERMINALS = np.array(
    [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
    ],
    dtype=float,
)

DEFAULT_AIRWAY_ROOT = np.array([0.50, 0.07], dtype=float)

DEFAULT_SPLIT_WEIGHTS = {
    "area": 2.0,
    "compactness": 2.0,
    "branch_length": 0.15,
}


BRANCHING_TEMPLATES = {
    "A_2plus2": {
        "template_id": "A",
        "description": "root bifurcation; both daughter branches bifurcate",
        "tree": {
            "node_id": "root",
            "node_type": "root",
            "children": [
                {
                    "node_id": "B0",
                    "node_type": "branch",
                    "children": [
                        {"node_id": "T0", "node_type": "terminal"},
                        {"node_id": "T1", "node_type": "terminal"},
                    ],
                },
                {
                    "node_id": "B1",
                    "node_type": "branch",
                    "children": [
                        {"node_id": "T2", "node_type": "terminal"},
                        {"node_id": "T3", "node_type": "terminal"},
                    ],
                },
            ],
        },
    },
    "B_root3_then2": {
        "template_id": "B",
        "description": "root trifurcation; one daughter branch bifurcates",
        "tree": {
            "node_id": "root",
            "node_type": "root",
            "children": [
                {"node_id": "T0", "node_type": "terminal"},
                {"node_id": "T1", "node_type": "terminal"},
                {
                    "node_id": "B0",
                    "node_type": "branch",
                    "children": [
                        {"node_id": "T2", "node_type": "terminal"},
                        {"node_id": "T3", "node_type": "terminal"},
                    ],
                },
            ],
        },
    },
    "C_2plus3": {
        "template_id": "C",
        "description": "root bifurcation into a bifurcating and a trifurcating branch",
        "tree": {
            "node_id": "root",
            "node_type": "root",
            "children": [
                {
                    "node_id": "B0",
                    "node_type": "branch",
                    "children": [
                        {"node_id": "T0", "node_type": "terminal"},
                        {"node_id": "T1", "node_type": "terminal"},
                    ],
                },
                {
                    "node_id": "B1",
                    "node_type": "branch",
                    "children": [
                        {"node_id": "T2", "node_type": "terminal"},
                        {"node_id": "T3", "node_type": "terminal"},
                        {"node_id": "T4", "node_type": "terminal"},
                    ],
                },
            ],
        },
    },
}


def generate_rectangular_base_seeds(
    bounds, grid_x=12, grid_y=12, irregularity=0.3, rng_seed=1
):
    """Reproduce the canonical seed population used by Mesh_Acinar_RVE.py."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        raise ValueError("The periodic cell must have positive width and height.")
    if grid_x < 2 or grid_y < 2:
        raise ValueError("grid_x and grid_y must both be at least two.")
    if not 0.0 <= irregularity < 1.0:
        raise ValueError("irregularity must satisfy 0 <= irregularity < 1.")

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
            x = xmin + ((x - xmin) % width)
            y = ymin + ((y - ymin) % height)
            seeds.append((x, y))
    return np.asarray(seeds, dtype=float)


def make_periodic_seed_cloud(base_seeds, a1, a2):
    """Create the central population and its eight nearest periodic copies."""
    points = []
    canonical_ids = []
    lattice_shifts = []
    n_cells = len(base_seeds)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            points.append(base_seeds + i * a1 + j * a2)
            canonical_ids.extend(range(n_cells))
            lattice_shifts.extend([(i, j)] * n_cells)
    return (
        np.vstack(points),
        np.asarray(canonical_ids, dtype=int),
        np.asarray(lattice_shifts, dtype=int),
    )


def polygon_area_and_centroid(vertices):
    """Return the unsigned area and centroid of a simple polygon."""
    xy = np.asarray(vertices, dtype=float)
    nxt = np.roll(xy, -1, axis=0)
    cross = xy[:, 0] * nxt[:, 1] - nxt[:, 0] * xy[:, 1]
    signed_area = 0.5 * np.sum(cross)
    if abs(signed_area) < 1e-14:
        raise RuntimeError("Degenerate Voronoi polygon encountered.")
    centroid = np.array(
        [
            np.sum((xy[:, 0] + nxt[:, 0]) * cross),
            np.sum((xy[:, 1] + nxt[:, 1]) * cross),
        ]
    ) / (6.0 * signed_area)
    return abs(float(signed_area)), centroid


def minimum_image_vector(delta, lattice_matrix, lattice_inverse):
    fractional = lattice_inverse @ np.asarray(delta, dtype=float)
    fractional -= np.round(fractional)
    return lattice_matrix @ fractional


def build_periodic_cell_graph(base_seeds, a1, a2, bounds):
    """Build canonical polygons and a periodic adjacency graph from ridges."""
    cloud, canonical_ids, shifts = make_periodic_seed_cloud(base_seeds, a1, a2)
    voronoi = Voronoi(cloud)
    central_indices = np.flatnonzero(np.all(shifts == 0, axis=1))
    if len(central_indices) != len(base_seeds):
        raise RuntimeError("Central periodic seed population is incomplete.")

    xmin, ymin, xmax, ymax = map(float, bounds)
    lengths = np.array([xmax - xmin, ymax - ymin])
    polygons = []
    areas = np.zeros(len(base_seeds), dtype=float)
    centroids = np.zeros_like(base_seeds)
    for point_index in central_indices:
        cell_id = int(canonical_ids[point_index])
        region = voronoi.regions[voronoi.point_region[point_index]]
        if not region or -1 in region:
            raise RuntimeError(f"Canonical cell {cell_id} has an unbounded region.")
        polygon = np.asarray([voronoi.vertices[k] for k in region], dtype=float)
        area, centroid = polygon_area_and_centroid(polygon)
        polygons.append(polygon)
        areas[cell_id] = area
        centroids[cell_id] = np.array(
            [xmin, ymin]
        ) + np.mod(centroid - np.array([xmin, ymin]), lengths)

    adjacency = [set() for _ in base_seeds]
    periodic_links = set()
    central_mask = np.all(shifts == 0, axis=1)
    for left, right in voronoi.ridge_points:
        if not (central_mask[left] or central_mask[right]):
            continue
        left_id = int(canonical_ids[left])
        right_id = int(canonical_ids[right])
        if left_id == right_id:
            continue
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)

        if central_mask[left]:
            shift = tuple(int(v) for v in shifts[right])
            periodic_links.add((left_id, right_id, shift[0], shift[1]))
        if central_mask[right]:
            shift = tuple(int(v) for v in shifts[left])
            periodic_links.add((right_id, left_id, shift[0], shift[1]))

    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    weighted_adjacency = []
    for cell_id, neighbors in enumerate(adjacency):
        weighted_neighbors = []
        for neighbor_id in sorted(neighbors):
            delta = minimum_image_vector(
                base_seeds[neighbor_id] - base_seeds[cell_id],
                lattice_matrix,
                lattice_inverse,
            )
            weighted_neighbors.append((neighbor_id, float(np.linalg.norm(delta))))
        weighted_adjacency.append(weighted_neighbors)

    return {
        "polygons": polygons,
        "areas": areas,
        "centroids": centroids,
        "adjacency": adjacency,
        "weighted_adjacency": weighted_adjacency,
        "periodic_links": sorted(periodic_links),
        "copy_canonical_ids": canonical_ids,
        "copy_shifts": shifts,
    }


def graph_is_connected(adjacency, allowed=None):
    nodes = set(range(len(adjacency))) if allowed is None else set(allowed)
    if not nodes:
        return False
    reached = set()
    stack = [min(nodes)]
    while stack:
        node = stack.pop()
        if node in reached:
            continue
        reached.add(node)
        stack.extend(neighbor for neighbor in adjacency[node] if neighbor in nodes)
    return reached == nodes


def periodic_distance(point, seeds, lattice_matrix, lattice_inverse):
    deltas = seeds - np.asarray(point, dtype=float)
    fractional = deltas @ lattice_inverse.T
    fractional -= np.round(fractional)
    wrapped = fractional @ lattice_matrix.T
    return np.linalg.norm(wrapped, axis=1)


def associate_terminals(terminal_coordinates, base_seeds, a1, a2):
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    terminals = []
    used_cells = set()
    for terminal_id, coordinate in enumerate(terminal_coordinates):
        distances = periodic_distance(
            coordinate, base_seeds, lattice_matrix, lattice_inverse
        )
        ranked = np.argsort(distances, kind="stable")
        cell_id = next(int(idx) for idx in ranked if int(idx) not in used_cells)
        used_cells.add(cell_id)
        terminals.append(
            {
                "terminal_id": terminal_id,
                "acinus_id": terminal_id,
                "terminal_coordinate": [float(v) for v in coordinate],
                "terminal_cell_id": cell_id,
                "distance_to_cell_seed": float(distances[cell_id]),
            }
        )
    return terminals


def build_airway_overlay(terminals, bounds):
    """Create a deterministic airway tree as visualization metadata only."""
    if len(terminals) != 4:
        raise ValueError("The current visualization tree requires four terminals.")

    xmin, ymin, xmax, ymax = map(float, bounds)
    width = xmax - xmin
    height = ymax - ymin
    terminal_by_id = {item["terminal_id"]: item for item in terminals}
    if set(terminal_by_id) != {0, 1, 2, 3}:
        raise ValueError("Terminal IDs 0, 1, 2, and 3 are required.")

    nodes = [
        {
            "node_id": "root",
            "node_type": "root",
            "coordinate": [xmin + 0.50 * width, ymin + 0.07 * height],
            "label": "Proximal airway root",
        },
        {
            "node_id": "B0",
            "node_type": "branch",
            "coordinate": [xmin + 0.50 * width, ymin + 0.20 * height],
            "label": "Branch B0",
        },
        {
            "node_id": "B1",
            "node_type": "branch",
            "coordinate": [xmin + 0.25 * width, ymin + 0.50 * height],
            "label": "Branch B1",
        },
        {
            "node_id": "B2",
            "node_type": "branch",
            "coordinate": [xmin + 0.75 * width, ymin + 0.50 * height],
            "label": "Branch B2",
        },
    ]
    for terminal_id in range(4):
        terminal = terminal_by_id[terminal_id]
        nodes.append(
            {
                "node_id": f"T{terminal_id}",
                "node_type": "terminal",
                "coordinate": terminal["terminal_coordinate"],
                "label": f"Terminal bronchiole T{terminal_id}",
                "terminal_id": terminal_id,
                "acinus_id": terminal["acinus_id"],
                "terminal_cell_id": terminal["terminal_cell_id"],
            }
        )

    edge_pairs = [
        ("root", "B0"),
        ("B0", "B1"),
        ("B0", "B2"),
        ("B1", "T0"),
        ("B1", "T2"),
        ("B2", "T1"),
        ("B2", "T3"),
    ]
    return {
        "representation": "visualization_and_anatomical_metadata_only",
        "affects_fe_geometry": False,
        "affects_mesh": False,
        "affects_material_regions": False,
        "affects_governing_equations": False,
        "periodic_rule": (
            "replicate every node and edge by integer combinations of a1 and a2"
        ),
        "root_node_id": "root",
        "nodes": nodes,
        "edges": [
            {"parent_node_id": parent, "child_node_id": child}
            for parent, child in edge_pairs
        ],
    }


def validate_airway_overlay(airway, terminals):
    """Fail if the overlay is not a connected tree with one terminal per acinus."""
    nodes = {node["node_id"]: node for node in airway["nodes"]}
    if len(nodes) != len(airway["nodes"]):
        raise RuntimeError("Airway node IDs must be unique.")

    root = airway["root_node_id"]
    if root not in nodes or nodes[root]["node_type"] != "root":
        raise RuntimeError("The airway root metadata is missing or inconsistent.")

    children = {node_id: [] for node_id in nodes}
    parent_count = {node_id: 0 for node_id in nodes}
    for edge in airway["edges"]:
        parent = edge["parent_node_id"]
        child = edge["child_node_id"]
        if parent not in nodes or child not in nodes:
            raise RuntimeError("An airway edge references an unknown node.")
        children[parent].append(child)
        parent_count[child] += 1

    reached = set()
    stack = [root]
    while stack:
        node_id = stack.pop()
        if node_id in reached:
            raise RuntimeError("The airway overlay contains a cycle.")
        reached.add(node_id)
        stack.extend(children[node_id])

    is_tree = (
        reached == set(nodes)
        and len(airway["edges"]) == len(nodes) - 1
        and parent_count[root] == 0
        and all(parent_count[node_id] == 1 for node_id in nodes if node_id != root)
    )
    if not is_tree:
        raise RuntimeError("The airway overlay must be one connected rooted tree.")

    terminal_nodes = [
        node for node in airway["nodes"] if node["node_type"] == "terminal"
    ]
    expected_pairs = {
        (item["terminal_id"], item["acinus_id"]) for item in terminals
    }
    actual_pairs = {
        (node["terminal_id"], node["acinus_id"]) for node in terminal_nodes
    }
    terminal_mapping_valid = (
        actual_pairs == expected_pairs
        and len(actual_pairs) == len(terminals)
        and all(len(children[node["node_id"]]) == 0 for node in terminal_nodes)
    )
    if not terminal_mapping_valid:
        raise RuntimeError("Each acinus must have exactly one terminal bronchiole.")

    metadata_only = all(
        airway[key] is False
        for key in (
            "affects_fe_geometry",
            "affects_mesh",
            "affects_material_regions",
            "affects_governing_equations",
        )
    )
    if not metadata_only:
        raise RuntimeError("The airway overlay must remain disconnected from FE physics.")

    return {
        "connected_acyclic_tree": True,
        "terminal_to_acinus_one_to_one": True,
        "metadata_only": True,
        "node_count": len(nodes),
        "edge_count": len(airway["edges"]),
        "root_count": 1,
        "branch_node_count": sum(
            node["node_type"] == "branch" for node in airway["nodes"]
        ),
        "terminal_count": len(terminal_nodes),
    }


def wrap_periodic_point(point, bounds):
    xmin, ymin, xmax, ymax = map(float, bounds)
    origin = np.array([xmin, ymin], dtype=float)
    lengths = np.array([xmax - xmin, ymax - ymin], dtype=float)
    return origin + np.mod(np.asarray(point, dtype=float) - origin, lengths)


def periodic_point_distance(point_a, point_b, lattice_matrix, lattice_inverse):
    delta = minimum_image_vector(
        np.asarray(point_b, dtype=float) - np.asarray(point_a, dtype=float),
        lattice_matrix,
        lattice_inverse,
    )
    return float(np.linalg.norm(delta))


def restricted_shortest_path_matrix(weighted_adjacency, parent_cell_ids):
    """All-pairs graph distances on the connected parent territory."""
    parent_ids = np.asarray(sorted(int(v) for v in parent_cell_ids), dtype=int)
    local_index = {cell_id: idx for idx, cell_id in enumerate(parent_ids)}
    distances = np.full((len(parent_ids), len(parent_ids)), np.inf)

    for source_local, source_id in enumerate(parent_ids):
        distances[source_local, source_local] = 0.0
        queue = [(0.0, int(source_id))]
        while queue:
            distance, cell_id = heapq.heappop(queue)
            cell_local = local_index[cell_id]
            if distance > distances[source_local, cell_local] + 1e-14:
                continue
            for neighbor_id, weight in weighted_adjacency[cell_id]:
                if neighbor_id not in local_index:
                    continue
                neighbor_local = local_index[neighbor_id]
                candidate = distance + weight
                if candidate < distances[source_local, neighbor_local] - 1e-14:
                    distances[source_local, neighbor_local] = candidate
                    heapq.heappush(queue, (candidate, int(neighbor_id)))

    if not np.all(np.isfinite(distances)):
        raise RuntimeError("The parent territory is disconnected on the periodic graph.")
    return parent_ids, distances


def optimize_binary_airway_split(
    parent_cell_ids,
    parent_airway_coordinate,
    graph,
    a1,
    a2,
    bounds,
    weights,
    minimum_cells_per_daughter=1,
):
    """Optimize one airway split and its two graph-connected daughter regions."""
    parent_ids, graph_distances = restricted_shortest_path_matrix(
        graph["weighted_adjacency"], parent_cell_ids
    )
    if len(parent_ids) < 2 * minimum_cells_per_daughter:
        raise RuntimeError("The parent territory contains too few cells to split.")

    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    coordinates = graph["centroids"][parent_ids]
    areas = graph["areas"][parent_ids]
    parent_area = float(np.sum(areas))
    length_scale = max(parent_area**0.5, 1e-14)

    squared_anchor_distances = np.zeros((len(parent_ids), len(parent_ids)))
    parent_branch_lengths = np.zeros(len(parent_ids))
    for anchor_local, coordinate in enumerate(coordinates):
        parent_branch_lengths[anchor_local] = periodic_point_distance(
            parent_airway_coordinate,
            coordinate,
            lattice_matrix,
            lattice_inverse,
        )
        for cell_local, cell_coordinate in enumerate(coordinates):
            distance = periodic_point_distance(
                coordinate,
                cell_coordinate,
                lattice_matrix,
                lattice_inverse,
            )
            squared_anchor_distances[anchor_local, cell_local] = distance**2

    best = None
    for left_local in range(len(parent_ids) - 1):
        for right_local in range(left_local + 1, len(parent_ids)):
            if (
                parent_branch_lengths[left_local] <= 1e-12
                or parent_branch_lengths[right_local] <= 1e-12
            ):
                continue
            left_distance = graph_distances[left_local]
            right_distance = graph_distances[right_local]
            left_mask = left_distance < right_distance - 1e-14
            ties = np.abs(left_distance - right_distance) <= 1e-14
            if parent_ids[left_local] < parent_ids[right_local]:
                left_mask |= ties
            right_mask = ~left_mask

            if (
                np.count_nonzero(left_mask) < minimum_cells_per_daughter
                or np.count_nonzero(right_mask) < minimum_cells_per_daughter
            ):
                continue

            left_ids = parent_ids[left_mask]
            right_ids = parent_ids[right_mask]
            if not graph_is_connected(graph["adjacency"], left_ids):
                continue
            if not graph_is_connected(graph["adjacency"], right_ids):
                continue

            left_area = float(np.sum(areas[left_mask]))
            right_area = float(np.sum(areas[right_mask]))
            area_cost = ((left_area - right_area) / parent_area) ** 2

            compactness_numerator = float(
                np.dot(areas[left_mask], squared_anchor_distances[left_local, left_mask])
                + np.dot(
                    areas[right_mask],
                    squared_anchor_distances[right_local, right_mask],
                )
            )
            compactness_cost = compactness_numerator / (parent_area * length_scale**2)
            branch_length_cost = (
                parent_branch_lengths[left_local]
                + parent_branch_lengths[right_local]
            ) / (2.0 * length_scale)
            total_cost = (
                weights["area"] * area_cost
                + weights["compactness"] * compactness_cost
                + weights["branch_length"] * branch_length_cost
            )
            score = (
                total_cost,
                area_cost,
                compactness_cost,
                branch_length_cost,
                int(parent_ids[left_local]),
                int(parent_ids[right_local]),
            )
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "daughter_cell_ids": [left_ids.tolist(), right_ids.tolist()],
                    "daughter_anchor_cell_ids": [
                        int(parent_ids[left_local]),
                        int(parent_ids[right_local]),
                    ],
                    "daughter_areas": [left_area, right_area],
                    "objective": {
                        "J": float(total_cost),
                        "J_area": float(area_cost),
                        "J_compactness": float(compactness_cost),
                        "J_branch_length": float(branch_length_cost),
                    },
                }

    if best is None:
        raise RuntimeError("No feasible connected binary airway split was found.")

    daughters = [set(ids) for ids in best["daughter_cell_ids"]]
    parent = set(int(v) for v in parent_ids)
    if daughters[0] & daughters[1] or daughters[0] | daughters[1] != parent:
        raise RuntimeError("Optimized daughters must be disjoint and cover the parent.")
    del best["score"]
    return best


def generate_airway_constrained_space_filling(
    graph,
    a1,
    a2,
    bounds,
    root_coordinate=DEFAULT_AIRWAY_ROOT,
    n_acini=4,
    weights=None,
):
    """Generate one binary airway tree and its territories in the same recursion."""
    if n_acini < 2 or n_acini & (n_acini - 1):
        raise ValueError("n_acini must be a power of two and at least two.")
    if weights is None:
        weights = DEFAULT_SPLIT_WEIGHTS.copy()
    weights = {key: float(weights[key]) for key in DEFAULT_SPLIT_WEIGHTS}
    if any(value < 0.0 for value in weights.values()) or not any(weights.values()):
        raise ValueError("Split weights must be non-negative and not all zero.")

    root_coordinate = wrap_periodic_point(root_coordinate, bounds)
    depth = int(np.log2(n_acini))
    nodes = [
        {
            "node_id": "root",
            "node_type": "root",
            "coordinate": root_coordinate.tolist(),
            "label": "Proximal airway root and first split",
            "generation": 0,
        }
    ]
    edges = []
    split_records = []
    labels = np.full(len(graph["centroids"]), -1, dtype=int)
    terminals = []
    branch_counter = 0
    terminal_counter = 0

    def recurse(parent_node_id, parent_coordinate, parent_cells, generation):
        nonlocal branch_counter, terminal_counter
        remaining_depth = depth - generation
        split = optimize_binary_airway_split(
            parent_cell_ids=parent_cells,
            parent_airway_coordinate=parent_coordinate,
            graph=graph,
            a1=a1,
            a2=a2,
            bounds=bounds,
            weights=weights,
            minimum_cells_per_daughter=2 ** max(remaining_depth - 1, 0),
        )

        child_node_ids = []
        for daughter_index in range(2):
            anchor_cell_id = split["daughter_anchor_cell_ids"][daughter_index]
            coordinate = graph["centroids"][anchor_cell_id]
            daughter_cells = split["daughter_cell_ids"][daughter_index]
            is_terminal = generation + 1 == depth
            if is_terminal:
                node_id = f"T{terminal_counter}"
                acinus_id = terminal_counter
                terminal_counter += 1
                node = {
                    "node_id": node_id,
                    "node_type": "terminal",
                    "coordinate": coordinate.tolist(),
                    "label": f"Terminal bronchiole {node_id}",
                    "generation": generation + 1,
                    "terminal_id": acinus_id,
                    "acinus_id": acinus_id,
                    "terminal_cell_id": anchor_cell_id,
                }
                labels[np.asarray(daughter_cells, dtype=int)] = acinus_id
                terminals.append(
                    {
                        "terminal_id": acinus_id,
                        "acinus_id": acinus_id,
                        "terminal_coordinate": coordinate.tolist(),
                        "terminal_cell_id": anchor_cell_id,
                    }
                )
            else:
                node_id = f"B{branch_counter}"
                branch_counter += 1
                node = {
                    "node_id": node_id,
                    "node_type": "branch",
                    "coordinate": coordinate.tolist(),
                    "label": f"Airway branch node {node_id}",
                    "generation": generation + 1,
                    "anchor_cell_id": anchor_cell_id,
                }

            nodes.append(node)
            child_node_ids.append(node_id)
            edges.append(
                {
                    "parent_node_id": parent_node_id,
                    "child_node_id": node_id,
                }
            )
            if not is_terminal:
                recurse(
                    node_id,
                    coordinate,
                    daughter_cells,
                    generation + 1,
                )

        split_records.append(
            {
                "parent_node_id": parent_node_id,
                "generation": generation,
                "parent_cell_count": len(parent_cells),
                "parent_area": float(np.sum(graph["areas"][list(parent_cells)])),
                "daughter_node_ids": child_node_ids,
                "daughter_anchor_cell_ids": split["daughter_anchor_cell_ids"],
                "daughter_cell_counts": [
                    len(ids) for ids in split["daughter_cell_ids"]
                ],
                "daughter_areas": split["daughter_areas"],
                "objective": split["objective"],
                "hard_constraints": {
                    "daughter_territories_connected": True,
                    "complete_canonical_cells_only": True,
                    "daughters_disjoint": True,
                    "daughters_cover_parent": True,
                },
            }
        )

    recurse(
        "root",
        root_coordinate,
        list(range(len(graph["centroids"]))),
        generation=0,
    )
    if np.any(labels < 0) or terminal_counter != n_acini:
        raise RuntimeError("Recursive airway generation did not label every cell.")

    airway = {
        "model": "airway_constrained_space_filling",
        "representation": "visualization_and_anatomical_metadata_only",
        "affects_fe_geometry": False,
        "affects_mesh": False,
        "affects_material_regions": False,
        "affects_governing_equations": False,
        "periodic_rule": (
            "all distances use minimum images; plot copies use integer a1/a2 shifts"
        ),
        "root_node_id": "root",
        "number_of_acini": n_acini,
        "split_weights": weights,
        "split_objective_definition": {
            "J": "w_A*J_area + w_C*J_compactness + w_L*J_branch_length",
            "J_area": "((A_left-A_right)/A_parent)^2",
            "J_compactness": (
                "area-weighted squared minimum-image distances from daughter "
                "cell centroids to their airway anchors, divided by A_parent^2"
            ),
            "J_branch_length": (
                "mean minimum-image parent-to-daughter branch length divided "
                "by sqrt(A_parent)"
            ),
            "daughter_assignment": (
                "two-source shortest-path Voronoi assignment restricted to the "
                "parent periodic cell graph"
            ),
        },
        "hard_split_constraints": [
            "both daughter territories connected on the periodic graph",
            "complete canonical Voronoi cells only",
            "daughter territories disjoint",
            "daughter territories exactly cover the parent territory",
            "nonzero parent-to-daughter branch lengths",
        ],
        "nodes": nodes,
        "edges": edges,
        "splits": sorted(split_records, key=lambda item: item["generation"]),
    }
    return labels, terminals, airway


def template_leaf_count(node_spec):
    if node_spec["node_type"] == "terminal":
        return 1
    return sum(template_leaf_count(child) for child in node_spec.get("children", []))


def template_terminal_ids(node_spec):
    if node_spec["node_type"] == "terminal":
        return [int(node_spec["node_id"][1:])]
    terminal_ids = []
    for child in node_spec.get("children", []):
        terminal_ids.extend(template_terminal_ids(child))
    return terminal_ids


def topology_implied_target_area_weights(tree_spec):
    """Leaf weights implied by sequential equal-area binary splits."""
    weights = {}

    def visit(node_spec, parent_weight):
        if node_spec["node_type"] == "terminal":
            weights[int(node_spec["node_id"][1:])] = float(parent_weight)
            return
        children = node_spec.get("children", [])
        if len(children) < 2:
            raise ValueError("Every non-terminal template node needs at least two children.")
        direct_weights = [0.5 ** (idx + 1) for idx in range(len(children) - 1)]
        direct_weights.append(0.5 ** (len(children) - 1))
        for child, relative_weight in zip(children, direct_weights):
            visit(child, parent_weight * relative_weight)

    visit(tree_spec, 1.0)
    return {terminal_id: weights[terminal_id] for terminal_id in sorted(weights)}


def validate_branching_template(template):
    tree = template["tree"]
    if tree["node_id"] != "root" or tree["node_type"] != "root":
        raise ValueError("A branching template must begin at the root node.")
    node_ids = []

    def visit(node):
        node_ids.append(node["node_id"])
        if node["node_type"] == "terminal":
            if node.get("children"):
                raise ValueError("Terminal template nodes cannot have children.")
            return
        if len(node.get("children", [])) < 2:
            raise ValueError("Root and branch template nodes need at least two children.")
        for child in node["children"]:
            visit(child)

    visit(tree)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Branching-template node IDs must be unique.")
    terminal_ids = sorted(template_terminal_ids(tree))
    if terminal_ids != list(range(len(terminal_ids))):
        raise ValueError("Template terminal IDs must be contiguous from T0.")
    return terminal_ids


def generate_airway_from_branching_template(
    template,
    graph,
    a1,
    a2,
    bounds,
    root_coordinate=DEFAULT_AIRWAY_ROOT,
    weights=None,
):
    """Apply the unchanged binary optimizer to a prescribed airway topology."""
    terminal_ids = validate_branching_template(template)
    if weights is None:
        weights = DEFAULT_SPLIT_WEIGHTS.copy()
    weights = {key: float(weights[key]) for key in DEFAULT_SPLIT_WEIGHTS}
    if weights != DEFAULT_SPLIT_WEIGHTS:
        raise ValueError(
            "Branching-template exploration must use the accepted objective weights."
        )

    root_coordinate = wrap_periodic_point(root_coordinate, bounds)
    nodes = [
        {
            "node_id": "root",
            "node_type": "root",
            "coordinate": root_coordinate.tolist(),
            "label": "Proximal airway root",
            "generation": 0,
        }
    ]
    edges = []
    split_records = []
    labels = np.full(len(graph["centroids"]), -1, dtype=int)
    terminals = []

    def split_direct_children(
        parent_spec, parent_node_id, parent_coordinate, parent_cells, generation
    ):
        remaining_specs = list(parent_spec["children"])
        remaining_cells = list(parent_cells)
        allocations = []
        binary_stage = 0
        while len(remaining_specs) > 1:
            left_spec = remaining_specs[0]
            right_specs = remaining_specs[1:]
            minimum_cells = max(
                template_leaf_count(left_spec),
                sum(template_leaf_count(spec) for spec in right_specs),
            )
            split = optimize_binary_airway_split(
                parent_cell_ids=remaining_cells,
                parent_airway_coordinate=parent_coordinate,
                graph=graph,
                a1=a1,
                a2=a2,
                bounds=bounds,
                weights=weights,
                minimum_cells_per_daughter=minimum_cells,
            )
            right_group_label = "+".join(spec["node_id"] for spec in right_specs)
            split_records.append(
                {
                    "topological_parent_node_id": parent_node_id,
                    "generation": generation,
                    "binary_stage_at_parent": binary_stage,
                    "binary_partition_groups": [
                        left_spec["node_id"],
                        right_group_label,
                    ],
                    "parent_cell_count": len(remaining_cells),
                    "parent_area": float(
                        np.sum(graph["areas"][np.asarray(remaining_cells, dtype=int)])
                    ),
                    "daughter_cell_counts": [
                        len(ids) for ids in split["daughter_cell_ids"]
                    ],
                    "daughter_areas": split["daughter_areas"],
                    "daughter_anchor_cell_ids": split[
                        "daughter_anchor_cell_ids"
                    ],
                    "objective": split["objective"],
                    "hard_constraints": {
                        "daughter_territories_connected": True,
                        "complete_canonical_cells_only": True,
                        "daughters_disjoint": True,
                        "daughters_cover_parent": True,
                        "nonzero_branch_lengths": True,
                    },
                }
            )
            allocations.append(
                (
                    left_spec,
                    split["daughter_cell_ids"][0],
                    split["daughter_anchor_cell_ids"][0],
                )
            )
            if len(right_specs) == 1:
                allocations.append(
                    (
                        right_specs[0],
                        split["daughter_cell_ids"][1],
                        split["daughter_anchor_cell_ids"][1],
                    )
                )
                break
            remaining_specs = right_specs
            remaining_cells = split["daughter_cell_ids"][1]
            binary_stage += 1

        for child_spec, child_cells, anchor_cell_id in allocations:
            coordinate = graph["centroids"][anchor_cell_id]
            node_id = child_spec["node_id"]
            node_type = child_spec["node_type"]
            node = {
                "node_id": node_id,
                "node_type": node_type,
                "coordinate": coordinate.tolist(),
                "generation": generation + 1,
                "anchor_cell_id": int(anchor_cell_id),
            }
            if node_type == "terminal":
                terminal_id = int(node_id[1:])
                node.update(
                    {
                        "label": f"Terminal bronchiole {node_id}",
                        "terminal_id": terminal_id,
                        "acinus_id": terminal_id,
                        "terminal_cell_id": int(anchor_cell_id),
                    }
                )
                labels[np.asarray(child_cells, dtype=int)] = terminal_id
                terminals.append(
                    {
                        "terminal_id": terminal_id,
                        "acinus_id": terminal_id,
                        "terminal_coordinate": coordinate.tolist(),
                        "terminal_cell_id": int(anchor_cell_id),
                    }
                )
            elif node_type == "branch":
                node["label"] = f"Airway branch node {node_id}"
            else:
                raise ValueError(f"Unsupported template node type: {node_type}")
            nodes.append(node)
            edges.append(
                {
                    "parent_node_id": parent_node_id,
                    "child_node_id": node_id,
                }
            )
            if node_type == "branch":
                split_direct_children(
                    child_spec,
                    node_id,
                    coordinate,
                    child_cells,
                    generation + 1,
                )

    split_direct_children(
        template["tree"],
        "root",
        root_coordinate,
        list(range(len(graph["centroids"]))),
        generation=0,
    )
    terminals.sort(key=lambda item: item["terminal_id"])
    if np.any(labels < 0) or [item["terminal_id"] for item in terminals] != terminal_ids:
        raise RuntimeError("Template generation did not create every requested acinus.")

    airway = {
        "model": "airway_constrained_space_filling",
        "branching_template": template["template_id"],
        "branching_template_description": template["description"],
        "representation": "visualization_and_anatomical_metadata_only",
        "affects_fe_geometry": False,
        "affects_mesh": False,
        "affects_material_regions": False,
        "affects_governing_equations": False,
        "periodic_rule": (
            "all distances use minimum images; plot copies use integer a1/a2 shifts"
        ),
        "root_node_id": "root",
        "number_of_acini": len(terminals),
        "split_weights": weights,
        "split_objective": "J = 2*J_area + 2*J_compactness + 0.15*J_branch_length",
        "topology_implied_target_area_weights": (
            topology_implied_target_area_weights(template["tree"])
        ),
        "multiway_partition_interpretation": (
            "nodes with more than two children use sequential calls to the "
            "unchanged equal-daughter binary optimizer at the same airway node"
        ),
        "nodes": nodes,
        "edges": edges,
        "splits": split_records,
    }
    return labels, terminals, airway


def multisource_partition(weighted_adjacency, terminals):
    """Assign cells by deterministic weighted multi-source Dijkstra."""
    n_cells = len(weighted_adjacency)
    distances = np.full(n_cells, np.inf)
    labels = np.full(n_cells, -1, dtype=int)
    queue = []
    for terminal in terminals:
        source = terminal["terminal_cell_id"]
        label = terminal["terminal_id"]
        distances[source] = 0.0
        labels[source] = label
        heapq.heappush(queue, (0.0, label, source))

    tolerance = 1e-14
    while queue:
        distance, label, cell_id = heapq.heappop(queue)
        if distance > distances[cell_id] + tolerance:
            continue
        if abs(distance - distances[cell_id]) <= tolerance and label != labels[cell_id]:
            continue
        for neighbor_id, weight in weighted_adjacency[cell_id]:
            candidate = distance + weight
            improve = candidate < distances[neighbor_id] - tolerance
            tie_break = (
                abs(candidate - distances[neighbor_id]) <= tolerance
                and (labels[neighbor_id] < 0 or label < labels[neighbor_id])
            )
            if improve or tie_break:
                distances[neighbor_id] = candidate
                labels[neighbor_id] = label
                heapq.heappush(queue, (candidate, label, neighbor_id))

    if np.any(labels < 0):
        missing = np.flatnonzero(labels < 0).tolist()
        raise RuntimeError(f"Unassigned canonical cells: {missing}")
    return labels, distances


def periodic_weighted_centroid(points, weights, a1, a2, bounds):
    """Compute a local Fréchet mean using minimum-image displacements."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    origin = np.asarray(bounds[:2], dtype=float)
    center = points[int(np.argmax(weights))].copy()
    for _ in range(50):
        deltas = np.asarray(
            [
                minimum_image_vector(point - center, lattice_matrix, lattice_inverse)
                for point in points
            ]
        )
        update = np.average(deltas, axis=0, weights=weights)
        center += update
        fractional = lattice_inverse @ (center - origin)
        center = origin + lattice_matrix @ np.mod(fractional, 1.0)
        if np.linalg.norm(update) < 1e-13:
            break
    return center


def territory_shape_metrics(cell_ids, graph, a1, a2, bounds):
    """Return area-weighted toroidal covariance and anisotropy metrics."""
    cell_ids = np.asarray(cell_ids, dtype=int)
    points = graph["centroids"][cell_ids]
    weights = graph["areas"][cell_ids]
    center = periodic_weighted_centroid(points, weights, a1, a2, bounds)
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    deltas = np.asarray(
        [
            minimum_image_vector(point - center, lattice_matrix, lattice_inverse)
            for point in points
        ]
    )
    covariance = np.einsum("n,ni,nj->ij", weights, deltas, deltas) / np.sum(weights)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    anisotropy = float(eigenvalues[-1] / max(eigenvalues[0], 1e-16))
    return {
        "periodic_centroid": center.tolist(),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "anisotropy_eigenvalue_ratio": anisotropy,
        "rms_radius": float(np.sqrt(np.trace(covariance))),
    }


def airway_root_to_terminal_path_lengths(airway, a1, a2):
    nodes = {node["node_id"]: node for node in airway["nodes"]}
    parent = {
        edge["child_node_id"]: edge["parent_node_id"] for edge in airway["edges"]
    }
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    root = airway["root_node_id"]
    results = {}
    for node in airway["nodes"]:
        if node["node_type"] != "terminal":
            continue
        node_id = node["node_id"]
        current = node_id
        path = [current]
        length = 0.0
        while current != root:
            parent_id = parent[current]
            length += periodic_point_distance(
                nodes[parent_id]["coordinate"],
                nodes[current]["coordinate"],
                lattice_matrix,
                lattice_inverse,
            )
            current = parent_id
            path.append(current)
        results[node_id] = {
            "terminal_id": node["terminal_id"],
            "acinus_id": node["acinus_id"],
            "node_path_root_to_terminal": list(reversed(path)),
            "path_length": float(length),
        }
    return results


def territory_statistics(labels, graph, terminals, a1, a2, bounds):
    stats = []
    territory_areas = []
    areas = graph["areas"]
    adjacency = graph["adjacency"]
    for terminal in terminals:
        acinus_id = terminal["terminal_id"]
        cells = np.flatnonzero(labels == acinus_id)
        area = float(np.sum(areas[cells]))
        territory_areas.append(area)
        neighboring_acini = sorted(
            {
                int(labels[neighbor])
                for cell_id in cells
                for neighbor in adjacency[cell_id]
                if labels[neighbor] != acinus_id
            }
        )
        stats.append(
            {
                "acinus_id": acinus_id,
                "number_of_cells": int(len(cells)),
                "area": area,
                "area_fraction": area / float(np.sum(areas)),
                "terminal_id": terminal["terminal_id"],
                "terminal_coordinate": terminal["terminal_coordinate"],
                "terminal_cell_id": terminal["terminal_cell_id"],
                "neighboring_acinus_ids": neighboring_acini,
                "number_of_neighboring_acini": len(neighboring_acini),
                "connected_on_periodic_graph": graph_is_connected(adjacency, cells),
                "shape": territory_shape_metrics(cells, graph, a1, a2, bounds),
            }
        )
    territory_areas = np.asarray(territory_areas)
    cv_area = float(np.std(territory_areas) / np.mean(territory_areas))
    return stats, cv_area


def summarize_partition(method, labels, graph, terminals, a1, a2, bounds):
    territory_stats, cv_area = territory_statistics(
        labels, graph, terminals, a1, a2, bounds
    )
    copy_ids = graph["copy_canonical_ids"]
    copy_labels = labels[copy_ids]
    periodic_labels_valid = bool(
        all(
            np.all(copy_labels[copy_ids == cell_id] == labels[cell_id])
            for cell_id in range(len(labels))
        )
    )
    return {
        "method": method,
        "number_of_acini": len(terminals),
        "total_cell_count": len(labels),
        "CV_area": cv_area,
        "all_territories_connected_on_periodic_graph": all(
            item["connected_on_periodic_graph"] for item in territory_stats
        ),
        "periodic_label_continuity_satisfied": periodic_labels_valid,
        "territories": territory_stats,
    }


def compare_partition_labels(reference_labels, candidate_labels, cell_areas):
    """Compare territories after removing arbitrary acinus-ID permutations."""
    reference_ids = sorted(int(v) for v in np.unique(reference_labels))
    candidate_ids = sorted(int(v) for v in np.unique(candidate_labels))
    if len(reference_ids) != len(candidate_ids):
        raise ValueError("Partition comparison requires equal territory counts.")

    best = None
    total_area = float(np.sum(cell_areas))
    for permutation in itertools.permutations(reference_ids):
        mapping = dict(zip(candidate_ids, permutation))
        mapped = np.asarray([mapping[int(v)] for v in candidate_labels])
        matching = mapped == reference_labels
        matching_area = float(np.sum(cell_areas[matching]))
        score = (matching_area, tuple(-value for value in permutation))
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "mapping_candidate_to_reference": mapping,
                "matching_cell_count": int(np.count_nonzero(matching)),
                "matching_area_fraction": matching_area / total_area,
            }
    del best["score"]
    best["different_area_fraction"] = 1.0 - best["matching_area_fraction"]
    return best


def draw_periodic_cells(
    ax, polygons, labels, a1, a2, tile_range, colored=True, linewidth=0.35
):
    colors = plt.get_cmap("tab10")
    for i in tile_range:
        for j in tile_range:
            shift = i * a1 + j * a2
            for cell_id, polygon in enumerate(polygons):
                facecolor = colors(int(labels[cell_id])) if colored else "white"
                ax.add_patch(
                    Polygon(
                        polygon + shift,
                        closed=True,
                        facecolor=facecolor,
                        edgecolor="0.25",
                        linewidth=linewidth,
                        alpha=0.72 if colored else 1.0,
                    )
                )


def airway_legend_handles():
    return [
        Line2D([0], [0], color="#7b1fa2", linewidth=3.0, label="Airway tree"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#e53935",
            markeredgecolor="white",
            markersize=8,
            label="Proximal root",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#7b1fa2",
            markeredgecolor="white",
            markersize=7,
            label="Branch nodes",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#ffd54f",
            markeredgecolor="black",
            markersize=11,
            label="Terminal bronchioles",
        ),
    ]


def draw_airway_overlay(
    ax,
    airway,
    a1,
    a2,
    tile_indices=((0, 0),),
    annotate_tile=(0, 0),
    linewidth=3.0,
    marker_scale=1.0,
    show_acinus_ids=False,
):
    """Draw the metadata-only airway tree without creating geometric entities."""
    nodes = {node["node_id"]: node for node in airway["nodes"]}
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    for i, j in tile_indices:
        shift = i * a1 + j * a2
        for edge in airway["edges"]:
            parent = np.asarray(nodes[edge["parent_node_id"]]["coordinate"]) + shift
            child_coordinate = np.asarray(
                nodes[edge["child_node_id"]]["coordinate"]
            )
            delta = minimum_image_vector(
                child_coordinate - np.asarray(
                    nodes[edge["parent_node_id"]]["coordinate"]
                ),
                lattice_matrix,
                lattice_inverse,
            )
            child = parent + delta
            ax.plot(
                [parent[0], child[0]],
                [parent[1], child[1]],
                color="#7b1fa2",
                linewidth=linewidth,
                solid_capstyle="round",
                zorder=7,
            )

        for node in airway["nodes"]:
            point = np.asarray(node["coordinate"]) + shift
            node_type = node["node_type"]
            if node_type == "root":
                marker, size, color, edgecolor = "D", 78, "#e53935", "white"
            elif node_type == "branch":
                marker, size, color, edgecolor = "o", 58, "#7b1fa2", "white"
            elif node_type == "terminal":
                marker, size, color, edgecolor = "*", 175, "#ffd54f", "black"
            else:
                raise RuntimeError(f"Unknown airway node type: {node_type}")
            ax.scatter(
                *point,
                marker=marker,
                s=size * marker_scale,
                c=color,
                edgecolors=edgecolor,
                linewidths=0.8,
                zorder=8,
            )

            if (i, j) != annotate_tile:
                continue
            if node_type == "root":
                text = "root"
            elif node_type == "branch":
                text = node["node_id"]
            elif show_acinus_ids:
                text = f"{node['node_id']} → A{node['acinus_id']}"
            else:
                text = node["node_id"]
            ax.annotate(
                text,
                point,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="black",
                zorder=9,
            )


def finish_axis(ax, extent, title):
    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def plot_central_structure(
    path, graph, labels, airway, a1, a2, bounds, colored, title=None
):
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    draw_periodic_cells(
        ax,
        graph["polygons"],
        labels,
        a1,
        a2,
        tile_range=range(-1, 2),
        colored=colored,
        linewidth=0.45,
    )
    draw_airway_overlay(
        ax,
        airway,
        a1,
        a2,
        tile_indices=tuple(
            (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
        ),
        show_acinus_ids=colored,
    )
    if title is None:
        title = (
            "Periodic Voronoi/alveolar structure with airway overlay"
            if not colored
            else "Acinar territories and their terminal airways"
        )
    finish_axis(ax, bounds, title)
    airway_legend = ax.legend(
        handles=airway_legend_handles(),
        loc="upper right",
        fontsize=7.5,
        framealpha=0.94,
    )
    ax.add_artist(airway_legend)
    if colored:
        colors = plt.get_cmap("tab10")
        acinus_handles = [
            Patch(
                facecolor=colors(int(acinus_id)),
                edgecolor="0.25",
                label=f"acinus_id {acinus_id}",
            )
            for acinus_id in sorted(np.unique(labels))
        ]
        ax.legend(
            handles=acinus_handles,
            loc="upper left",
            fontsize=7.5,
            framealpha=0.94,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_tiled_partition(path, graph, labels, airway, a1, a2, bounds):
    xmin, ymin, xmax, ymax = map(float, bounds)
    width = xmax - xmin
    height = ymax - ymin
    extent = (xmin - width, ymin - height, xmax + width, ymax + height)
    fig, ax = plt.subplots(figsize=(10.5, 10.0))
    draw_periodic_cells(
        ax,
        graph["polygons"],
        labels,
        a1,
        a2,
        tile_range=range(-2, 3),
        colored=True,
        linewidth=0.25,
    )
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            origin = np.array([xmin, ymin]) + i * a1 + j * a2
            rectangle = plt.Rectangle(
                origin,
                width,
                height,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
                linestyle="--",
                zorder=4,
            )
            ax.add_patch(rectangle)
    draw_airway_overlay(
        ax,
        airway,
        a1,
        a2,
        tile_indices=tuple(
            (i, j) for i in range(-2, 3) for j in range(-2, 3)
        ),
        annotate_tile=(0, 0),
        linewidth=1.7,
        marker_scale=0.32,
        show_acinus_ids=True,
    )
    finish_axis(ax, extent, "3×3 airway-constrained periodic acinar territories")
    ax.legend(
        handles=airway_legend_handles(),
        loc="upper right",
        fontsize=7.5,
        framealpha=0.94,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_partition_comparison(
    path,
    graph,
    reference_labels,
    reference_airway,
    space_filling_labels,
    space_filling_airway,
    a1,
    a2,
    bounds,
):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), sharex=True, sharey=True)
    methods = (
        (
            reference_labels,
            reference_airway,
            "Reference: prescribed terminals + Dijkstra",
        ),
        (
            space_filling_labels,
            space_filling_airway,
            "Main model: airway-constrained space-filling",
        ),
    )
    airway_tiles = tuple(
        (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
    )
    for ax, (labels, airway, title) in zip(axes, methods):
        draw_periodic_cells(
            ax,
            graph["polygons"],
            labels,
            a1,
            a2,
            tile_range=range(-1, 2),
            colored=True,
            linewidth=0.35,
        )
        draw_airway_overlay(
            ax,
            airway,
            a1,
            a2,
            tile_indices=airway_tiles,
            linewidth=2.5,
            marker_scale=0.75,
            show_acinus_ids=True,
        )
        finish_axis(ax, bounds, title)
    axes[1].legend(
        handles=airway_legend_handles(),
        loc="upper right",
        fontsize=7.5,
        framealpha=0.94,
    )
    fig.suptitle("Periodic acinar partition methods", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def draw_airway_topology_diagram(ax, airway):
    children = {node["node_id"]: [] for node in airway["nodes"]}
    nodes = {node["node_id"]: node for node in airway["nodes"]}
    for edge in airway["edges"]:
        children[edge["parent_node_id"]].append(edge["child_node_id"])
    for node_id in children:
        children[node_id].sort()

    terminal_nodes = sorted(
        (
            node for node in airway["nodes"] if node["node_type"] == "terminal"
        ),
        key=lambda node: node["terminal_id"],
    )
    x_positions = {
        node["node_id"]: value
        for node, value in zip(
            terminal_nodes,
            np.linspace(0.08, 0.92, len(terminal_nodes)),
        )
    }

    def place(node_id):
        if node_id in x_positions:
            return x_positions[node_id]
        child_positions = [place(child_id) for child_id in children[node_id]]
        x_positions[node_id] = float(np.mean(child_positions))
        return x_positions[node_id]

    root = airway["root_node_id"]
    place(root)
    y_positions = {
        node_id: -float(nodes[node_id].get("generation", 0))
        for node_id in nodes
    }
    for edge in airway["edges"]:
        parent = edge["parent_node_id"]
        child = edge["child_node_id"]
        ax.plot(
            [x_positions[parent], x_positions[child]],
            [y_positions[parent], y_positions[child]],
            color="#7b1fa2",
            linewidth=2.2,
            zorder=1,
        )
    for node_id, node in nodes.items():
        if node["node_type"] == "root":
            marker, size, color = "D", 95, "#e53935"
        elif node["node_type"] == "branch":
            marker, size, color = "o", 80, "#7b1fa2"
        else:
            marker, size, color = "*", 160, "#ffd54f"
        ax.scatter(
            x_positions[node_id],
            y_positions[node_id],
            marker=marker,
            s=size,
            c=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=2,
        )
        ax.annotate(
            node_id,
            (x_positions[node_id], y_positions[node_id]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )
    maximum_generation = max(-value for value in y_positions.values())
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-maximum_generation - 0.35, 0.35)
    ax.axis("off")


def format_split_lines(airway):
    lines = []
    for split in airway["splits"]:
        parent = split.get(
            "topological_parent_node_id", split.get("parent_node_id")
        )
        groups = split.get(
            "binary_partition_groups", split.get("daughter_node_ids")
        )
        counts = split["daughter_cell_counts"]
        lines.append(
            f"{parent} split: {groups[0]} ({counts[0]} cells) | "
            f"{groups[1]} ({counts[1]} cells)"
        )
    return lines


def plot_detailed_airway_summary(
    path, graph, labels, airway, partition_summary, a1, a2, bounds
):
    fig = plt.figure(figsize=(16.0, 9.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.35, 1.0),
        height_ratios=(0.42, 0.58),
        wspace=0.16,
        hspace=0.16,
    )
    structure_ax = fig.add_subplot(grid[:, 0])
    topology_ax = fig.add_subplot(grid[0, 1])
    metrics_ax = fig.add_subplot(grid[1, 1])

    draw_periodic_cells(
        structure_ax,
        graph["polygons"],
        labels,
        a1,
        a2,
        tile_range=range(-1, 2),
        colored=True,
        linewidth=0.42,
    )
    draw_airway_overlay(
        structure_ax,
        airway,
        a1,
        a2,
        tile_indices=tuple(
            (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
        ),
        linewidth=3.2,
        show_acinus_ids=True,
    )
    finish_axis(
        structure_ax,
        bounds,
        "Periodic Voronoi territories and generated airway tree",
    )
    structure_ax.legend(
        handles=airway_legend_handles(),
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
    )

    draw_airway_topology_diagram(topology_ax, airway)
    topology_ax.set_title("Airway hierarchy / topology", fontsize=12, weight="bold")
    topology_ax.text(
        0.5,
        -0.02,
        r"$J = 2J_{area} + 2J_{compactness} + 0.15J_{branch\ length}$",
        transform=topology_ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        bbox={"facecolor": "#f4f0f7", "edgecolor": "#7b1fa2", "pad": 7},
    )

    metrics_ax.axis("off")
    target_weights = airway["topology_implied_target_area_weights"]
    rows = []
    for territory in partition_summary["territories"]:
        acinus_id = territory["acinus_id"]
        rows.append(
            [
                f"A{acinus_id}",
                f"T{territory['terminal_id']}",
                f"{target_weights[acinus_id]:.3f}",
                f"{territory['area_fraction']:.3f}",
                str(territory["number_of_cells"]),
                f"{territory['shape']['anisotropy_eigenvalue_ratio']:.3f}",
                f"{territory['root_to_terminal_path_length']:.3f}",
            ]
        )
    table = metrics_ax.table(
        cellText=rows,
        colLabels=(
            "Acinus",
            "Terminal",
            "Target area",
            "Area fraction",
            "Cells",
            "Anisotropy",
            "Path length",
        ),
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.43, 1.0, 0.52),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.35)
    metrics_ax.text(
        0.0,
        0.36,
        f"CV_area = {partition_summary['CV_area']:.5f}    "
        "All territories connected: yes    Periodic labels: yes",
        fontsize=10,
        weight="bold",
        transform=metrics_ax.transAxes,
    )
    metrics_ax.text(
        0.0,
        0.29,
        "Target acinar area weights: "
        + ", ".join(
            f"A{acinus_id}={weight:.3f}"
            for acinus_id, weight in target_weights.items()
        ),
        fontsize=9.5,
        transform=metrics_ax.transAxes,
    )
    metrics_ax.text(
        0.0,
        0.22,
        "Recursive split structure:",
        fontsize=10,
        weight="bold",
        transform=metrics_ax.transAxes,
    )
    metrics_ax.text(
        0.02,
        0.18,
        "\n".join(format_split_lines(airway)),
        fontsize=9,
        va="top",
        linespacing=1.45,
        transform=metrics_ax.transAxes,
    )
    fig.suptitle(
        "Airway-constrained space-filling model — current 2+2 baseline",
        fontsize=17,
        weight="bold",
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_branching_template_comparison(path, template_results, graph, a1, a2, bounds):
    fig = plt.figure(figsize=(18.5, 10.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(0.72, 0.28), hspace=0.12)
    airway_tiles = tuple(
        (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
    )
    for column, (template_key, result) in enumerate(template_results.items()):
        ax = fig.add_subplot(grid[0, column])
        draw_periodic_cells(
            ax,
            graph["polygons"],
            result["labels"],
            a1,
            a2,
            tile_range=range(-1, 2),
            colored=True,
            linewidth=0.30,
        )
        draw_airway_overlay(
            ax,
            result["airway"],
            a1,
            a2,
            tile_indices=airway_tiles,
            linewidth=2.4,
            marker_scale=0.65,
            show_acinus_ids=True,
        )
        finish_axis(
            ax,
            bounds,
            {
                "A": "Template A — 2+2",
                "B": "Template B — 1+1+(2)",
                "C": "Template C — 2+3",
            }[result["template_id"]],
        )
        if column == 2:
            ax.legend(
                handles=airway_legend_handles(),
                loc="upper right",
                fontsize=7,
                framealpha=0.94,
            )

    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")
    rows = []
    for result in template_results.values():
        summary = result["summary"]
        territories = summary["territories"]
        rows.append(
            [
                result["template_id"],
                str(summary["number_of_acini"]),
                ", ".join(
                    f"{value:.3f}"
                    for value in result["airway"][
                        "topology_implied_target_area_weights"
                    ].values()
                ),
                ", ".join(f"{item['area_fraction']:.3f}" for item in territories),
                f"{summary['CV_area']:.4f}",
                ", ".join(
                    f"{item['shape']['anisotropy_eigenvalue_ratio']:.2f}"
                    for item in territories
                ),
                ", ".join(
                    f"{item['root_to_terminal_path_length']:.3f}"
                    for item in territories
                ),
                "yes" if summary["all_territories_connected_on_periodic_graph"] else "no",
                "yes" if summary["periodic_label_continuity_satisfied"] else "no",
            ]
        )
    table = table_ax.table(
        cellText=rows,
        colLabels=(
            "Template",
            "N",
            "Implied target weights",
            "Actual area fractions",
            "CV_area",
            "Anisotropy by acinus",
            "Path lengths",
            "Connected",
            "Periodic",
        ),
        cellLoc="center",
        colLoc="center",
        colWidths=(
            0.055,
            0.035,
            0.15,
            0.15,
            0.060,
            0.16,
            0.16,
            0.060,
            0.060,
        ),
        bbox=(0.0, 0.04, 1.0, 0.90),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    table.scale(1.0, 1.55)
    fig.suptitle(
        "Branching-template exploration with unchanged split objective",
        fontsize=17,
        weight="bold",
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/acinar_partition"))
    parser.add_argument("--grid-x", type=int, default=12)
    parser.add_argument("--grid-y", type=int, default=12)
    parser.add_argument("--irregularity", type=float, default=0.3)
    parser.add_argument("--rng-seed", type=int, default=1)
    parser.add_argument("--root-x", type=float, default=float(DEFAULT_AIRWAY_ROOT[0]))
    parser.add_argument("--root-y", type=float, default=float(DEFAULT_AIRWAY_ROOT[1]))
    parser.add_argument("--w-area", type=float, default=DEFAULT_SPLIT_WEIGHTS["area"])
    parser.add_argument(
        "--w-compactness",
        type=float,
        default=DEFAULT_SPLIT_WEIGHTS["compactness"],
    )
    parser.add_argument(
        "--w-branch-length",
        type=float,
        default=DEFAULT_SPLIT_WEIGHTS["branch_length"],
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    bounds = (0.0, 0.0, 1.0, 1.0)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.0, 1.0])
    base_seeds = generate_rectangular_base_seeds(
        bounds,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        irregularity=args.irregularity,
        rng_seed=args.rng_seed,
    )
    graph = build_periodic_cell_graph(base_seeds, a1, a2, bounds)
    if not graph_is_connected(graph["adjacency"]):
        raise RuntimeError("The canonical periodic cell graph is disconnected.")

    reference_terminals = associate_terminals(DEFAULT_TERMINALS, base_seeds, a1, a2)
    reference_airway = build_airway_overlay(reference_terminals, bounds)
    reference_airway_validation = validate_airway_overlay(
        reference_airway, reference_terminals
    )
    reference_labels, reference_graph_distances = multisource_partition(
        graph["weighted_adjacency"], reference_terminals
    )
    reference_summary = summarize_partition(
        "terminal_seeded_periodic_dijkstra_reference",
        reference_labels,
        graph,
        reference_terminals,
        a1,
        a2,
        bounds,
    )
    reference_summary["airway_overlay_validation"] = reference_airway_validation

    split_weights = {
        "area": args.w_area,
        "compactness": args.w_compactness,
        "branch_length": args.w_branch_length,
    }
    if split_weights != DEFAULT_SPLIT_WEIGHTS:
        raise ValueError(
            "This exploration keeps the accepted objective weights exactly: "
            "2, 2, and 0.15."
        )

    template_results = {}
    for template_key, template in BRANCHING_TEMPLATES.items():
        labels, terminals, airway = generate_airway_from_branching_template(
            template=template,
            graph=graph,
            a1=a1,
            a2=a2,
            bounds=bounds,
            root_coordinate=np.array([args.root_x, args.root_y]),
            weights=split_weights,
        )
        airway_validation = validate_airway_overlay(airway, terminals)
        partition_summary = summarize_partition(
            f"airway_constrained_space_filling_template_{template['template_id']}",
            labels,
            graph,
            terminals,
            a1,
            a2,
            bounds,
        )
        partition_summary["airway_validation"] = airway_validation
        path_lengths_for_template = airway_root_to_terminal_path_lengths(
            airway, a1, a2
        )
        path_by_acinus = {
            item["acinus_id"]: item
            for item in path_lengths_for_template.values()
        }
        for territory in partition_summary["territories"]:
            path = path_by_acinus[territory["acinus_id"]]
            territory["root_to_terminal_node_path"] = path[
                "node_path_root_to_terminal"
            ]
            territory["root_to_terminal_path_length"] = path["path_length"]
        if not partition_summary["all_territories_connected_on_periodic_graph"]:
            raise RuntimeError(f"Template {template['template_id']} is disconnected.")
        if not partition_summary["periodic_label_continuity_satisfied"]:
            raise RuntimeError(f"Template {template['template_id']} is not periodic.")
        template_results[template_key] = {
            "template_key": template_key,
            "template_id": template["template_id"],
            "description": template["description"],
            "labels": labels,
            "terminals": terminals,
            "airway": airway,
            "summary": partition_summary,
            "root_to_terminal_paths": path_lengths_for_template,
        }

    baseline_result = template_results["A_2plus2"]
    space_filling_labels = baseline_result["labels"]
    space_filling_terminals = baseline_result["terminals"]
    space_filling_airway = baseline_result["airway"]
    space_filling_summary = baseline_result["summary"]
    path_lengths = baseline_result["root_to_terminal_paths"]

    expected_area = abs(float(np.linalg.det(np.column_stack((a1, a2)))))
    total_area = float(np.sum(graph["areas"]))
    area_partition_valid = abs(total_area - expected_area) <= 1e-10 * expected_area
    periodic_link_count = sum(
        1 for _, _, i, j in graph["periodic_links"] if i != 0 or j != 0
    )
    if not area_partition_valid or periodic_link_count == 0:
        raise RuntimeError("The canonical periodic-cell construction is invalid.")
    partition_comparison = compare_partition_labels(
        reference_labels, space_filling_labels, graph["areas"]
    )
    partition_comparison.update(
        {
            "reference_method": reference_summary["method"],
            "main_method": space_filling_summary["method"],
            "reference_CV_area": reference_summary["CV_area"],
            "space_filling_CV_area": space_filling_summary["CV_area"],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_figure = args.output_dir / "terminal_seeded_reference_partition.png"
    structure_figure = args.output_dir / "space_filling_voronoi_airway_central.png"
    central_figure = args.output_dir / "space_filling_acinar_airway_central.png"
    tiled_figure = args.output_dir / "space_filling_acinar_airway_3x3.png"
    comparison_figure = args.output_dir / "partition_method_comparison.png"
    detailed_summary_figure = args.output_dir / "airway_space_filling_summary.png"
    template_comparison_figure = (
        args.output_dir / "branching_template_comparison.png"
    )
    template_output_dir = args.output_dir / "branching_templates"
    metadata_path = args.output_dir / "periodic_acinar_metadata.json"
    terminals_path = args.output_dir / "periodic_acinar_terminals.json"
    airway_path = args.output_dir / "periodic_airway_metadata.json"
    summary_path = args.output_dir / "periodic_acinar_summary.json"
    reference_summary_path = args.output_dir / "terminal_seeded_reference_summary.json"
    comparison_path = args.output_dir / "partition_method_comparison.json"
    template_comparison_path = (
        args.output_dir / "branching_template_comparison.json"
    )

    template_output_dir.mkdir(parents=True, exist_ok=True)

    plot_central_structure(
        reference_figure,
        graph,
        reference_labels,
        reference_airway,
        a1,
        a2,
        bounds,
        colored=True,
        title="Terminal-seeded Dijkstra reference partition",
    )
    plot_central_structure(
        structure_figure,
        graph,
        space_filling_labels,
        space_filling_airway,
        a1,
        a2,
        bounds,
        colored=False,
        title="Airway-constrained space-filling tree on periodic Voronoi cells",
    )
    plot_central_structure(
        central_figure,
        graph,
        space_filling_labels,
        space_filling_airway,
        a1,
        a2,
        bounds,
        colored=True,
        title="Airway-constrained space-filling acinar territories",
    )
    plot_tiled_partition(
        tiled_figure,
        graph,
        space_filling_labels,
        space_filling_airway,
        a1,
        a2,
        bounds,
    )
    plot_partition_comparison(
        comparison_figure,
        graph,
        reference_labels,
        reference_airway,
        space_filling_labels,
        space_filling_airway,
        a1,
        a2,
        bounds,
    )
    plot_detailed_airway_summary(
        detailed_summary_figure,
        graph,
        space_filling_labels,
        space_filling_airway,
        space_filling_summary,
        a1,
        a2,
        bounds,
    )

    template_report = {}
    for template_key, result in template_results.items():
        file_stem = template_key.lower()
        template_structure_figure = (
            template_output_dir / f"{file_stem}_voronoi_airway_central.png"
        )
        template_central_figure = (
            template_output_dir / f"{file_stem}_acinar_airway_central.png"
        )
        template_tiled_figure = (
            template_output_dir / f"{file_stem}_acinar_airway_3x3.png"
        )
        template_summary_path = template_output_dir / f"{file_stem}_summary.json"
        plot_central_structure(
            template_structure_figure,
            graph,
            result["labels"],
            result["airway"],
            a1,
            a2,
            bounds,
            colored=False,
            title=(
                f"Template {result['template_id']}: periodic Voronoi + airway tree"
            ),
        )
        plot_central_structure(
            template_central_figure,
            graph,
            result["labels"],
            result["airway"],
            a1,
            a2,
            bounds,
            colored=True,
            title=(
                f"Template {result['template_id']}: acinar territories + airway tree"
            ),
        )
        plot_tiled_partition(
            template_tiled_figure,
            graph,
            result["labels"],
            result["airway"],
            a1,
            a2,
            bounds,
        )
        figures = {
            "central_voronoi_with_airway": str(template_structure_figure),
            "central_acini_with_airway": str(template_central_figure),
            "periodic_3x3_acini_with_airway": str(template_tiled_figure),
        }
        serializable_result = {
            "template_key": template_key,
            "template_id": result["template_id"],
            "description": result["description"],
            "objective": result["airway"]["split_objective"],
            "objective_weights": result["airway"]["split_weights"],
            "target_area_weights": result["airway"][
                "topology_implied_target_area_weights"
            ],
            "airway_topology": {
                "root_node_id": result["airway"]["root_node_id"],
                "nodes": result["airway"]["nodes"],
                "edges": result["airway"]["edges"],
                "recursive_splits": result["airway"]["splits"],
            },
            "root_to_terminal_paths": result["root_to_terminal_paths"],
            "partition_summary": result["summary"],
            "cell_assignments": [
                {"cell_id": int(cell_id), "acinus_id": int(acinus_id)}
                for cell_id, acinus_id in enumerate(result["labels"])
            ],
            "figures": figures,
        }
        template_summary_path.write_text(
            json.dumps(serializable_result, indent=2, default=json_ready) + "\n"
        )
        serializable_result["summary_file"] = str(template_summary_path)
        template_report[template_key] = serializable_result

    plot_branching_template_comparison(
        template_comparison_figure,
        template_results,
        graph,
        a1,
        a2,
        bounds,
    )

    cells = []
    for cell_id in range(len(base_seeds)):
        cells.append(
            {
                "cell_id": cell_id,
                "acinus_id": int(space_filling_labels[cell_id]),
                "reference_acinus_id": int(reference_labels[cell_id]),
                "seed_coordinate": base_seeds[cell_id].tolist(),
                "cell_centroid": graph["centroids"][cell_id].tolist(),
                "cell_area": float(graph["areas"][cell_id]),
                "neighbor_cell_ids": sorted(int(v) for v in graph["adjacency"][cell_id]),
                "reference_distance_to_terminal_on_graph": float(
                    reference_graph_distances[cell_id]
                ),
            }
        )

    seed_digest = hashlib.sha256(base_seeds.astype("<f8").tobytes()).hexdigest()
    metadata = {
        "schema_version": 2,
        "purpose": "Anatomical labels and regional post-processing only",
        "main_partition_method": "airway_constrained_space_filling",
        "reference_partition_method": "terminal_seeded_periodic_dijkstra",
        "canonical_cell_definition": (
            "cell_id is the zero-based index of its base seed in the half-open "
            "central periodic cell [xmin,xmax) x [ymin,ymax); periodic seed copies "
            "map back to this same cell_id"
        ),
        "periodic_cell": {
            "bounds": list(bounds),
            "a1": a1.tolist(),
            "a2": a2.tolist(),
        },
        "seed_generation": {
            "grid_x": args.grid_x,
            "grid_y": args.grid_y,
            "irregularity": args.irregularity,
            "rng_seed": args.rng_seed,
            "base_seed_sha256": seed_digest,
        },
        "airway": space_filling_airway,
        "graph": {
            "vertex_count": len(base_seeds),
            "edge_count": sum(len(v) for v in graph["adjacency"]) // 2,
            "edge_weight": "minimum-image Euclidean distance between base seeds",
            "periodic_neighbor_links": [
                {
                    "cell_id": left,
                    "neighbor_cell_id": right,
                    "neighbor_lattice_shift": [i, j],
                }
                for left, right, i, j in graph["periodic_links"]
                if i != 0 or j != 0
            ],
        },
        "cells": cells,
    }
    summary = {
        "main_model": space_filling_summary,
        "reference_model": reference_summary,
        "comparison": partition_comparison,
        "reported_metric_definitions": {
            "CV_area": "standard deviation / mean of the four acinar areas",
            "territory_anisotropy": (
                "largest/smallest eigenvalue of the area-weighted covariance "
                "of cell centroids about the periodic territory centroid"
            ),
            "root_to_terminal_path_length": (
                "sum of minimum-image Euclidean airway-edge lengths"
            ),
        },
        "airway_topology": {
            "root_node_id": space_filling_airway["root_node_id"],
            "nodes": space_filling_airway["nodes"],
            "edges": space_filling_airway["edges"],
            "recursive_splits": space_filling_airway["splits"],
        },
        "root_to_terminal_paths": path_lengths,
        "total_area": total_area,
        "expected_periodic_cell_area": expected_area,
        "cell_areas_partition_periodic_domain": area_partition_valid,
        "periodic_neighbor_link_count": periodic_link_count,
        "figures": {
            "terminal_seeded_reference": str(reference_figure),
            "space_filling_voronoi_with_airway": str(structure_figure),
            "space_filling_territories_with_airway": str(central_figure),
            "space_filling_3x3": str(tiled_figure),
            "method_comparison": str(comparison_figure),
            "detailed_airway_summary": str(detailed_summary_figure),
            "branching_template_comparison": str(template_comparison_figure),
        },
        "branching_template_exploration": template_report,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=json_ready) + "\n")
    terminals_path.write_text(json.dumps(space_filling_terminals, indent=2) + "\n")
    airway_path.write_text(
        json.dumps(space_filling_airway, indent=2, default=json_ready) + "\n"
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=json_ready) + "\n")
    reference_summary_path.write_text(
        json.dumps(reference_summary, indent=2, default=json_ready) + "\n"
    )
    comparison_path.write_text(
        json.dumps(partition_comparison, indent=2, default=json_ready) + "\n"
    )
    template_comparison_path.write_text(
        json.dumps(template_report, indent=2, default=json_ready) + "\n"
    )

    print(json.dumps(summary, indent=2, default=json_ready))
    print(f"Metadata: {metadata_path}")
    print(f"Terminals: {terminals_path}")
    print(f"Airway: {airway_path}")
    print(f"Reference: {reference_summary_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Detailed summary figure: {detailed_summary_figure}")
    print(f"Template comparison: {template_comparison_path}")


if __name__ == "__main__":
    main()
