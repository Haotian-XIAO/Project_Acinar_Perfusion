#!/usr/bin/env python3
"""Partition the accepted periodic Voronoi patch into labelled acinar territories.

The partition is metadata only.  It does not create mesh entities, physical
groups, interfaces, boundary conditions, or solver coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
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


def territory_statistics(labels, areas, adjacency, terminals):
    stats = []
    territory_areas = []
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
            }
        )
    territory_areas = np.asarray(territory_areas)
    cv_area = float(np.std(territory_areas) / np.mean(territory_areas))
    return stats, cv_area


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
    for i, j in tile_indices:
        shift = i * a1 + j * a2
        for edge in airway["edges"]:
            parent = np.asarray(nodes[edge["parent_node_id"]]["coordinate"]) + shift
            child = np.asarray(nodes[edge["child_node_id"]]["coordinate"]) + shift
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


def plot_central_structure(path, graph, labels, airway, a1, a2, bounds, colored):
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
        show_acinus_ids=colored,
    )
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
            (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
        ),
        annotate_tile=(0, 0),
        linewidth=1.7,
        marker_scale=0.32,
        show_acinus_ids=True,
    )
    finish_axis(ax, extent, "3×3 periodic acinar and airway metadata")
    ax.legend(
        handles=airway_legend_handles(),
        loc="upper right",
        fontsize=7.5,
        framealpha=0.94,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240)
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

    terminals = associate_terminals(DEFAULT_TERMINALS, base_seeds, a1, a2)
    airway = build_airway_overlay(terminals, bounds)
    airway_validation = validate_airway_overlay(airway, terminals)
    labels, graph_distances = multisource_partition(
        graph["weighted_adjacency"], terminals
    )
    territory_stats, cv_area = territory_statistics(
        labels, graph["areas"], graph["adjacency"], terminals
    )

    expected_area = abs(float(np.linalg.det(np.column_stack((a1, a2)))))
    total_area = float(np.sum(graph["areas"]))
    area_partition_valid = abs(total_area - expected_area) <= 1e-10 * expected_area
    periodic_link_count = sum(
        1 for _, _, i, j in graph["periodic_links"] if i != 0 or j != 0
    )
    copy_ids = graph["copy_canonical_ids"]
    copy_labels = labels[copy_ids]
    periodic_labels_valid = bool(
        all(
            np.all(copy_labels[copy_ids == cell_id] == labels[cell_id])
            for cell_id in range(len(base_seeds))
        )
        and periodic_link_count > 0
    )
    all_territories_connected = all(
        item["connected_on_periodic_graph"] for item in territory_stats
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    structure_figure = args.output_dir / "periodic_voronoi_airway_overlay.png"
    central_figure = args.output_dir / "periodic_acinar_airway_central.png"
    tiled_figure = args.output_dir / "periodic_acinar_airway_3x3.png"
    metadata_path = args.output_dir / "periodic_acinar_metadata.json"
    terminals_path = args.output_dir / "periodic_acinar_terminals.json"
    airway_path = args.output_dir / "periodic_airway_metadata.json"
    summary_path = args.output_dir / "periodic_acinar_summary.json"

    plot_central_structure(
        structure_figure, graph, labels, airway, a1, a2, bounds, colored=False
    )
    plot_central_structure(
        central_figure, graph, labels, airway, a1, a2, bounds, colored=True
    )
    plot_tiled_partition(tiled_figure, graph, labels, airway, a1, a2, bounds)

    cells = []
    for cell_id in range(len(base_seeds)):
        cells.append(
            {
                "cell_id": cell_id,
                "acinus_id": int(labels[cell_id]),
                "seed_coordinate": base_seeds[cell_id].tolist(),
                "cell_centroid": graph["centroids"][cell_id].tolist(),
                "cell_area": float(graph["areas"][cell_id]),
                "neighbor_cell_ids": sorted(int(v) for v in graph["adjacency"][cell_id]),
                "distance_to_terminal_on_graph": float(graph_distances[cell_id]),
            }
        )

    seed_digest = hashlib.sha256(base_seeds.astype("<f8").tobytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "purpose": "Anatomical labels and regional post-processing only",
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
        "airway_overlay": airway,
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
        "number_of_acini": len(terminals),
        "total_cell_count": len(base_seeds),
        "total_area": total_area,
        "expected_periodic_cell_area": expected_area,
        "cell_areas_partition_periodic_domain": area_partition_valid,
        "CV_area": cv_area,
        "all_territories_connected_on_periodic_graph": all_territories_connected,
        "periodic_label_continuity_satisfied": periodic_labels_valid,
        "periodic_neighbor_link_count": periodic_link_count,
        "airway_overlay_validation": airway_validation,
        "territories": territory_stats,
        "figures": {
            "voronoi_with_airway": str(structure_figure),
            "central_labels_with_airway": str(central_figure),
            "tiled_labels_and_airway_3x3": str(tiled_figure),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=json_ready) + "\n")
    terminals_path.write_text(json.dumps(terminals, indent=2) + "\n")
    airway_path.write_text(json.dumps(airway, indent=2) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, default=json_ready) + "\n")

    print(json.dumps(summary, indent=2, default=json_ready))
    print(f"Metadata: {metadata_path}")
    print(f"Terminals: {terminals_path}")
    print(f"Airway overlay: {airway_path}")


if __name__ == "__main__":
    main()
