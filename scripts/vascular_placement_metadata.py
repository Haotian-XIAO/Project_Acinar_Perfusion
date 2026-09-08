#!/usr/bin/env python3
"""Generate periodic arterial and shared-venous placement metadata.

The vascular objects produced here are anatomical/post-processing metadata.
They do not create Gmsh entities, finite-element regions, boundary conditions,
source terms, or changes to the governing equations.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import Voronoi

from partition_periodic_acini import (
    BRANCHING_TEMPLATES,
    DEFAULT_AIRWAY_ROOT,
    DEFAULT_SPLIT_WEIGHTS,
    airway_legend_handles,
    build_periodic_cell_graph,
    draw_airway_overlay,
    draw_periodic_cells,
    generate_airway_from_branching_template,
    generate_rectangular_base_seeds,
    json_ready,
    make_periodic_seed_cloud,
    minimum_image_vector,
    summarize_partition,
    validate_airway_overlay,
    wrap_periodic_point,
)


VENOUS_OBJECTIVE_WEIGHTS = {
    "drainage": 1.0,
    "balance": 0.25,
    "separation": 0.10,
    "junction_preference": 0.02,
}
SEPARATION_THRESHOLD_FRACTION = 0.20


def closest_point_on_segment(point, endpoint_a, endpoint_b):
    segment = np.asarray(endpoint_b, dtype=float) - np.asarray(endpoint_a, dtype=float)
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-28:
        return np.asarray(endpoint_a, dtype=float).copy()
    parameter = float(
        np.dot(np.asarray(point, dtype=float) - endpoint_a, segment) / denominator
    )
    parameter = min(1.0, max(0.0, parameter))
    return np.asarray(endpoint_a, dtype=float) + parameter * segment


def segment_copy_nearest_point(point, endpoint_a, endpoint_b, a1, a2):
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    midpoint = 0.5 * (np.asarray(endpoint_a) + np.asarray(endpoint_b))
    nearest_midpoint = np.asarray(point) + minimum_image_vector(
        midpoint - np.asarray(point), lattice_matrix, lattice_inverse
    )
    shift = nearest_midpoint - midpoint
    shifted_a = np.asarray(endpoint_a) + shift
    shifted_b = np.asarray(endpoint_b) + shift
    projection = closest_point_on_segment(point, shifted_a, shifted_b)
    return projection, float(np.linalg.norm(projection - point))


def place_arterial_sources(terminals, labels, graph, a1, a2, bounds):
    """Project each terminal onto the nearest wall ridge in its own acinus."""
    sources = []
    for terminal in sorted(terminals, key=lambda item: item["terminal_id"]):
        terminal_id = int(terminal["terminal_id"])
        acinus_id = int(terminal["acinus_id"])
        terminal_coordinate = np.asarray(terminal["terminal_coordinate"], dtype=float)
        candidate_cells = np.flatnonzero(labels == acinus_id)
        best = None
        for cell_id in candidate_cells:
            polygon = graph["polygons"][int(cell_id)]
            for edge_index, endpoint_a in enumerate(polygon):
                endpoint_b = polygon[(edge_index + 1) % len(polygon)]
                projection, distance = segment_copy_nearest_point(
                    terminal_coordinate, endpoint_a, endpoint_b, a1, a2
                )
                score = (distance, int(cell_id), int(edge_index))
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "coordinate": wrap_periodic_point(projection, bounds),
                        "projection_distance": distance,
                        "associated_canonical_cell_id": int(cell_id),
                        "associated_polygon_edge_index": int(edge_index),
                    }
        if best is None:
            raise RuntimeError(f"No admissible wall point found for acinus {acinus_id}.")
        sources.append(
            {
                "arterial_id": f"Pa{terminal_id}",
                "terminal_id": terminal_id,
                "acinus_id": acinus_id,
                "terminal_coordinate": terminal_coordinate.tolist(),
                "arterial_coordinate": best["coordinate"].tolist(),
                "projection_distance": float(best["projection_distance"]),
                "associated_canonical_cell_id": best[
                    "associated_canonical_cell_id"
                ],
                "associated_polygon_edge_index": best[
                    "associated_polygon_edge_index"
                ],
                "admissibility": "existing Voronoi ridge in the physical wall domain",
                "periodic_consistency": True,
            }
        )
    if len(sources) != len(terminals):
        raise RuntimeError("Arterial placement must be one-to-one with terminals.")
    return sources


def canonical_periodic_ridges(base_seeds, a1, a2, bounds):
    """Extract one canonical identity for each finite periodic Voronoi ridge."""
    cloud, canonical_ids, shifts = make_periodic_seed_cloud(base_seeds, a1, a2)
    voronoi = Voronoi(cloud)
    central = np.all(shifts == 0, axis=1)
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    origin = np.asarray(bounds[:2], dtype=float)
    ridge_by_identity = {}

    for point_pair, vertex_ids in zip(
        voronoi.ridge_points, voronoi.ridge_vertices
    ):
        left_point, right_point = (int(value) for value in point_pair)
        if not (central[left_point] or central[right_point]):
            continue
        if len(vertex_ids) != 2 or -1 in vertex_ids:
            continue
        left_id = int(canonical_ids[left_point])
        right_id = int(canonical_ids[right_point])
        if left_id == right_id:
            continue
        relative_shift = shifts[right_point] - shifts[left_point]
        forward = (left_id, right_id, int(relative_shift[0]), int(relative_shift[1]))
        reverse = (right_id, left_id, -int(relative_shift[0]), -int(relative_shift[1]))
        identity = min(forward, reverse)
        if identity in ridge_by_identity:
            continue

        endpoints = np.asarray(
            [voronoi.vertices[int(vertex_id)] for vertex_id in vertex_ids],
            dtype=float,
        )
        midpoint = np.mean(endpoints, axis=0)
        fractional_midpoint = lattice_inverse @ (midpoint - origin)
        lattice_shift = np.floor(fractional_midpoint).astype(int)
        endpoints -= lattice_matrix @ lattice_shift
        midpoint -= lattice_matrix @ lattice_shift

        first_cell, second_cell, shift_i, shift_j = identity
        ridge_by_identity[identity] = {
            "canonical_identity": [first_cell, second_cell, shift_i, shift_j],
            "associated_canonical_cells": [first_cell, second_cell],
            "neighbor_lattice_shift": [shift_i, shift_j],
            "endpoints": endpoints.tolist(),
            "midpoint": wrap_periodic_point(midpoint, bounds).tolist(),
            "length": float(np.linalg.norm(endpoints[1] - endpoints[0])),
        }

    ridges = []
    for ridge_id, identity in enumerate(sorted(ridge_by_identity)):
        ridge = ridge_by_identity[identity]
        ridge["ridge_id"] = f"R{ridge_id:03d}"
        ridges.append(ridge)
    return ridges


def canonical_coordinate_key(coordinate, bounds, digits=11):
    wrapped = wrap_periodic_point(coordinate, bounds)
    return tuple(float(value) for value in np.round(wrapped, digits))


def identify_interacinar_venous_candidates(ridges, labels, bounds):
    """Create ridge-midpoint and multi-acinus junction candidates."""
    interacinar_ridges = []
    for ridge in ridges:
        left_cell, right_cell = ridge["associated_canonical_cells"]
        adjacent_acini = sorted({int(labels[left_cell]), int(labels[right_cell])})
        if len(adjacent_acini) != 2:
            continue
        record = dict(ridge)
        record["adjacent_acini"] = adjacent_acini
        interacinar_ridges.append(record)

    candidates = []
    for ridge in interacinar_ridges:
        candidates.append(
            {
                "coordinate": ridge["midpoint"],
                "adjacent_acini": ridge["adjacent_acini"],
                "candidate_type": "two_acini_boundary",
                "associated_canonical_cells": ridge[
                    "associated_canonical_cells"
                ],
                "associated_ridge_ids": [ridge["ridge_id"]],
                "associated_ridges": [ridge],
            }
        )

    junctions = {}
    for ridge in interacinar_ridges:
        for endpoint in ridge["endpoints"]:
            key = canonical_coordinate_key(endpoint, bounds)
            junction = junctions.setdefault(
                key,
                {
                    "coordinate": list(key),
                    "acini": set(),
                    "cells": set(),
                    "ridge_ids": set(),
                    "ridges": {},
                },
            )
            junction["acini"].update(ridge["adjacent_acini"])
            junction["cells"].update(ridge["associated_canonical_cells"])
            junction["ridge_ids"].add(ridge["ridge_id"])
            junction["ridges"][ridge["ridge_id"]] = ridge

    for key in sorted(junctions):
        junction = junctions[key]
        adjacent_acini = sorted(junction["acini"])
        if len(adjacent_acini) < 3:
            continue
        count_name = {
            3: "three",
            4: "four",
            5: "five",
            6: "six",
        }.get(len(adjacent_acini), str(len(adjacent_acini)))
        candidates.append(
            {
                "coordinate": junction["coordinate"],
                "adjacent_acini": adjacent_acini,
                "candidate_type": f"{count_name}_acini_junction",
                "associated_canonical_cells": sorted(junction["cells"]),
                "associated_ridge_ids": sorted(junction["ridge_ids"]),
                "associated_ridges": [
                    junction["ridges"][ridge_id]
                    for ridge_id in sorted(junction["ridge_ids"])
                ],
            }
        )

    candidates.sort(
        key=lambda item: (
            item["candidate_type"],
            item["coordinate"][0],
            item["coordinate"][1],
            item["associated_ridge_ids"],
        )
    )
    for candidate_index, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"V{candidate_index:03d}"
        candidate["canonical_identity"] = {
            "candidate_type": candidate["candidate_type"],
            "wrapped_coordinate": candidate["coordinate"],
            "associated_ridge_ids": candidate["associated_ridge_ids"],
        }
        candidate["lies_on_existing_wall_network"] = True
        candidate["periodic_consistency"] = True
    return interacinar_ridges, candidates


def candidate_graph_distances(candidate, graph, a1, a2):
    """Graph-topological cell distances from one ridge/junction candidate."""
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    coordinate = np.asarray(candidate["coordinate"], dtype=float)
    distances = np.full(len(graph["centroids"]), np.inf)
    queue = []
    for cell_id in candidate["associated_canonical_cells"]:
        delta = minimum_image_vector(
            graph["centroids"][cell_id] - coordinate,
            lattice_matrix,
            lattice_inverse,
        )
        distance = float(np.linalg.norm(delta))
        if distance < distances[cell_id]:
            distances[cell_id] = distance
            heapq.heappush(queue, (distance, int(cell_id)))
    while queue:
        distance, cell_id = heapq.heappop(queue)
        if distance > distances[cell_id] + 1e-14:
            continue
        for neighbor_id, edge_length in graph["weighted_adjacency"][cell_id]:
            candidate_distance = distance + edge_length
            if candidate_distance < distances[neighbor_id] - 1e-14:
                distances[neighbor_id] = candidate_distance
                heapq.heappush(queue, (candidate_distance, int(neighbor_id)))
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("A venous candidate cannot reach the full periodic graph.")
    return distances


def pairwise_periodic_candidate_distances(candidates, a1, a2):
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    coordinates = np.asarray([item["coordinate"] for item in candidates])
    distances = np.zeros((len(candidates), len(candidates)))
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            delta = minimum_image_vector(
                coordinates[right] - coordinates[left],
                lattice_matrix,
                lattice_inverse,
            )
            distances[left, right] = distances[right, left] = np.linalg.norm(delta)
    return distances


def optimize_venous_sites(
    number_of_sites,
    candidates,
    candidate_distances,
    candidate_separations,
    cell_areas,
    periodic_area,
    weights=VENOUS_OBJECTIVE_WEIGHTS,
):
    """Globally enumerate the finite candidate set for a fixed N_v."""
    if not 1 <= number_of_sites <= len(candidates):
        raise ValueError("Invalid number of venous sites.")
    total_area = float(np.sum(cell_areas))
    length_scale = periodic_area**0.5
    separation_threshold = SEPARATION_THRESHOLD_FRACTION * length_scale
    best = None

    for selection in itertools.combinations(range(len(candidates)), number_of_sites):
        selected_distances = candidate_distances[np.asarray(selection, dtype=int)]
        assignments = np.argmin(selected_distances, axis=0)
        minimum_distances = np.min(selected_distances, axis=0)
        drainage_cost = float(
            np.dot(cell_areas, minimum_distances) / (total_area * length_scale)
        )
        catchment_areas = np.bincount(
            assignments,
            weights=cell_areas,
            minlength=number_of_sites,
        )
        catchment_cv = float(np.std(catchment_areas) / np.mean(catchment_areas))
        balance_cost = catchment_cv**2

        if number_of_sites == 1:
            minimum_separation = None
            separation_cost = 0.0
        else:
            selected_pair_distances = [
                candidate_separations[left, right]
                for left, right in itertools.combinations(selection, 2)
            ]
            minimum_separation = float(min(selected_pair_distances))
            separation_cost = float(
                np.mean(
                    [
                        max(0.0, 1.0 - distance / separation_threshold) ** 2
                        for distance in selected_pair_distances
                    ]
                )
            )

        junction_cost = float(
            np.mean(
                [
                    1.0 if len(candidates[index]["adjacent_acini"]) == 2 else 0.0
                    for index in selection
                ]
            )
        )
        objective = (
            weights["drainage"] * drainage_cost
            + weights["balance"] * balance_cost
            + weights["separation"] * separation_cost
            + weights["junction_preference"] * junction_cost
        )
        score = (
            objective,
            drainage_cost,
            balance_cost,
            separation_cost,
            junction_cost,
            selection,
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "selection": selection,
                "assignments": assignments,
                "catchment_areas": catchment_areas,
                "normalized_drainage_cost": drainage_cost,
                "catchment_area_CV": catchment_cv,
                "minimum_pairwise_separation": minimum_separation,
                "objective_terms": {
                    "J_vein": float(objective),
                    "J_drainage": drainage_cost,
                    "J_balance": float(balance_cost),
                    "J_separation": separation_cost,
                    "J_junction_preference": junction_cost,
                },
            }

    if best is None:
        raise RuntimeError("No venous-site configuration was found.")
    selection = best.pop("selection")
    best.pop("score")
    assignments = best.pop("assignments")
    best["number_of_venous_sinks"] = number_of_sites
    best["selected_candidate_ids"] = [
        candidates[index]["candidate_id"] for index in selection
    ]
    best["selected_sinks"] = []
    for local_index, candidate_index in enumerate(selection):
        candidate = candidates[candidate_index]
        assigned_cells = np.flatnonzero(assignments == local_index)
        best["selected_sinks"].append(
            {
                "candidate_id": candidate["candidate_id"],
                "coordinate": candidate["coordinate"],
                "adjacent_acini": candidate["adjacent_acini"],
                "number_of_directly_adjacent_acini": len(
                    candidate["adjacent_acini"]
                ),
                "candidate_type": candidate["candidate_type"],
                "drainage_catchment_area": float(best["catchment_areas"][local_index]),
                "drainage_catchment_cell_count": int(len(assigned_cells)),
                "assigned_canonical_cell_ids": assigned_cells.tolist(),
            }
        )
    best["catchment_areas"] = [float(value) for value in best["catchment_areas"]]
    best["all_cells_assigned_once"] = bool(len(assignments) == len(cell_areas))
    return best


def vascular_legend_handles():
    return airway_legend_handles() + [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="#ff6d00",
            markeredgecolor="black",
            markersize=9,
            label="Arterial source",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color="#00acc1",
            markersize=6,
            label="Venous candidate",
        ),
        Line2D(
            [0],
            [0],
            marker="H",
            linestyle="none",
            markerfacecolor="#0d47a1",
            markeredgecolor="white",
            markersize=10,
            label="Selected venous sink",
        ),
    ]


def draw_vascular_overlay(
    ax,
    arterial_sources,
    venous_candidates,
    selected_sinks,
    a1,
    a2,
    tile_indices=((0, 0),),
    annotate_tile=(0, 0),
    show_all_candidates=True,
):
    selected_ids = {sink["candidate_id"] for sink in selected_sinks}
    lattice_matrix = np.column_stack((a1, a2))
    lattice_inverse = np.linalg.inv(lattice_matrix)
    for i, j in tile_indices:
        shift = i * a1 + j * a2
        if show_all_candidates:
            boundary_points = []
            junction_points = []
            for candidate in venous_candidates:
                point = np.asarray(candidate["coordinate"]) + shift
                if "junction" in candidate["candidate_type"]:
                    junction_points.append(point)
                else:
                    boundary_points.append(point)
            if boundary_points:
                points = np.asarray(boundary_points)
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    marker="x",
                    s=16,
                    c="#00acc1",
                    linewidths=0.8,
                    alpha=0.75,
                    zorder=9,
                )
            if junction_points:
                points = np.asarray(junction_points)
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    marker="P",
                    s=26,
                    c="#00acc1",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.9,
                    zorder=9,
                )

        for source in arterial_sources:
            terminal_point = np.asarray(source["terminal_coordinate"]) + shift
            arterial_delta = minimum_image_vector(
                np.asarray(source["arterial_coordinate"])
                - np.asarray(source["terminal_coordinate"]),
                lattice_matrix,
                lattice_inverse,
            )
            point = terminal_point + arterial_delta
            ax.plot(
                [terminal_point[0], point[0]],
                [terminal_point[1], point[1]],
                color="#ff6d00",
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
                zorder=10,
            )
            ax.scatter(
                *point,
                marker="^",
                s=105,
                c="#ff6d00",
                edgecolors="black",
                linewidths=0.9,
                zorder=11,
            )
            if (i, j) == annotate_tile:
                ax.annotate(
                    source["arterial_id"],
                    point,
                    xytext=(5, -12),
                    textcoords="offset points",
                    fontsize=7.5,
                    weight="bold",
                    color="#bf360c",
                    zorder=12,
                )

        for sink in selected_sinks:
            point = np.asarray(sink["coordinate"]) + shift
            ax.scatter(
                *point,
                marker="H",
                s=155,
                c="#0d47a1",
                edgecolors="white",
                linewidths=1.2,
                zorder=13,
            )
            if (i, j) == annotate_tile:
                ax.annotate(
                    sink["candidate_id"],
                    point,
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    weight="bold",
                    color="#0d47a1",
                    zorder=14,
                )

    if not selected_ids.issubset(
        {candidate["candidate_id"] for candidate in venous_candidates}
    ):
        raise RuntimeError("A selected sink is not an admissible candidate.")


def finish_central_axis(ax, bounds, title):
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=12, weight="bold")


def draw_central_layout(
    ax,
    graph,
    labels,
    airway,
    arterial_sources,
    venous_candidates,
    solution,
    a1,
    a2,
    bounds,
    show_all_candidates=True,
):
    tile_indices = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1))
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
        tile_indices=tile_indices,
        linewidth=2.8,
        marker_scale=0.8,
        show_acinus_ids=True,
    )
    draw_vascular_overlay(
        ax,
        arterial_sources,
        venous_candidates,
        solution["selected_sinks"],
        a1,
        a2,
        tile_indices=tile_indices,
        show_all_candidates=show_all_candidates,
    )
    finish_central_axis(
        ax,
        bounds,
        f"Shared venous drainage: $N_v={solution['number_of_venous_sinks']}$",
    )


def plot_single_configuration(
    path,
    graph,
    labels,
    airway,
    arterial_sources,
    venous_candidates,
    solution,
    a1,
    a2,
    bounds,
):
    fig, ax = plt.subplots(figsize=(8.8, 8.1))
    draw_central_layout(
        ax,
        graph,
        labels,
        airway,
        arterial_sources,
        venous_candidates,
        solution,
        a1,
        a2,
        bounds,
    )
    ax.legend(
        handles=vascular_legend_handles(),
        loc="upper right",
        fontsize=7.2,
        framealpha=0.95,
    )
    ax.text(
        0.02,
        0.02,
        f"Drainage={solution['normalized_drainage_cost']:.4f}  "
        f"Catchment CV={solution['catchment_area_CV']:.4f}",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.4", "alpha": 0.9},
        zorder=20,
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_detailed_layout(
    path,
    graph,
    labels,
    airway,
    arterial_sources,
    venous_candidates,
    solution,
    a1,
    a2,
    bounds,
):
    fig = plt.figure(figsize=(16.2, 9.0))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0), wspace=0.13)
    ax = fig.add_subplot(grid[0, 0])
    info_ax = fig.add_subplot(grid[0, 1])
    draw_central_layout(
        ax,
        graph,
        labels,
        airway,
        arterial_sources,
        venous_candidates,
        solution,
        a1,
        a2,
        bounds,
    )
    ax.legend(
        handles=vascular_legend_handles(),
        loc="upper right",
        fontsize=7.2,
        framealpha=0.95,
    )

    info_ax.axis("off")
    info_ax.text(
        0.0,
        0.97,
        "Arterial sources",
        transform=info_ax.transAxes,
        fontsize=13,
        weight="bold",
        va="top",
    )
    arterial_rows = [
        [
            item["arterial_id"],
            f"T{item['terminal_id']} / A{item['acinus_id']}",
            f"({item['arterial_coordinate'][0]:.4f}, "
            f"{item['arterial_coordinate'][1]:.4f})",
            f"{item['projection_distance']:.5f}",
        ]
        for item in arterial_sources
    ]
    arterial_table = info_ax.table(
        cellText=arterial_rows,
        colLabels=("Source", "Terminal / acinus", "Coordinate", "Projection"),
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.66, 1.0, 0.25),
    )
    arterial_table.auto_set_font_size(False)
    arterial_table.set_fontsize(8.2)

    info_ax.text(
        0.0,
        0.59,
        f"Selected shared venous sinks ($N_v={solution['number_of_venous_sinks']}$)",
        transform=info_ax.transAxes,
        fontsize=13,
        weight="bold",
    )
    venous_rows = [
        [
            sink["candidate_id"],
            f"({sink['coordinate'][0]:.4f}, {sink['coordinate'][1]:.4f})",
            ",".join(f"A{value}" for value in sink["adjacent_acini"]),
            sink["candidate_type"].replace("_", " "),
            f"{sink['drainage_catchment_area']:.3f}",
        ]
        for sink in solution["selected_sinks"]
    ]
    venous_table = info_ax.table(
        cellText=venous_rows,
        colLabels=("ID", "Coordinate", "Adjacent", "Type", "Catchment"),
        cellLoc="center",
        colLoc="center",
        colWidths=(0.10, 0.24, 0.15, 0.32, 0.16),
        bbox=(0.0, 0.30, 1.0, 0.23),
    )
    venous_table.auto_set_font_size(False)
    venous_table.set_fontsize(7.8)
    info_ax.text(
        0.0,
        0.23,
        f"Candidates: {len(venous_candidates)}   "
        f"Normalized drainage: {solution['normalized_drainage_cost']:.5f}   "
        f"Catchment CV: {solution['catchment_area_CV']:.5f}",
        transform=info_ax.transAxes,
        fontsize=9.5,
        weight="bold",
    )
    info_ax.text(
        0.0,
        0.16,
        "Placement metadata only — no vascular channel, FE region, source term, "
        "or boundary condition is created.",
        transform=info_ax.transAxes,
        fontsize=9.2,
        wrap=True,
        bbox={"facecolor": "#f4f7fb", "edgecolor": "#0d47a1", "pad": 7},
    )
    fig.suptitle(
        "Anatomically motivated arterial-source and shared-venous placement",
        fontsize=17,
        weight="bold",
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_configuration_comparison(
    path,
    graph,
    labels,
    airway,
    arterial_sources,
    venous_candidates,
    solutions,
    a1,
    a2,
    bounds,
):
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 13.0))
    for ax, solution in zip(axes.flat, solutions):
        draw_central_layout(
            ax,
            graph,
            labels,
            airway,
            arterial_sources,
            venous_candidates,
            solution,
            a1,
            a2,
            bounds,
        )
        ax.text(
            0.02,
            0.02,
            f"Drainage={solution['normalized_drainage_cost']:.4f}; "
            f"CV={solution['catchment_area_CV']:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.9},
            zorder=20,
        )
    axes[0, 1].legend(
        handles=vascular_legend_handles(),
        loc="upper right",
        fontsize=6.7,
        framealpha=0.95,
    )
    fig.suptitle(
        "Fixed-$N_v$ shared venous-site optimization",
        fontsize=17,
        weight="bold",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_diminishing_returns(path, solutions):
    numbers = np.asarray([item["number_of_venous_sinks"] for item in solutions])
    drainage = np.asarray([item["normalized_drainage_cost"] for item in solutions])
    catchment_cv = np.asarray([item["catchment_area_CV"] for item in solutions])
    relative = drainage / drainage[0]

    fig = plt.figure(figsize=(11.5, 7.8))
    grid = fig.add_gridspec(2, 1, height_ratios=(0.68, 0.32), hspace=0.16)
    ax = fig.add_subplot(grid[0, 0])
    cv_ax = ax.twinx()
    drainage_line = ax.plot(
        numbers,
        relative,
        marker="o",
        linewidth=2.5,
        color="#0d47a1",
        label="Drainage cost / $N_v=1$",
    )[0]
    cv_line = cv_ax.plot(
        numbers,
        catchment_cv,
        marker="s",
        linewidth=2.0,
        color="#d84315",
        label="Catchment-area CV",
    )[0]
    ax.set_xticks(numbers)
    ax.set_xlabel("Number of shared venous sinks $N_v$")
    ax.set_ylabel("Relative normalized drainage cost", color="#0d47a1")
    cv_ax.set_ylabel("Drainage catchment-area CV", color="#d84315")
    ax.grid(alpha=0.25)
    ax.legend(handles=[drainage_line, cv_line], loc="upper right")
    ax.set_title("Geometric drainage improvement without an $N_v$ penalty")

    table_ax = fig.add_subplot(grid[1, 0])
    table_ax.axis("off")
    previous = None
    rows = []
    for solution, relative_cost in zip(solutions, relative):
        improvement = None if previous is None else previous - solution[
            "normalized_drainage_cost"
        ]
        rows.append(
            [
                str(solution["number_of_venous_sinks"]),
                f"{solution['normalized_drainage_cost']:.5f}",
                f"{relative_cost:.3f}",
                "—" if improvement is None else f"{improvement:.5f}",
                f"{solution['catchment_area_CV']:.4f}",
                ", ".join(solution["selected_candidate_ids"]),
            ]
        )
        previous = solution["normalized_drainage_cost"]
    table = table_ax.table(
        cellText=rows,
        colLabels=(
            "$N_v$",
            "Drainage cost",
            "Relative cost",
            "Marginal reduction",
            "Catchment CV",
            "Selected candidates",
        ),
        cellLoc="center",
        colLoc="center",
        colWidths=(0.07, 0.17, 0.15, 0.18, 0.16, 0.27),
        bbox=(0.0, 0.02, 1.0, 0.94),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_periodic_tiling(
    path,
    graph,
    labels,
    airway,
    arterial_sources,
    venous_candidates,
    solution,
    a1,
    a2,
):
    fig, ax = plt.subplots(figsize=(12.5, 11.5))
    tile_indices = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1))
    draw_periodic_cells(
        ax,
        graph["polygons"],
        labels,
        a1,
        a2,
        tile_range=range(-1, 2),
        colored=True,
        linewidth=0.28,
    )
    draw_airway_overlay(
        ax,
        airway,
        a1,
        a2,
        tile_indices=tile_indices,
        linewidth=2.2,
        marker_scale=0.55,
        show_acinus_ids=False,
    )
    draw_vascular_overlay(
        ax,
        arterial_sources,
        venous_candidates,
        solution["selected_sinks"],
        a1,
        a2,
        tile_indices=tile_indices,
        show_all_candidates=False,
    )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.axvline(1.0, color="black", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(-1.0, 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"3×3 periodic vascular metadata continuation ($N_v={solution['number_of_venous_sinks']}$)",
        fontsize=14,
        weight="bold",
    )
    ax.legend(
        handles=vascular_legend_handles(),
        loc="upper right",
        fontsize=7.4,
        framealpha=0.95,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/vascular_placement")
    )
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
    labels, terminals, airway = generate_airway_from_branching_template(
        BRANCHING_TEMPLATES["A_2plus2"],
        graph,
        a1,
        a2,
        bounds,
        root_coordinate=DEFAULT_AIRWAY_ROOT,
        weights=DEFAULT_SPLIT_WEIGHTS,
    )
    airway_validation = validate_airway_overlay(airway, terminals)
    partition_summary = summarize_partition(
        "accepted_airway_constrained_space_filling_template_A",
        labels,
        graph,
        terminals,
        a1,
        a2,
        bounds,
    )
    if not partition_summary["all_territories_connected_on_periodic_graph"]:
        raise RuntimeError("The accepted acinar partition is disconnected.")
    if not partition_summary["periodic_label_continuity_satisfied"]:
        raise RuntimeError("The accepted acinar labels are not periodic.")

    arterial_sources = place_arterial_sources(
        terminals, labels, graph, a1, a2, bounds
    )
    ridges = canonical_periodic_ridges(base_seeds, a1, a2, bounds)
    interacinar_ridges, venous_candidates = identify_interacinar_venous_candidates(
        ridges, labels, bounds
    )
    if not venous_candidates:
        raise RuntimeError("No interacinar venous candidates were identified.")

    graph_edge_count = sum(len(neighbors) for neighbors in graph["adjacency"]) // 2
    label_crossing_edge_count = sum(
        1
        for cell_id, neighbors in enumerate(graph["adjacency"])
        for neighbor_id in neighbors
        if cell_id < neighbor_id and labels[cell_id] != labels[neighbor_id]
    )
    if len(ridges) != graph_edge_count:
        raise RuntimeError("Canonical ridge identities do not match graph adjacencies.")
    if len(interacinar_ridges) != label_crossing_edge_count:
        raise RuntimeError("An interacinar graph adjacency lacks a canonical ridge.")
    candidate_ids = [item["candidate_id"] for item in venous_candidates]
    candidate_identity_strings = [
        json.dumps(item["canonical_identity"], sort_keys=True)
        for item in venous_candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)) or len(
        candidate_identity_strings
    ) != len(set(candidate_identity_strings)):
        raise RuntimeError("Venous candidate identities are not unique.")
    if any(
        not (bounds[0] <= item["coordinate"][0] < bounds[2])
        or not (bounds[1] <= item["coordinate"][1] < bounds[3])
        for item in venous_candidates
    ):
        raise RuntimeError("A venous candidate is outside the canonical cell.")
    if any(
        labels[item["associated_canonical_cell_id"]] != item["acinus_id"]
        for item in arterial_sources
    ):
        raise RuntimeError("An arterial source is not associated with its own acinus.")

    candidate_distances = np.asarray(
        [
            candidate_graph_distances(candidate, graph, a1, a2)
            for candidate in venous_candidates
        ]
    )
    candidate_separations = pairwise_periodic_candidate_distances(
        venous_candidates, a1, a2
    )
    periodic_area = abs(float(np.linalg.det(np.column_stack((a1, a2)))))
    solutions = [
        optimize_venous_sites(
            number_of_sites,
            venous_candidates,
            candidate_distances,
            candidate_separations,
            graph["areas"],
            periodic_area,
        )
        for number_of_sites in (1, 2, 3, 4)
    ]
    candidate_id_set = set(candidate_ids)
    for solution in solutions:
        if len(solution["selected_candidate_ids"]) != solution[
            "number_of_venous_sinks"
        ]:
            raise RuntimeError("A fixed-N_v solution has the wrong number of sinks.")
        if not set(solution["selected_candidate_ids"]).issubset(candidate_id_set):
            raise RuntimeError("A fixed-N_v solution selected an inadmissible site.")
        if abs(sum(solution["catchment_areas"]) - periodic_area) > 1e-12:
            raise RuntimeError("Venous catchments do not cover the periodic cell.")

    # N_v is not optimized. N_v=2 is used only as the central/tiled geometric
    # efficiency example; every fixed-N solution remains available for review.
    example_solution = solutions[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration_figures = {}
    for solution in solutions:
        number_of_sites = solution["number_of_venous_sinks"]
        path = args.output_dir / f"vascular_layout_Nv{number_of_sites}.png"
        plot_single_configuration(
            path,
            graph,
            labels,
            airway,
            arterial_sources,
            venous_candidates,
            solution,
            a1,
            a2,
            bounds,
        )
        configuration_figures[str(number_of_sites)] = str(path)

    detailed_figure = args.output_dir / "vascular_layout_detailed_Nv2.png"
    comparison_figure = args.output_dir / "venous_configuration_comparison.png"
    diminishing_returns_figure = args.output_dir / "venous_diminishing_returns.png"
    tiled_figure = args.output_dir / "vascular_layout_Nv2_periodic_3x3.png"
    metadata_path = args.output_dir / "vascular_placement_metadata.json"
    plot_detailed_layout(
        detailed_figure,
        graph,
        labels,
        airway,
        arterial_sources,
        venous_candidates,
        example_solution,
        a1,
        a2,
        bounds,
    )
    plot_configuration_comparison(
        comparison_figure,
        graph,
        labels,
        airway,
        arterial_sources,
        venous_candidates,
        solutions,
        a1,
        a2,
        bounds,
    )
    plot_diminishing_returns(diminishing_returns_figure, solutions)
    plot_periodic_tiling(
        tiled_figure,
        graph,
        labels,
        airway,
        arterial_sources,
        venous_candidates,
        example_solution,
        a1,
        a2,
    )

    candidate_type_counts = {}
    for candidate in venous_candidates:
        candidate_type = candidate["candidate_type"]
        candidate_type_counts[candidate_type] = (
            candidate_type_counts.get(candidate_type, 0) + 1
        )
    metadata = {
        "schema_version": 1,
        "purpose": "anatomical vascular placement metadata only",
        "affects_fe_geometry": False,
        "affects_mesh": False,
        "affects_material_regions": False,
        "affects_governing_equations": False,
        "affects_boundary_conditions": False,
        "periodic_cell": {
            "bounds": list(bounds),
            "a1": a1.tolist(),
            "a2": a2.tolist(),
        },
        "accepted_acinar_model": {
            "template": "A_2plus2",
            "partition_summary": partition_summary,
            "airway_validation": airway_validation,
        },
        "arterial_placement_rule": (
            "minimum-image projection from each terminal bronchiole to the "
            "nearest Voronoi ridge belonging to a canonical cell with the same "
            "acinus_id"
        ),
        "arterial_sources": arterial_sources,
        "venous_candidate_rule": (
            "one candidate at every label-crossing canonical ridge midpoint plus "
            "one canonical candidate at each junction incident to at least three acini"
        ),
        "canonical_periodic_ridge_count": len(ridges),
        "interacinar_ridge_count": len(interacinar_ridges),
        "venous_candidate_count": len(venous_candidates),
        "venous_candidate_type_counts": candidate_type_counts,
        "validation": {
            "arterial_source_to_terminal_one_to_one": True,
            "arterial_sources_on_existing_wall_ridges": True,
            "arterial_sources_in_matching_acini": True,
            "canonical_ridges_match_periodic_graph_edges": True,
            "every_interacinar_graph_edge_has_a_ridge": True,
            "candidate_identities_unique_and_canonically_wrapped": True,
            "all_selected_sinks_are_admissible_candidates": True,
            "all_fixed_Nv_catchments_cover_the_periodic_cell": True,
            "vascular_objects_are_metadata_only": True,
        },
        "interacinar_ridges": interacinar_ridges,
        "venous_candidates": venous_candidates,
        "venous_objective": {
            "formula": (
                "J_vein = 1.0*J_drainage + 0.25*J_balance + "
                "0.10*J_separation + 0.02*J_junction_preference"
            ),
            "weights": VENOUS_OBJECTIVE_WEIGHTS,
            "J_drainage": (
                "area-weighted nearest-candidate periodic graph distance, "
                "normalized by sqrt(periodic cell area)"
            ),
            "J_balance": "squared CV of drainage catchment areas",
            "J_separation": (
                "mean squared hinge penalty below 0.20*sqrt(periodic cell area)"
            ),
            "J_junction_preference": (
                "fraction of selected sites that are two-acinus ridge candidates"
            ),
            "number_of_sites_penalty": None,
        },
        "fixed_number_solutions": solutions,
        "example_configuration_number_of_sinks": 2,
        "figures": {
            "individual_configurations": configuration_figures,
            "detailed_example": str(detailed_figure),
            "configuration_comparison": str(comparison_figure),
            "diminishing_returns": str(diminishing_returns_figure),
            "periodic_3x3_example": str(tiled_figure),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=json_ready) + "\n")

    compact_solutions = [
        {
            key: value
            for key, value in solution.items()
            if key not in {"selected_sinks"}
        }
        | {
            "selected_sinks": [
                {
                    key: value
                    for key, value in sink.items()
                    if key != "assigned_canonical_cell_ids"
                }
                for sink in solution["selected_sinks"]
            ]
        }
        for solution in solutions
    ]
    compact_report = {
        "arterial_sources": arterial_sources,
        "venous_candidate_count": len(venous_candidates),
        "venous_candidate_type_counts": candidate_type_counts,
        "solutions": compact_solutions,
        "metadata": str(metadata_path),
        "figures": metadata["figures"],
    }
    print(json.dumps(compact_report, indent=2, default=json_ready))


if __name__ == "__main__":
    main()
