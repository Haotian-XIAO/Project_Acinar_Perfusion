#!/usr/bin/env python3
"""Generate and validate finite FE vascular source/sink subdomains.

This script performs mesh/marker diagnostics only.  It never constructs a
Darcy problem or evaluates a pressure/flux field.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Mesh_Acinar_RVE imports dolfin/dolfin_mech for its legacy entry points, while
# the periodic generator used here requires only Gmsh/meshio.  Stubs keep this
# validation utility independent of an MPI-enabled FEniCS runtime.
sys.modules.setdefault("dolfin", types.ModuleType("dolfin"))
sys.modules.setdefault("dolfin_mech", types.ModuleType("dolfin_mech"))
import Mesh_Acinar_RVE as mesh_generator
import partition_periodic_acini as partition


METADATA = ROOT / "results/vascular_placement/vascular_placement_metadata.json"
ACINAR = ROOT / "results/acinar_partition/periodic_acinar_metadata.json"
MESH_BASE = ROOT / "mesh/Mesh_Acinar_Perfusion_VascularMarkers"
OUT = ROOT / "results/vascular_placement"


def triangle_geometry(mesh):
    points = np.asarray(mesh.points[:, :2], dtype=float)
    triangles = np.asarray(mesh.get_cells_type("triangle"), dtype=int)
    physical = np.asarray(mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=int)
    vertices = points[triangles]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    areas = np.abs(cross) / 2.0
    centroids = vertices.mean(axis=1)
    return points, triangles, physical, vertices, areas, centroids


def patch_metrics(ids, areas, centroids, marker_mapping, patch_radius, bounds):
    xmin, ymin, xmax, ymax = bounds
    metrics = {}
    for vascular_id, marker in marker_mapping.items():
        mask = ids == int(marker["physical_tag"])
        if not np.any(mask):
            raise RuntimeError(f"Marker {vascular_id} has no FE cells.")
        area = float(areas[mask].sum())
        cell_count = int(mask.sum())
        if cell_count < 3:
            raise RuntimeError(f"{vascular_id} patch has too few FE cells ({cell_count}).")
        centroid = np.average(centroids[mask], axis=0, weights=areas[mask])
        target = np.asarray(marker["coordinate"], dtype=float)
        offset = float(np.linalg.norm(centroid - target))
        image_centers = mesh_generator._periodic_patch_centers(
            target, patch_radius, xmin, ymin, xmax, ymax
        )
        distances = np.min(
            np.linalg.norm(centroids[mask, None, :] - np.asarray(image_centers)[None, :, :], axis=2),
            axis=1,
        )
        if float(np.max(distances)) > patch_radius + 5e-6:
            raise RuntimeError(f"{vascular_id} contains a cell centroid outside its patch disk.")
        disk_area_fraction = area / (np.pi * patch_radius**2)
        if not 0.5 <= disk_area_fraction <= 1.05:
            raise RuntimeError(
                f"{vascular_id} patch has unsuitable tissue fraction {disk_area_fraction:.3f}."
            )
        if offset > 0.5 * patch_radius:
            raise RuntimeError(f"{vascular_id} patch centroid is too far from metadata point.")
        metrics[vascular_id] = {
            "physical_tag": int(marker["physical_tag"]),
            "vascular_type": marker["vascular_type"],
            "cell_count": cell_count,
            "assembled_reference_area": area,
            "centroid": centroid.tolist(),
            "metadata_coordinate": target.tolist(),
            "metadata_centroid_offset": offset,
            "disk_area_fraction": disk_area_fraction,
        }
    return metrics


def validate_overlap(marker_mapping, patch_radius, bounds):
    sites = list(marker_mapping.values())
    ids = list(marker_mapping)
    xmin, ymin, xmax, ymax = bounds
    for i in range(len(sites)):
        for j in range(i):
            p = np.asarray(sites[i]["coordinate"])
            q = np.asarray(sites[j]["coordinate"])
            images = mesh_generator._periodic_patch_centers(q, patch_radius, xmin, ymin, xmax, ymax)
            if min(np.linalg.norm(p - image) for image in images) < 2.0 * patch_radius:
                raise RuntimeError(f"Vascular disks overlap: {ids[j]} and {ids[i]}.")


def draw_airway(ax, airway, bounds, tile_range=(-1, 0, 1), annotate=False):
    xmin, ymin, xmax, ymax = bounds
    a1 = np.array([xmax - xmin, 0.0])
    a2 = np.array([0.0, ymax - ymin])
    nodes = {node["node_id"]: node for node in airway["nodes"]}
    for i in tile_range:
        for j in tile_range:
            shift = i * a1 + j * a2
            for edge in airway["edges"]:
                p0 = np.asarray(nodes[edge["parent_node_id"]]["coordinate"])
                p1 = np.asarray(nodes[edge["child_node_id"]]["coordinate"])
                delta = p1 - p0
                delta -= np.round(delta / np.array([xmax - xmin, ymax - ymin])) * np.array(
                    [xmax - xmin, ymax - ymin]
                )
                p0 = p0 + shift
                p1 = p0 + delta
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#7b1fa2", lw=2.6, zorder=8)
            for node in airway["nodes"]:
                p = np.asarray(node["coordinate"]) + shift
                kind = node["node_type"]
                if kind == "root":
                    style = dict(marker="D", s=70, c="#e53935", edgecolors="white")
                elif kind == "branch":
                    style = dict(marker="o", s=48, c="#7b1fa2", edgecolors="white")
                else:
                    style = dict(marker="*", s=125, c="#ffd54f", edgecolors="black")
                ax.scatter(*p, linewidths=0.8, zorder=9, **style)
                if annotate and (i, j) == (0, 0):
                    label = "root" if kind == "root" else node["node_id"]
                    ax.annotate(label, p, xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")


def add_vascular_overlay(ax, marker_mapping, metrics, bounds, patch_radius, tiled=False):
    xmin, ymin, xmax, ymax = bounds
    shifts = [(0, 0)] if not tiled else [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    colors = {"arterial": "#d62728", "venous": "#1f77b4"}
    for i, j in shifts:
        shift = np.array([i * (xmax - xmin), j * (ymax - ymin)])
        for vascular_id, marker in marker_mapping.items():
            point = np.asarray(marker["coordinate"]) + shift
            color = colors[marker["vascular_type"]]
            ax.add_patch(Circle(point, patch_radius, facecolor=color, edgecolor=color, alpha=0.24, lw=1.0, zorder=5))
            ax.scatter(*point, marker="^" if marker["vascular_type"] == "arterial" else "v", s=52,
                       c=color, edgecolors="black", linewidths=0.6, zorder=10)
            if not tiled or (i, j) == (0, 0):
                ax.annotate(vascular_id, point, xytext=(4, -10), textcoords="offset points", fontsize=7, zorder=11)
    if not tiled:
        for vascular_id, item in metrics.items():
            point = np.asarray(item["centroid"])
            ax.scatter(*point, marker="x", s=38, c="black", linewidths=1.1, zorder=11)


def central_figure(mesh, marker_mapping, metrics, acinar_metadata, airway, patch_radius, path):
    _, _, ids, vertices, _, _ = triangle_geometry(mesh)
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    cmap = plt.get_cmap("tab10")
    labels = np.asarray([cell["acinus_id"] for cell in acinar_metadata["cells"]], dtype=int)
    seeds = partition.generate_rectangular_base_seeds((0, 0, 1, 1), 12, 12, 0.3, 1)
    graph = partition.build_periodic_cell_graph(seeds, np.array([1.0, 0.0]), np.array([0.0, 1.0]), (0, 0, 1, 1))
    for polygon, label in zip(graph["polygons"], labels):
        ax.add_patch(Polygon(polygon, facecolor=cmap(int(label)), alpha=0.23, edgecolor="#777777", lw=0.35, zorder=1))
    for triangle, marker_id in zip(vertices, ids):
        color = "#c7c7c7" if marker_id == 2 else {int(v["physical_tag"]): ("#d62728" if v["vascular_type"] == "arterial" else "#1f77b4") for v in marker_mapping.values()}.get(int(marker_id), "#c7c7c7")
        ax.add_patch(Polygon(triangle, facecolor=color, edgecolor="none", alpha=0.65 if marker_id != 2 else 0.18, zorder=2))
    draw_airway(ax, airway, (0, 0, 1, 1), annotate=True)
    add_vascular_overlay(ax, marker_mapping, metrics, (0, 0, 1, 1), patch_radius)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title("Template A periodic acini with FE vascular source/sink patches")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    handles = [
        Patch(facecolor="#c7c7c7", alpha=0.6, label="ordinary wall/tissue"),
        Line2D([0], [0], color="#7b1fa2", lw=2.6, label="airway tree (metadata only)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#d62728", markeredgecolor="black", label="arterial FE patch"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#1f77b4", markeredgecolor="black", label="venous FE patch"),
        Line2D([0], [0], marker="x", color="black", label="area centroid"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.92)
    fig.tight_layout(); fig.savefig(path, dpi=240); plt.close(fig)


def marker_figure(mesh, marker_mapping, path):
    _, _, ids, vertices, _, _ = triangle_geometry(mesh)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {2: "#bdbdbd"}
    for item in marker_mapping.values():
        colors[int(item["physical_tag"])] = "#d62728" if item["vascular_type"] == "arterial" else "#1f77b4"
    for triangle, marker_id in zip(vertices, ids):
        ax.add_patch(Polygon(triangle, facecolor=colors.get(int(marker_id), "#eeeeee"), edgecolor="none", alpha=0.92))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title("Finite-element domain markers")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    handles = [Patch(facecolor="#bdbdbd", label="ordinary wall")]
    handles += [Patch(facecolor=("#d62728" if v["vascular_type"] == "arterial" else "#1f77b4"), label=k) for k, v in marker_mapping.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout(); fig.savefig(path, dpi=240); plt.close(fig)


def tiled_figure(marker_mapping, metrics, airway, patch_radius, path):
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    xmin, ymin, xmax, ymax = 0, 0, 1, 1
    acinar = json.loads(ACINAR.read_text())
    labels = np.asarray([cell["acinus_id"] for cell in acinar["cells"]], dtype=int)
    seeds = partition.generate_rectangular_base_seeds((0, 0, 1, 1), 12, 12, 0.3, 1)
    graph = partition.build_periodic_cell_graph(seeds, np.array([1.0, 0.0]), np.array([0.0, 1.0]), (0, 0, 1, 1))
    cmap = plt.get_cmap("tab10")
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shift = np.array([i, j])
            for polygon, label in zip(graph["polygons"], labels):
                ax.add_patch(Polygon(np.asarray(polygon) + shift, facecolor=cmap(int(label)), alpha=0.25, edgecolor="#777777", lw=0.25))
    draw_airway(ax, airway, (0, 0, 1, 1), tile_range=(-1, 0, 1), annotate=False)
    add_vascular_overlay(ax, marker_mapping, metrics, (0, 0, 1, 1), patch_radius, tiled=True)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor="black", lw=0.55, ls="--"))
    ax.set_xlim(-1, 2); ax.set_ylim(-1, 2); ax.set_aspect("equal")
    ax.set_title("3×3 periodic continuation of acini, airway, and vascular metadata")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(handles=[Line2D([0], [0], color="#7b1fa2", lw=2.4, label="airway metadata"),
                       Line2D([0], [0], marker="^", color="w", markerfacecolor="#d62728", markeredgecolor="black", label="arterial patch"),
                       Line2D([0], [0], marker="v", color="w", markerfacecolor="#1f77b4", markeredgecolor="black", label="venous patch")],
              loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=240); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mesh_generator.run_PeriodicVoronoi_Mesh({
        "grid_x": 12, "grid_y": 12, "DoI": 0.3, "rng_seed": 1,
        "voronoi_offset": 0.013, "l": 0.01,
        "vascular_metadata_json": str(METADATA),
        "number_of_venous_sinks": 2, "patch_radius": 0.003,
        "mesh_filebasename": str(MESH_BASE), "gmsh_verbose": False,
    })
    mesh = meshio.read(str(MESH_BASE) + ".msh")
    _, _, physical, vertices, areas, centroids = triangle_geometry(mesh)
    mapping = json.loads((Path(str(MESH_BASE) + "_vascular_markers.json")).read_text())["markers"]
    metrics = patch_metrics(physical, areas, centroids, mapping, 0.003, (0, 0, 1, 1))
    patch_areas = [item["assembled_reference_area"] for item in metrics.values()]
    if max(patch_areas) / min(patch_areas) > 2.0:
        raise RuntimeError("Vascular patch areas are severely disproportionate.")
    validate_overlap(mapping, 0.003, (0, 0, 1, 1))
    ordinary_mask = physical == 2
    ordinary_area = float(areas[ordinary_mask].sum())
    arterial_ids = [key for key, value in mapping.items() if value["vascular_type"] == "arterial"]
    venous_ids = [key for key, value in mapping.items() if value["vascular_type"] == "venous"]
    q_targets = {key: 0.25 for key in arterial_ids} | {key: 0.5 for key in venous_ids}
    theta = {key: q_targets[key] / metrics[key]["assembled_reference_area"] for key in mapping}
    q_in = sum(theta[key] * metrics[key]["assembled_reference_area"] for key in arterial_ids)
    q_out = sum(theta[key] * metrics[key]["assembled_reference_area"] for key in venous_ids)
    mass_residual = q_in - q_out
    acinar_metadata = json.loads(ACINAR.read_text())
    airway = acinar_metadata["airway"]
    central_figure(mesh, mapping, metrics, acinar_metadata, airway, 0.003, OUT / "finite_vascular_acinar_central.png")
    marker_figure(mesh, mapping, OUT / "finite_vascular_mesh_markers.png")
    tiled_figure(mapping, metrics, airway, 0.003, OUT / "finite_vascular_acinar_3x3.png")
    report = {
        "mesh_base": str(MESH_BASE),
        "patch_radius": 0.003,
        "ordinary_wall_cell_count": int(ordinary_mask.sum()),
        "ordinary_wall_area": ordinary_area,
        "patches": metrics,
        "theta_for_Q_total_1": theta,
        "target_patch_flows": q_targets,
        "Q_in": q_in,
        "Q_out": q_out,
        "mass_balance_residual": mass_residual,
        "mass_balance_tolerance": 1e-12,
        "macro_pressure_gradient_for_future_run": [0.0, 0.0],
        "figures": {
            "central": str(OUT / "finite_vascular_acinar_central.png"),
            "markers": str(OUT / "finite_vascular_mesh_markers.png"),
            "periodic_3x3": str(OUT / "finite_vascular_acinar_3x3.png"),
        },
    }
    (OUT / "finite_marker_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
