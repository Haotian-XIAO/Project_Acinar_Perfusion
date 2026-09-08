#!/usr/bin/env python3
"""PyVista postprocessing for the retained healthy vascular baseline solution."""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "vascular_baseline"
VTU = RESULTS / "final_fields.vtu"
MARKERS = ROOT / "mesh" / "Mesh_Acinar_Perfusion_VascularMarkers_vascular_markers.json"


def read_solution_grid(path):
    source = meshio.read(path)
    triangles = np.asarray(source.get_cells_type("triangle"), dtype=int)
    cells = np.column_stack([np.full(len(triangles), 3, dtype=int), triangles]).ravel()
    cell_types = np.full(len(triangles), int(pv.CellType.TRIANGLE), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, np.asarray(source.points, dtype=float))
    for name, values_by_block in source.cell_data.items():
        grid.cell_data[name] = np.asarray(values_by_block[0])
    return grid


def patch_mesh(grid, tag):
    return grid.threshold([tag - 0.1, tag + 0.1], scalars="domain_id")


def add_patch_overlays(plotter, grid, markers, labels=None):
    label_points = []
    label_text = []
    for vascular_id, marker in markers.items():
        color = "#d62728" if marker["vascular_type"] == "arterial" else "#1f77b4"
        plotter.add_mesh(
            patch_mesh(grid, int(marker["physical_tag"])),
            color=color, opacity=0.90, show_edges=True, line_width=1.0,
        )
        point = [*marker["coordinate"], 0.002]
        label_points.append(point)
        label_text.append(labels.get(vascular_id, vascular_id) if labels else vascular_id)
    plotter.add_point_labels(
        np.asarray(label_points), label_text, font_size=12, point_size=7,
        text_color="black", shape_color="white", shape_opacity=0.75,
        always_visible=True,
    )


def setup_plotter(title):
    plotter = pv.Plotter(off_screen=True, window_size=(1500, 1250))
    plotter.set_background("white")
    plotter.add_text(title, font_size=13, color="black")
    return plotter


def finish(plotter, path):
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.screenshot(path)
    plotter.close()


def pressure_figure(grid, markers):
    plotter = setup_plotter("Total liquid pressure p_l")
    plotter.add_mesh(grid, scalars="p_l", cmap="coolwarm", show_edges=False,
                     scalar_bar_args={"title": "p_l"})
    add_patch_overlays(plotter, grid, markers)
    finish(plotter, RESULTS / "pressure_field.png")


def flux_figure(grid, markers):
    plotter = setup_plotter("Darcy flux magnitude |q_l|")
    plotter.add_mesh(grid, scalars="q_l_magnitude", cmap="viridis", show_edges=False,
                     scalar_bar_args={"title": "|q_l|"})
    add_patch_overlays(plotter, grid, markers)
    finish(plotter, RESULTS / "flux_magnitude.png")

    positive = np.asarray(grid.cell_data["q_l_magnitude"], dtype=float)
    safe_floor = max(float(positive.max()) * 1e-12, np.finfo(float).tiny)
    grid_log = grid.copy()
    grid_log.cell_data["log10_q_l_magnitude"] = np.log10(np.maximum(positive, safe_floor))
    plotter = setup_plotter("Darcy flux magnitude log10(|q_l|)")
    plotter.add_mesh(grid_log, scalars="log10_q_l_magnitude", cmap="viridis",
                     show_edges=False, scalar_bar_args={"title": "log10(|q_l|)"})
    add_patch_overlays(plotter, grid_log, markers)
    finish(plotter, RESULTS / "flux_magnitude_log.png")


def pressure_flux_overlay(grid, markers):
    plotter = setup_plotter("Liquid pressure with subsampled Darcy flux vectors")
    plotter.add_mesh(grid, scalars="p_l", cmap="coolwarm", show_edges=False,
                     scalar_bar_args={"title": "p_l"})
    centers = grid.cell_centers()
    stride = max(1, centers.n_points // 350)
    indices = np.arange(0, centers.n_points, stride)
    selected = pv.PolyData(np.asarray(centers.points)[indices])
    selected.point_data["q_l"] = np.asarray(centers.point_data["q_l"])[indices]
    selected.point_data["q_l_magnitude"] = np.asarray(
        centers.point_data["q_l_magnitude"]
    )[indices]
    glyphs = selected.glyph(orient="q_l", scale=False, factor=0.018)
    plotter.add_mesh(glyphs, color="#17202a")
    add_patch_overlays(plotter, grid, markers)
    finish(plotter, RESULTS / "pressure_flux_overlay.png")


def source_sink_map(grid, markers):
    plotter = setup_plotter("Finite vascular source/sink subdomains")
    ordinary = grid.threshold([1.9, 2.1], scalars="domain_id")
    plotter.add_mesh(ordinary, color="#bdbdbd", opacity=0.75, show_edges=False)
    labels = {key: f"{key}: {'+' if key.startswith('Pa') else '-'}{0.25 if key.startswith('Pa') else 0.50:.2f}"
              for key in markers}
    add_patch_overlays(plotter, grid, markers, labels=labels)
    finish(plotter, RESULTS / "vascular_source_sink_map.png")


def pressure_with_values(grid, markers, pressure_report):
    plotter = setup_plotter("Liquid pressure with patch-averaged values")
    plotter.add_mesh(grid, scalars="p_l", cmap="coolwarm", show_edges=False,
                     scalar_bar_args={"title": "p_l"})
    values = pressure_report["patch_area_averaged_pressure"]
    labels = {key: f"{key}: {values[key]:.4g}" for key in markers}
    add_patch_overlays(plotter, grid, markers, labels=labels)
    finish(plotter, RESULTS / "pressure_with_patch_values.png")


def numerical_sanity(grid, markers, pressure_report):
    centers = np.asarray(grid.cell_centers().points[:, :2])
    q_l = np.asarray(grid.cell_data["q_l"])[:, :2]
    magnitudes = np.asarray(grid.cell_data["q_l_magnitude"])
    displacement_magnitude = np.linalg.norm(
        np.asarray(grid.cell_data["displacement"]), axis=1
    )
    venous_points = np.asarray([
        marker["coordinate"] for marker in markers.values()
        if marker["vascular_type"] == "venous"
    ])
    deltas = venous_points[None, :, :] - centers[:, None, :]
    deltas -= np.round(deltas)
    nearest = deltas[np.arange(len(centers)), np.argmin(np.linalg.norm(deltas, axis=2), axis=1)]
    directional = np.einsum("ij,ij->i", q_l, nearest)
    moving = magnitudes > max(float(magnitudes.max()) * 1e-10, np.finfo(float).eps)
    directional_fraction = float(np.mean(directional[moving] > 0.0)) if np.any(moving) else None
    p = pressure_report["patch_area_averaged_pressure"]
    arterial_mean = float(np.mean([p[key] for key in p if key.startswith("Pa")]))
    venous_mean = float(np.mean([p[key] for key in p if key.startswith("V")]))
    p99 = float(np.percentile(magnitudes, 99.0))
    report = {
        "arterial_patch_pressure_mean": arterial_mean,
        "venous_patch_pressure_mean": venous_mean,
        "arterial_mean_exceeds_venous_mean": arterial_mean > venous_mean,
        "fraction_of_moving_cells_with_flux_component_toward_nearest_venous_site": directional_fraction,
        "max_flux_to_99th_percentile_ratio": float(magnitudes.max() / p99) if p99 > 0 else None,
        "displacement_magnitude": {
            "min": float(displacement_magnitude.min()),
            "max": float(displacement_magnitude.max()),
            "mean": float(displacement_magnitude.mean()),
        },
        "interpretation_note": "Nearest-site alignment is only a qualitative geometric diagnostic; the wall-network path need not point directly at the nearest venous coordinate.",
    }
    (RESULTS / "postprocess_sanity.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    if not VTU.exists():
        raise FileNotFoundError(f"Run the vascular baseline solve first: missing {VTU}")
    marker_document = json.loads(MARKERS.read_text())
    markers = marker_document["markers"]
    pressure_report = json.loads((RESULTS / "patch_pressures.json").read_text())
    grid = read_solution_grid(VTU)
    pressure_figure(grid, markers)
    flux_figure(grid, markers)
    pressure_flux_overlay(grid, markers)
    source_sink_map(grid, markers)
    pressure_with_values(grid, markers, pressure_report)
    report = numerical_sanity(grid, markers, pressure_report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    pv.global_theme.allow_empty_mesh = True
    main()
