#!/usr/bin/env python3
"""Compare the retained healthy result with complete compensated Pa2 suppression."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista as pv

import postprocess_vascular_baseline as vascular_figures


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "vascular_baseline"
SUPPRESSION = ROOT / "results" / "vascular_suppression_Pa2_000"
MARKERS = ROOT / "mesh" / "Mesh_Acinar_Perfusion_VascularMarkers_vascular_markers.json"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def compare_metrics():
    baseline = load_json(BASELINE / "regional_perfusion.json")
    suppression = load_json(SUPPRESSION / "regional_perfusion.json")
    baseline_pressure = load_json(BASELINE / "patch_pressures.json")
    suppression_pressure = load_json(SUPPRESSION / "patch_pressures.json")
    baseline_by_id = {int(row["acinus_id"]): row for row in baseline["acini"]}
    suppression_by_id = {int(row["acinus_id"]): row for row in suppression["acini"]}

    rows = []
    for acinus_id in range(4):
        healthy = baseline_by_id[acinus_id]
        suppressed = suppression_by_id[acinus_id]
        ratio = suppressed["Pi_i"] / healthy["Pi_i"]
        rows.append({
            "acinus_id": acinus_id,
            "Pi_baseline": healthy["Pi_i"],
            "Pi_suppressed": suppressed["Pi_i"],
            "Pi_ratio": ratio,
            "Pi_percent_change": 100.0 * (ratio - 1.0),
            "fP_baseline": healthy["fP_i"],
            "fP_suppressed": suppressed["fP_i"],
            "fP_change": suppressed["fP_i"] - healthy["fP_i"],
        })

    report = {
        "case": "complete Pa2 suppression with imposed equal compensation at Pa0, Pa1, Pa3",
        "acini": rows,
        "CV_Pi": {
            "baseline": baseline["global"]["CV_Pi"],
            "suppressed": suppression["global"]["CV_Pi"],
            "change": suppression["global"]["CV_Pi"] - baseline["global"]["CV_Pi"],
        },
        "Delta_p_av": {
            "baseline": baseline_pressure["Delta_p_av"],
            "suppressed": suppression_pressure["Delta_p_av"],
            "ratio": suppression_pressure["Delta_p_av"] / baseline_pressure["Delta_p_av"],
        },
        "R_hyd": {
            "baseline": baseline_pressure["R_hyd"],
            "suppressed": suppression_pressure["R_hyd"],
            "ratio": suppression_pressure["R_hyd"] / baseline_pressure["R_hyd"],
        },
    }
    (SUPPRESSION / "baseline_comparison.json").write_text(json.dumps(report, indent=2))
    with (SUPPRESSION / "baseline_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return report


def metric_figures(report):
    rows = report["acini"]
    labels = [f"A{row['acinus_id']}" for row in rows]
    positions = np.arange(4)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    width = 0.36
    ax.bar(positions - width / 2, [row["Pi_baseline"] for row in rows], width,
           label="Healthy", color="#4c78a8")
    ax.bar(positions + width / 2, [row["Pi_suppressed"] for row in rows], width,
           label="Pa2 = 0, compensated", color="#f58518")
    ax.set_xticks(positions, labels)
    ax.set_ylabel(r"Regional perfusion intensity $\Pi_i$")
    ax.set_title("Healthy baseline versus complete Pa2 suppression")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SUPPRESSION / "baseline_vs_suppression_Pi.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ratios = [row["Pi_ratio"] for row in rows]
    bars = ax.bar(labels, ratios, color="#f58518")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
               label="healthy reference")
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio, f"{ratio:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"$\Pi_i^{suppressed}/\Pi_i^{healthy}$")
    ax.set_title("Relative regional response to complete Pa2 suppression")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SUPPRESSION / "relative_regional_change.png", dpi=250)
    plt.close(fig)


def read_cell_field(path, field):
    mesh = meshio.read(path)
    return (
        np.asarray(mesh.points[:, :2], dtype=float),
        np.asarray(mesh.get_cells_type("triangle"), dtype=int),
        np.asarray(mesh.cell_data_dict[field]["triangle"], dtype=float),
    )


def comparable_flux_figure():
    data = [
        ("Healthy", *read_cell_field(BASELINE / "final_fields.vtu", "q_l_magnitude")),
        ("Pa2 = 0, compensated", *read_cell_field(SUPPRESSION / "final_fields.vtu", "q_l_magnitude")),
    ]
    vmin = min(float(values.min()) for _, _, _, values in data)
    vmax = max(float(values.max()) for _, _, _, values in data)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), constrained_layout=True)
    image = None
    for ax, (title, points, triangles, values) in zip(axes, data):
        image = ax.tripcolor(points[:, 0], points[:, 1], triangles,
                             facecolors=values, shading="flat", cmap="viridis",
                             vmin=vmin, vmax=vmax)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="x", ylabel="y", title=title)
    fig.colorbar(image, ax=axes, label=r"$|q_l|$", shrink=0.88)
    fig.suptitle("Darcy flux magnitude on a shared scalar range")
    fig.savefig(SUPPRESSION / "baseline_vs_suppression_flux.png", dpi=250)
    plt.close(fig)


def vascular_field_figures():
    marker_document = load_json(MARKERS)
    grid = vascular_figures.read_solution_grid(SUPPRESSION / "final_fields.vtu")
    vascular_figures.RESULTS = SUPPRESSION
    vascular_figures.pressure_figure(grid, marker_document["markers"])
    vascular_figures.flux_figure(grid, marker_document["markers"])
    vascular_figures.pressure_flux_overlay(grid, marker_document["markers"])


def main():
    pv.global_theme.allow_empty_mesh = True
    report = compare_metrics()
    metric_figures(report)
    comparable_flux_figure()
    vascular_field_figures()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
