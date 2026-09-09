#!/usr/bin/env python3
"""Postprocess retained healthy-flow deformation cases and create PyVista figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "results" / "deformation_healthy_sweep"
ACINAR_SUMMARY = ROOT / "results" / "acinar_partition" / "periodic_acinar_summary.json"
CASES = {
    0.00: SWEEP_ROOT / "stretch_000",
    0.05: SWEEP_ROOT / "stretch_005",
    0.10: SWEEP_ROOT / "stretch_010",
    0.15: SWEEP_ROOT / "stretch_015",
    0.20: SWEEP_ROOT / "stretch_020",
}


def solution_path(stretch, case_dir):
    basename = "vascular_baseline" if stretch == 0.0 else case_dir.name
    return case_dir / f"{basename}.xdmf"


def parse_hdf_reference(text, xdmf_path):
    filename, dataset = text.strip().split(":", 1)
    return xdmf_path.parent / filename, dataset


def read_final_fields(xdmf_path):
    root = ET.parse(xdmf_path).getroot()
    grids = root.findall('.//Grid[@GridType="Uniform"]')
    grid = max(grids, key=lambda item: float(item.find("Time").attrib["Value"]))

    def read(item):
        h5_path, dataset = parse_hdf_reference(item.text, xdmf_path)
        with h5py.File(h5_path, "r") as stream:
            return np.asarray(stream[dataset])

    coordinates = read(grid.find("Geometry/DataItem"))[:, :2]
    topology = read(grid.find("Topology/DataItem")).astype(int)
    wanted = {"U_tot", "J_tot", "pl_tot", "q_l", "Phif", "Phis"}
    fields = {}
    for attribute in grid.findall("Attribute"):
        if attribute.attrib.get("Center") != "Node":
            continue
        name = attribute.attrib["Name"]
        if name in wanted:
            fields[name] = read(attribute.find("DataItem"))
    missing = {"U_tot", "J_tot", "pl_tot", "q_l", "Phif"} - set(fields)
    if missing:
        raise RuntimeError(f"Missing retained nodal fields in {xdmf_path}: {sorted(missing)}")
    fields["U_tot"] = fields["U_tot"][:, :2]
    fields["q_l"] = fields["q_l"][:, :2]
    fields["J_tot"] = fields["J_tot"].reshape(-1)
    fields["pl_tot"] = fields["pl_tot"].reshape(-1)
    fields["Phif"] = fields["Phif"].reshape(-1)
    if "Phis" in fields:
        fields["Phis"] = fields["Phis"].reshape(-1)
    return coordinates, topology, fields


def make_grid(coordinates, topology, point_data):
    points = np.column_stack([coordinates, np.zeros(len(coordinates))])
    cells = np.column_stack([np.full(len(topology), 3, dtype=np.int64), topology]).ravel()
    cell_types = np.full(len(topology), int(pv.CellType.TRIANGLE), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    for name, values in point_data.items():
        if np.asarray(values).ndim == 2 and np.asarray(values).shape[1] == 2:
            values = np.column_stack([values, np.zeros(len(values))])
        grid.point_data[name] = np.asarray(values)
    return grid


def interface_polydata(topology, coordinates, acinus_ids):
    owners = {}
    for triangle_id, triangle in enumerate(topology):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(triangle_id)
    edges = [edge for edge, cells in owners.items()
             if len(cells) == 2 and acinus_ids[cells[0]] != acinus_ids[cells[1]]]
    if not edges:
        return None
    points = np.column_stack([coordinates, np.full(len(coordinates), 0.001)])
    lines = np.asarray([[2, a, b] for a, b in edges], dtype=np.int64).ravel()
    return pv.PolyData(points, lines=lines)


def setup(title):
    plotter = pv.Plotter(off_screen=True, window_size=(1500, 1200))
    plotter.set_background("white")
    plotter.add_text(title, color="black", font_size=13)
    return plotter


def finish(plotter, path):
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.screenshot(path)
    plotter.close()


def scalar_figure(grid, scalar, title, scalar_title, path, cmap="viridis"):
    plotter = setup(title)
    plotter.add_mesh(grid, scalars=scalar, cmap=cmap, show_edges=False,
                     scalar_bar_args={"title": scalar_title})
    finish(plotter, path)


def make_case_figures(stretch, case_dir, coordinates, topology, fields, acinus_ids):
    deformed = coordinates + fields["U_tot"]
    displacement_magnitude = np.linalg.norm(fields["U_tot"], axis=1)
    flux_magnitude = np.linalg.norm(fields["q_l"], axis=1)
    data = dict(fields)
    data["displacement_magnitude"] = displacement_magnitude
    data["q_l_magnitude"] = flux_magnitude
    grid = make_grid(deformed, topology, data)

    scalar_figure(grid, "displacement_magnitude",
                  f"Deformed wall network, s={stretch:.2f}", "|u|",
                  case_dir / "deformed_shape_displacement.png", cmap="magma")

    reference_grid = make_grid(coordinates, topology, {})
    reference_edges = reference_grid.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False,
        non_manifold_edges=False,
    )
    deformed_edges = grid.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False,
        non_manifold_edges=False,
    )
    plotter = setup(f"Reference/deformed geometry overlay, s={stretch:.2f}")
    plotter.add_mesh(reference_edges, color="#777777", line_width=2,
                     opacity=0.65, label="reference")
    plotter.add_mesh(deformed_edges, color="#d62728", line_width=2,
                     opacity=0.85, label="deformed")
    plotter.add_legend(face="rectangle", bcolor="white")
    finish(plotter, case_dir / "deformation_overlay.png")

    scalar_figure(grid, "J_tot", f"Jacobian field, s={stretch:.2f}", "J",
                  case_dir / "J_field.png", cmap="coolwarm")
    scalar_figure(grid, "pl_tot", f"Liquid pressure on deformed geometry, s={stretch:.2f}",
                  "p_l", case_dir / "pressure_field.png", cmap="coolwarm")
    scalar_figure(grid, "q_l_magnitude", f"Spatial Darcy flux on deformed geometry, s={stretch:.2f}",
                  "|q_l|", case_dir / "flux_magnitude.png")
    scalar_figure(grid, "Phif", f"Fluid porosity on deformed geometry, s={stretch:.2f}",
                  "Phif", case_dir / "porosity_field.png", cmap="cividis")

    plotter = setup(f"Regional perfusion on deformed geometry, s={stretch:.2f}")
    plotter.add_mesh(grid, scalars="q_l_magnitude", cmap="viridis", show_edges=False,
                     scalar_bar_args={"title": "|q_l|"})
    interfaces = interface_polydata(topology, deformed, acinus_ids)
    if interfaces is not None:
        plotter.add_mesh(interfaces, color="black", line_width=2.5)
    centers = deformed[topology].mean(axis=1)
    label_points, labels = [], []
    for acinus_id in range(4):
        center = centers[acinus_ids == acinus_id].mean(axis=0)
        label_points.append([center[0], center[1], 0.003])
        labels.append(f"A{acinus_id}")
    plotter.add_point_labels(np.asarray(label_points), labels, font_size=15,
                             text_color="black", shape_color="white",
                             shape_opacity=0.8, always_visible=True)
    finish(plotter, case_dir / "regional_perfusion_map.png")


def load_case(stretch, case_dir):
    coordinates, topology, fields = read_final_fields(solution_path(stretch, case_dir))
    mapping = json.loads((case_dir / "fe_acinar_mapping.json").read_text())
    acinus_ids = np.asarray(mapping["acinus_id_by_fe_cell"], dtype=int)
    regional = json.loads((case_dir / "regional_perfusion.json").read_text())
    diagnostics = json.loads((case_dir / "diagnostics.json").read_text())
    pressures = json.loads((case_dir / "patch_pressures.json").read_text())
    balance = json.loads((case_dir / "mass_balance.json").read_text())
    by_id = {int(item["acinus_id"]): item for item in regional["acini"]}
    u_magnitude = np.linalg.norm(fields["U_tot"], axis=1)
    row = {
        "deformation_s": stretch,
        "lambda_x": 1.0 + stretch,
        "nonlinear_converged": diagnostics["nonlinear_converged"],
        "time_load_increments": diagnostics["time_load_increments"],
        "total_nonlinear_iterations": diagnostics["total_nonlinear_iterations"],
        "global_mean_abs_q_l": regional["global"]["global_current_area_weighted_mean_abs_q"],
        "Pi0": by_id[0]["Pi_i"], "Pi1": by_id[1]["Pi_i"],
        "Pi2": by_id[2]["Pi_i"], "Pi3": by_id[3]["Pi_i"],
        "fP0": by_id[0]["fP_i"], "fP1": by_id[1]["fP_i"],
        "fP2": by_id[2]["fP_i"], "fP3": by_id[3]["fP_i"],
        "CV_Pi": regional["global"]["CV_Pi"],
        "pa_eff": pressures["pa_eff"], "pv_eff": pressures["pv_eff"],
        "Delta_p_av": pressures["Delta_p_av"], "R_hyd": pressures["R_hyd"],
        "max_displacement": float(u_magnitude.max()),
        "mean_displacement": float(u_magnitude.mean()),
        "min_J": float(fields["J_tot"].min()),
        "max_J": float(fields["J_tot"].max()),
        "mean_J": float(fields["J_tot"].mean()),
        "Q_in": balance["Q_in"], "Q_out": balance["Q_out"],
        "mass_balance_residual": balance["mass_balance_residual"],
        "ptilde_l_volume_average": diagnostics["ptilde_l_volume_average"],
        "lambda_p": diagnostics["lambda_p"],
        "regional_max_consistency_residual": max(
            abs(value) for value in regional["global"]["residuals"].values()
        ),
        "case_directory": str(case_dir.resolve()),
    }
    return row, coordinates, topology, fields, acinus_ids


def write_summary(rows):
    reference = rows[0]
    for row in rows:
        for acinus_id in range(4):
            row[f"Pi{acinus_id}_relative"] = row[f"Pi{acinus_id}"] / reference[f"Pi{acinus_id}"]
        row["R_hyd_relative"] = row["R_hyd"] / reference["R_hyd"]
    report = {
        "deformation_family": {
            "parameter": "s",
            "U_bar": "[[s, 0], [0, 0]]",
            "F_bar": "[[1+s, 0], [0, 1]]",
            "interpretation": "uniaxial macroscopic x-extension with transverse macroscopic stretch fixed",
        },
        "fixed_vascular_loading": {
            "Pa0": 0.25, "Pa1": 0.25, "Pa2": 0.25, "Pa3": 0.25,
            "V002": 0.50, "V006": 0.50, "Q_total": 1.0,
        },
        "statistics_convention": "Displacement and J extrema/means are over retained final nodal fields; global mean |q_l| uses the accepted current-area-weighted regional convention.",
        "cases": rows,
    }
    (SWEEP_ROOT / "deformation_healthy_sweep.json").write_text(json.dumps(report, indent=2))
    with (SWEEP_ROOT / "deformation_healthy_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return report


def make_summary_figures(rows):
    stretch = np.asarray([row["deformation_s"] for row in rows])
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    labels = ["A0", "A1", "A2", "A3"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for acinus_id, color, label in zip(range(4), colors, labels):
        ax.plot(stretch, [row[f"Pi{acinus_id}_relative"] for row in rows], "o-", color=color, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Macroscopic extension s", ylabel=r"$\Pi_i(s)/\Pi_i(0)$",
           title="Relative regional perfusion under deformation")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(SWEEP_ROOT / "Pi_relative_vs_deformation.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(stretch, [row["CV_Pi"] for row in rows], "o-", color="#7a3e9d")
    ax.set(xlabel="Macroscopic extension s", ylabel=r"$CV_{\Pi}$", title="Inter-acinar heterogeneity")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(SWEEP_ROOT / "CV_Pi_vs_deformation.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(stretch, [row["R_hyd_relative"] for row in rows], "o-", color="#2f4b7c")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Macroscopic extension s", ylabel=r"$R_{hyd}(s)/R_{hyd}(0)$",
           title="Relative hydraulic resistance")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(SWEEP_ROOT / "Rhyd_relative_vs_deformation.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(stretch, [row["max_displacement"] for row in rows], "o-", label="max |u|")
    ax.plot(stretch, [row["mean_displacement"] for row in rows], "o-", label="mean |u|")
    ax.set(xlabel="Macroscopic extension s", ylabel="Displacement magnitude", title="Retained nodal displacement")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(SWEEP_ROOT / "displacement_vs_deformation.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(stretch, [row["min_J"] for row in rows], "o-", label="min J")
    ax.plot(stretch, [row["mean_J"] for row in rows], "o-", label="mean J")
    ax.plot(stretch, [row["max_J"] for row in rows], "o-", label="max J")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Macroscopic extension s", ylabel="J", title="Jacobian range")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(SWEEP_ROOT / "J_range_vs_deformation.png", dpi=250); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    for acinus_id, color, label in zip(range(4), colors, labels):
        axes[0].plot(stretch, [row[f"Pi{acinus_id}_relative"] for row in rows], "o-", color=color, label=label)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel(r"$\Pi_i/\Pi_i(0)$"); axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(stretch, [row["CV_Pi"] for row in rows], "o-", color="#7a3e9d")
    axes[1].set_ylabel(r"$CV_{\Pi}$")
    axes[2].plot(stretch, [row["R_hyd_relative"] for row in rows], "o-", color="#2f4b7c")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel(r"$R_{hyd}/R_{hyd}(0)$")
    for axis, title in zip(axes, ("Regional intensity", "Heterogeneity", "Hydraulic resistance")):
        axis.set_xlabel("Macroscopic extension s"); axis.set_title(title); axis.grid(alpha=0.25)
    fig.suptitle("Healthy-flow deformation sweep")
    fig.savefig(SWEEP_ROOT / "deformation_healthy_summary.png", dpi=250)
    plt.close(fig)


def main():
    pv.global_theme.allow_empty_mesh = True
    rows = []
    for stretch, case_dir in CASES.items():
        row, coordinates, topology, fields, acinus_ids = load_case(stretch, case_dir)
        rows.append(row)
        make_case_figures(stretch, case_dir, coordinates, topology, fields, acinus_ids)
    report = write_summary(rows)
    make_summary_figures(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
