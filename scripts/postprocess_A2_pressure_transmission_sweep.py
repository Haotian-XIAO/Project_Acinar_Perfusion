#!/usr/bin/env python3
"""Summarize the fixed-open-pressure A2 transmission sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import postprocess_regional_gas_expansion as gas


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "results" / "A2_pressure_transmission_sweep"
BETA_VALUES = (1.00, 0.75, 0.50, 0.25, 0.00)
CASE_DIRS = {beta: SWEEP_ROOT / f"beta_{int(round(100 * beta)):03d}" for beta in BETA_VALUES}
PG0_REFERENCE = ROOT / "results" / "regional_pg_validation" / "regional_zero" / "regional_gas_expansion.json"
REPRESENTATIVE = (1.00, 0.50, 0.00)


def beta_name(beta):
    return f"beta_{int(round(100 * beta)):03d}"


def read_case(beta):
    case_dir = CASE_DIRS[beta]
    name = case_dir.name
    config = json.loads((case_dir / "run_config.json").read_text())
    perfusion = json.loads((case_dir / "regional_perfusion.json").read_text())
    expansion = json.loads((case_dir / "regional_gas_expansion.json").read_text())
    pressure = json.loads((case_dir / "patch_pressures.json").read_text())
    diagnostics = json.loads((case_dir / "diagnostics.json").read_text())
    balance = json.loads((case_dir / "mass_balance.json").read_text())
    pressures = config.get("regional_gas_pressures", [0.0] * 4)
    if config.get("gas_pressure_mode", "regional") == "global":
        pressures = [config.get("global_gas_pressure", 0.0)] * 4
    p_rows = {int(row["acinus_id"]): row for row in perfusion["acini"]}
    e_rows = {int(row["acinus_id"]): row for row in expansion["acini"]}
    row = {
        "beta": beta,
        "pg0": float(pressures[0]), "pg1": float(pressures[1]),
        "pg2": float(pressures[2]), "pg3": float(pressures[3]),
        "CV_Pi": float(perfusion["global"]["CV_Pi"]),
        "R_hyd": float(pressure["R_hyd"]),
        "pa_eff": float(pressure["pa_eff"]), "pv_eff": float(pressure["pv_eff"]),
        "Delta_p_av": float(pressure["Delta_p_av"]),
        "min_J": float(expansion["global"]["min_J"]),
        "mean_J": float(expansion["global"]["mean_J"]),
        "max_J": float(expansion["global"]["max_J"]),
        "mean_displacement": float(expansion["global"]["mean_displacement"]),
        "max_displacement": float(expansion["global"]["max_displacement"]),
        "current_wall_area": float(expansion["global"]["global_current_wall_area"]),
        "geometric_wall_area": float(expansion["global"]["periodically_canonicalized_geometric_wall_area"]),
        "total_gas_area": float(expansion["validation"]["current_gas_area"]),
        "gas_loop_closure": float(expansion["validation"]["max_current_loop_closure_residual"]),
        "convergence": bool(diagnostics["nonlinear_converged"]),
        "time_load_increments": int(diagnostics["time_load_increments"]),
        "total_nonlinear_iterations": int(diagnostics["total_nonlinear_iterations"]),
        "mass_balance_residual": float(balance["mass_balance_residual"]),
        "ptilde_l_volume_average": float(diagnostics["ptilde_l_volume_average"]),
        "lambda_p": float(diagnostics["lambda_p"]),
        "regional_consistency_residual": max(abs(value) for value in perfusion["global"]["residuals"].values()),
        "case_directory": str(case_dir.resolve()),
    }
    for i in range(4):
        row[f"Ag{i}"] = float(e_rows[i]["current_gas_area"])
        row[f"E{i}"] = float(e_rows[i]["E_i"])
        row[f"Pi{i}"] = float(p_rows[i]["Pi_i"])
        row[f"fP{i}"] = float(p_rows[i]["fP_i"])
        row[f"P{i}"] = float(p_rows[i]["P_i"])
        row[f"PiQ{i}"] = float(p_rows[i]["PiQ_i"])
        row[f"CV_q{i}"] = float(p_rows[i]["CV_q_i"])
        row[f"mean_J{i}"] = float(e_rows[i]["mean_J"])
    row["mean_open_expansion"] = (row["E0"] + row["E1"] + row["E3"]) / 3.0
    row["D_E2"] = row["mean_open_expansion"] - row["E2"]
    return row


def plot_curves(rows):
    beta = np.asarray([row["beta"] for row in rows])
    colors = plt.get_cmap("tab10").colors[:4]
    def finish(ax, xlabel, ylabel, title, path, reference=None):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if reference is not None:
            ax.axhline(reference, color="black", linestyle="--", linewidth=.9)
        ax.grid(alpha=.25); ax.legend(frameon=False); plt.tight_layout()
        plt.savefig(path, dpi=250); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i, color in enumerate(colors): ax.plot(beta, [r[f"E{i}"] for r in rows], "o-", color=color, label=f"A{i}")
    finish(ax, r"$β$ (A2 pressure transmission)", r"$E_i$", "Regional gas-area expansion", SWEEP_ROOT / "regional_expansion_vs_beta.png")

    fig, ax = plt.subplots(figsize=(7, 4.8)); ax.plot(beta, [r["D_E2"] for r in rows], "o-", color="#d62728", label=r"$D_{E2}$")
    finish(ax, r"$β$", r"$D_{E2}$", "A2 expansion mismatch relative to open acini", SWEEP_ROOT / "A2_expansion_deficit_vs_beta.png")

    beta1 = rows[0]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i, color in enumerate(colors): ax.plot(beta, [r[f"Pi{i}"] / beta1[f"Pi{i}"] for r in rows], "o-", color=color, label=f"A{i}")
    finish(ax, r"$β$", r"$Π_i(β)/Π_i(1)$", "Regional perfusion relative to beta=1", SWEEP_ROOT / "regional_perfusion_vs_beta.png", 1.0)

    fig, ax = plt.subplots(figsize=(7, 4.8)); ax.plot(beta, [r["Pi2"] / beta1["Pi2"] for r in rows], "o-", color="#d62728", label="A2")
    finish(ax, r"$β$", r"$Π_2(β)/Π_2(1)$", "A2 perfusion response", SWEEP_ROOT / "A2_perfusion_response_vs_beta.png", 1.0)

    fig, ax = plt.subplots(figsize=(7, 4.8)); ax.plot(beta, [r["CV_Pi"] for r in rows], "o-", color="#9467bd", label=r"$CV_Π$")
    finish(ax, r"$β$", r"$CV_Π$", "Inter-acinar perfusion heterogeneity", SWEEP_ROOT / "CV_Pi_vs_beta.png")

    fig, ax = plt.subplots(figsize=(7, 4.8)); ax.plot(beta, [r["R_hyd"] / beta1["R_hyd"] for r in rows], "o-", color="#2ca02c", label=r"$R_{hyd}/R_{hyd}(1)$")
    finish(ax, r"$β$", r"$R_{hyd}(β)/R_{hyd}(1)$", "Effective hydraulic resistance", SWEEP_ROOT / "Rhyd_vs_beta.png", 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.8)); ax.plot([r["D_E2"] for r in rows], [r["Pi2"] / beta1["Pi2"] for r in rows], "o-", color="#17becf", label="beta sweep")
    for r in rows: ax.annotate(f"{r['beta']:.2g}", (r["D_E2"], r["Pi2"] / beta1["Pi2"]), xytext=(4, 4), textcoords="offset points")
    finish(ax, r"$D_{E2}$", r"$Π_2(β)/Π_2(1)$", "A2 perfusion versus expansion mismatch", SWEEP_ROOT / "A2_perfusion_vs_expansion_deficit.png")


def make_representative_figures(rows):
    loaded = {}
    for beta in REPRESENTATIVE:
        case_dir = CASE_DIRS[beta]
        coordinates, topology, fields = gas.read_final_fields(case_dir / f"{case_dir.name}.xdmf")
        mapping = json.loads((case_dir / "fe_acinar_mapping.json").read_text())
        loaded[beta] = (coordinates, topology, fields,
                        np.asarray(mapping["acinus_id_by_fe_cell"], dtype=int))

    def setup(title):
        p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2100, 750)); p.set_background("white")
        return p
    def finish(p, path):
        for i in range(3): p.subplot(0, i); p.view_xy(); p.camera.parallel_projection = True; p.reset_camera()
        p.screenshot(path); p.close()
    def add_region_labels(plotter, coordinates, topology, acinus_ids, fields):
        deformed = coordinates + fields["U_tot"]
        centers = deformed[topology].mean(axis=1)
        points = np.asarray([[*centers[acinus_ids == i].mean(axis=0), .004] for i in range(4)])
        plotter.add_point_labels(points, [f"A{i}" for i in range(4)], font_size=12,
                                 text_color="black", shape_color="white",
                                 shape_opacity=.8, always_visible=True)
    def scalar_comparison(field_name, title, filename, cmap="viridis"):
        def array_for(fields):
            if field_name == "q_l_magnitude":
                return np.linalg.norm(fields["q_l"], axis=1)
            if field_name == "displacement_magnitude":
                return np.linalg.norm(fields["U_tot"], axis=1)
            return np.asarray(fields[field_name]).reshape(-1)
        values = [array_for(loaded[b][2]) for b in REPRESENTATIVE]
        lo, hi = min(v.min() for v in values), max(v.max() for v in values)
        p = setup(title)
        for index, beta in enumerate(REPRESENTATIVE):
            p.subplot(0, index); coordinates, topology, fields, acinus_ids = loaded[beta]
            deformed = coordinates + fields["U_tot"]
            grid = gas.pyvista_grid(deformed, topology, {field_name: array_for(fields)})
            p.add_mesh(grid, scalars=field_name, cmap=cmap, clim=(lo, hi), show_edges=False,
                       scalar_bar_args={"title": field_name})
            add_region_labels(p, coordinates, topology, acinus_ids, fields)
            p.add_text(f"{title}\nbeta={beta:.2f}", color="black", font_size=11)
        finish(p, SWEEP_ROOT / filename)
    # Overlay uses reference/deformed feature edges and is deliberately shared across cases.
    p = setup("Reference/deformed geometry: A2 pressure transmission")
    for index, beta in enumerate(REPRESENTATIVE):
        p.subplot(0, index); coordinates, topology, fields, acinus_ids = loaded[beta]
        ref = gas.pyvista_grid(coordinates, topology, {})
        deformed = gas.pyvista_grid(coordinates + fields["U_tot"], topology, {})
        p.add_mesh(ref.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False), color="#888888", line_width=2, label="reference")
        p.add_mesh(deformed.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False), color="#d62728", line_width=2, label="deformed")
        add_region_labels(p, coordinates, topology, acinus_ids, fields)
        p.add_text(f"reference/deformed\nbeta={beta:.2f}", color="black", font_size=11)
    finish(p, SWEEP_ROOT / "beta_comparison_deformation.png")
    scalar_comparison("displacement_magnitude", "Displacement magnitude", "beta_comparison_displacement.png", "magma")
    scalar_comparison("J_tot", "Jacobian", "beta_comparison_J.png", "coolwarm")
    scalar_comparison("q_l_magnitude", "Spatial Darcy flux magnitude", "beta_comparison_flux.png")
    scalar_comparison("pl_tot", "Liquid pressure", "beta_comparison_pressure.png", "coolwarm")


def main():
    if any("periodically_canonicalized_geometric_wall_area" not in
           json.loads((CASE_DIRS[beta] / "regional_gas_expansion.json").read_text()).get("global", {})
           for beta in BETA_VALUES):
        metadata = json.loads(gas.ACINAR_METADATA.read_text())
        bx, bt, bf = gas.read_final_fields(gas.BASELINE_XDMF)
        reference_rows, _ = gas.regional_gas_areas(bx, bt, bf["U_tot"], metadata, np.eye(2))
        reference_areas = {row["acinus_id"]: row["current_gas_area"] for row in reference_rows}
        for beta in BETA_VALUES:
            case_dir = CASE_DIRS[beta]
            expansion_path = case_dir / "regional_gas_expansion.json"
            if expansion_path.is_symlink():
                expansion_path.unlink()
            gas.process_case(beta_name(beta), case_dir,
                             case_dir / f"{case_dir.name}.xdmf", reference_areas)
    rows = [read_case(beta) for beta in BETA_VALUES]
    pg0 = json.loads(PG0_REFERENCE.read_text())
    reference_total_gas = float(pg0["validation"]["current_gas_area"])
    reference_wall = float(pg0["global"]["global_current_wall_area"])
    if "periodically_canonicalized_geometric_wall_area" in pg0["global"]:
        reference_geometric_wall = float(pg0["global"]["periodically_canonicalized_geometric_wall_area"])
    else:
        metadata = json.loads(gas.ACINAR_METADATA.read_text())
        bx, bt, bf = gas.read_final_fields(gas.BASELINE_XDMF)
        reference_geometric_wall = gas.regional_gas_areas(
            bx, bt, bf["U_tot"], metadata, np.eye(2)
        )[1]["periodically_canonicalized_geometric_wall_area"]
    beta1 = rows[0]
    for row in rows:
        for i in range(4):
            row[f"E{i}_from_beta1"] = row[f"E{i}"] - beta1[f"E{i}"]
            row[f"Pi{i}_relative_beta1"] = row[f"Pi{i}"] / beta1[f"Pi{i}"]
        row["R_hyd_relative_beta1"] = row["R_hyd"] / beta1["R_hyd"]
        row["total_gas_area_change_from_pg0"] = row["total_gas_area"] - reference_total_gas
        row["wall_area_change_from_pg0"] = row["current_wall_area"] - reference_wall
        row["geometric_wall_area_change_from_pg0"] = row["geometric_wall_area"] - reference_geometric_wall
        row["fixed_cell_area_identity_residual"] = row["total_gas_area_change_from_pg0"] + row["geometric_wall_area_change_from_pg0"]
    plot_curves(rows)
    make_representative_figures(rows)
    report = {
        "experiment": {
            "p_open": 0.20,
            "beta_definition": "A2 regional inspiratory alveolar-pressure transmission factor; not ventilation fraction",
            "Ubar": [[0.0, 0.0], [0.0, 0.0]],
            "vascular_loading": {"Pa0": .25, "Pa1": .25, "Pa2": .25, "Pa3": .25, "V002": .5, "V006": .5},
            "reference_for_normalized_obstruction_metrics": "beta=1, pg=[0.20,0.20,0.20,0.20]",
            "secondary_gas_area_reference": "accepted pg=0 healthy baseline",
        },
        "validation": {
            "all_cases_converged": all(row["convergence"] for row in rows),
            "max_abs_mass_balance_residual": max(abs(row["mass_balance_residual"]) for row in rows),
            "max_regional_consistency_residual": max(row["regional_consistency_residual"] for row in rows),
            "max_abs_fixed_cell_area_identity_residual": max(abs(row["fixed_cell_area_identity_residual"]) for row in rows),
            "max_gas_loop_closure_residual": max(row["gas_loop_closure"] for row in rows),
            "all_J_positive": all(row["min_J"] > 0 for row in rows),
            "sum_fP_minus_one": [sum(row[f"fP{i}"] for i in range(4)) - 1.0 for row in rows],
        },
        "cases": rows,
        "terminology": {
            "D_E2": "mean(E0,E1,E3)-E2; expansion mismatch relative to open acini, not a ventilation deficit",
            "Pi_i": "predicted regional perfusion intensity conditional on prescribed healthy vascular flow",
        },
    }
    (SWEEP_ROOT / "A2_pressure_transmission_sweep.json").write_text(json.dumps(report, indent=2))
    keys = list(rows[0])
    with (SWEEP_ROOT / "A2_pressure_transmission_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
