#!/usr/bin/env python3
"""Summarize and visualize the retained 3 x 5 deformation/Pa2 matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import postprocess_deformation_healthy_sweep as deformation


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = ROOT / "results" / "deformation_Pa2_matrix"
DEFORM_ROOT = ROOT / "results" / "deformation_healthy_sweep"
SUPPRESSION_ROOT = ROOT / "results"
S_VALUES = (0.00, 0.10, 0.20)
A_VALUES = (1.00, 0.75, 0.50, 0.25, 0.00)


def source_directory(s, alpha):
    if s == 0.0:
        return SUPPRESSION_ROOT / ("vascular_baseline" if alpha == 1.0 else f"vascular_suppression_Pa2_{int(round(alpha*100)):03d}")
    if alpha == 1.0:
        return DEFORM_ROOT / f"stretch_{int(round(s*100)):03d}"
    return MATRIX_ROOT / f"s_{int(round(s*100)):03d}_alpha_{int(round(alpha*100)):03d}"


def xdmf_path(s, alpha, directory):
    if s == 0.0:
        basename = "vascular_baseline" if alpha == 1.0 else directory.name
    elif alpha == 1.0:
        basename = directory.name
    else:
        basename = directory.name
    return directory / f"{basename}.xdmf"


def load_state(s, alpha):
    directory = source_directory(s, alpha)
    required = ["regional_perfusion.json", "run_config.json", "diagnostics.json",
                "patch_pressures.json", "mass_balance.json", "fe_acinar_mapping.json"]
    missing = [name for name in required if not (directory / name).exists()]
    path = xdmf_path(s, alpha, directory)
    if missing or not path.exists():
        raise FileNotFoundError(f"State (s={s}, alpha={alpha}) missing {missing + ([str(path)] if not path.exists() else [])}")
    regional = json.loads((directory / "regional_perfusion.json").read_text())
    config = json.loads((directory / "run_config.json").read_text())
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    pressures = json.loads((directory / "patch_pressures.json").read_text())
    balance = json.loads((directory / "mass_balance.json").read_text())
    mapping = json.loads((directory / "fe_acinar_mapping.json").read_text())
    fields = deformation.read_final_fields(path.resolve())
    by_id = {int(row["acinus_id"]): row for row in regional["acini"]}
    u_magnitude = np.linalg.norm(fields[2]["U_tot"], axis=1)
    flows = config["target_flows"]
    row = {
        "s": float(s), "alpha": float(alpha),
        "Qa0": flows["Pa0"], "Qa1": flows["Pa1"], "Qa2": flows["Pa2"], "Qa3": flows["Pa3"],
        "Pi0": by_id[0]["Pi_i"], "Pi1": by_id[1]["Pi_i"], "Pi2": by_id[2]["Pi_i"], "Pi3": by_id[3]["Pi_i"],
        "fP0": by_id[0]["fP_i"], "fP1": by_id[1]["fP_i"], "fP2": by_id[2]["fP_i"], "fP3": by_id[3]["fP_i"],
        "CV_Pi": regional["global"]["CV_Pi"],
        "R_hyd": pressures["R_hyd"], "pa_eff": pressures["pa_eff"],
        "pv_eff": pressures["pv_eff"], "Delta_p_av": pressures["Delta_p_av"],
        "mean_J": float(fields[2]["J_tot"].mean()),
        "min_J": float(fields[2]["J_tot"].min()), "max_J": float(fields[2]["J_tot"].max()),
        "mean_displacement": float(u_magnitude.mean()), "max_displacement": float(u_magnitude.max()),
        "global_mean_abs_q_l": regional["global"]["global_current_area_weighted_mean_abs_q"],
        "Q_in": balance["Q_in"], "Q_out": balance["Q_out"],
        "mass_balance_residual": balance["mass_balance_residual"],
        "nonlinear_converged": diagnostics["nonlinear_converged"],
        "time_load_increments": diagnostics["time_load_increments"],
        "total_nonlinear_iterations": diagnostics["total_nonlinear_iterations"],
        "ptilde_l_volume_average": diagnostics["ptilde_l_volume_average"],
        "lambda_p": diagnostics["lambda_p"],
        "regional_max_consistency_residual": max(abs(value) for value in regional["global"]["residuals"].values()),
        "regional_share_sum": regional["global"]["normalized_share_sum"],
        "directory": str(directory.resolve()),
        "xdmf": str(path.resolve()),
        "mapping_cell_count": mapping["fe_triangle_count"],
    }
    return row, fields[0], fields[1], fields[2]


def matrix(rows):
    return np.asarray([[next(row for row in rows if row["s"] == s and row["alpha"] == a)
                        for a in A_VALUES] for s in S_VALUES])


def add_normalized(rows, reference):
    for row in rows:
        for i in range(4):
            row[f"Pi{i}_relative"] = row[f"Pi{i}"] / reference[f"Pi{i}"]
        row["R_hyd_relative"] = row["R_hyd"] / reference["R_hyd"]


def interaction(rows):
    def get(s, a):
        return next(row for row in rows if row["s"] == s and row["alpha"] == a)
    ref = get(0.0, 1.0)
    outputs = {}
    for quantity in ("Pi2_relative", "R_hyd_relative"):
        outputs[quantity] = [[get(s, a)[quantity] - get(s, 1.0)[quantity] - get(0.0, a)[quantity] + 1.0
                              for a in A_VALUES] for s in S_VALUES]
    outputs["CV_Pi_absolute"] = [[get(s, a)["CV_Pi"] - get(s, 1.0)["CV_Pi"] - get(0.0, a)["CV_Pi"] + ref["CV_Pi"]
                                  for a in A_VALUES] for s in S_VALUES]
    return outputs


def linearity_diagnostics(rows):
    """Descriptive straight-line and endpoint-interpolation diagnostics."""
    diagnostics = {}
    x = np.asarray(A_VALUES, dtype=float)
    for quantity in ("Pi2", "CV_Pi", "R_hyd"):
        per_s = {}
        for s in S_VALUES:
            y = np.asarray([next(r for r in rows if r["s"] == s and r["alpha"] == a)[quantity]
                            for a in A_VALUES], dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            fitted = slope * x + intercept
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 if ss_tot == 0.0 else 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot
            endpoint = np.interp(x, [x[-1], x[0]], [y[-1], y[0]])
            per_s[f"{s:.2f}"] = {
                "slope": float(slope), "intercept": float(intercept),
                "R2": float(r2),
                "max_abs_deviation_from_endpoint_interpolation": float(np.max(np.abs(y - endpoint))),
            }
        diagnostics[quantity] = per_s
    return diagnostics


def write_outputs(rows, interactions, linearity):
    clean_rows = [{key: value for key, value in row.items() if key not in {"directory", "xdmf"}} for row in rows]
    fields = list(clean_rows[0].keys())
    with (MATRIX_ROOT / "deformation_Pa2_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(clean_rows)
    validation = [{
        "s": row["s"], "alpha": row["alpha"], "converged": row["nonlinear_converged"],
        "Q_in": row["Q_in"], "Q_out": row["Q_out"], "mass_balance_residual": row["mass_balance_residual"],
        "ptilde_l_volume_average": row["ptilde_l_volume_average"], "lambda_p": row["lambda_p"],
        "min_J": row["min_J"], "regional_max_consistency_residual": row["regional_max_consistency_residual"],
        "regional_share_sum_minus_one": row["regional_share_sum"] - 1.0,
        "time_load_increments": row["time_load_increments"], "total_nonlinear_iterations": row["total_nonlinear_iterations"],
    } for row in rows]
    report = {
        "deformation": {"parameter": "s", "U_bar": "[[s,0],[0,0]]", "F_bar": "[[1+s,0],[0,1]]"},
        "suppression": {"alpha": "Qa2/0.25", "Q_total": 1.0,
                        "compensation": "equal prescribed redistribution to Pa0, Pa1, Pa3"},
        "reference": {"s": 0.0, "alpha": 1.0},
        "states": clean_rows,
        "validation": validation,
        "interaction_residuals": interactions,
        "linearity_diagnostics": linearity,
    }
    (MATRIX_ROOT / "deformation_Pa2_matrix.json").write_text(json.dumps(report, indent=2))
    return report


def heatmap(values, title, cbar, path, fmt=".3f", cmap="viridis", center=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    image = ax.imshow(values, aspect="auto", cmap=cmap, origin="upper")
    if center is not None:
        image.set_clim(center[0], center[1])
    ax.set_xticks(range(5), ["1.00", "0.75", "0.50", "0.25", "0.00"])
    ax.set_yticks(range(3), ["0.00", "0.10", "0.20"])
    ax.set(xlabel=r"$\alpha_{Pa2}$", ylabel="deformation s", title=title)
    for i in range(3):
        for j in range(5):
            value = values[i, j]
            normalized = (value - image.norm.vmin) / (image.norm.vmax - image.norm.vmin) if image.norm.vmax > image.norm.vmin else 0.5
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    color="white" if normalized < 0.45 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label=cbar)
    fig.tight_layout(); fig.savefig(path, dpi=250); plt.close(fig)


def curve_figures(rows):
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    labels = ["s=0.00", "s=0.10", "s=0.20"]
    for quantity, ylabel, filename, normalized in (
        ("Pi2_relative", r"$\Pi_2/\Pi_2(0,1)$", "Pi2_vs_alpha_by_deformation.png", True),
        ("CV_Pi", r"$CV_{\Pi}$", "CV_Pi_vs_alpha_by_deformation.png", False),
        ("R_hyd_relative", r"$R_{hyd}/R_{hyd}(0,1)$", "Rhyd_vs_alpha_by_deformation.png", True),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for s, color, label in zip(S_VALUES, colors, labels):
            values = [next(row for row in rows if row["s"] == s and row["alpha"] == a)[quantity] for a in A_VALUES]
            ax.plot(A_VALUES, values, "o-", color=color, label=label)
        if normalized: ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel=r"$\alpha_{Pa2}$", ylabel=ylabel, title=f"{ylabel} versus Pa2 suppression")
        ax.set_xlim(1.0, 0.0); ax.grid(alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(MATRIX_ROOT / filename, dpi=250); plt.close(fig)


def compact_summary(rows):
    arrays = {
        "Pi2": np.asarray([[next(r for r in rows if r["s"] == s and r["alpha"] == a)["Pi2_relative"] for a in A_VALUES] for s in S_VALUES]),
        "CV": np.asarray([[next(r for r in rows if r["s"] == s and r["alpha"] == a)["CV_Pi"] for a in A_VALUES] for s in S_VALUES]),
        "R": np.asarray([[next(r for r in rows if r["s"] == s and r["alpha"] == a)["R_hyd_relative"] for a in A_VALUES] for s in S_VALUES]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    for ax, key, title, label in zip(axes, ("Pi2", "CV", "R"), (r"$\Pi_2/\Pi_2(0,1)$", r"$CV_{\Pi}$", r"$R_{hyd}/R_{hyd}(0,1)$"), ("normalized Pi2", "CV_Pi", "normalized R_hyd")):
        image = ax.imshow(arrays[key], aspect="auto", origin="upper", cmap="viridis")
        ax.set_xticks(range(5), ["1", ".75", ".5", ".25", "0"]); ax.set_yticks(range(3), ["0", ".1", ".2"])
        ax.set_xlabel("alpha"); ax.set_ylabel("s"); ax.set_title(title)
        fig.colorbar(image, ax=ax, label=label)
    fig.suptitle("Deformation x Pa2-suppression response matrix")
    fig.savefig(MATRIX_ROOT / "deformation_Pa2_interaction_summary.png", dpi=250); plt.close(fig)


def pyvista_four_corner(rows_and_fields):
    states = [(0.0, 1.0, "A: reference"), (0.2, 1.0, "B: deformation"),
              (0.0, 0.0, "C: Pa2 impairment"), (0.2, 0.0, "D: combined")]
    grids = []
    for s, alpha, _ in states:
        row, coordinates, topology, fields = rows_and_fields[(s, alpha)]
        deformed = coordinates + fields["U_tot"]
        values = {"p_l": fields["pl_tot"], "q_l_magnitude": np.linalg.norm(fields["q_l"], axis=1)}
        grids.append(deformation.make_grid(deformed, topology, values))

    def panel_figure(scalar, title, filename, cmap):
        lo = min(float(grid.point_data[scalar].min()) for grid in grids)
        hi = max(float(grid.point_data[scalar].max()) for grid in grids)
        plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1800, 1400))
        plotter.set_background("white")
        for index, ((_, _, label), grid) in enumerate(zip(states, grids)):
            plotter.subplot(index // 2, index % 2)
            plotter.add_text(label, color="black", font_size=12)
            plotter.add_mesh(grid, scalars=scalar, cmap=cmap, clim=[lo, hi], show_edges=False,
                             scalar_bar_args={"title": scalar})
            plotter.view_xy(); plotter.camera.parallel_projection = True; plotter.reset_camera()
        plotter.screenshot(MATRIX_ROOT / filename); plotter.close()

    panel_figure("q_l_magnitude", "Four-corner spatial Darcy flux", "four_corner_flux_comparison.png", "viridis")
    panel_figure("p_l", "Four-corner liquid pressure", "four_corner_pressure_comparison.png", "coolwarm")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True, constrained_layout=True)
    for ax, (s, alpha, label) in zip(axes.flat, states):
        row = rows_and_fields[(s, alpha)][0]
        ax.bar(["A0", "A1", "A2", "A3"], [row[f"Pi{i}"] for i in range(4)], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
        ax.set_title(label); ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel(r"$\Pi_i$"); axes[1, 0].set_ylabel(r"$\Pi_i$")
    fig.suptitle("Four-corner regional perfusion intensities")
    fig.savefig(MATRIX_ROOT / "four_corner_regional_perfusion.png", dpi=250); plt.close(fig)


def main():
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    state_data = {}
    for s in S_VALUES:
        for alpha in A_VALUES:
            row, coordinates, topology, fields = load_state(s, alpha)
            rows.append(row)
            state_data[(s, alpha)] = (row, coordinates, topology, fields)
    rows.sort(key=lambda row: (-row["s"], -row["alpha"]))
    # Reference must be (s=0, alpha=1), independent of row presentation order.
    reference = next(row for row in rows if row["s"] == 0.0 and row["alpha"] == 1.0)
    add_normalized(rows, reference)
    interactions = interaction(rows)
    linearity = linearity_diagnostics(rows)
    write_outputs(rows, interactions, linearity)

    values = matrix(rows)
    heatmap(np.asarray([[item["Pi2_relative"] for item in row] for row in values]), r"$\Pi_2(s,\alpha)/\Pi_2(0,1)$", "normalized Pi2", MATRIX_ROOT / "Pi2_matrix.png")
    heatmap(np.asarray([[item["CV_Pi"] for item in row] for row in values]), r"$CV_{\Pi}(s,\alpha)$", "CV_Pi", MATRIX_ROOT / "CV_Pi_matrix.png")
    heatmap(np.asarray([[item["R_hyd_relative"] for item in row] for row in values]), r"$R_{hyd}(s,\alpha)/R_{hyd}(0,1)$", "normalized R_hyd", MATRIX_ROOT / "Rhyd_matrix.png")
    heatmap(np.asarray([[item["fP2"] for item in row] for row in values]), r"$fP_2(s,\alpha)$", "fP2", MATRIX_ROOT / "fP2_matrix.png")
    heatmap(np.asarray(interactions["Pi2_relative"]), "Pi2 descriptive interaction residual", "I_Pi2", MATRIX_ROOT / "interaction_Pi2.png", fmt="+.3f", cmap="coolwarm")
    heatmap(np.asarray(interactions["R_hyd_relative"]), "R_hyd descriptive interaction residual", "I_R_hyd", MATRIX_ROOT / "interaction_Rhyd.png", fmt="+.3f", cmap="coolwarm")
    heatmap(np.asarray(interactions["CV_Pi_absolute"]), "CV_Pi absolute interaction residual", "I_CV_Pi", MATRIX_ROOT / "interaction_CV_Pi.png", fmt="+.3f", cmap="coolwarm")
    curve_figures(rows); compact_summary(rows)

    # Four corners are represented by tuples (row, coordinates, topology, fields); retain this small
    # adapter so the plotter can use the same retained state data.
    pyvista_four_corner(state_data)
    print(json.dumps({"states": len(rows), "output": str(MATRIX_ROOT.resolve()), "interactions": interactions}, indent=2))


if __name__ == "__main__":
    pv.global_theme.allow_empty_mesh = True
    main()
