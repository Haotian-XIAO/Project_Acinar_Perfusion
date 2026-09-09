#!/usr/bin/env python3
"""Summarize retained healthy/Pa2-suppression results without solving."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "vascular_suppression_Pa2_sweep"
CASES = {
    1.00: ROOT / "results" / "vascular_baseline",
    0.75: ROOT / "results" / "vascular_suppression_Pa2_075",
    0.50: ROOT / "results" / "vascular_suppression_Pa2_050",
    0.25: ROOT / "results" / "vascular_suppression_Pa2_025",
    0.00: ROOT / "results" / "vascular_suppression_Pa2_000",
}


def load_case(alpha, directory):
    required = ["regional_perfusion.json", "run_config.json", "mass_balance.json",
                "diagnostics.json", "patch_pressures.json"]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"alpha={alpha:.2f} missing {missing} in {directory}")
    regional = json.loads((directory / "regional_perfusion.json").read_text())
    config = json.loads((directory / "run_config.json").read_text())
    balance = json.loads((directory / "mass_balance.json").read_text())
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    pressures = json.loads((directory / "patch_pressures.json").read_text())
    flows = config["target_flows"]
    by_id = {int(row["acinus_id"]): row for row in regional["acini"]}
    row = {
        "alpha_Pa2": float(alpha),
        "Qa0": float(flows["Pa0"]), "Qa1": float(flows["Pa1"]),
        "Qa2": float(flows["Pa2"]), "Qa3": float(flows["Pa3"]),
        "Pi0": by_id[0]["Pi_i"], "Pi1": by_id[1]["Pi_i"],
        "Pi2": by_id[2]["Pi_i"], "Pi3": by_id[3]["Pi_i"],
        "fP0": by_id[0]["fP_i"], "fP1": by_id[1]["fP_i"],
        "fP2": by_id[2]["fP_i"], "fP3": by_id[3]["fP_i"],
        "CV_Pi": regional["global"]["CV_Pi"],
        "pa_eff": pressures["pa_eff"], "pv_eff": pressures["pv_eff"],
        "Delta_p_av": pressures["Delta_p_av"], "R_hyd": pressures["R_hyd"],
        "Q_in": balance["Q_in"], "Q_out": balance["Q_out"],
        "mass_balance_residual": balance["mass_balance_residual"],
        "nonlinear_converged": diagnostics["nonlinear_converged"],
        "time_load_increments": diagnostics["time_load_increments"],
        "total_nonlinear_iterations": diagnostics["total_nonlinear_iterations"],
        "ptilde_l_volume_average": diagnostics["ptilde_l_volume_average"],
        "lambda_p": diagnostics["lambda_p"],
        "regional_max_consistency_residual": max(
            abs(value) for value in regional["global"]["residuals"].values()
        ),
        "case_directory": str(directory.resolve()),
    }
    return row


def add_normalized_columns(rows, baseline):
    for row in rows:
        for acinus_id in range(4):
            row[f"Pi{acinus_id}_relative"] = row[f"Pi{acinus_id}"] / baseline[f"Pi{acinus_id}"]
        row["R_hyd_relative"] = row["R_hyd"] / baseline["R_hyd"]


def linearity(values, alpha, endpoint_values=None):
    values = np.asarray(values, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    coefficients = np.polyfit(alpha, values, 1)
    fitted = np.polyval(coefficients, alpha)
    residual = values - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    result = {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "R2": float(r_squared),
        "maximum_absolute_deviation_from_fit": float(np.max(np.abs(residual))),
    }
    if endpoint_values is not None:
        endpoint_values = np.asarray(endpoint_values, dtype=float)
        endpoint_line = endpoint_values[1] + alpha * (endpoint_values[0] - endpoint_values[1])
        result["maximum_absolute_deviation_from_endpoint_interpolation"] = float(
            np.max(np.abs(values - endpoint_line))
        )
    return result


def write_summary(rows, output):
    json_rows = [{key: value for key, value in row.items() if key != "case_directory"} for row in rows]
    fieldnames = list(json_rows[0].keys())
    with (output / "Pa2_suppression_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_rows)
    baseline = rows[0]
    alpha = np.asarray([row["alpha_Pa2"] for row in rows])
    nonlinearity = {}
    for name in ("Pi2", "CV_Pi", "R_hyd"):
        endpoint = [baseline[name], rows[-1][name]]
        nonlinearity[name] = linearity([row[name] for row in rows], alpha, endpoint)
    validation = [
        {
            "alpha_Pa2": row["alpha_Pa2"],
            "Q_in": row["Q_in"], "Q_out": row["Q_out"],
            "mass_balance_residual": row["mass_balance_residual"],
            "nonlinear_converged": row["nonlinear_converged"],
            "time_load_increments": row["time_load_increments"],
            "total_nonlinear_iterations": row["total_nonlinear_iterations"],
            "ptilde_l_volume_average": row["ptilde_l_volume_average"],
            "lambda_p": row["lambda_p"],
            "regional_max_consistency_residual": row["regional_max_consistency_residual"],
        }
        for row in rows
    ]
    report = {
        "loading_definition": {
            "alpha_Pa2": "Qa2 / 0.25",
            "Qa2": "0.25 * alpha_Pa2",
            "Qa0_Qa1_Qa3": "0.25 + 0.25 * (1-alpha_Pa2) / 3",
            "Qv002": 0.5, "Qv006": 0.5, "Q_total": 1.0,
            "compensation_interpretation": "prescribed equal redistribution, not predicted collateral recruitment",
        },
        "baseline_alpha": 1.0,
        "cases": json_rows,
        "validation": validation,
        "linearity_diagnostics": nonlinearity,
    }
    (output / "Pa2_suppression_sweep.json").write_text(json.dumps(report, indent=2))
    return report


def make_figures(rows, output):
    alpha = np.asarray([row["alpha_Pa2"] for row in rows])
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    labels = ["A0", "A1", "A2", "A3"]

    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    for acinus_id, color, label in zip(range(4), colors, labels):
        ax.plot(alpha, [row[f"Pi{acinus_id}_relative"] for row in rows], "o-", color=color, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel=r"$\alpha_{Pa2}$", ylabel=r"$\Pi_i(\alpha)/\Pi_i(1)$",
           title="Relative regional perfusion versus Pa2 suppression")
    ax.set_xlim(1.0, 0.0); ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "Pi_relative_vs_alpha.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    ax.plot(alpha, [row["CV_Pi"] for row in rows], "o-", color="#7a3e9d")
    ax.set(xlabel=r"$\alpha_{Pa2}$", ylabel=r"$CV_{\Pi}$", title="Inter-acinar perfusion heterogeneity")
    ax.set_xlim(1.0, 0.0); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "CV_Pi_vs_alpha.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    baseline = rows[0]["R_hyd"]
    ax.plot(alpha, [row["R_hyd"] / baseline for row in rows], "o-", color="#2f4b7c")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel=r"$\alpha_{Pa2}$", ylabel=r"$R_{hyd}(\alpha)/R_{hyd}(1)$",
           title="Relative effective hydraulic resistance")
    ax.set_xlim(1.0, 0.0); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "Rhyd_relative_vs_alpha.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    for acinus_id, color, label in zip(range(4), colors, labels):
        ax.plot(alpha, [row[f"fP{acinus_id}"] for row in rows], "o-", color=color, label=label)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1, label="equal-share reference")
    ax.set(xlabel=r"$\alpha_{Pa2}$", ylabel=r"$fP_i$", title="Normalized regional transport shares")
    ax.set_xlim(1.0, 0.0); ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "normalized_perfusion_share_vs_alpha.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    response = [row["Pi2_relative"] for row in rows]
    ax.plot(alpha, response, "o-", color="#e45756", label="A2 response")
    endpoint_line = response[-1] + alpha * (response[0] - response[-1])
    ax.plot(alpha, endpoint_line, "--", color="black", label="endpoint linear reference")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set(xlabel=r"$\alpha_{Pa2}$", ylabel=r"$\Pi_2(\alpha)/\Pi_2(1)$",
           title="A2 response to Pa2 suppression")
    ax.set_xlim(1.0, 0.0); ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "Pa2_response.png", dpi=250); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    for acinus_id, color, label in zip(range(4), colors, labels):
        axes[0].plot(alpha, [row[f"Pi{acinus_id}_relative"] for row in rows], "o-", color=color, label=label)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel(r"$\Pi_i/\Pi_i(1)$"); axes[0].set_title("Regional intensity")
    axes[1].plot(alpha, [row["CV_Pi"] for row in rows], "o-", color="#7a3e9d")
    axes[1].set_ylabel(r"$CV_{\Pi}$"); axes[1].set_title("Heterogeneity")
    axes[2].plot(alpha, [row["R_hyd"] / baseline for row in rows], "o-", color="#2f4b7c")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel(r"$R_{hyd}/R_{hyd}(1)$"); axes[2].set_title("Hydraulic resistance")
    for axis in axes:
        axis.set_xlabel(r"$\alpha_{Pa2}$"); axis.set_xlim(1.0, 0.0); axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Compensated Pa2 suppression summary")
    fig.savefig(output / "Pa2_suppression_summary.png", dpi=250)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [load_case(alpha, directory) for alpha, directory in CASES.items()]
    add_normalized_columns(rows, rows[0])
    report = write_summary(rows, OUTPUT)
    make_figures(rows, OUTPUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
