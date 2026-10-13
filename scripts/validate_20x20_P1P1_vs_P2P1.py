#!/usr/bin/env python3
"""Controlled FE-order comparison for the frozen 20x20 vascular baseline.

This is intentionally project-local: it does not edit or monkey-patch files in
``dolfin_mech_clean``.  The subclass only overrides subsolution construction
so liquid-pressure degree can be selected independently of displacement degree.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import dolfin
import meshio
import numpy as np

import test_acinar_perfusion_vascular_baseline as baseline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fe_order_validation_20x20"
CASES = {"p2p1": (2, 1), "p1p1": (1, 1)}


def fixed_liquid_pressure_problem(u_degree, p_degree):
    """Return a project-side class with independently specified CG degrees."""
    parent = baseline.dmech.MicroPoroFlowHyperelasticityProblem

    class ProjectLocalMicroPoroFlow(parent):
        last_instance = None

        def set_subsols(self, displacement_perturbation_degree=None,
                        solid_pressure_degree=None,
                        liquid_pressure_perturbation_degree=None,
                        porosity_degree=None, porosity_init_val=None,
                        porosity_init_fun=None):
            if displacement_perturbation_degree != u_degree:
                raise RuntimeError("Validation driver received an unexpected displacement degree")
            self.add_macroscopic_stretch_subsol(symmetry=None)
            self.add_displacement_perturbation_subsol(degree=u_degree)
            if self.w_solid_incompressibility:
                degree = u_degree - 1 if solid_pressure_degree is None else solid_pressure_degree
                self.add_pressure_subsol(degree=degree)
            self.add_surface_area_subsol()
            self.add_pressure_liquid_perturbation_subsol(degree=p_degree)
            self.add_pressure_liquid_perturbation_zero_mean_subsol()

        def set_solution_functions(self):
            super().set_solution_functions()
            type(self).last_instance = self

    ProjectLocalMicroPoroFlow.__name__ = f"ProjectLocalMicroPoroFlow_U{u_degree}_P{p_degree}"
    return ProjectLocalMicroPoroFlow


def element_description(subsol):
    element = subsol.fs.ufl_element()
    return {"family": element.family(), "degree": int(element.degree()),
            "value_shape": list(element.value_shape()), "dofs": int(subsol.fs.dim())}


def add_j_to_vtu(mesh, path, problem):
    """Preserve baseline fields and add cellwise J for cross-order comparison."""
    saved = meshio.read(path)
    j = baseline.project_scalar_cell_data(problem.kinematics.J, mesh)
    u = baseline.project_vector_cell_data(problem.U_tot, mesh)
    saved.cell_data["J"] = [j]
    saved.cell_data["U_magnitude"] = [np.linalg.norm(u, axis=1)]
    meshio.write(path, saved)


def run_case(name):
    u_degree, p_degree = CASES[name]
    case_dir = OUT / name
    if case_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing validation case: {case_dir}")
    case_dir.mkdir(parents=True)

    custom_class = fixed_liquid_pressure_problem(u_degree, p_degree)
    original_class = baseline.dmech.MicroPoroFlowHyperelasticityProblem
    original_results, original_base = baseline.RESULTS, baseline.RESULT_BASE
    original_write = baseline.write_final_vtu
    baseline.RESULTS, baseline.RESULT_BASE = case_dir, case_dir / name
    baseline.dmech.MicroPoroFlowHyperelasticityProblem = custom_class

    captured = {}
    def write_final(mesh, domains, problem, darcy):
        original_write(mesh, domains, problem, darcy)
        add_j_to_vtu(mesh, case_dir / "final_fields.vtu", problem)
        captured["problem"] = problem

    baseline.write_final_vtu = write_final
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        baseline.run()
    finally:
        baseline.write_final_vtu = original_write
        baseline.dmech.MicroPoroFlowHyperelasticityProblem = original_class
        baseline.RESULTS, baseline.RESULT_BASE = original_results, original_base
    elapsed = time.perf_counter() - started
    problem = captured.get("problem") or custom_class.last_instance
    if problem is None:
        raise RuntimeError("Could not retain instantiated validation problem")
    fields = {
        "U_tilde": element_description(problem.displacement_perturbation_subsol),
        "p_l_perturbation": element_description(problem.pl_perturbation_subsol),
        "U_bar": element_description(problem.macroscopic_stretch_subsol),
        "S_area": element_description(problem.surface_area_subsol),
        "lambda_p_zero_mean": element_description(problem.lambda_pl_perturbation_zero_mean_subsol),
    }
    if fields["U_tilde"]["degree"] != u_degree or fields["p_l_perturbation"]["degree"] != p_degree:
        raise RuntimeError(f"Wrong instantiated FE order: {fields}")
    report = {
        "case": name, "requested": {"U_tilde": ["CG", u_degree], "p_l": ["CG", p_degree]},
        "instantiated_subspaces": fields,
        "mixed_dofs": int(problem.sol_func.function_space().dim()),
        "wall_time_seconds": elapsed,
        "ru_maxrss_before": int(rss_before),
        "ru_maxrss_after": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "ru_maxrss_units": "platform resource.getrusage units; on macOS this is bytes",
    }
    (case_dir / "space_and_performance.json").write_text(json.dumps(report, indent=2) + "\n")

    # Same regional postprocessor/metadata as the baseline; only output directory differs.
    import postprocess_regional_perfusion as regional
    regional.run(case_dir, case_dir / f"{name}.xdmf", case_dir / "final_fields.vtu",
                 baseline.ACINAR_METADATA_JSON, ROOT / "results" / "acinar_partition_20x20" / "periodic_acinar_summary.json")
    return report


def load_json(path):
    return json.loads(Path(path).read_text())


def relative(b, a):
    return None if a == 0 else (b - a) / a


def arrays(case):
    data = meshio.read(OUT / case / "final_fields.vtu")
    # The 20x20 baseline uses one triangle cell block.
    return data.points[:, :2], data.cells[0].data, {k: np.asarray(v[0]) for k, v in data.cell_data.items()}


def write_difference_fields_and_figures():
    pa, ta, fa = arrays("p2p1")
    pb, tb, fb = arrays("p1p1")
    if not (np.array_equal(ta, tb) and np.allclose(pa, pb)):
        raise RuntimeError("Cases do not share identical mesh coordinates/connectivity")
    diff = {"p_l_difference_P1P1_minus_P2P1": fb["p_l"] - fa["p_l"],
            "U_magnitude_difference_P1P1_minus_P2P1": fb["U_magnitude"] - fa["U_magnitude"],
            "J_difference_P1P1_minus_P2P1": fb["J"] - fa["J"]}
    meshio.write(OUT / "difference_fields_P1P1_minus_P2P1.vtu",
                 meshio.Mesh(points=np.column_stack([pa, np.zeros(len(pa))]), cells=[("triangle", ta)],
                             cell_data={key: [value] for key, value in diff.items()}))
    import matplotlib.pyplot as plt
    for key, values in diff.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.tripcolor(pa[:, 0], pa[:, 1], ta, facecolors=values, shading="flat", cmap="coolwarm")
        fig.colorbar(image, ax=ax, label=key)
        ax.set_aspect("equal"); ax.set_title("P1/P1 minus P2/P1")
        fig.tight_layout(); fig.savefig(OUT / f"{key}.png", dpi=220); plt.close(fig)
    return {key: {"min": float(value.min()), "max": float(value.max()), "l2_cell_rms": float(np.sqrt(np.mean(value**2)))} for key, value in diff.items()}


def compare():
    a, b = OUT / "p2p1", OUT / "p1p1"
    sa, sb = load_json(a / "space_and_performance.json"), load_json(b / "space_and_performance.json")
    ha, hb = load_json(a / "patch_pressures.json"), load_json(b / "patch_pressures.json")
    da, db = load_json(a / "diagnostics.json"), load_json(b / "diagnostics.json")
    ra, rb = load_json(a / "regional_perfusion.json"), load_json(b / "regional_perfusion.json")
    global_keys = ["pa_eff", "pv_eff", "Delta_p_av", "R_hyd", "Q_total"]
    hydraulic = {key: {"P2P1": ha[key], "P1P1": hb[key], "absolute": hb[key] - ha[key], "relative": relative(hb[key], ha[key])} for key in global_keys}
    regional = []
    for left, right in zip(ra["acini"], rb["acini"]):
        row = {"acinus_id": left["acinus_id"]}
        for key in ("Pi_i", "P_i", "fP_i", "CV_q_i"):
            row[key] = {"P2P1": left[key], "P1P1": right[key], "absolute": right[key] - left[key], "relative": relative(right[key], left[key])}
        regional.append(row)
    mechanical = {}
    for key in ("J", "U_magnitude"):
        _, _, fa = arrays("p2p1"); _, _, fb = arrays("p1p1")
        for statistic, func in (("min", np.min), ("mean", np.mean), ("max", np.max)):
            av, bv = float(func(fa[key])), float(func(fb[key]))
            mechanical[f"{key}_{statistic}"] = {"P2P1": av, "P1P1": bv, "absolute": bv-av, "relative": relative(bv, av)}
    convergence = {"P2P1": {"wall_time_seconds": sa["wall_time_seconds"], "mixed_dofs": sa["mixed_dofs"], "newton_iterations": da["total_nonlinear_iterations"], "increments": da["time_load_increments"], "rejected_increments": sum(not row["success"] for row in da["increments"])},
                   "P1P1": {"wall_time_seconds": sb["wall_time_seconds"], "mixed_dofs": sb["mixed_dofs"], "newton_iterations": db["total_nonlinear_iterations"], "increments": db["time_load_increments"], "rejected_increments": sum(not row["success"] for row in db["increments"])}}
    diff_stats = write_difference_fields_and_figures()
    ranking_a = [r["acinus_id"] for r in sorted(ra["acini"], key=lambda x: x["Pi_i"], reverse=True)]
    ranking_b = [r["acinus_id"] for r in sorted(rb["acini"], key=lambda x: x["Pi_i"], reverse=True)]
    report = {"hydraulic": hydraulic, "regional": regional, "mechanical": mechanical, "convergence": convergence,
              "field_difference_statistics": diff_stats, "perfusion_ranking": {"P2P1": ranking_a, "P1P1": ranking_b, "unchanged": ranking_a == ranking_b},
              "criteria": {"global_pressure_drop_and_resistance_relative": 0.03, "regional_relative": 0.05, "J_mean_relative": 0.02, "J_extrema_relative": 0.10, "U_relative": 0.05}}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.case:
        run_case(args.case)
    if args.compare:
        compare()
    if not args.case and not args.compare:
        parser.error("provide --case or --compare")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
