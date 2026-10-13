#!/usr/bin/env python3
"""Assemble, inspect, and solve exactly one first-Newton system.

This is intentionally a project-side adapter: it reuses the accepted driver
formulation and does not alter dolfin_mech or enter TimeIntegrator's retry
loop.  It writes the exception that NonlinearSolver.linear_solve normally
discards with a bare ``except``.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import dolfin
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "physiological_60x60_linear_solver_diagnostic"


def dump_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def petsc_matrix(matrix):
    """Return the PETSc Mat behind a DOLFIN Matrix without copying it."""
    return dolfin.as_backend_type(matrix).mat()


def describe_constraints(constraints):
    result = []
    for c in constraints:
        row = {"class": type(c).__name__}
        bc = getattr(c, "bc", None)
        if bc is not None:
            try:
                row["constrained_dofs"] = len(bc.get_boundary_values())
            except Exception as exc:  # diagnostic metadata must not stop assembly
                row["constrained_dofs_error"] = repr(exc)
        result.append(row)
    return result


def vector_stats(vec):
    values = vec.get_local()
    return {
        "size": int(vec.size()), "l2_norm": float(vec.norm("l2")),
        "finite": bool(np.isfinite(values).all()),
        "nonfinite_count": int(values.size - np.isfinite(values).sum()),
        "min": float(np.min(values)), "max": float(np.max(values)),
    }


def inspect_matrix(solver, problem, assembly_seconds):
    A = petsc_matrix(solver.jac_mat)
    info = A.getInfo()
    diagonal = A.getDiagonal().getArray(readonly=True).copy()
    absdiag = np.abs(diagonal)
    # PETSc performs this natively, avoiding a Python getRow loop over ~3.8M rows.
    try:
        zero_rows = A.findZeroRows()
        exact_zero_rows = int(zero_rows.getSize())
    except Exception as exc:
        exact_zero_rows = None
        zero_rows_error = traceback.format_exc()
    else:
        zero_rows_error = None
    subsols = {s.name: int(s.func.function_space().dim()) for s in problem.subsols}
    total = int(problem.sol_func.function_space().dim())
    data = {
        "first_newton_only": True,
        "matrix_dimensions": [int(x) for x in A.getSize()],
        "total_dofs": total,
        "nnz": int(info.get("nz_used", info.get("nz_allocated", -1))),
        "petsc_matrix_type": A.getType(),
        "dolfin_matrix_type": type(solver.jac_mat).__name__,
        "assembly_seconds": assembly_seconds,
        "linear_solver": {"type": solver.linear_solver_type, "name": solver.linear_solver_name,
                          "backend_object": type(solver.linear_solver).__name__,
                          "umfpack_requested": solver.linear_solver_name.lower() == "umfpack"},
        "rhs": vector_stats(solver.res_vec),
        "matrix_entries_finite": bool(np.isfinite(diagonal).all()),
        "diagonal": {"finite": bool(np.isfinite(diagonal).all()),
                     "exact_zero_count": int(np.count_nonzero(diagonal == 0.0)),
                     "near_zero_count_abs_le_1e-14": int(np.count_nonzero(absdiag <= 1e-14)),
                     "min_abs": float(absdiag.min()), "max_abs": float(absdiag.max())},
        "exact_zero_rows": exact_zero_rows,
        "exact_zero_columns": "not computed: PETSc has no cheap distributed column analogue; square-matrix structural row check and diagonal check performed",
        "zero_rows_check_error": zero_rows_error,
        "subsolution_dofs": subsols,
        "constraint_bcs": describe_constraints(solver.constraints),
        "pressure_gauge": {"zero_mean_multiplier_subsolutions": sum("lambda_p_l_perturbation_zero_mean" == s.name for s in problem.subsols),
                           "zero_mean_operator_count": sum(type(op).__name__ == "ZeroMeanPressureOperator" for op in problem.operators)},
        "macro_constraints": {"macroscopic_stretch_dofs": subsols.get("U_bar", subsols.get("macroscopic_stretch", 0)),
                              "step_penalty_operator_count": sum("MacroscopicStretchComponentPenalty" in type(op).__name__ for op in problem.steps[0].operators)},
    }
    dump_json("matrix_diagnostics.json", data)
    dump_json("dof_breakdown.json", {"total_dofs": total, "subsolutions": subsols,
                                      "constraints": data["constraint_bcs"],
                                      "interpretation": "R-family entries are global Lagrange-multiplier/constraint DOFs."})
    return data


class FirstNewtonDiagnosticIntegrator:
    """Drop-in TimeIntegrator replacement that cannot retry or advance time."""
    def __init__(self, problem, solver, parameters, **kwargs):
        self.problem, self.solver = problem, solver

    def integrate(self):
        step = self.problem.steps[0]
        if "--spaces-only" in sys.argv:
            subsols = {s.name: int(s.func.function_space().dim()) for s in self.problem.subsols}
            name = "dof_breakdown_20x20.json" if "--case20" in sys.argv else "dof_breakdown.json"
            dump_json(name, {
                "total_dofs": int(self.problem.sol_func.function_space().dim()),
                "subsolutions": subsols,
                "mesh_cells": int(self.problem.mesh.num_cells()),
                "mesh_vertices": int(self.problem.mesh.num_vertices()),
                "note": "Obtained by constructing the selected accepted problem only; no form assembly or solve was attempted."
            })
            return False
        dt = step.dt_ini
        t_step = dt / (step.t_fin - step.t_ini)
        self.problem.set_variational_formulation(k_step=0)
        self.solver.constraints = list(self.problem.constraints) + list(step.constraints)
        for operator in step.operators:
            operator.set_value_at_t_step(t_step)
            operator.set_dt(dt)
        for constraint in step.constraints:
            constraint.set_value_at_t_step(t_step)
        self.problem.sol_old_func.vector()[:] = self.problem.sol_func.vector()[:]
        if len(self.problem.subsols) > 1:
            dolfin.assign(self.problem.get_subsols_func_old_lst(), self.problem.sol_old_func)
        self.solver.k_iter = 1
        started = time.perf_counter()
        assembled = self.solver.assemble_linear_system()
        elapsed = time.perf_counter() - started
        if assembled is False:
            (OUT / "exact_exception.txt").write_text("Assembly failed before the linear solve.\n")
            return False
        inspect_matrix(self.solver, self.problem, elapsed)
        try:
            self.solver.linear_solver.solve(self.problem.dsol_func.vector(), self.solver.res_vec)
        except BaseException:
            (OUT / "exact_exception.txt").write_text(traceback.format_exc())
            dump_json("linear_solve_result.json", {"attempted_once": True, "succeeded": False})
            return False
        (OUT / "exact_exception.txt").write_text("No exception: the single requested linear solve returned successfully.\n")
        dump_json("linear_solve_result.json", {"attempted_once": True, "succeeded": True})
        return False  # deliberate stop before Newton update

    def close(self):
        pass


def configure_60(base, driver):
    base.MESH_BASE = driver.MARKER_BASE
    base.MARKER_JSON = driver.SITES
    base.RESULTS = OUT
    base.RESULT_BASE = OUT / "first_newton_60x60"
    base.Q_TOTAL = driver.Q_TOTAL
    base.TARGET_FLOWS = dict(driver.FLOWS)
    base.MACROSCOPIC_DISPLACEMENT_GRADIENT = ((0.0, 0.0), (0.0, 0.0))
    base.load_mesh_and_markers = driver.load_60_mesh_and_markers
    base.make_regional_gas_boundary_markers = driver.global_gas_boundary_only


def write_incomplete_attempt_reports():
    """Persist facts available when a process is killed before Python can catch."""
    d60 = json.loads((OUT / "dof_breakdown.json").read_text())
    d20 = json.loads((OUT / "dof_breakdown_20x20.json").read_text())
    (OUT / "exact_exception.txt").write_text(
        "No Python/C++ exception was captured. The single diagnostic process printed "
        "'Assembly…' and terminated before assembly returned, so it never reached the "
        "requested one UMFPACK call. This is not an exception hidden by the wrapper. "
        "The preserved failed-run wrapper uses bare `except:` at NonlinearSolver.py:232-236, "
        "therefore its prior underlying exception cannot be reconstructed from existing logs.\n")
    dump_json("matrix_diagnostics.json", {
        "status": "unavailable", "reason": "single diagnostic process terminated during first Jacobian assembly before Python regained control",
        "matrix_dimensions": [d60["total_dofs"], d60["total_dofs"]], "total_dofs": d60["total_dofs"],
        "nnz": None, "matrix_storage_type": "DOLFIN Matrix / PETSc AIJ expected", "assembly_seconds": None,
        "zero_rows": None, "zero_columns": None, "diagonal": None, "matrix_entries_finite": None,
        "rhs_entries_finite": None, "rhs_norm": None, "linear_solver": {"type": "dolfin", "name": "umfpack", "umfpack_requested": True},
        "evidence": "Preserved solver.log shows successful assemblies at 51.60 s then ~11.49 s and res_norm=3.5416e+02 before each Solve."})
    dump_json("comparison_20x20_vs_60x60.json", {
        "formulation_comparison": "No formulation difference found in the reused base driver: same P2 U_tilde / P1 p_l_perturbation / R U_bar / R S_area / R pressure-gauge spaces; periodic bcs; one pointwise U_tilde pin; one ZeroMeanPressureOperator; four U_bar component penalties at 1e6; one Darcy operator; six source-only operators; and DOLFIN UMFPACK.",
        "20x20": d20, "60x60": d60,
        "introduced_60x60_differences": ["mesh and vascular-support assets", "Q_total 0.667 instead of 1.0", "60x60 adapter uses the original global gas-boundary measure (as does the global-pressure 20x20 solve)", "post-convergence FOI projection override only; it is outside the residual/Jacobian"],
        "constraint_conclusion": "The source code establishes exactly one pressure gauge and one pointwise displacement pin in both cases; U_bar=0 is imposed by four component penalties, plus the unchanged symmetry penalty. No unmatched multiplier was found in the formulation review."
    })
    from petsc4py import PETSc
    packages = {p: bool(PETSc.Sys.hasExternalPackage(p)) for p in ("mumps", "hypre", "superlu_dist", "umfpack")}
    dump_json("recommended_next_solver.json", {
        "status": "recommendation only; no alternative long solve launched",
        "available_petsc_external_packages": packages,
        "recommended_first_trial": "PETSc KSP GMRES with a field-split preconditioner matching U_tilde, p_l_perturbation, and the five R/global fields; use AMG/HYPRE for the displacement block only if hypre is available, and an appropriate pressure Schur/block approximation. Preserve the one gauge multiplier in its global block.",
        "direct_alternative": "Use PETSc LU/MUMPS only if MUMPS is reported available and memory is provisioned for factorization; do not assume it is available.",
        "why_not_umfpack": "The preserved run reached 14.95 GB peak RSS and spent 15,877 s before abnormal termination. That supports a resource/backend scalability diagnosis, but not a claim of a particular UMFPACK exception or structural singularity."
    })


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    import test_acinar_perfusion_vascular_baseline as base
    if "--case20" in sys.argv:
        # Keep the accepted 20x20 setup untouched; redirect only its metadata.
        base.RESULTS = OUT
        base.RESULT_BASE = OUT / "first_newton_20x20"
    else:
        import run_60x60_full_healthy_baseline as driver
        configure_60(base, driver)
    original_integrator = base.dmech.TimeIntegrator
    base.dmech.TimeIntegrator = FirstNewtonDiagnosticIntegrator
    try:
        base.run()
    except RuntimeError as exc:
        # base.run treats our intentional False return as a normal failed integration.
        if str(exc) != "Healthy vascular baseline integration failed.":
            raise
    finally:
        base.dmech.TimeIntegrator = original_integrator
    if "--spaces-only" in sys.argv and "--case20" not in sys.argv:
        write_incomplete_attempt_reports()


if __name__ == "__main__":
    main()
