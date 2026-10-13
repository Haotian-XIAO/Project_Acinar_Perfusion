#!/usr/bin/env python3
"""One first-Newton 60x60 solve using PETSc LU/MUMPS, then stop.

This project-side adapter reuses the accepted healthy-baseline formulation.
It deliberately never calls the normal TimeIntegrator retry loop and never
modifies dolfin_mech.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import dolfin
import numpy as np
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "physiological_60x60_mumps_first_newton"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def exception_file(exc):
    (OUT / "exception.txt").write_text(
        "type: " + type(exc).__name__ + "\nmessage: " + str(exc) + "\n\n" + traceback.format_exc())


def mat_info(A):
    info = A.getInfo()
    return {"dimensions": [int(v) for v in A.getSize()],
            "nnz": int(info.get("nz_used", info.get("nz_allocated", -1))),
            "petsc_matrix_type": A.getType()}


class OneNewtonMumpsIntegrator:
    """Minimal TimeIntegrator protocol: one load state, one solve, one update."""
    def __init__(self, problem, solver, parameters, **kwargs):
        self.problem, self.solver = problem, solver

    def close(self):
        pass

    def configure_mumps(self):
        if not PETSc.Sys.hasExternalPackage("mumps"):
            raise RuntimeError("PETSc was built without MUMPS; refusing any fallback.")
        ksp = self.solver.linear_solver.ksp()
        pc = ksp.getPC()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        ksp.setFromOptions()
        return ksp, pc

    def integrate(self):
        result = {"status": "started", "one_newton_only": True}
        try:
            step = self.problem.steps[0]
            dt = step.dt_ini
            t_step = dt / (step.t_fin - step.t_ini)
            self.problem.set_variational_formulation(k_step=0)
            self.solver.constraints = list(self.problem.constraints) + list(step.constraints)
            for op in step.operators:
                op.set_value_at_t_step(t_step)
                op.set_dt(dt)
            for constraint in step.constraints:
                constraint.set_value_at_t_step(t_step)
            self.problem.sol_old_func.vector()[:] = self.problem.sol_func.vector()[:]
            if len(self.problem.subsols) > 1:
                dolfin.assign(self.problem.get_subsols_func_old_lst(), self.problem.sol_old_func)

            ksp, pc = self.configure_mumps()
            write("solver_configuration.json", {
                "requested": {"ksp_type": "preonly", "pc_type": "lu", "factor_solver": "mumps"},
                "mumps_available": bool(PETSc.Sys.hasExternalPackage("mumps")),
                "petsc_version": list(PETSc.Sys.getVersion()),
                "linear_solver_object": type(self.solver.linear_solver).__name__})

            self.solver.k_iter = 1
            t0 = time.perf_counter()
            if self.solver.assemble_linear_system() is False:
                raise RuntimeError("First Jacobian/residual assembly reported failure.")
            assembly_seconds = time.perf_counter() - t0
            A = dolfin.as_backend_type(self.solver.jac_mat).mat()
            b = dolfin.as_backend_type(self.solver.res_vec).vec()
            ksp.setOperators(A)
            ksp.setUp()
            selected = pc.getFactorSolverType()
            if str(selected).lower() != "mumps":
                raise RuntimeError("PETSc did not select MUMPS (selected %r); refusing fallback." % (selected,))
            before = float(self.solver.res_vec.norm("l2"))
            write("matrix_information.json", {**mat_info(A), "total_dofs": int(self.problem.sol_func.function_space().dim()),
                                              "rhs_l2_norm": before, "assembly_seconds": assembly_seconds})
            write("solver_configuration.json", {
                "requested": {"ksp_type": "preonly", "pc_type": "lu", "factor_solver": "mumps"},
                "confirmed": {"ksp_type": str(ksp.getType()), "pc_type": str(pc.getType()),
                              "factor_solver": str(selected)},
                "mumps_available": True, "petsc_version": list(PETSc.Sys.getVersion())})

            solve_start = time.perf_counter()
            self.solver.linear_solver.solve(self.problem.dsol_func.vector(), self.solver.res_vec)
            solve_seconds = time.perf_counter() - solve_start
            increment = self.problem.dsol_func.vector().get_local()
            if not np.isfinite(increment).all():
                raise RuntimeError("MUMPS returned a non-finite Newton increment.")

            # This follows the normal solver's first-update order: homogenize,
            # then add the (unrelaxed) first increment.  No second solve occurs.
            for constraint in self.solver.constraints:
                constraint.homogenize()
            self.problem.sol_func.vector().axpy(1.0, self.problem.dsol_func.vector())
            if len(self.problem.subsols) > 1:
                dolfin.assign(self.problem.get_subsols_func_lst(), self.problem.sol_func)

            # Reassemble only to evaluate the post-update residual; do not solve it.
            self.solver.k_iter = 2
            after_start = time.perf_counter()
            if self.solver.assemble_linear_system() is False:
                raise RuntimeError("Post-update residual assembly reported failure.")
            post_assembly_seconds = time.perf_counter() - after_start
            after = float(self.solver.res_vec.norm("l2"))
            usage = resource.getrusage(resource.RUSAGE_SELF)
            result.update({"status": "success", "R_before": before, "R_after": after,
                           "R_after_over_R_before": after / before if before else None,
                           "increment_finite": True, "increment_l2_norm": float(self.problem.dsol_func.vector().norm("l2")),
                           "solve_seconds": solve_seconds, "post_update_assembly_seconds": post_assembly_seconds})
            write("first_newton_result.json", result)
            write("performance.json", {"jacobian_assembly_seconds": assembly_seconds,
                                        "mumps_analysis_factor_solve_seconds": solve_seconds,
                                        "post_update_residual_assembly_seconds": post_assembly_seconds,
                                        "total_seconds": time.perf_counter() - PROCESS_START,
                                        "ru_maxrss_platform_units": int(usage.ru_maxrss),
                                        "note": "MUMPS does not expose analysis/factor/solve split through this DOLFIN call; PETSc/MUMPS stdout remains in run.log."})
        except BaseException as exc:
            exception_file(exc)
            result.update({"status": "failed", "exception_type": type(exc).__name__, "exception_message": str(exc)})
            write("first_newton_result.json", result)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            write("performance.json", {"total_seconds": time.perf_counter() - PROCESS_START,
                                        "ru_maxrss_platform_units": int(usage.ru_maxrss)})
        return False  # intentional stop: base.run raises its ordinary failed-integrator error


def configure_same_60_problem(base, driver):
    base.MESH_BASE = driver.MARKER_BASE
    base.MARKER_JSON = driver.SITES
    base.RESULTS = OUT
    base.RESULT_BASE = OUT / "first_newton_60x60_mumps"
    base.Q_TOTAL = driver.Q_TOTAL
    base.TARGET_FLOWS = dict(driver.FLOWS)
    base.MACROSCOPIC_DISPLACEMENT_GRADIENT = ((0.0, 0.0), (0.0, 0.0))
    base.load_mesh_and_markers = driver.load_60_mesh_and_markers
    base.make_regional_gas_boundary_markers = driver.global_gas_boundary_only
    base.SOLVER_PARAMETERS = {**base.SOLVER_PARAMETERS, "linear_solver_type": "petsc", "linear_solver_name": "mumps"}


def main():
    global PROCESS_START
    PROCESS_START = time.perf_counter()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    import test_acinar_perfusion_vascular_baseline as base
    import run_60x60_full_healthy_baseline as driver
    configure_same_60_problem(base, driver)
    original = base.dmech.TimeIntegrator
    base.dmech.TimeIntegrator = OneNewtonMumpsIntegrator
    try:
        base.run()
    except RuntimeError as exc:
        if str(exc) != "Healthy vascular baseline integration failed.":
            exception_file(exc)
            raise
    finally:
        base.dmech.TimeIntegrator = original


if __name__ == "__main__":
    main()
