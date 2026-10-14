"""Opt-in linear-solver, FOI-projection and diagnostics options for the vascular drivers.

Imported only when one of the corresponding command-line options is given, so
the default driver path is unchanged.  Nothing here modifies dolfin_mech files:
solver objects are wrapped per instance, and the FOI projection method is
replaced on the class for the duration of one ``run()``.

Fail loud: dolfin_mech's ``NonlinearSolver.linear_solve`` discards linear-solver
exceptions with a bare ``except`` and retries with a smaller time step.  With
these options the original exception is recorded and the run is aborted after
the nonlinear solve returns, instead of being retried.
"""
from __future__ import annotations

import json
import resource
import sys
import time
import traceback
from pathlib import Path

import dolfin
import dolfin_mech as dmech
from petsc4py import PETSc

# MUMPS 5 INFOG/RINFOG entries recorded after analysis and factorisation (MUMPS user guide numbering).
INFOG_LABELS = {
    1: "status (0 = ok, <0 error)",
    2: "additional error information",
    3: "estimated real workspace for factors (negative: millions of entries)",
    7: "ordering actually used",
    9: "real workspace allocated for factors (negative: millions of entries)",
    16: "estimated in-core factorisation memory, MB, max over processes",
    17: "estimated in-core factorisation memory, MB, sum over processes",
    18: "memory allocated during factorisation, MB, max over processes",
    19: "memory allocated during factorisation, MB, sum over processes",
    20: "estimated number of entries in factors (negative: millions)",
    21: "memory effectively used during factorisation, MB, max over processes",
    22: "memory effectively used during factorisation, MB, sum over processes",
    28: "number of null pivots",
    29: "effective number of entries in factors (negative: millions)",
}
RINFOG_LABELS = {
    1: "estimated flops for elimination",
    2: "flops for assembly",
    3: "flops for elimination",
}
ICNTL_RECORDED = (4, 7, 13, 14, 23, 28, 29, 33, 35)


def peak_rss_bytes():
    # macOS reports ru_maxrss in bytes.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def iterative_foi_projection(self):
    """CG/Jacobi L2 projection of one field of interest (same as the G60 script)."""
    dolfin.project(v=self.expr, V=self.fs, function=self.func,
                   form_compiler_parameters=self.form_compiler_parameters,
                   solver_type="cg", preconditioner_type="jacobi")


class AnalysisOnlyFinished(Exception):
    """Raised inside the linear solve to stop after a MUMPS analysis-only pass."""


class Diagnostics:
    """Phase timings, peak RSS, Newton residual history and MUMPS statistics."""

    def __init__(self, results_dir, options):
        self.path = Path(results_dir) / "solver_diagnostics.json"
        self.start = time.perf_counter()
        self.data = {"options": dict(options), "phases": [], "foi_projection": {"calls": 0, "seconds": 0.0},
                     "assembly": [], "linear_solves": [], "nonlinear_solves": [], "mumps": [],
                     "petsc_options": {}, "linear_solver_error": None}
        self.write()

    def write(self):
        self.data["peak_rss_bytes"] = peak_rss_bytes()
        self.data["elapsed_seconds"] = time.perf_counter() - self.start
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def mark(self, name):
        self.data["phases"].append({"name": name, "t_seconds": time.perf_counter() - self.start,
                                    "peak_rss_bytes": peak_rss_bytes()})
        print(f"[solver-diagnostics] phase {name} t={self.data['phases'][-1]['t_seconds']:.1f}s "
              f"peak_rss={self.data['phases'][-1]['peak_rss_bytes'] / 2**30:.2f}GiB", flush=True)
        self.write()


def mumps_statistics(factor):
    """Read INFOG/RINFOG/ICNTL from a MUMPS factor matrix; errors are recorded, not hidden."""
    stats = {"infog": {}, "rinfog": {}, "icntl": {}, "errors": []}
    for table, getter, keys in (("infog", factor.getMumpsInfog, INFOG_LABELS),
                                ("rinfog", factor.getMumpsRinfog, RINFOG_LABELS),
                                ("icntl", factor.getMumpsIcntl, ICNTL_RECORDED)):
        for key in keys:
            try:
                stats[table][str(key)] = getter(key)
            except PETSc.Error as exc:
                stats[table][str(key)] = None
                stats["errors"].append(f"{table}({key}): {exc!r}")
    return stats


class LinearSolverProxy:
    """Stands in for ``NonlinearSolver.linear_solver``: times, records and never hides errors."""

    def __init__(self, solver, inner, options, diagnostics):
        self._solver = solver
        self._inner = inner
        self._options = options
        self._diagnostics = diagnostics
        self.error = None
        self.analysis_only_finished = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _mumps_factor(self):
        return self._inner.ksp().getPC().getFactorMatrix()

    def _analysis_only(self, x, b):
        """MUMPS analysis without factorisation.

        petsc4py 3.21 does not implement Mat.factorSymbolicLU, so the analysis is run through
        the normal solve with ICNTL(23) = 1 MB: MUMPS completes the analysis, then refuses the
        factorisation with INFOG(1) = -19 before allocating it.  Any other outcome is an error.
        """
        PETSc.Options()["mat_mumps_icntl_23"] = 1
        start = time.perf_counter()
        try:
            self._inner.solve(x, b)
        except RuntimeError as exc:
            lines = [line for line in str(exc).splitlines() if "Reason:" in line]
            outcome = f"{type(exc).__name__}: {lines[0].strip() if lines else str(exc)[:300]}"
        else:
            raise RuntimeError("MUMPS analysis-only pass: solve unexpectedly succeeded with ICNTL(23)=1.")
        seconds = time.perf_counter() - start
        stats = mumps_statistics(self._mumps_factor())
        if stats["infog"].get("1") != -19:
            raise RuntimeError(f"MUMPS analysis-only pass: expected INFOG(1) = -19, got {stats['infog'].get('1')}; "
                               f"solve error was {outcome}")
        A = self._solver.jac_mat.mat()
        self._diagnostics.data["mumps_analysis_only"] = {
            "method": "solve with ICNTL(23)=1 MB; analysis completed, factorisation refused (INFOG(1)=-19)",
            "analysis_seconds": seconds, "matrix_size": [int(v) for v in A.getSize()],
            "matrix_nnz": int(A.getInfo()["nz_used"]), "expected_solve_error": outcome, **stats}
        self._diagnostics.mark("mumps_analysis_done")
        raise AnalysisOnlyFinished("MUMPS analysis-only pass finished; no factorisation performed.")

    def solve(self, x, b):
        start = time.perf_counter()
        try:
            if self._options["linear_solver"] == "mumps":
                A = self._solver.jac_mat.mat()
                # Operators are re-attached explicitly: the NonlinearSolver sets them once,
                # before the first assembly creates the matrix layout.
                self._inner.ksp().setOperators(A)
                if self._options.get("mumps_analysis_only"):
                    self._analysis_only(x, b)
            result = self._inner.solve(x, b)
        except AnalysisOnlyFinished:
            self.analysis_only_finished = True
            raise
        except BaseException as exc:
            record = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                      "k_iter": getattr(self._solver, "k_iter", None)}
            if self._options["linear_solver"] == "mumps":
                try:
                    record["mumps"] = mumps_statistics(self._mumps_factor())
                except BaseException as stats_exc:
                    record["mumps"] = None
                    record["mumps_statistics_error"] = repr(stats_exc)
            self.error = record
            self._diagnostics.data["linear_solver_error"] = record
            self._diagnostics.write()
            print("[solver-diagnostics] LINEAR SOLVER ERROR (original exception follows):\n" + record["traceback"],
                  file=sys.stderr, flush=True)
            raise
        seconds = time.perf_counter() - start
        self._diagnostics.data["linear_solves"].append(
            {"k_iter": getattr(self._solver, "k_iter", None), "seconds": seconds, "peak_rss_bytes": peak_rss_bytes()})
        if self._options["linear_solver"] == "mumps":
            self._diagnostics.data["mumps"].append(
                {"k_iter": getattr(self._solver, "k_iter", None), **mumps_statistics(self._mumps_factor())})
        self._diagnostics.write()
        return result


def install(problem, solver, options, diagnostics):
    """Wrap one NonlinearSolver instance; returns the linear-solver proxy."""
    proxy = LinearSolverProxy(solver, solver.linear_solver, options, diagnostics)
    solver.linear_solver = proxy

    original_assemble = solver.assemble_linear_system

    def assemble_linear_system():
        start = time.perf_counter()
        result = original_assemble()
        diagnostics.data["assembly"].append({
            "k_iter": solver.k_iter, "seconds": time.perf_counter() - start,
            "res_norm": float(solver.res_norm) if hasattr(solver, "res_norm") else None,
            "returned": None if result is None else bool(result), "peak_rss_bytes": peak_rss_bytes()})
        diagnostics.write()
        return result

    solver.assemble_linear_system = assemble_linear_system

    original_solve = solver.solve

    def solve(*args, **kwargs):
        start = time.perf_counter()
        success, n_iter = original_solve(*args, **kwargs)
        diagnostics.data["nonlinear_solves"].append({
            "args": [repr(a) for a in args], "success": bool(success), "n_iter": int(n_iter),
            "seconds": time.perf_counter() - start, "peak_rss_bytes": peak_rss_bytes()})
        diagnostics.write()
        if proxy.analysis_only_finished:
            diagnostics.mark("stopped_after_mumps_analysis")
            raise SystemExit(0)
        if proxy.error is not None:
            raise RuntimeError("Linear solver failed (dolfin_mech would retry with a smaller time step); "
                               f"original error: {proxy.error['type']}: {proxy.error['message']}")
        return success, n_iter

    solver.solve = solve
    if options["linear_solver"] == "mumps":
        diagnostics.data["petsc_options"] = PETSc.Options().getAll()
        diagnostics.write()
    return proxy


class FoiProjection:
    """Context manager selecting (and, with diagnostics, timing) FOI L2 projections."""

    def __init__(self, mode, diagnostics):
        if mode not in ("direct", "iterative"):
            raise ValueError(f"Unknown FOI projection mode {mode!r}.")
        self.mode = mode
        self.diagnostics = diagnostics
        self.original = None

    def __enter__(self):
        self.original = dmech.FOI.update_project
        base = iterative_foi_projection if self.mode == "iterative" else self.original
        diagnostics = self.diagnostics

        def update_project(foi):
            start = time.perf_counter()
            base(foi)
            if diagnostics is not None:
                diagnostics.data["foi_projection"]["calls"] += 1
                diagnostics.data["foi_projection"]["seconds"] += time.perf_counter() - start

        dmech.FOI.update_project = update_project
        return self

    def __exit__(self, *exc):
        dmech.FOI.update_project = self.original
        return False


def start_petsc_log():
    PETSc.Log.begin()


def write_petsc_log(results_dir):
    viewer = PETSc.Viewer().createASCII(str(Path(results_dir) / "petsc_log.txt"))
    PETSc.Log.view(viewer)
    viewer.destroy()
