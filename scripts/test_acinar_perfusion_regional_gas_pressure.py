#!/usr/bin/env python3
"""Run the four regional gas-pressure validation states.

The project-level adapter configures either one existing global surface-
pressure operator or four existing operators on a second regional facet
MeshFunction.  dolfin_mech and the FE formulation are not modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import postprocess_regional_gas_expansion as gas_expansion
import postprocess_regional_perfusion as regional
import test_acinar_perfusion_vascular_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "regional_pg_validation"
ACINAR_METADATA = ROOT / "results" / "acinar_partition" / "periodic_acinar_metadata.json"
ACINAR_SUMMARY = ROOT / "results" / "acinar_partition" / "periodic_acinar_summary.json"
HEALTHY_FLOWS = {
    "Pa0": 0.25, "Pa1": 0.25, "Pa2": 0.25, "Pa3": 0.25,
    "V002": 0.50, "V006": 0.50,
}
CASES = (
    ("regional_zero", "regional", (0.0, 0.0, 0.0, 0.0)),
    ("global_uniform_002", "global", (0.02, 0.02, 0.02, 0.02)),
    ("regional_uniform_002", "regional", (0.02, 0.02, 0.02, 0.02)),
    ("A2_only_002", "regional", (0.0, 0.0, 0.02, 0.0)),
)


def complete_case(case_name, case_dir):
    required = (
        f"{case_name}.xdmf", f"{case_name}.h5", "final_fields.vtu",
        "run_config.json", "diagnostics.json", "patch_pressures.json",
        "mass_balance.json", "increments.sta", "regional_gas_facets.json",
    )
    return all((case_dir / name).exists() for name in required)


def solve_case(case_name, mode, pg):
    case_dir = RESULTS_ROOT / case_name
    if complete_case(case_name, case_dir):
        print(f"Reusing complete regional-pg case: {case_dir}")
        return case_dir
    if case_dir.exists() and any(case_dir.iterdir()):
        raise RuntimeError(f"Partial case directory; refusing to overwrite: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)
    baseline.RESULTS = case_dir
    baseline.RESULT_BASE = case_dir / case_name
    baseline.Q_TOTAL = 1.0
    baseline.TARGET_FLOWS = dict(HEALTHY_FLOWS)
    baseline.MACROSCOPIC_DISPLACEMENT_GRADIENT = ((0.0, 0.0), (0.0, 0.0))
    baseline.GAS_PRESSURE_MODE = mode
    # Global reference retains the original tag-0 measure and one original
    # surface-pressure operator. Regional cases use the second marker field.
    baseline.USE_REGIONAL_GAS_BOUNDARY_MEASURE_FOR_GLOBAL = False
    baseline.GLOBAL_GAS_PRESSURE = float(pg[0]) if mode == "global" else 0.0
    baseline.REGIONAL_GAS_PRESSURES = tuple(float(value) for value in pg)
    baseline.run()
    config_path = case_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = {
        "case_name": case_name,
        "purpose": "regional acinar gas-pressure validation",
        "gas_pressure_mode": mode,
        "pg": list(pg),
        "gas_pressure_units": "model units; not physiologically calibrated",
        "Ubar": [[0.0, 0.0], [0.0, 0.0]],
        "healthy_vascular_loading": HEALTHY_FLOWS,
    }
    config_path.write_text(json.dumps(config, indent=2))
    return case_dir


def postprocess_perfusion(case_name, case_dir):
    regional_path = case_dir / "regional_perfusion.json"
    if regional_path.exists():
        print(f"Reusing regional perfusion metrics: {regional_path}")
        return
    regional.run(
        result_dir=case_dir,
        solution_xdmf=(case_dir / f"{case_name}.xdmf").resolve(),
        final_vtu=(case_dir / "final_fields.vtu").resolve(),
        acinar_metadata_path=ACINAR_METADATA.resolve(),
        acinar_summary_path=ACINAR_SUMMARY.resolve(),
    )


def accepted_reference_gas_areas():
    metadata = json.loads(ACINAR_METADATA.read_text())
    coordinates, topology, fields = gas_expansion.read_final_fields(
        gas_expansion.BASELINE_XDMF
    )
    rows, validation = gas_expansion.regional_gas_areas(
        coordinates, topology, fields["U_tot"], metadata, np.eye(2)
    )
    print(
        "accepted pg=0 gas-area reference: "
        f"area={validation['current_gas_area']:.16e}, "
        f"closure={validation['max_current_loop_closure_residual']:.3e}"
    )
    return {row["acinus_id"]: row["current_gas_area"] for row in rows}


def postprocess_expansion(case_name, case_dir, reference):
    output = case_dir / "regional_gas_expansion.json"
    if output.exists():
        return json.loads(output.read_text())
    return gas_expansion.process_case(
        case_name, case_dir, case_dir / f"{case_name}.xdmf", reference
    )[0]


def assert_uniform_equivalence(reference):
    name_b, name_c = "global_uniform_002", "regional_uniform_002"
    dir_b, dir_c = RESULTS_ROOT / name_b, RESULTS_ROOT / name_c
    expansion_b = postprocess_expansion(name_b, dir_b, reference)
    expansion_c = postprocess_expansion(name_c, dir_c, reference)
    row_b = gas_expansion.case_summary(name_b, dir_b, expansion_b)
    row_c = gas_expansion.case_summary(name_c, dir_c, expansion_c)
    errors = gas_expansion.compare_case_fields(
        name_b, dir_b / f"{name_b}.xdmf", name_c, dir_c / f"{name_c}.xdmf"
    )
    comparison = gas_expansion.comparison_metrics(row_b, row_c, errors)
    (RESULTS_ROOT / "uniform_equivalence_precheck.json").write_text(
        json.dumps(comparison, indent=2)
    )
    max_field_relative = max(value["relative_l2"] for value in errors.values())
    scalar_keys = [*(f"Pi{i}" for i in range(4)), *(f"fP{i}" for i in range(4)),
                   "CV_Pi", "R_hyd", "global_current_wall_area", "mean_displacement",
                   "max_displacement", "min_J", "mean_J", "max_J"]
    max_scalar_relative = max(
        abs(comparison["scalar_relative_differences"][key]["relative"])
        for key in scalar_keys
    )
    tolerance = 1e-7
    print(
        f"global/regional uniform precheck: max field relative L2={max_field_relative:.3e}, "
        f"max scalar relative={max_scalar_relative:.3e}, tolerance={tolerance:.1e}"
    )
    if max(max_field_relative, max_scalar_relative) > tolerance:
        raise RuntimeError(
            "Global-vs-regional uniform equivalence failed; refusing to run A2-only case."
        )


def run():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    reference = accepted_reference_gas_areas()
    # A--C first. The heterogeneous case is gated by the uniform decomposition test.
    for case_name, mode, pg in CASES[:3]:
        case_dir = solve_case(case_name, mode, pg)
        postprocess_perfusion(case_name, case_dir)
        postprocess_expansion(case_name, case_dir, reference)
    assert_uniform_equivalence(reference)
    case_name, mode, pg = CASES[3]
    case_dir = solve_case(case_name, mode, pg)
    postprocess_perfusion(case_name, case_dir)
    postprocess_expansion(case_name, case_dir, reference)
    gas_expansion.main()


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
