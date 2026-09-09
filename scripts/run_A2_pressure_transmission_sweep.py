#!/usr/bin/env python3
"""Run the fixed-open-pressure A2 airway-transmission sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import postprocess_regional_gas_expansion as gas
import postprocess_regional_perfusion as regional
import test_acinar_perfusion_vascular_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "results" / "A2_pressure_transmission_sweep"
REFERENCE_CASE = ROOT / "results" / "regional_pg_validation" / "regional_uniform_002"
ACINAR_METADATA = ROOT / "results" / "acinar_partition" / "periodic_acinar_metadata.json"
ACINAR_SUMMARY = ROOT / "results" / "acinar_partition" / "periodic_acinar_summary.json"
P_OPEN = 0.20
BETA_VALUES = (1.00, 0.75, 0.50, 0.25, 0.00)
HEALTHY_FLOWS = {
    "Pa0": 0.25, "Pa1": 0.25, "Pa2": 0.25, "Pa3": 0.25,
    "V002": 0.50, "V006": 0.50,
}


def beta_name(beta):
    return f"beta_{int(round(100 * beta)):03d}"


def pressure_vector(beta):
    return (P_OPEN, P_OPEN, P_OPEN * beta, P_OPEN)


def link_reference_case(case_dir):
    """Create a local retained-result view without copying the large HDF5 file."""
    case_dir.mkdir(parents=True, exist_ok=True)
    source_names = (
        "regional_uniform_002.xdmf", "regional_uniform_002.h5", "final_fields.vtu",
        "run_config.json", "diagnostics.json", "patch_pressures.json", "mass_balance.json",
        "increments.sta", "regional_perfusion.json", "regional_perfusion.csv",
        "fe_acinar_mapping.json", "regional_gas_facets.json", "regional_gas_expansion.json",
    )
    for name in source_names:
        source = REFERENCE_CASE / name
        destination = case_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(Path(os.path.relpath(source, case_dir)))
    for alias, source_name in (("beta_100.xdmf", "regional_uniform_002.xdmf"),
                               ("beta_100.h5", "regional_uniform_002.h5")):
        destination = case_dir / alias
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(Path(os.path.relpath(case_dir / source_name, case_dir)))
    definition = {
        "case_name": beta_name(1.0),
        "beta_A2_pressure_transmission": 1.0,
        "p_open": P_OPEN,
        "regional_gas_pressures": list(pressure_vector(1.0)),
        "interpretation": "regional inspiratory alveolar-pressure transmission factor; not ventilation fraction",
        "reference_source": str(REFERENCE_CASE.resolve()),
    }
    (case_dir / "case_definition.json").write_text(json.dumps(definition, indent=2))


def complete_case(case_name, case_dir):
    return all((case_dir / filename).exists() for filename in (
        f"{case_name}.xdmf", f"{case_name}.h5", "final_fields.vtu", "run_config.json",
        "diagnostics.json", "patch_pressures.json", "mass_balance.json", "increments.sta",
        "regional_perfusion.json", "regional_perfusion.csv", "regional_gas_expansion.json",
    ))


def solve_case(beta):
    case_name = beta_name(beta)
    case_dir = SWEEP_ROOT / case_name
    if beta == 1.0:
        if not complete_case(case_name, case_dir):
            if case_dir.exists() and any(case_dir.iterdir()):
                raise RuntimeError(f"Partial beta=1 directory; refusing overwrite: {case_dir}")
            link_reference_case(case_dir)
        return case_dir
    if complete_case(case_name, case_dir):
        print(f"Reusing complete beta={beta:.2f}: {case_dir}")
        return case_dir
    if case_dir.exists() and any(case_dir.iterdir()):
        raise RuntimeError(f"Partial beta directory; refusing overwrite: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)
    pg = pressure_vector(beta)
    baseline.RESULTS = case_dir
    baseline.RESULT_BASE = case_dir / case_name
    baseline.Q_TOTAL = 1.0
    baseline.TARGET_FLOWS = dict(HEALTHY_FLOWS)
    baseline.MACROSCOPIC_DISPLACEMENT_GRADIENT = ((0.0, 0.0), (0.0, 0.0))
    baseline.GAS_PRESSURE_MODE = "regional"
    baseline.USE_REGIONAL_GAS_BOUNDARY_MEASURE_FOR_GLOBAL = False
    baseline.GLOBAL_GAS_PRESSURE = 0.0
    baseline.REGIONAL_GAS_PRESSURES = tuple(pg)
    baseline.run()
    config_path = case_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = {
        "case_name": case_name,
        "purpose": "fixed-open-pressure A2 airway-pressure transmission sweep",
        "beta_A2_pressure_transmission": beta,
        "p_open": P_OPEN,
        "regional_gas_pressures": list(pg),
        "pressure_units": "model units; not physiologically calibrated",
        "Ubar": [[0.0, 0.0], [0.0, 0.0]],
        "healthy_vascular_loading": HEALTHY_FLOWS,
    }
    config_path.write_text(json.dumps(config, indent=2))
    regional.run(
        result_dir=case_dir,
        solution_xdmf=(case_dir / f"{case_name}.xdmf").resolve(),
        final_vtu=(case_dir / "final_fields.vtu").resolve(),
        acinar_metadata_path=ACINAR_METADATA.resolve(),
        acinar_summary_path=ACINAR_SUMMARY.resolve(),
    )
    reference_metadata = json.loads(gas.ACINAR_METADATA.read_text())
    bx, bt, bf = gas.read_final_fields(gas.BASELINE_XDMF)
    reference_rows, _ = gas.regional_gas_areas(bx, bt, bf["U_tot"], reference_metadata, np.eye(2))
    reference_areas = {row["acinus_id"]: row["current_gas_area"] for row in reference_rows}
    gas.process_case(case_name, case_dir, case_dir / f"{case_name}.xdmf", reference_areas)
    (case_dir / "case_definition.json").write_text(json.dumps({
        "case_name": case_name, "beta_A2_pressure_transmission": beta,
        "p_open": P_OPEN, "regional_gas_pressures": list(pg),
        "interpretation": "regional inspiratory alveolar-pressure transmission factor; not ventilation fraction",
    }, indent=2))
    return case_dir


def run():
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    for beta in BETA_VALUES:
        solve_case(beta)
    print("A2 pressure-transmission cases are ready for postprocessing.")


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
