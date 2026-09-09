#!/usr/bin/env python3
"""Run the healthy-flow uniaxial macroscopic deformation sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path

import postprocess_regional_perfusion as regional
import test_acinar_perfusion_vascular_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "results" / "deformation_healthy_sweep"
REFERENCE_RESULTS = ROOT / "results" / "vascular_baseline"
ACINAR_METADATA = ROOT / "results" / "acinar_partition" / "periodic_acinar_metadata.json"
ACINAR_SUMMARY = ROOT / "results" / "acinar_partition" / "periodic_acinar_summary.json"
SWEEP_VALUES = (0.00, 0.05, 0.10, 0.15, 0.20)
HEALTHY_FLOWS = {
    "Pa0": 0.25, "Pa1": 0.25, "Pa2": 0.25, "Pa3": 0.25,
    "V002": 0.50, "V006": 0.50,
}


def case_name(stretch):
    return f"stretch_{int(round(100 * stretch)):03d}"


def result_basename(case_dir, stretch):
    return "vascular_baseline" if stretch == 0.0 else case_dir.name


def case_definition(stretch, case_dir):
    return {
        "case_id": case_dir.name,
        "deformation_parameter": "s",
        "deformation_family": "uniaxial macroscopic x-extension with transverse displacement gradient fixed",
        "s": stretch,
        "F_bar": [[1.0 + stretch, 0.0], [0.0, 1.0]],
        "U_bar": [[stretch, 0.0], [0.0, 0.0]],
        "healthy_vascular_loading": HEALTHY_FLOWS,
        "Q_total": 1.0,
        "alpha_Pa2": 1.0,
    }


def link_reference_case(case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "vascular_baseline.xdmf", "vascular_baseline.h5", "final_fields.vtu",
        "run_config.json", "diagnostics.json", "patch_pressures.json",
        "mass_balance.json", "increments.sta", "regional_perfusion.json",
        "regional_perfusion.csv", "fe_acinar_mapping.json",
    )
    for name in names:
        source = REFERENCE_RESULTS / name
        destination = case_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(Path(os.path.relpath(source, case_dir)))
    (case_dir / "case_definition.json").write_text(
        json.dumps(case_definition(0.0, case_dir), indent=2)
    )


def complete_case(case_dir, stretch):
    basename = result_basename(case_dir, stretch)
    names = (
        f"{basename}.xdmf", f"{basename}.h5", "final_fields.vtu",
        "run_config.json", "diagnostics.json", "patch_pressures.json",
        "mass_balance.json", "increments.sta",
    )
    return all((case_dir / name).exists() for name in names)


def solve_case(stretch, case_dir):
    definition = case_definition(stretch, case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_definition.json").write_text(json.dumps(definition, indent=2))
    baseline.RESULTS = case_dir
    baseline.RESULT_BASE = case_dir / case_dir.name
    baseline.Q_TOTAL = 1.0
    baseline.TARGET_FLOWS = dict(HEALTHY_FLOWS)
    baseline.MACROSCOPIC_DISPLACEMENT_GRADIENT = (
        (stretch, 0.0),
        (0.0, 0.0),
    )
    baseline.run()
    config_path = case_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = definition
    config_path.write_text(json.dumps(config, indent=2))


def postprocess_case(stretch, case_dir):
    basename = result_basename(case_dir, stretch)
    regional.run(
        result_dir=case_dir,
        solution_xdmf=(case_dir / f"{basename}.xdmf").resolve(),
        final_vtu=(case_dir / "final_fields.vtu").resolve(),
        acinar_metadata_path=ACINAR_METADATA.resolve(),
        acinar_summary_path=ACINAR_SUMMARY.resolve(),
    )


def run():
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    for stretch in SWEEP_VALUES:
        case_dir = SWEEP_ROOT / case_name(stretch)
        if stretch == 0.0:
            print(f"Linking accepted healthy reference into {case_dir}")
            link_reference_case(case_dir)
        elif not complete_case(case_dir, stretch):
            if case_dir.exists() and any(case_dir.iterdir()):
                raise RuntimeError(f"Partial case directory; refusing to overwrite: {case_dir}")
            print(f"Solving deformation s={stretch:.2f}: {case_dir}")
            solve_case(stretch, case_dir)
        else:
            print(f"Reusing complete deformation s={stretch:.2f}: {case_dir}")

        if not complete_case(case_dir, stretch):
            raise RuntimeError(f"Incomplete deformation case: {case_dir}")
        if not (case_dir / "regional_perfusion.json").exists():
            postprocess_case(stretch, case_dir)
        else:
            print(f"Reusing regional metrics s={stretch:.2f}")


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
