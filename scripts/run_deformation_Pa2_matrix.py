#!/usr/bin/env python3
"""Run the missing healthy-flow deformation x Pa2-suppression states."""

from __future__ import annotations

import json
import os
from pathlib import Path

import postprocess_regional_perfusion as regional
import test_acinar_perfusion_vascular_suppression_Pa2 as vascular_case


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = ROOT / "results" / "deformation_Pa2_matrix"
ACINAR_METADATA = ROOT / "results" / "acinar_partition" / "periodic_acinar_metadata.json"
ACINAR_SUMMARY = ROOT / "results" / "acinar_partition" / "periodic_acinar_summary.json"
DEFORMATIONS = (0.00, 0.10, 0.20)
ALPHAS = (1.00, 0.75, 0.50, 0.25, 0.00)


def case_name(s, alpha):
    return f"s_{int(round(100*s)):03d}_alpha_{int(round(100*alpha)):03d}"


def flows(alpha):
    return vascular_case.case_flows(alpha)


def definition(s, alpha, case_dir):
    return {
        "case_id": case_dir.name,
        "deformation_parameter": "s",
        "deformation_family": "uniaxial macroscopic x-extension with transverse displacement gradient fixed",
        "s": s,
        "U_bar": [[s, 0.0], [0.0, 0.0]],
        "F_bar": [[1.0+s, 0.0], [0.0, 1.0]],
        "alpha_Pa2": alpha,
        "target_flows": flows(alpha),
        "Q_total": 1.0,
        "compensation_interpretation": "Pa2 attenuation and equal compensation are prescribed inputs, not predicted collateral recruitment.",
    }


def complete(case_dir):
    names = (
        f"{case_dir.name}.xdmf", f"{case_dir.name}.h5", "final_fields.vtu",
        "run_config.json", "diagnostics.json", "patch_pressures.json",
        "mass_balance.json", "increments.sta",
    )
    return all((case_dir / name).exists() for name in names)


def solve_case(s, alpha, case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    case_def = definition(s, alpha, case_dir)
    (case_dir / "case_definition.json").write_text(json.dumps(case_def, indent=2))
    vascular_case.accepted_baseline.MACROSCOPIC_DISPLACEMENT_GRADIENT = (
        (s, 0.0), (0.0, 0.0)
    )
    vascular_case.run_case(alpha_pa2=alpha, results=case_dir)
    config_path = case_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = case_def
    config_path.write_text(json.dumps(config, indent=2))


def postprocess(case_dir):
    regional.run(
        result_dir=case_dir,
        solution_xdmf=(case_dir / f"{case_dir.name}.xdmf").resolve(),
        final_vtu=(case_dir / "final_fields.vtu").resolve(),
        acinar_metadata_path=ACINAR_METADATA.resolve(),
        acinar_summary_path=ACINAR_SUMMARY.resolve(),
    )


def run():
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    missing = [(s, alpha) for s in DEFORMATIONS for alpha in ALPHAS
               if not (s == 0.0 or alpha == 1.0)]
    for s, alpha in missing:
        case_dir = MATRIX_ROOT / case_name(s, alpha)
        if complete(case_dir):
            print(f"Reusing complete state s={s:.2f}, alpha={alpha:.2f}")
            continue
        if case_dir.exists() and any(case_dir.iterdir()):
            raise RuntimeError(f"Partial matrix case; refusing to overwrite: {case_dir}")
        print(f"Solving state s={s:.2f}, alpha={alpha:.2f}: {case_dir}")
        solve_case(s, alpha, case_dir)
        if not complete(case_dir):
            raise RuntimeError(f"Incomplete matrix solution: {case_dir}")
        postprocess(case_dir)


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
