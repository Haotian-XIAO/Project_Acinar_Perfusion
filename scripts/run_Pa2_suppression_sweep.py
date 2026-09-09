#!/usr/bin/env python3
"""Run and postprocess the missing compensated Pa2 suppression cases."""

from __future__ import annotations

import json
import os
from pathlib import Path

import postprocess_regional_perfusion as regional
import test_acinar_perfusion_vascular_suppression_Pa2 as suppression


ROOT = Path(__file__).resolve().parents[1]
ACINAR_METADATA = ROOT / "results" / "acinar_partition" / "periodic_acinar_metadata.json"
ACINAR_SUMMARY = ROOT / "results" / "acinar_partition" / "periodic_acinar_summary.json"
CASES = {
    1.00: ROOT / "results" / "vascular_baseline",
    0.75: ROOT / "results" / "vascular_suppression_Pa2_075",
    0.50: ROOT / "results" / "vascular_suppression_Pa2_050",
    0.25: ROOT / "results" / "vascular_suppression_Pa2_025",
    0.00: ROOT / "results" / "vascular_suppression_Pa2_000",
}

REQUIRED_CASE_FILES = (
    "mass_balance.json", "diagnostics.json", "patch_pressures.json",
    "run_config.json", "increments.sta", "final_fields.vtu",
)


def complete_case(case_dir):
    xdmf = case_dir / f"{case_dir.name}.xdmf"
    h5 = case_dir / f"{case_dir.name}.h5"
    return all((case_dir / name).exists() for name in REQUIRED_CASE_FILES) and xdmf.exists() and h5.exists()


def postprocess_case(alpha, case_dir):
    xdmf = case_dir / f"{case_dir.name}.xdmf"
    final_vtu = case_dir / "final_fields.vtu"
    return regional.run(
        result_dir=case_dir,
        solution_xdmf=xdmf.resolve(),
        final_vtu=final_vtu.resolve(),
        acinar_metadata_path=ACINAR_METADATA.resolve(),
        acinar_summary_path=ACINAR_SUMMARY.resolve(),
    )


def run():
    for alpha, case_dir in CASES.items():
        case_dir.mkdir(parents=True, exist_ok=True)
        if alpha in (1.0, 0.0):
            if not complete_case(case_dir):
                raise FileNotFoundError(f"Accepted endpoint case is incomplete: {case_dir}")
            print(f"Reusing retained endpoint alpha={alpha:.2f}: {case_dir}")
        else:
            existing = list(case_dir.iterdir())
            if existing and not complete_case(case_dir):
                raise RuntimeError(f"Intermediate case directory is partial; refusing to overwrite: {case_dir}")
            if not complete_case(case_dir):
                print(f"Solving alpha={alpha:.2f}: {case_dir}")
                suppression.run_case(alpha_pa2=alpha, results=case_dir)
            else:
                print(f"Reusing existing complete alpha={alpha:.2f}: {case_dir}")

        regional_json = case_dir / "regional_perfusion.json"
        if regional_json.exists():
            print(f"Reusing regional metrics alpha={alpha:.2f}")
        else:
            postprocess_case(alpha, case_dir)

        # Keep the alpha explicitly available even if an older accepted result
        # predates the parameterized case adapter.
        config_path = case_dir / "run_config.json"
        config = json.loads(config_path.read_text())
        config.setdefault("experiment", {})["alpha_Pa2"] = alpha
        config_path.write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
