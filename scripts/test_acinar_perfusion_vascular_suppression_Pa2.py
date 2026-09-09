#!/usr/bin/env python3
"""Fixed-total-flow complete Pa2 suppression using the accepted baseline solver.

This is a deliberately thin case adapter.  The FE problem, Darcy operator,
material model, time integration, and output implementation remain those in
``test_acinar_perfusion_vascular_baseline``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import test_acinar_perfusion_vascular_baseline as accepted_baseline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "vascular_suppression_Pa2_000"
TARGET_FLOWS = {}
CASE_DEFINITION = {}


def case_flows(alpha_pa2):
    alpha_pa2 = float(alpha_pa2)
    if not 0.0 <= alpha_pa2 <= 1.0:
        raise ValueError("alpha_Pa2 must lie in [0, 1].")
    delta = 0.25 * (1.0 - alpha_pa2)
    compensated = 0.25 + delta / 3.0
    return {
        "Pa0": compensated,
        "Pa1": compensated,
        "Pa2": 0.25 * alpha_pa2,
        "Pa3": compensated,
        "V002": 0.50,
        "V006": 0.50,
    }


def case_definition(alpha_pa2, results):
    return {
        "case_id": results.name,
        "suppressed_arterial_source": "Pa2",
        "alpha_Pa2": float(alpha_pa2),
        "Q_total": 1.0,
        "target_flows": case_flows(alpha_pa2),
        "loading_interpretation": (
            "Pa2 attenuation and equal compensation by Pa0, Pa1, and Pa3 are "
            "prescribed inputs; they are not predicted collateral recruitment."
        ),
    }


def run_case(alpha_pa2=0.0, results=RESULTS):
    global RESULTS, TARGET_FLOWS
    RESULTS = Path(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    TARGET_FLOWS = case_flows(alpha_pa2)
    definition = case_definition(alpha_pa2, results)
    (RESULTS / "case_definition.json").write_text(json.dumps(definition, indent=2))

    accepted_baseline.RESULTS = RESULTS
    accepted_baseline.RESULT_BASE = RESULTS / RESULTS.name
    accepted_baseline.Q_TOTAL = 1.0
    accepted_baseline.TARGET_FLOWS = dict(TARGET_FLOWS)
    accepted_baseline.run()

    run_config_path = RESULTS / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["experiment"] = definition
    run_config_path.write_text(json.dumps(run_config, indent=2))


def run():
    """Backward-compatible complete Pa2 suppression endpoint (alpha=0)."""
    run_case(alpha_pa2=0.0, results=RESULTS)


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
