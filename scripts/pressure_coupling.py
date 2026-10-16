"""Pressure-driven vascular coupling (opt-in) for the vascular driver.

Each vascular support k (tags 3-8) exchanges liquid with a reservoir at
pressure P_k through a feeding conductance G_k:

    s_k = gamma_k (P_k(t) - p_tot)  on Omega_k,   gamma_k = G_k / A_k,

added to the liquid equation as  -int_{Omega_k} s_k v dX  (reference measure,
the same dx(tag) as the prescribed-flow sources).  A_k is the assembled
reference area of the support; G_k does not depend on the deformation
(feeding vessels are outside the model).  P_k(t) is ramped from 0 to P_k over
the load increments, like the gas pressure.  p_tot is the installed Darcy
operator's pl_tot, so the exchange makes no assumption on K or porosity.

The Robin terms fix the pressure level, so the zero-mean gauge of dolfin_mech
is replaced by a pin of its multiplier (lambda_test * lambda / V0); the mixed
space, the periodic map and the checkpoint format are unchanged.

The exchange flows Q_k = G_k (P_k - p_hat_k) (p_hat_k: reference-area mean of
p_tot on Omega_k; positive = into the tissue) are outputs.  Testing the liquid
equation with v = 1 leaves sum_k Q_k, which is linear in the unknowns: it is
zero up to the linear-solver residual after every Newton solve.

Configurations (all values are resolved and recorded before the problem build):
  matched  P_k = p_hat_k^ref + Q_k^ref / G_k with G_k = eps * G_{a|v}(phi, s_a)
  common   P_a, P_v, G_a, G_v from a feeding-resistance fraction phi and the
           arterial share s_a of the feeding drop, at the reference level
           (phi and s_a have no defaults: both are uncalibrated sensitivity parameters)
  explicit P_a, P_v, G_a, G_v given directly
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import dolfin
import dolfin_mech as dmech
import numpy as np

SUPPORTS = ("Pa0", "Pa1", "Pa2", "Pa3", "V002", "V006")
# |sum_k Q_k| / Q_a^tot after the solve; the v = 1 component of the liquid
# residual is linear, so only the linear-solver residual is left.
CONSERVATION_TOLERANCE = 1e-10
REFERENCE_FILES = ("mass_balance.json", "patch_pressures.json", "run_config.json")
MODELLING_NOTES = [
    "exchange per reference support area (same dx(tag) as the prescribed-flow sources)",
    "feeding conductances G_k do not depend on the deformation (feeding vessels are outside the model)",
    "reservoir pressures P_k are ramped linearly from 0 over the load increments, like the gas pressure",
    "each support has a private conductance to an ideal reservoir (no shared resistance)",
]


# ----------------------------------------------------------------- CLI -----

def add_arguments(parser):
    group = parser.add_argument_group("vascular coupling (opt-in; default: prescribed flows)")
    group.add_argument("--vascular-coupling", choices=("flow", "pressure"), default="flow",
                       help="flow: prescribed support flows (default, unchanged path); "
                            "pressure: supports exchange with reservoir pressures through feeding conductances")
    group.add_argument("--exchange-config", choices=("matched", "common", "explicit"),
                       help="pressure mode: matched per-support (regression), common reservoirs, or explicit values")
    group.add_argument("--reference-run", type=Path,
                       help="matched/common: prescribed-flow run directory (repository-relative) with its "
                            "checkpoint, mass_balance.json, patch_pressures.json and run_config.json")
    group.add_argument("--feeding-fraction", type=float,
                       help="matched/common: feeding pressure drop / total drop, 0 < phi < 1 (required, no default)")
    group.add_argument("--arterial-share", type=float,
                       help="matched/common: arterial share s_a of the feeding drop, 0 < s_a < 1 (required, no default)")
    group.add_argument("--conductance-scale", type=float,
                       help="matched: G_k = scale * common-mode conductance (default 1)")
    group.add_argument("--P-a", dest="P_a", type=float, help="explicit: arterial reservoir pressure")
    group.add_argument("--P-v", dest="P_v", type=float, help="explicit: venous reservoir pressure")
    group.add_argument("--G-a", dest="G_a", type=float, help="explicit: feeding conductance of each arterial support")
    group.add_argument("--G-v", dest="G_v", type=float, help="explicit: feeding conductance of each venous support")
    group.add_argument("--support-multiplier", action="append", default=[], metavar="NAME=M",
                       help="common/explicit: G_k <- M * G_k for support NAME (repeatable; M >= 0)")


def _finite(value):
    return value is not None and math.isfinite(value)


def spec_from_args(args, parser):
    """Validated, unresolved request (None for the default prescribed-flow path). Exits on bad combinations."""
    pressure_only = {"--exchange-config": args.exchange_config, "--reference-run": args.reference_run,
                     "--feeding-fraction": args.feeding_fraction, "--arterial-share": args.arterial_share,
                     "--conductance-scale": args.conductance_scale, "--P-a": args.P_a, "--P-v": args.P_v,
                     "--G-a": args.G_a, "--G-v": args.G_v,
                     "--support-multiplier": args.support_multiplier or None}
    given = sorted(name for name, value in pressure_only.items() if value is not None)
    if args.vascular_coupling == "flow":
        if given:
            parser.error(f"{', '.join(given)}: only valid with --vascular-coupling pressure")
        return None
    config = args.exchange_config
    if config is None:
        parser.error("--vascular-coupling pressure requires --exchange-config")
    allowed = {
        "matched": {"--exchange-config", "--reference-run", "--feeding-fraction", "--arterial-share", "--conductance-scale"},
        "common": {"--exchange-config", "--reference-run", "--feeding-fraction", "--arterial-share", "--support-multiplier"},
        "explicit": {"--exchange-config", "--P-a", "--P-v", "--G-a", "--G-v", "--support-multiplier"},
    }[config]
    extra = sorted(set(given) - allowed)
    if extra:
        parser.error(f"--exchange-config {config} does not accept {', '.join(extra)}")
    spec = {"mode": "pressure", "config": config}
    if config in ("matched", "common"):
        if args.reference_run is None:
            parser.error(f"--exchange-config {config} requires --reference-run")
        if args.feeding_fraction is None:
            parser.error(f"--exchange-config {config} requires --feeding-fraction (no default)")
        if args.arterial_share is None:
            parser.error(f"--exchange-config {config} requires --arterial-share (no default)")
        phi = args.feeding_fraction
        s_a = args.arterial_share
        if not (_finite(phi) and 0.0 < phi < 1.0):
            parser.error(f"--feeding-fraction must satisfy 0 < phi < 1, got {phi}")
        if not (_finite(s_a) and 0.0 < s_a < 1.0):
            parser.error(f"--arterial-share must satisfy 0 < s_a < 1, got {s_a}")
        spec.update({"reference_run": str(args.reference_run), "feeding_fraction": phi, "arterial_share": s_a})
        if config == "matched":
            eps = 1.0 if args.conductance_scale is None else args.conductance_scale
            if not (_finite(eps) and eps > 0.0):
                parser.error(f"--conductance-scale must be positive, got {eps}")
            spec["conductance_scale"] = eps
    else:
        missing = [name for name, value in (("--P-a", args.P_a), ("--P-v", args.P_v), ("--G-a", args.G_a), ("--G-v", args.G_v))
                   if value is None]
        if missing:
            parser.error(f"--exchange-config explicit requires {', '.join(missing)}")
        for name, value in (("--P-a", args.P_a), ("--P-v", args.P_v)):
            if not _finite(value):
                parser.error(f"{name} must be finite, got {value}")
        for name, value in (("--G-a", args.G_a), ("--G-v", args.G_v)):
            if not (_finite(value) and value > 0.0):
                parser.error(f"{name} must be positive, got {value}")
        spec["explicit"] = {"P_a": args.P_a, "P_v": args.P_v, "G_a": args.G_a, "G_v": args.G_v}
    multipliers = {}
    for item in args.support_multiplier:
        name, sep, value = item.partition("=")
        if not sep or name not in SUPPORTS:
            parser.error(f"--support-multiplier expects NAME=M with NAME in {SUPPORTS}, got {item!r}")
        if name in multipliers:
            parser.error(f"--support-multiplier given twice for {name}")
        try:
            m = float(value)
        except ValueError:
            parser.error(f"--support-multiplier {item!r}: M is not a number")
        if not (math.isfinite(m) and m >= 0.0):
            parser.error(f"--support-multiplier {item!r}: M must be finite and >= 0")
        multipliers[name] = m
    if config != "matched":
        spec["support_multipliers"] = multipliers
    return spec


# ----------------------------------------------------------- reference -----

def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(spec, root, geometry, legacy_periodic_tolerance, macro_gradient, gas_pressure):
    """Read and check the reference prescribed-flow run (matched/common). Raises on any mismatch."""
    if spec["config"] == "explicit":
        return None
    directory = Path(spec["reference_run"])
    directory = directory if directory.is_absolute() else root / directory
    if not directory.is_dir():
        raise RuntimeError(f"Reference run directory not found: {directory}")
    checkpoints = sorted(directory.glob("*_checkpoint.h5"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Reference run {directory} must hold exactly one *_checkpoint.h5, found {[p.name for p in checkpoints]}")
    files = {name: directory / name for name in REFERENCE_FILES}
    files["checkpoint"] = checkpoints[0]
    files["checkpoint_sidecar"] = checkpoints[0].with_suffix(".json")
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Reference run is missing {missing}")
    config = json.loads(files["run_config.json"].read_text())
    mass = json.loads(files["mass_balance.json"].read_text())
    pressures = json.loads(files["patch_pressures.json"].read_text())
    problems = []
    if "vascular_coupling" in config:
        problems.append("reference is not a prescribed-flow run (run_config.json has vascular_coupling)")
    if config.get("geometry_inputs") != geometry:
        problems.append(f"geometry {config.get('geometry_inputs')} != current {geometry}")
    if bool(config.get("periodic_map", {}).get("legacy_periodic_tolerance")) != bool(legacy_periodic_tolerance):
        problems.append("periodic map (legacy tolerance) differs from the current run")
    reference_gradient = config.get("macroscopic_displacement_gradient")
    if reference_gradient != [list(map(float, row)) for row in macro_gradient]:
        problems.append(f"macroscopic displacement gradient {reference_gradient} != current {[list(r) for r in macro_gradient]}")
    if config.get("gas_pressure_mode") != "global":
        problems.append(f"reference gas pressure mode {config.get('gas_pressure_mode')!r} is not 'global'")
    if config.get("global_gas_pressure") != float(gas_pressure):
        problems.append(f"reference gas pressure {config.get('global_gas_pressure')} != current {float(gas_pressure)}")
    if set(mass.get("patches", {})) != set(SUPPORTS) or set(pressures.get("patch_area_averaged_pressure", {})) != set(SUPPORTS):
        problems.append("reference supports differ from Pa0-Pa3, V002, V006")
    if problems:
        raise RuntimeError(f"Reference run {directory} is not consistent with this run: " + "; ".join(problems))
    supports = {}
    for name in SUPPORTS:
        row = mass["patches"][name]
        sign = 1.0 if row["vascular_type"] == "arterial" else -1.0
        supports[name] = {"vascular_type": row["vascular_type"], "A_k": row["assembled_reference_area"],
                          "Q_k": sign * row["prescribed_integrated_Q"],
                          "p_hat_k": pressures["patch_area_averaged_pressure"][name]}
    return {"run_dir": str(directory.relative_to(root)) if directory.is_relative_to(root) else str(directory),
            "md5": {key: _md5(path) for key, path in files.items()},
            "files": {key: path.name for key, path in files.items()},
            "Q_total": mass["Q_total"], "pa_eff": pressures["pa_eff"], "pv_eff": pressures["pv_eff"],
            "Delta_p_av": pressures["Delta_p_av"], "supports": supports}


# ---------------------------------------------------------- resolution -----

def _common_conductances(reference, phi, s_a):
    supports = reference["supports"]
    q_a = {supports[k]["Q_k"] for k in SUPPORTS if supports[k]["vascular_type"] == "arterial"}
    q_v = {-supports[k]["Q_k"] for k in SUPPORTS if supports[k]["vascular_type"] == "venous"}
    if len(q_a) != 1 or len(q_v) != 1:
        raise RuntimeError(f"Common reservoirs need equal reference flows per support type; got arterial {sorted(q_a)}, venous {sorted(q_v)}")
    q_a, q_v = q_a.pop(), q_v.pop()
    if not (q_a > 0 and q_v > 0):
        raise RuntimeError(f"Reference flows must be positive (arterial {q_a}, venous {q_v})")
    # pa_eff / pv_eff of the reference are flow-weighted patch means; recompute as a consistency check
    pa = sum(supports[k]["Q_k"] * supports[k]["p_hat_k"] for k in SUPPORTS if supports[k]["vascular_type"] == "arterial") / reference["Q_total"]
    pv = sum(-supports[k]["Q_k"] * supports[k]["p_hat_k"] for k in SUPPORTS if supports[k]["vascular_type"] == "venous") / reference["Q_total"]
    for name, recomputed, recorded in (("pa_eff", pa, reference["pa_eff"]), ("pv_eff", pv, reference["pv_eff"])):
        if abs(recomputed - recorded) > 1e-12 * max(1.0, abs(recorded)):
            raise RuntimeError(f"Reference {name} {recorded} does not match its patch pressures ({recomputed})")
    delta_p = reference["pa_eff"] - reference["pv_eff"]
    if not delta_p > 0:
        raise RuntimeError(f"Reference Delta_p_av must be positive, got {delta_p}")
    delta_tot = phi * delta_p / (1.0 - phi)
    delta_a, delta_v = s_a * delta_tot, (1.0 - s_a) * delta_tot
    return {"Q_a_ref": q_a, "Q_v_ref": q_v, "Delta_p_ref": delta_p, "delta_tot": delta_tot, "delta_a": delta_a, "delta_v": delta_v,
            "G_a": q_a / delta_a, "G_v": q_v / delta_v,
            "P_a": reference["pa_eff"] + delta_a, "P_v": reference["pv_eff"] - delta_v}


def resolve(spec, reference, rows, gas_pressure):
    """Per-support A_k, G_k, gamma_k, P_k from the request; rows: {name: physical_tag, vascular_type, assembled_reference_area}."""
    if set(rows) != set(SUPPORTS):
        raise RuntimeError(f"Mesh supports {sorted(rows)} differ from {SUPPORTS}")
    if reference is not None:
        for name in SUPPORTS:
            current, recorded = rows[name]["assembled_reference_area"], reference["supports"][name]["A_k"]
            if rows[name]["vascular_type"] != reference["supports"][name]["vascular_type"] or abs(current - recorded) > 1e-12 * recorded:
                raise RuntimeError(f"Support {name} differs from the reference run (area {current} vs {recorded})")
    resolved = {"mode": "pressure", "config": spec["config"], "modelling_notes": MODELLING_NOTES,
                "conservation_tolerance": CONSERVATION_TOLERANCE, "global_gas_pressure": float(gas_pressure)}
    multipliers = spec.get("support_multipliers", {})
    if spec["config"] in ("matched", "common"):
        common = _common_conductances(reference, spec["feeding_fraction"], spec["arterial_share"])
        resolved.update({"feeding_fraction": spec["feeding_fraction"], "arterial_share": spec["arterial_share"],
                         "split_assumption": "delta_tot = phi Delta_p_ref / (1 - phi); delta_a = s_a delta_tot; delta_v = (1 - s_a) delta_tot",
                         "common_mode_values": common,
                         "reference": {key: reference[key] for key in ("run_dir", "md5", "files", "Q_total", "pa_eff", "pv_eff", "Delta_p_av")}})
    if spec["config"] == "common":
        values = {key: common[key] for key in ("P_a", "P_v", "G_a", "G_v")}
    elif spec["config"] == "explicit":
        values = dict(spec["explicit"])
    else:
        values = None
        resolved["conductance_scale"] = spec["conductance_scale"]
    if values is not None:
        resolved["reservoirs"] = {key: float(value) for key, value in values.items()}
        resolved["reservoirs_repr"] = {key: repr(float(value)) for key, value in values.items()}
    supports = {}
    for name in SUPPORTS:
        row = rows[name]
        arterial = row["vascular_type"] == "arterial"
        area = float(row["assembled_reference_area"])
        if spec["config"] == "matched":
            g = spec["conductance_scale"] * (common["G_a"] if arterial else common["G_v"])
            ref = reference["supports"][name]
            p = ref["p_hat_k"] + ref["Q_k"] / g
            m = 1.0
        else:
            m = float(multipliers.get(name, 1.0))
            g = m * (values["G_a"] if arterial else values["G_v"])
            p = values["P_a"] if arterial else values["P_v"]
        entry = {"physical_tag": int(row["physical_tag"]), "vascular_type": row["vascular_type"], "A_k": area,
                 "m_k": m, "G_k": float(g), "gamma_k": float(g) / area, "P_k": float(p), "P_k_repr": repr(float(p)),
                 "P_k_minus_p_g": float(p) - float(gas_pressure)}
        if reference is not None:
            entry["reference"] = {"Q_k": reference["supports"][name]["Q_k"], "p_hat_k": reference["supports"][name]["p_hat_k"]}
        supports[name] = entry
    resolved["supports"] = supports
    resolved["zone_indicators_note"] = "P_k - p_g recorded as indicators only; no zone claim"
    return resolved


# ------------------------------------------------------------ operators -----

class LambdaPinOperator(dmech.Operator):
    """lambda_test * lambda / V0: pins the zero-mean multiplier to 0 (no gauge term in the liquid equation)."""

    def __init__(self, lam, lam_test, measure):
        self.measure = measure
        V0 = dolfin.assemble(1.0 * measure)
        self.res_form = lam_test * (lam / V0) * self.measure


def lambda_pinned_problem_class(parent):
    """Problem subclass whose zero-mean operator (added by dolfin_mech in __init__) is replaced by the lambda pin."""

    class LambdaPinnedMicroPoroFlowProblem(parent):
        def add_pressure_liquid_perturbation_zero_mean_operator(self):
            subsol = self.lambda_pl_perturbation_zero_mean_subsol
            return self.add_operator(LambdaPinOperator(subsol.subfunc, subsol.dsubtest, self.dV))

    LambdaPinnedMicroPoroFlowProblem.__name__ = f"LambdaPinned{parent.__name__}"
    return LambdaPinnedMicroPoroFlowProblem


class RobinExchangeOperator(dmech.Operator):
    """-gamma_k (P_k - p_tot) v on one support.

    ``pressure`` provides ``val`` (UFL scalar) and ``set_value_at_t_step``; a
    dmech.TimeVaryingConstant ramp now, a shared-reservoir unknown later.
    """

    def __init__(self, name, p_test, pl_tot, measure, gamma, pressure):
        self.name = name
        self.measure = measure
        self.gamma = float(gamma)
        self.pressure = pressure
        self.pl_tot = pl_tot
        self.exchange = dolfin.Constant(self.gamma) * (pressure.val - pl_tot)
        self.res_form = -self.exchange * p_test * self.measure

    def set_value_at_t_step(self, t_step):
        self.pressure.set_value_at_t_step(t_step)


def add_exchange_operators(problem, darcy_operator, resolved, k_step):
    operators = []
    for name in SUPPORTS:
        entry = resolved["supports"][name]
        ramp = dmech.TimeVaryingConstant(val_ini=0.0, val_fin=float(entry["P_k"]))
        operators.append(problem.add_operator(RobinExchangeOperator(
            name=name, p_test=problem.pl_perturbation_subsol.dsubtest, pl_tot=darcy_operator.pl_tot,
            measure=problem.get_subdomain_measure(entry["physical_tag"]), gamma=entry["gamma_k"], pressure=ramp), k_step=k_step))
    return operators


# -------------------------------------------------------------- outputs -----

def exchange_report(problem, darcy_operator, operators, resolved):
    supports = {}
    for operator in operators:
        entry = resolved["supports"][operator.name]
        q = float(dolfin.assemble(operator.exchange * operator.measure))
        p_hat = float(dolfin.assemble(darcy_operator.pl_tot * operator.measure)) / entry["A_k"]
        p_now = float(operator.pressure.val)
        arterial = entry["vascular_type"] == "arterial"
        supports[operator.name] = {
            "vascular_type": entry["vascular_type"], "A_k": entry["A_k"], "G_k": entry["G_k"], "gamma_k": entry["gamma_k"],
            "m_k": entry["m_k"], "P_k": p_now, "Q_k": q, "p_hat_k": p_hat, "P_k_minus_p_hat_k": p_now - p_hat,
            "Q_k_from_G_k": entry["G_k"] * (p_now - p_hat),
            "sign_flag": ("arterial support drains (Q_k < 0)" if arterial and q < 0
                          else "venous support supplies (Q_k > 0)" if (not arterial) and q > 0 else None)}
    q_a = sum(v["Q_k"] for v in supports.values() if v["vascular_type"] == "arterial")
    q_v = sum(v["Q_k"] for v in supports.values() if v["vascular_type"] == "venous")
    total = q_a + q_v
    lam = problem.lambda_pl_perturbation_zero_mean_subsol.func.vector().get_local()
    area = float(dolfin.assemble(dolfin.Constant(1.0) * problem.dV))
    return {
        "definitions": {"Q_k": "gamma_k int_{Omega_k} (P_k - p_tot) dX = G_k (P_k - p_hat_k); positive = into the tissue; "
                               "a support exchange flow, not an acinar inflow",
                        "p_hat_k": "reference-area mean of p_tot on Omega_k"},
        "supports": supports,
        "Q_a_tot": q_a, "Q_v_tot": q_v, "sum_Q_k": total,
        "sum_Q_k_relative_to_Q_a_tot": total / q_a if q_a != 0 else None,
        "conservation_tolerance": CONSERVATION_TOLERANCE,
        "sign_flags": {name: v["sign_flag"] for name, v in supports.items() if v["sign_flag"]},
        "lambda_dof_max_abs": float(np.abs(lam).max()) if lam.size else 0.0,
        "p_tilde_mean": float(dolfin.assemble(problem.pl_perturbation_subsol.subfunc * problem.dV)) / area,
        "p_tot_mean": float(dolfin.assemble(darcy_operator.pl_tot * problem.dV)) / area,
        "global_gas_pressure": resolved["global_gas_pressure"],
        "P_minus_p_g": {name: v["P_k"] - resolved["global_gas_pressure"] for name, v in supports.items()},
    }


def cell_exchange(mesh, operators):
    """Per-FE-cell exchange integrals int_{cell} s_k dX of each support (mesh cell order), as sparse arrays."""
    space = dolfin.FunctionSpace(mesh, "DG", 0)
    test = dolfin.TestFunction(space)
    dofmap = space.dofmap()
    cell_to_dof = np.array([dofmap.cell_dofs(cell)[0] for cell in range(mesh.num_cells())], dtype=np.int64)
    arrays = {}
    for operator in operators:
        values = dolfin.assemble(operator.exchange * test * operator.measure).get_local()[cell_to_dof]
        cells = np.nonzero(values)[0]
        arrays[f"{operator.name}_cells"] = cells
        arrays[f"{operator.name}_values"] = values[cells]
    return arrays


def add_acinus_exchange(results_dir, exchange_json="vascular_exchange.json", cells_npz="vascular_exchange_cells.npz",
                        mapping_json="fe_acinar_mapping.json"):
    """Per acinus: arterial supply, venous drainage and net source (exact cell integrals over Omega_k and A_i)."""
    results_dir = Path(results_dir)
    report = json.loads((results_dir / exchange_json).read_text())
    mapping = json.loads((results_dir / mapping_json).read_text())
    acinus = np.asarray(mapping["acinus_id_by_fe_cell"])
    arrays = np.load(results_dir / cells_npz)
    acini = sorted(int(a) for a in np.unique(acinus))
    table = {f"A{i}": {"arterial_supply": 0.0, "venous_drainage": 0.0, "by_support": {}} for i in acini}
    for name, support in report["supports"].items():
        cells, values = arrays[f"{name}_cells"], arrays[f"{name}_values"]
        for i in acini:
            q = float(values[acinus[cells] == i].sum())
            if q != 0.0:
                table[f"A{i}"]["by_support"][name] = q
                key = "arterial_supply" if support["vascular_type"] == "arterial" else "venous_drainage"
                table[f"A{i}"][key] += q if key == "arterial_supply" else -q
    for row in table.values():
        row["net_source"] = row["arterial_supply"] - row["venous_drainage"]
    report["per_acinus"] = {"definition": "cell-ownership split of the support exchange integrals (fe_acinar_mapping.json); "
                                          "support-based supply/drainage, not acinar inflow",
                            "acini": table,
                            "sum_net_source": sum(row["net_source"] for row in table.values())}
    (results_dir / exchange_json).write_text(json.dumps(report, indent=2) + "\n")
    return report["per_acinus"]
