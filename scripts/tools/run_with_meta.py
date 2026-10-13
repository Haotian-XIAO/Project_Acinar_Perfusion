#!/usr/bin/env python3
"""Run one target script inside a protocol run directory.

Before the target is imported, this records ``run_meta.json`` (code versions,
environment, command, parameters, geometry fingerprints) and aborts if
``dolfin_mech`` is not imported from the expected checkout, if a geometry
input does not match ``geometry_manifest/geometry.md5``, or if any metadata
item cannot be determined (recorded as null plus an error, never a default).
Git state is read from ``git_state.json``, written by ``start_run.sh`` in the
native shell: under Rosetta, Python child processes cannot run the system git
(xcrun) tools.  The final ``STATUS`` is written by the calling shell layer,
which also sees signal deaths that Python cannot report.

Usage (from the repository root):
    python scripts/tools/run_with_meta.py --run-dir results/<study>/<run_id> \
        --expected-dolfin-mech <checkout> [--geometry-file PATTERN ...] \
        [--param KEY=VALUE ...] -- <target.py> [target args...]

Target arguments may contain ``{run_dir}``, replaced by the run directory.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABORT_EXIT_CODE = 2


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load_git_state(run_dir, errors):
    """Git state captured by start_run.sh; any missing item is an error."""
    path = run_dir / "git_state.json"
    if not path.exists():
        errors.append(f"missing {path.name} (written by start_run.sh)")
        return {"repo": None, "dolfin_mech": None}
    state = json.loads(path.read_text())
    for key in ("repo", "dolfin_mech"):
        entry = state.get(key)
        if not entry or entry.get("commit") is None or entry.get("dirty") is None:
            errors.append(f"git state of {key} unknown: {entry.get('error') if entry else 'not recorded'}")
    return state


def rosetta_translated(errors):
    if sys.platform != "darwin":
        return {"value": None, "error": f"not applicable on {sys.platform}"}
    result = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True)
    if result.returncode != 0 or result.stdout.strip() not in ("0", "1"):
        error = f"sysctl.proc_translated failed (exit {result.returncode}): {result.stderr.strip()}"
        errors.append(error)
        return {"value": None, "error": error}
    return {"value": result.stdout.strip() == "1", "error": None}


def required_env(name, errors):
    value = os.environ.get(name)
    if not value:
        errors.append(f"environment variable {name} is not set")
    return value


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_geometry(patterns, manifest_path):
    """Match each pattern against manifest entries and compare md5 values."""
    manifest = {}
    for line in Path(manifest_path).read_text().splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            manifest[name] = digest
    files, errors = {}, []
    for pattern in patterns:
        matches = sorted(name for name in manifest if fnmatch.fnmatchcase(name, pattern))
        if not matches:
            errors.append(f"no manifest entry matches {pattern!r}")
        for name in matches:
            path = ROOT / name
            actual = md5(path) if path.exists() else None
            files[name] = {"md5": actual, "manifest_md5": manifest[name], "ok": actual == manifest[name]}
            if actual is None:
                errors.append(f"missing geometry file {name}")
            elif actual != manifest[name]:
                errors.append(f"md5 mismatch for {name}: {actual} != manifest {manifest[name]}")
    return files, errors


def parse_args(argv):
    if "--" not in argv:
        raise SystemExit("usage: run_with_meta.py [options] -- <target.py> [target args...]")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-dolfin-mech", type=Path, required=True,
                        help="dolfin_mech checkout that must provide the imported package")
    parser.add_argument("--manifest", type=Path, default=ROOT / "geometry_manifest" / "geometry.md5")
    parser.add_argument("--geometry-file", action="append", default=[],
                        help="repository-relative path or glob of manifest entries used by the run")
    parser.add_argument("--param", action="append", default=[], help="KEY=VALUE recorded in run_meta.json")
    parser.add_argument("--study", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args(argv[:split])
    target = argv[split + 1:]
    if not target:
        parser.error("missing target script after --")
    return args, target


def main(argv):
    args, target = parse_args(argv)
    run_dir = (ROOT / args.run_dir).resolve()
    try:
        return run(args, target, run_dir)
    except Exception:
        # Failures of the wrapper itself (e.g. dolfin import) before the target runs.
        (run_dir / "traceback.txt").write_text(traceback.format_exc())
        raise


def run(args, target, run_dir):
    expected = args.expected_dolfin_mech.resolve()
    replacements = [(str(run_dir), "<RUN_DIR>"), (str(ROOT), "<REPO>"), (str(expected), "<DOLFIN_MECH>"),
                    (str(Path.home()), "~")]

    def redact(value):
        # Keep usernames and machine paths out of metadata that may be shared.
        if value is None:
            return None
        text = str(value)
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    target_args = [arg.replace("{run_dir}", os.path.relpath(run_dir, ROOT)) for arg in target[1:]]
    target_path = (ROOT / target[0]).resolve()
    errors = []
    params = {}
    for item in args.param:
        if "=" not in item:
            errors.append(f"--param {item!r} is not KEY=VALUE")
            continue
        key, value = item.split("=", 1)
        params[key] = value
    git = load_git_state(run_dir, errors)

    import dolfin
    import dolfin_mech
    dolfin_mech_file = Path(dolfin_mech.__file__).resolve()
    dolfin_mech_ok = expected in dolfin_mech_file.parents
    if not dolfin_mech_ok:
        errors.append(f"dolfin_mech imported from {dolfin_mech_file}, not inside {expected}")
    geometry_files, geometry_errors = check_geometry(args.geometry_file, args.manifest)
    errors += geometry_errors

    meta = {
        "study": args.study,
        "run_id": args.run_id,
        "host_label": args.host_label,
        "start_time": now(),
        "pid": os.getpid(),
        "command": [redact(arg) for arg in [sys.executable, *sys.argv]],
        "target": {"script": redact(target_path), "args": [redact(arg) for arg in target_args]},
        "parameters": params,
        "repo": git["repo"],
        "dolfin_mech": {
            "file": redact(dolfin_mech_file),
            "expected_checkout": "<DOLFIN_MECH>",
            "inside_expected_checkout": dolfin_mech_ok,
            "git": git["dolfin_mech"],
        },
        "dolfin_version": dolfin.__version__,
        "environment": {
            "conda_env": required_env("CONDA_DEFAULT_ENV", errors),
            "conda_prefix": redact(required_env("CONDA_PREFIX", errors)),
            "python": platform.python_version(),
            "python_executable": redact(sys.executable),
            "platform_machine": platform.machine(),
            "rosetta_translated": rosetta_translated(errors),
            "pythonpath": redact(required_env("PYTHONPATH", errors)),
        },
        "geometry_manifest": os.path.relpath(args.manifest.resolve(), ROOT),
        "geometry_check": "requested" if args.geometry_file else "not requested",
        "geometry_files": geometry_files,
        "preflight_errors": errors,
    }
    meta_path = run_dir / "run_meta.json"

    for entry in (git["repo"], git["dolfin_mech"]):
        if entry and entry.get("error"):
            entry["error"] = redact(entry["error"])

    def write_meta():
        shown = dict(meta, preflight_errors=[redact(error) for error in errors])
        meta_path.write_text(json.dumps(shown, indent=2) + "\n")

    write_meta()
    print("run_meta.json written:", os.path.relpath(meta_path, ROOT), flush=True)
    if errors:
        print("PRE-FLIGHT ABORT:\n  " + "\n  ".join(errors), file=sys.stderr, flush=True)
        meta.update(end_time=now(), exit_code=ABORT_EXIT_CODE)
        write_meta()
        return ABORT_EXIT_CODE

    sys.argv = [str(target_path), *target_args]
    sys.path.insert(0, str(target_path.parent))
    exit_code = 0
    try:
        runpy.run_path(str(target_path), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        exit_code = 1
        (run_dir / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    meta.update(end_time=now(), exit_code=exit_code)
    write_meta()
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
