#!/usr/bin/env python3
"""Launcher smoke test: import dolfin and dolfin_mech, sleep, exit with a chosen code. No FE solve."""
import argparse
import os
import sys
import time

import dolfin
import dolfin_mech

parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=float, default=5.0)
parser.add_argument("--exit-code", type=int, default=0)
parser.add_argument("--signal", type=int, help="kill this process with the given signal instead of exiting")
parser.add_argument("--raise-error", action="store_true", help="raise a RuntimeError instead of exiting")
args = parser.parse_args()
print("python", sys.version.split()[0])
print("dolfin", dolfin.__version__)
print("dolfin_mech __version__:", getattr(dolfin_mech, "__version__", None))
time.sleep(args.sleep)
if args.signal:
    print("sending signal", args.signal, flush=True)
    os.kill(os.getpid(), args.signal)
if args.raise_error:
    raise RuntimeError("smoke-test exception")
print("exiting with", args.exit_code)
sys.exit(args.exit_code)
