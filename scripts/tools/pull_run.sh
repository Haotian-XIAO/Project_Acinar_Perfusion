#!/bin/bash
# Copy one remote run directory back to the same relative path in this checkout.
#
# Usage (from the local repository):
#   pull_run.sh --host SSH_HOST --remote-repo DIR results/<study>/<run_id> [--no-fields]
#
# --no-fields skips large field files (*.h5, *.xdmf, *.pvd, *.vtu).
set -euo pipefail

host= remote_repo= run_rel= excludes=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host) host=$2; shift 2 ;;
        --remote-repo) remote_repo=$2; shift 2 ;;
        --no-fields) excludes=(--exclude '*.h5' --exclude '*.xdmf' --exclude '*.pvd' --exclude '*.vtu'); shift ;;
        *) run_rel=${1%/}; shift ;;
    esac
done
[ -n "$host" ] && [ -n "$remote_repo" ] && [ -n "$run_rel" ] || { echo "usage: pull_run.sh --host H --remote-repo DIR results/<study>/<run_id> [--no-fields]" >&2; exit 64; }
case "$run_rel" in results/*/*) ;; *) echo "pull_run.sh: expected results/<study>/<run_id>, got $run_rel" >&2; exit 64 ;; esac

mkdir -p "$run_rel"
rsync -av --partial ${excludes[@]+"${excludes[@]}"} "$host:$remote_repo/$run_rel/" "$run_rel/"
