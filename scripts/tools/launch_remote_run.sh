#!/bin/bash
# Launch one protocol run on a remote machine from this checkout.
#
# Usage (from the local repository):
#   launch_remote_run.sh --host SSH_HOST --remote-repo DIR --conda-sh FILE --dolfin-mech DIR \
#       --study NAME --label LABEL [--env NAME] [--host-label LABEL] \
#       [--geometry-file PATTERN ...] [--param KEY=VALUE ...] -- <target.py> [target args...]
#
# Machine-specific values are arguments only; nothing host-specific is stored here.
# Refuses to launch unless the remote working tree is clean and its HEAD equals
# the local HEAD. The run itself is started by start_run.sh on the remote, which
# returns immediately; check results/<study>/<run_id>/STATUS later.
set -euo pipefail

host= remote_repo=
forward=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host) host=$2; shift 2 ;;
        --remote-repo) remote_repo=$2; shift 2 ;;
        --) forward+=("$@"); break ;;
        *) forward+=("$1"); shift ;;
    esac
done
[ -n "$host" ] && [ -n "$remote_repo" ] || { echo "launch_remote_run.sh: --host and --remote-repo are required" >&2; exit 64; }

local_head=$(git rev-parse HEAD)
remote_state=$(ssh -o BatchMode=yes "$host" "cd $(printf '%q' "$remote_repo") && git rev-parse HEAD && git status --porcelain")
remote_head=$(printf '%s\n' "$remote_state" | head -1)
remote_dirty=$(printf '%s\n' "$remote_state" | tail -n +2)
if [ -n "$remote_dirty" ]; then
    echo "launch_remote_run.sh: refusing to launch, remote working tree is dirty:" >&2
    printf '%s\n' "$remote_dirty" >&2
    exit 65
fi
if [ "$remote_head" != "$local_head" ]; then
    echo "launch_remote_run.sh: refusing to launch, remote HEAD $remote_head != local HEAD $local_head" >&2
    exit 65
fi
echo "remote HEAD = local HEAD = $local_head, remote tree clean"

ssh -o BatchMode=yes "$host" "cd $(printf '%q' "$remote_repo") && /bin/bash scripts/tools/start_run.sh $(printf '%q ' "${forward[@]}")"
