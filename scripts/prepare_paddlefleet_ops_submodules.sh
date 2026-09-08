#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Retry source-tree network preparation before setup/build, retaining the
# pinned Fleet helper's exact and nested gitlink verification.
set -euo pipefail
ops_path="${1:?ops-path}"
setup_script="${2:?pinned-setup-script}"
if [[ ! -d "$ops_path" ]]; then
    echo "[ops-submodules] wheel input; no source preparation"
    exit 0
fi
command -v timeout >/dev/null
[[ -f "$setup_script" ]] || { echo "[ops-submodules] pinned setup script missing" >&2; exit 1; }
for attempt in 1 2 3; do
    echo "[ops-submodules] prepare and verify recorded gitlinks, attempt $attempt/3"
    if timeout --kill-after=30s 15m bash "$setup_script" --prepare-ops-submodules "$ops_path"; then
        echo "[ops-submodules] recorded gitlinks prepared and verified"
        exit 0
    else
        rc=$?
    fi
    if [[ "$attempt" == 3 ]]; then
        echo "[ops-submodules] preparation failed after 3 attempts (exit $rc); stop before setup/build/training" >&2
        exit "$rc"
    fi
    delay=$((attempt * 15))
    echo "[ops-submodules] preparation exited $rc; retry in ${delay}s"
    sleep "$delay"
done
