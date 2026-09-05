#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Called from Get Whl after select_paddlefleet_alignment_pin.sh.
# Keep this file free of nested quotes so the workflow docker exec
# single-quoted -c script can invoke it without breaking the host shell.
# A selector error receipt is not an env-handoff failure.

set -euo pipefail

DEST="${1:-/workspace}"
REC="${DEST}/paddlefleet_alignment_pin_receipt.json"
ENVF="${DEST}/paddlefleet_alignment_pin.env"

if [[ ! -f "${REC}" ]]; then
  echo "::error:: missing selector receipt ${REC}" >&2
  exit 1
fi

python3 - "${REC}" <<'PY'
import json
import sys

path = sys.argv[1]
doc = json.load(open(path, encoding="utf-8"))
status = doc.get("status")
if status != "ok":
    raise SystemExit(
        f"selector receipt status={status!r} is not ok; "
        "Get Whl must stop (do not download remaining wheels or build)"
    )
print("selector receipt status=ok")
PY

if [[ ! -f "${ENVF}" ]]; then
  echo "::error:: selector did not write ${ENVF}" >&2
  exit 1
fi

cat "${REC}"
