#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Consume selector output in a later docker exec. Source-mode paths must
# survive the step boundary; stack-paired must not fall back to the
# hardcoded 0.0.0 wheel filenames used by unpaired develop.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: consume_paddlefleet_alignment_pin.sh [--env FILE] [--out FILE] [--self-test]

Reads paddlefleet_alignment_pin.env written by select_paddlefleet_alignment_pin.sh
and writes a consumed env file for setup_venvs.sh. Stack-paired refuses the
0.0.0 develop filename and requires existing file-or-directory paths.
EOF
}

ENVFILE="${PADDLEFLEET_PIN_ENV:-/workspace/paddlefleet_alignment_pin.env}"
OUTFILE="${PADDLEFLEET_CONSUMED_ENV:-/workspace/paddlefleet_alignment_pin.consumed.env}"
RUN_SELF_TEST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENVFILE="${2:?}"; shift 2 ;;
    --out) OUTFILE="${2:?}"; shift 2 ;;
    --self-test) RUN_SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

hardcoded_develop_wheel() {
  case "$1" in
    */paddlefleet-0.0.0-py3-none-linux_x86_64.whl) return 0 ;;
    */paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl) return 0 ;;
    *) return 1 ;;
  esac
}

write_consumed() {
  mkdir -p "$(dirname "${OUTFILE}")"
  cat >"${OUTFILE}" <<EOF
PADDLEFLEET_WHEEL_PATH=${PADDLEFLEET_WHEEL_PATH}
PADDLEFLEET_OPS_WHEEL_PATH=${PADDLEFLEET_OPS_WHEEL_PATH}
ALIGNMENT_PADDLEFLEET_MODE=${ALIGNMENT_PADDLEFLEET_MODE}
PADDLEFLEET_PIN_RECEIPT=${PADDLEFLEET_PIN_RECEIPT:-}
PADDLEFLEET_SOURCE_COMMIT=${PADDLEFLEET_SOURCE_COMMIT:-}
PADDLEFLEET_WHEEL_ORIGIN=${PADDLEFLEET_WHEEL_ORIGIN:-}
PADDLEFLEET_OPS_ORIGIN=${PADDLEFLEET_OPS_ORIGIN:-}
EOF
  echo "[paddlefleet-pin-consume] wrote ${OUTFILE}" >&2
  echo "[paddlefleet-pin-consume] wheel=${PADDLEFLEET_WHEEL_PATH} ops=${PADDLEFLEET_OPS_WHEEL_PATH} mode=${ALIGNMENT_PADDLEFLEET_MODE} origin=${PADDLEFLEET_WHEEL_ORIGIN:-}" >&2
}

consume() {
  local mode="${ALIGNMENT_PADDLEFLEET_MODE:-develop}"
  if [[ -f "${ENVFILE}" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck disable=SC1090
    . "${ENVFILE}"
    set +a
    echo "[paddlefleet-pin-consume] loaded ${ENVFILE} receipt=${PADDLEFLEET_PIN_RECEIPT:-}" >&2
    mode="${ALIGNMENT_PADDLEFLEET_MODE:-${mode}}"
  fi
  ALIGNMENT_PADDLEFLEET_MODE="${mode}"

  if [[ "${mode}" == "stack-paired" ]]; then
    [[ -f "${ENVFILE}" ]] || {
      echo "::error:: stack-paired missing ${ENVFILE}; selector export did not cross docker exec" >&2
      exit 1
    }
    [[ -n "${PADDLEFLEET_WHEEL_PATH:-}" && -n "${PADDLEFLEET_OPS_WHEEL_PATH:-}" ]] || {
      echo "::error:: stack-paired env missing PADDLEFLEET_WHEEL_PATH or OPS path" >&2
      exit 1
    }
    if hardcoded_develop_wheel "${PADDLEFLEET_WHEEL_PATH}" || hardcoded_develop_wheel "${PADDLEFLEET_OPS_WHEEL_PATH}"; then
      echo "::error:: stack-paired refused hardcoded 0.0.0 wheel fallback: ${PADDLEFLEET_WHEEL_PATH} ${PADDLEFLEET_OPS_WHEEL_PATH}" >&2
      exit 1
    fi
  else
    PADDLEFLEET_WHEEL_PATH="${PADDLEFLEET_WHEEL_PATH:-/workspace/paddlefleet-0.0.0-py3-none-linux_x86_64.whl}"
    PADDLEFLEET_OPS_WHEEL_PATH="${PADDLEFLEET_OPS_WHEEL_PATH:-/workspace/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl}"
  fi

  if [[ ! -e "${PADDLEFLEET_WHEEL_PATH}" ]]; then
    echo "::error:: missing paddlefleet path: ${PADDLEFLEET_WHEEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -e "${PADDLEFLEET_OPS_WHEEL_PATH}" ]]; then
    echo "::error:: missing paddlefleet_ops path: ${PADDLEFLEET_OPS_WHEEL_PATH}" >&2
    exit 1
  fi
  write_consumed
}

run_self_test() {
  local root self
  self="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
  root="$(mktemp -d)"
  trap 'rm -rf "${root}"' RETURN
  mkdir -p "${root}/PaddleFleet/packages/paddlefleet_ops"
  echo tree >"${root}/PaddleFleet/pyproject.toml"
  echo ops >"${root}/PaddleFleet/packages/paddlefleet_ops/pyproject.toml"

  cat >"${root}/pin.env" <<EOF
PADDLEFLEET_WHEEL_PATH=${root}/PaddleFleet
PADDLEFLEET_OPS_WHEEL_PATH=${root}/PaddleFleet/packages/paddlefleet_ops
ALIGNMENT_PADDLEFLEET_MODE=stack-paired
PADDLEFLEET_PIN_RECEIPT=${root}/receipt.json
PADDLEFLEET_SOURCE_COMMIT=abc
PADDLEFLEET_WHEEL_ORIGIN=source_tree
PADDLEFLEET_OPS_ORIGIN=source_tree
EOF

  ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
    bash "${self}" --env "${root}/pin.env" --out "${root}/consumed.env"
  grep -q "PADDLEFLEET_WHEEL_PATH=${root}/PaddleFleet$" "${root}/consumed.env"
  grep -q "PADDLEFLEET_WHEEL_ORIGIN=source_tree" "${root}/consumed.env"

  env -u PADDLEFLEET_WHEEL_PATH -u PADDLEFLEET_OPS_WHEEL_PATH \
    ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
    bash "${self}" --env "${root}/pin.env" --out "${root}/consumed2.env"
  grep -q "PADDLEFLEET_WHEEL_PATH=${root}/PaddleFleet$" "${root}/consumed2.env"

  cat >"${root}/missing.env" <<EOF
PADDLEFLEET_WHEEL_PATH=/workspace/paddlefleet-0.0.0-py3-none-linux_x86_64.whl
PADDLEFLEET_OPS_WHEEL_PATH=/workspace/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl
ALIGNMENT_PADDLEFLEET_MODE=stack-paired
EOF
  if ALIGNMENT_PADDLEFLEET_MODE=stack-paired bash "${self}" --env "${root}/missing.env" --out "${root}/bad.env"; then
    echo "self-test FAIL: 0.0.0 fallback should be rejected" >&2
    exit 1
  fi
  echo "consume_paddlefleet_alignment_pin self-test OK"
}

if [[ "${RUN_SELF_TEST}" == 1 ]]; then
  run_self_test
  exit 0
fi
consume
