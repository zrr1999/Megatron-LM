#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Consume selector output in a later docker exec. The caller mode/pin stay
# authoritative: a leftover develop env must not silently downgrade
# stack-paired. A 0.0.0 filename is allowed when the receipt proves an
# explicit URL+sha256 (or a source tree / in-invocation build). Unproven
# develop/latest fallback is refused.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: consume_paddlefleet_alignment_pin.sh [--env FILE] [--out FILE] [--self-test]

Reads paddlefleet_alignment_pin.env + receipt from select_paddlefleet_alignment_pin.sh
and writes a consumed env file for setup_venvs.sh.

Caller ALIGNMENT_PADDLEFLEET_MODE / PADDLEFLEET_PIN_SHA are the request.
The env file must match that request; it does not override them.
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

fail() {
  echo "::error:: $*" >&2
  exit 1
}

parse_envfile() {
  FILE_MODE=""
  FILE_PIN=""
  FILE_SOURCE=""
  FILE_RECEIPT=""
  FILE_WHEEL=""
  FILE_OPS=""
  FILE_WHEEL_ORIGIN=""
  FILE_OPS_ORIGIN=""
  FILE_WHEEL_DIGEST=""
  FILE_OPS_DIGEST=""
  [[ -f "${ENVFILE}" ]] || return 0
  local line k v
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    k="${line%%=*}"
    v="${line#*=}"
    case "${k}" in
      ALIGNMENT_PADDLEFLEET_MODE) FILE_MODE="${v}" ;;
      PADDLEFLEET_PIN_SHA) FILE_PIN="${v}" ;;
      PADDLEFLEET_SOURCE_COMMIT) FILE_SOURCE="${v}" ;;
      PADDLEFLEET_PIN_RECEIPT) FILE_RECEIPT="${v}" ;;
      PADDLEFLEET_WHEEL_PATH) FILE_WHEEL="${v}" ;;
      PADDLEFLEET_OPS_WHEEL_PATH) FILE_OPS="${v}" ;;
      PADDLEFLEET_WHEEL_ORIGIN) FILE_WHEEL_ORIGIN="${v}" ;;
      PADDLEFLEET_OPS_ORIGIN) FILE_OPS_ORIGIN="${v}" ;;
      PADDLEFLEET_WHEEL_DIGEST_VERIFIED) FILE_WHEEL_DIGEST="${v}" ;;
      PADDLEFLEET_OPS_DIGEST_VERIFIED) FILE_OPS_DIGEST="${v}" ;;
    esac
  done <"${ENVFILE}"
}

# Unproven develop fallback: develop_latest origin, or a default 0.0.0
# filename with no digest proof. A verified ci_metadata/build artifact may
# legally keep the 0.0.0 filename.
unproven_develop_fallback() {
  local path="$1" origin="$2" digest="$3"
  case "${origin}" in
    develop_latest) return 0 ;;
    source_tree|build|ci_metadata)
      [[ "${origin}" == develop_latest ]] && return 0
      return 1
      ;;
  esac
  case "${path}" in
    */paddlefleet-0.0.0-py3-none-linux_x86_64.whl|*/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl)
      [[ "${digest}" == "true" ]] && return 1
      return 0
      ;;
  esac
  return 1
}

check_receipt() {
  local receipt="$1" requested_mode="$2" requested_pin="$3"
  [[ -f "${receipt}" ]] || fail "stack-paired missing receipt ${receipt}"
  python3 - "${receipt}" "${requested_mode}" "${requested_pin}" <<'PY'
import json, sys
path, requested_mode, requested_pin = sys.argv[1:4]
doc = json.load(open(path, encoding="utf-8"))
if doc.get("status") != "ok":
    raise SystemExit(f"receipt status={doc.get('status')!r} is not ok")
if requested_mode == "stack-paired":
    if doc.get("mode") != "stack-paired":
        raise SystemExit(
            f"requested stack-paired but receipt mode={doc.get('mode')!r}"
        )
    src = doc.get("source") or {}
    if requested_pin:
        exp = src.get("expected_commit") or ""
        act = src.get("actual_commit") or ""
        if requested_pin not in (exp, act):
            raise SystemExit(
                f"receipt source pin mismatch requested={requested_pin} "
                f"expected={exp} actual={act}"
            )
        if src.get("commit_verified") is not True:
            raise SystemExit("receipt source commit_verified is not true")
    pairing = (doc.get("pairing") or {}).get("status")
    if pairing == "unpaired_default":
        raise SystemExit("receipt pairing.status=unpaired_default")
    for art in doc.get("artifacts") or []:
        origin = art.get("origin") or ""
        url = art.get("url") or ""
        if origin == "develop_latest":
            raise SystemExit(f"artifact {art.get('name')} origin=develop_latest")
        if "/develop/latest/" in url or "CodeSync/develop/" in url:
            raise SystemExit(f"artifact {art.get('name')} unpaired develop URL")
print("receipt matches request")
PY
}

write_consumed() {
  mkdir -p "$(dirname "${OUTFILE}")"
  cat >"${OUTFILE}" <<EOF
PADDLEFLEET_WHEEL_PATH=${PADDLEFLEET_WHEEL_PATH}
PADDLEFLEET_OPS_WHEEL_PATH=${PADDLEFLEET_OPS_WHEEL_PATH}
ALIGNMENT_PADDLEFLEET_MODE=${ALIGNMENT_PADDLEFLEET_MODE}
PADDLEFLEET_PIN_RECEIPT=${PADDLEFLEET_PIN_RECEIPT:-}
PADDLEFLEET_SOURCE_COMMIT=${PADDLEFLEET_SOURCE_COMMIT:-}
PADDLEFLEET_PIN_SHA=${PADDLEFLEET_PIN_SHA:-}
PADDLEFLEET_WHEEL_ORIGIN=${PADDLEFLEET_WHEEL_ORIGIN:-}
PADDLEFLEET_OPS_ORIGIN=${PADDLEFLEET_OPS_ORIGIN:-}
EOF
  echo "[paddlefleet-pin-consume] wrote ${OUTFILE}" >&2
  echo "[paddlefleet-pin-consume] requested_mode=${ALIGNMENT_PADDLEFLEET_MODE} pin=${PADDLEFLEET_PIN_SHA:-} wheel=${PADDLEFLEET_WHEEL_PATH} origin=${PADDLEFLEET_WHEEL_ORIGIN:-}" >&2
}

consume() {
  local requested_mode="${ALIGNMENT_PADDLEFLEET_MODE:-develop}"
  local requested_pin="${PADDLEFLEET_PIN_SHA:-}"

  parse_envfile

  if [[ "${requested_mode}" == "stack-paired" ]]; then
    [[ -f "${ENVFILE}" ]] || fail "stack-paired missing ${ENVFILE}; selector export did not cross docker exec"
    [[ "${FILE_MODE}" == "stack-paired" ]] || fail "requested stack-paired but env mode=${FILE_MODE:-empty} (will not consume a develop leftover)"
    if [[ -n "${requested_pin}" ]]; then
      local file_id="${FILE_PIN:-${FILE_SOURCE}}"
      [[ "${file_id}" == "${requested_pin}" ]] || fail "requested pin ${requested_pin} != env pin/source ${file_id:-empty}"
    fi
    local receipt="${FILE_RECEIPT:-}"
    if [[ -z "${receipt}" && -f "${ENVFILE%/*}/paddlefleet_alignment_pin_receipt.json" ]]; then
      receipt="${ENVFILE%/*}/paddlefleet_alignment_pin_receipt.json"
    fi
    check_receipt "${receipt}" "${requested_mode}" "${requested_pin}"
    PADDLEFLEET_WHEEL_PATH="${FILE_WHEEL}"
    PADDLEFLEET_OPS_WHEEL_PATH="${FILE_OPS}"
    PADDLEFLEET_WHEEL_ORIGIN="${FILE_WHEEL_ORIGIN}"
    PADDLEFLEET_OPS_ORIGIN="${FILE_OPS_ORIGIN}"
    PADDLEFLEET_PIN_RECEIPT="${receipt}"
    PADDLEFLEET_SOURCE_COMMIT="${FILE_SOURCE}"
    PADDLEFLEET_PIN_SHA="${requested_pin:-${FILE_PIN}}"
    ALIGNMENT_PADDLEFLEET_MODE="stack-paired"
    [[ -n "${PADDLEFLEET_WHEEL_PATH}" && -n "${PADDLEFLEET_OPS_WHEEL_PATH}" ]] \
      || fail "stack-paired env missing PADDLEFLEET_WHEEL_PATH or OPS path"
    if unproven_develop_fallback "${PADDLEFLEET_WHEEL_PATH}" "${FILE_WHEEL_ORIGIN}" "${FILE_WHEEL_DIGEST}"; then
      fail "stack-paired refused unproven develop fallback for paddlefleet: path=${PADDLEFLEET_WHEEL_PATH} origin=${FILE_WHEEL_ORIGIN:-empty} digest_verified=${FILE_WHEEL_DIGEST:-false}"
    fi
    if unproven_develop_fallback "${PADDLEFLEET_OPS_WHEEL_PATH}" "${FILE_OPS_ORIGIN}" "${FILE_OPS_DIGEST}"; then
      fail "stack-paired refused unproven develop fallback for paddlefleet_ops: path=${PADDLEFLEET_OPS_WHEEL_PATH} origin=${FILE_OPS_ORIGIN:-empty} digest_verified=${FILE_OPS_DIGEST:-false}"
    fi
  else
    ALIGNMENT_PADDLEFLEET_MODE="develop"
    PADDLEFLEET_WHEEL_PATH="${FILE_WHEEL:-/workspace/paddlefleet-0.0.0-py3-none-linux_x86_64.whl}"
    PADDLEFLEET_OPS_WHEEL_PATH="${FILE_OPS:-/workspace/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl}"
    PADDLEFLEET_WHEEL_ORIGIN="${FILE_WHEEL_ORIGIN:-develop_latest}"
    PADDLEFLEET_OPS_ORIGIN="${FILE_OPS_ORIGIN:-develop_latest}"
    PADDLEFLEET_PIN_RECEIPT="${FILE_RECEIPT:-}"
    PADDLEFLEET_SOURCE_COMMIT="${FILE_SOURCE:-}"
    PADDLEFLEET_PIN_SHA="${requested_pin}"
  fi

  if [[ ! -e "${PADDLEFLEET_WHEEL_PATH}" ]]; then
    fail "missing paddlefleet path: ${PADDLEFLEET_WHEEL_PATH}"
  fi
  if [[ ! -e "${PADDLEFLEET_OPS_WHEEL_PATH}" ]]; then
    fail "missing paddlefleet_ops path: ${PADDLEFLEET_OPS_WHEEL_PATH}"
  fi
  write_consumed
}

write_min_receipt() {
  local path="$1" mode="$2" pin="$3" wheel="$4" ops="$5" origin="$6" digest="$7" pairing="$8"
  python3 - "${path}" "${mode}" "${pin}" "${wheel}" "${ops}" "${origin}" "${digest}" "${pairing}" <<'PY'
import json, sys
path, mode, pin, wheel, ops, origin, digest, pairing = sys.argv[1:9]
digest_ok = digest == "true"
doc = {
    "schema": "paddlefleet-alignment-pin/v1",
    "status": "ok",
    "mode": mode,
    "source": {
        "expected_commit": pin or None,
        "actual_commit": pin or None,
        "commit_verified": bool(pin) and mode == "stack-paired",
    },
    "artifacts": [
        {"name": "paddlefleet", "path": wheel, "url": None, "origin": origin,
         "digest_verified": digest_ok, "expected_sha256": "abc" if digest_ok else None,
         "actual_sha256": "abc" if digest_ok else None},
        {"name": "paddlefleet_ops", "path": ops, "url": None, "origin": origin,
         "digest_verified": digest_ok, "expected_sha256": "def" if digest_ok else None,
         "actual_sha256": "def" if digest_ok else None},
    ],
    "pairing": {"stack_paired_proven": False, "status": pairing, "reason": "fixture"},
}
open(path, "w", encoding="utf-8").write(json.dumps(doc, indent=2) + "\n")
PY
}

run_self_test() {
  local root self
  self="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
  root="$(mktemp -d)"
  trap 'rm -rf "${root}"' RETURN
  mkdir -p "${root}/PaddleFleet/packages/paddlefleet_ops"
  echo tree >"${root}/PaddleFleet/pyproject.toml"
  echo ops >"${root}/PaddleFleet/packages/paddlefleet_ops/pyproject.toml"
  local pin="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

  write_min_receipt "${root}/receipt.json" stack-paired "${pin}" \
    "${root}/PaddleFleet" "${root}/PaddleFleet/packages/paddlefleet_ops" \
    source_tree false source_tree_from_checked_out_pin
  cat >"${root}/pin.env" <<EOF
PADDLEFLEET_WHEEL_PATH=${root}/PaddleFleet
PADDLEFLEET_OPS_WHEEL_PATH=${root}/PaddleFleet/packages/paddlefleet_ops
ALIGNMENT_PADDLEFLEET_MODE=stack-paired
PADDLEFLEET_PIN_RECEIPT=${root}/receipt.json
PADDLEFLEET_SOURCE_COMMIT=${pin}
PADDLEFLEET_PIN_SHA=${pin}
PADDLEFLEET_WHEEL_ORIGIN=source_tree
PADDLEFLEET_OPS_ORIGIN=source_tree
PADDLEFLEET_WHEEL_DIGEST_VERIFIED=false
PADDLEFLEET_OPS_DIGEST_VERIFIED=false
EOF

  ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA="${pin}" \
    bash "${self}" --env "${root}/pin.env" --out "${root}/consumed.env"
  grep -q "PADDLEFLEET_WHEEL_PATH=${root}/PaddleFleet$" "${root}/consumed.env"
  grep -q "ALIGNMENT_PADDLEFLEET_MODE=stack-paired" "${root}/consumed.env"
  grep -q "PADDLEFLEET_PIN_SHA=${pin}" "${root}/consumed.env"

  env -u PADDLEFLEET_WHEEL_PATH -u PADDLEFLEET_OPS_WHEEL_PATH \
    ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA="${pin}" \
    bash "${self}" --env "${root}/pin.env" --out "${root}/consumed2.env"
  grep -q "PADDLEFLEET_WHEEL_PATH=${root}/PaddleFleet$" "${root}/consumed2.env"

  # 1) leftover develop env must not silently satisfy stack-paired.
  mkdir -p "${root}/devwhl"
  echo dummy >"${root}/devwhl/paddlefleet-0.0.0-py3-none-linux_x86_64.whl"
  echo dummy >"${root}/devwhl/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl"
  write_min_receipt "${root}/develop-receipt.json" develop "" \
    "${root}/devwhl/paddlefleet-0.0.0-py3-none-linux_x86_64.whl" \
    "${root}/devwhl/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl" \
    develop_latest false unpaired_default
  cat >"${root}/develop.env" <<EOF
PADDLEFLEET_WHEEL_PATH=${root}/devwhl/paddlefleet-0.0.0-py3-none-linux_x86_64.whl
PADDLEFLEET_OPS_WHEEL_PATH=${root}/devwhl/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl
ALIGNMENT_PADDLEFLEET_MODE=develop
PADDLEFLEET_PIN_RECEIPT=${root}/develop-receipt.json
PADDLEFLEET_WHEEL_ORIGIN=develop_latest
PADDLEFLEET_OPS_ORIGIN=develop_latest
EOF
  if ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA="${pin}" \
      bash "${self}" --env "${root}/develop.env" --out "${root}/downgrade.env"; then
    echo "self-test FAIL: develop leftover env overrode stack-paired" >&2
    exit 1
  fi

  # 2) verified URL+hash artifact may keep the 0.0.0 filename.
  write_min_receipt "${root}/named-receipt.json" stack-paired "${pin}" \
    "${root}/devwhl/paddlefleet-0.0.0-py3-none-linux_x86_64.whl" \
    "${root}/devwhl/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl" \
    ci_metadata true unproven
  cat >"${root}/named.env" <<EOF
PADDLEFLEET_WHEEL_PATH=${root}/devwhl/paddlefleet-0.0.0-py3-none-linux_x86_64.whl
PADDLEFLEET_OPS_WHEEL_PATH=${root}/devwhl/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl
ALIGNMENT_PADDLEFLEET_MODE=stack-paired
PADDLEFLEET_PIN_RECEIPT=${root}/named-receipt.json
PADDLEFLEET_SOURCE_COMMIT=${pin}
PADDLEFLEET_PIN_SHA=${pin}
PADDLEFLEET_WHEEL_ORIGIN=ci_metadata
PADDLEFLEET_OPS_ORIGIN=ci_metadata
PADDLEFLEET_WHEEL_DIGEST_VERIFIED=true
PADDLEFLEET_OPS_DIGEST_VERIFIED=true
EOF
  ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA="${pin}" \
    bash "${self}" --env "${root}/named.env" --out "${root}/named.consumed.env"
  grep -q "paddlefleet-0.0.0-py3-none-linux_x86_64.whl" "${root}/named.consumed.env"

  # Unproven 0.0.0 fallback (no origin, no digest) still fails.
  cat >"${root}/bare.env" <<EOF
PADDLEFLEET_WHEEL_PATH=${root}/devwhl/paddlefleet-0.0.0-py3-none-linux_x86_64.whl
PADDLEFLEET_OPS_WHEEL_PATH=${root}/devwhl/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl
ALIGNMENT_PADDLEFLEET_MODE=stack-paired
PADDLEFLEET_PIN_RECEIPT=${root}/named-receipt.json
PADDLEFLEET_SOURCE_COMMIT=${pin}
PADDLEFLEET_PIN_SHA=${pin}
EOF
  if ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA="${pin}" \
      bash "${self}" --env "${root}/bare.env" --out "${root}/bare.consumed.env"; then
    echo "self-test FAIL: unproven 0.0.0 fallback should be rejected" >&2
    exit 1
  fi

  echo "consume_paddlefleet_alignment_pin self-test OK"
}

if [[ "${RUN_SELF_TEST}" == 1 ]]; then
  run_self_test
  exit 0
fi
consume
