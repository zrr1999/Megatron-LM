#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Select PaddleFleet source + wheels for alignment_model_accuracy.
#
# Default (ALIGNMENT_PADDLEFLEET_MODE=develop or unset):
#   historical CodeSync/develop tarball + BOS develop/latest wheels.
#   Cases are not filtered.
#
# Explicit (ALIGNMENT_PADDLEFLEET_MODE=stack-paired): fail-closed.
#   Checkout PADDLEFLEET_PIN_SHA; git rev-parse HEAD must equal the pin.
#   Artifacts: caller URL+sha256 (from Build Fleet whl / Actions metadata)
#   or build from the checked-out tree. Independent digest matches do not
#   prove the wheels were produced from that commit — receipt records
#   source_commit vs artifact sha256 separately and pairing as unproven
#   unless this invocation built the files from the pin.
#   git/download/checkout failures still write an error receipt.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: select_paddlefleet_alignment_pin.sh [--dest DIR] [--self-test]

--self-test ignores --dest and uses an offline fixture (no network).

Env:
  ALIGNMENT_PADDLEFLEET_MODE   develop (default) | stack-paired
  PADDLEFLEET_PIN_SHA          required 40-hex commit in stack-paired
  PADDLEFLEET_GIT_URL          git remote or local repo (stack-paired)
  PADDLEFLEET_WHEEL_URL        optional explicit wheel (https or local path)
  PADDLEFLEET_WHEEL_SHA256     required with WHEEL_URL
  PADDLEFLEET_OPS_WHEEL_URL    optional explicit ops wheel
  PADDLEFLEET_OPS_WHEEL_SHA256 required with OPS URL
  PADDLEFLEET_BUILD_CMD        optional; default uv build paddlefleet
  PADDLEFLEET_BUILD_OPS_CMD    optional; default uv build paddlefleet-ops
  ALIGNMENT_PADDLEFLEET_DEST   output directory (default /workspace)
EOF
}

MODE="${ALIGNMENT_PADDLEFLEET_MODE:-develop}"
DEST="${ALIGNMENT_PADDLEFLEET_DEST:-/workspace}"
RUN_SELF_TEST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="${2:?--dest requires a path}"
      shift 2
      ;;
    --self-test)
      RUN_SELF_TEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

BOS="${PADDLEFLEET_BOS:-https://paddle-github-action.bj.bcebos.com}"
DEFAULT_TAR_URL="https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddleFleet.tar"
DEFAULT_WHL_URL="${BOS}/PaddleFleet/develop/latest/paddlefleet-0.0.0-py3-none-linux_x86_64.whl"
DEFAULT_OPS_URL="${BOS}/PaddleFleet/develop/latest/cu130/paddle-release/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl"
GIT_URL="${PADDLEFLEET_GIT_URL:-https://github.com/PaddlePaddle/PaddleFleet.git}"
PIN_SHA="${PADDLEFLEET_PIN_SHA:-}"
WHEEL_URL="${PADDLEFLEET_WHEEL_URL:-}"
WHEEL_SHA="${PADDLEFLEET_WHEEL_SHA256:-}"
OPS_URL="${PADDLEFLEET_OPS_WHEEL_URL:-}"
OPS_SHA="${PADDLEFLEET_OPS_WHEEL_SHA256:-}"
BUILD_CMD="${PADDLEFLEET_BUILD_CMD:-}"
BUILD_OPS_CMD="${PADDLEFLEET_BUILD_OPS_CMD:-}"

ACTUAL_SHA=""
SOURCE_VERIFIED=false
PADDLEFLEET_WHEEL_PATH=""
PADDLEFLEET_OPS_WHEEL_PATH=""
ACTUAL_WHEEL_SHA=""
ACTUAL_OPS_SHA=""
WHEEL_DIGEST_VERIFIED=false
OPS_DIGEST_VERIFIED=false
WHEEL_ORIGIN=""
OPS_ORIGIN=""
WHEEL_BUILT_FROM_COMMIT=""
OPS_BUILT_FROM_COMMIT=""
LOADED_FROM=""
RECEIPT_WRITTEN=0

log() { echo "[paddlefleet-pin] $*" >&2; }

sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

unpaired_url() {
  case "$1" in
    *"/develop/latest/"*|*"CodeSync/develop/"*) return 0 ;;
    *) return 1 ;;
  esac
}

pairing_fields() {
  local proven=false
  local status="unproven"
  local reason="wheel/ops digest match does not prove production from source_commit"
  if [[ "${MODE}" == "develop" ]]; then
    status="unpaired_default"
    reason="develop tarball and develop/latest wheels; not a stack pin"
  elif [[ "${WHEEL_ORIGIN}" == "source_tree" && "${OPS_ORIGIN}" == "source_tree" \
      && "${SOURCE_VERIFIED}" == "true" ]]; then
    status="source_tree_from_checked_out_pin"
    reason="this invocation exported checked-out source trees; not a wheel digest proof"
  elif [[ "${WHEEL_ORIGIN}" == "build" && "${OPS_ORIGIN}" == "build" \
      && "${WHEEL_BUILT_FROM_COMMIT}" == "${ACTUAL_SHA}" \
      && "${OPS_BUILT_FROM_COMMIT}" == "${ACTUAL_SHA}" \
      && "${SOURCE_VERIFIED}" == "true" ]]; then
    status="built_from_checked_out_pin"
    reason="this invocation built both artifacts from checked-out source_commit; not a remote-stack proof"
  fi
  printf '%s\t%s\t%s\n' "${proven}" "${status}" "${reason}"
}

write_receipt() {
  local status="$1" detail="${2:-}"
  mkdir -p "${DEST}"
  local receipt="${DEST}/paddlefleet_alignment_pin_receipt.json"
  local pair
  pair="$(pairing_fields)"
  local stack_proven pairing_status pairing_reason
  stack_proven="${pair%%$'\t'*}"
  pair="${pair#*$'\t'}"
  pairing_status="${pair%%$'\t'*}"
  pairing_reason="${pair#*$'\t'}"
  if ! command -v python3 >/dev/null 2>&1; then
    printf '{"schema":"paddlefleet-alignment-pin/v1","status":"%s","detail":"%s"}\n' \
      "${status}" "${detail}" >"${receipt}"
    RECEIPT_WRITTEN=1
    return 0
  fi
  python3 - "${receipt}" "${status}" "${detail}" "${stack_proven}" \
    "${pairing_status}" "${pairing_reason}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, status, detail, stack_proven, pairing_status, pairing_reason = sys.argv[1:7]

def art(name, pth, url, exp, act, digest_ok, origin, built_from):
    if not pth:
        return None
    return {
        "name": name,
        "path": pth,
        "url": url or None,
        "expected_sha256": exp or None,
        "actual_sha256": act or None,
        "digest_verified": digest_ok == "true",
        "origin": origin or None,
        "built_from_commit": built_from or None,
    }

arts = [a for a in (
    art("paddlefleet", os.environ.get("PADDLEFLEET_WHEEL_PATH", ""),
        os.environ.get("WHEEL_URL", ""), os.environ.get("WHEEL_SHA", ""),
        os.environ.get("ACTUAL_WHEEL_SHA", ""), os.environ.get("WHEEL_DIGEST_VERIFIED", "false"),
        os.environ.get("WHEEL_ORIGIN", ""), os.environ.get("WHEEL_BUILT_FROM_COMMIT", "")),
    art("paddlefleet_ops", os.environ.get("PADDLEFLEET_OPS_WHEEL_PATH", ""),
        os.environ.get("OPS_URL", ""), os.environ.get("OPS_SHA", ""),
        os.environ.get("ACTUAL_OPS_SHA", ""), os.environ.get("OPS_DIGEST_VERIFIED", "false"),
        os.environ.get("OPS_ORIGIN", ""), os.environ.get("OPS_BUILT_FROM_COMMIT", "")),
) if a]

doc = {
    "schema": "paddlefleet-alignment-pin/v1",
    "status": status,
    "detail": detail,
    "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mode": os.environ.get("MODE"),
    "dest": os.environ.get("DEST"),
    "source": {
        "git_url": os.environ.get("GIT_URL") or None,
        "expected_commit": os.environ.get("PIN_SHA") or None,
        "actual_commit": os.environ.get("ACTUAL_SHA") or None,
        "commit_verified": os.environ.get("SOURCE_VERIFIED") == "true",
    },
    "loaded_from": os.environ.get("LOADED_FROM") or None,
    "artifacts": arts,
    "pairing": {
        "stack_paired_proven": stack_proven == "true",
        "status": pairing_status,
        "reason": pairing_reason,
    },
    "default_urls": {
        "source_tar": os.environ.get("DEFAULT_TAR_URL"),
        "paddlefleet_wheel": os.environ.get("DEFAULT_WHL_URL"),
        "paddlefleet_ops_wheel": os.environ.get("DEFAULT_OPS_URL"),
    },
    "cases_preserved": ["MinimaxV2.5_EP2", "GLM45Air_EP2"],
    "unpaired_develop_rejected_in_stack_paired": True,
}
open(path, "w", encoding="utf-8").write(json.dumps(doc, indent=2) + "\n")
print("[paddlefleet-pin] receipt", path, file=sys.stderr)
PY
  RECEIPT_WRITTEN=1
}

export_receipt_env() {
  export MODE DEST GIT_URL PIN_SHA ACTUAL_SHA SOURCE_VERIFIED LOADED_FROM
  export PADDLEFLEET_WHEEL_PATH PADDLEFLEET_OPS_WHEEL_PATH
  export WHEEL_URL WHEEL_SHA ACTUAL_WHEEL_SHA WHEEL_DIGEST_VERIFIED WHEEL_ORIGIN WHEEL_BUILT_FROM_COMMIT
  export OPS_URL OPS_SHA ACTUAL_OPS_SHA OPS_DIGEST_VERIFIED OPS_ORIGIN OPS_BUILT_FROM_COMMIT
  export DEFAULT_TAR_URL DEFAULT_WHL_URL DEFAULT_OPS_URL
}

fail() {
  trap - ERR
  local msg="$1"
  log "FAIL: ${msg}"
  export_receipt_env
  write_receipt "error" "${msg}"
  echo "::error:: ${msg}" >&2
  exit 1
}

on_err() {
  local rc=$?
  if [[ "${RECEIPT_WRITTEN}" == 1 || "${RUN_SELF_TEST}" == 1 ]]; then
    return "${rc}"
  fi
  fail "command failed rc=${rc}"
}
trap 'on_err' ERR

write_envfile() {
  cat >"${DEST}/paddlefleet_alignment_pin.env" <<EOF
PADDLEFLEET_WHEEL_PATH=${PADDLEFLEET_WHEEL_PATH}
PADDLEFLEET_OPS_WHEEL_PATH=${PADDLEFLEET_OPS_WHEEL_PATH}
ALIGNMENT_PADDLEFLEET_MODE=${MODE}
PADDLEFLEET_PIN_RECEIPT=${DEST}/paddlefleet_alignment_pin_receipt.json
PADDLEFLEET_SOURCE_COMMIT=${ACTUAL_SHA}
PADDLEFLEET_PIN_SHA=${PIN_SHA}
PADDLEFLEET_WHEEL_ORIGIN=${WHEEL_ORIGIN}
PADDLEFLEET_OPS_ORIGIN=${OPS_ORIGIN}
PADDLEFLEET_WHEEL_DIGEST_VERIFIED=${WHEEL_DIGEST_VERIFIED}
PADDLEFLEET_OPS_DIGEST_VERIFIED=${OPS_DIGEST_VERIFIED}
EOF
}

# Local paths and file:// are copied. http(s) goes through wget so a PATH stub
# can keep fixtures offline. Failures call fail() in this shell (not $()).
download() {
  local url="$1" out="$2"
  log "download ${url} -> ${out}"
  mkdir -p "$(dirname "${out}")"
  if [[ "${url}" == file://* ]]; then
    local src="${url#file://}"
    [[ -f "${src}" ]] || fail "download failed, local file missing: ${src}"
    cp -- "${src}" "${out}" || fail "download copy failed: ${src}"
    return 0
  fi
  if [[ "${url}" == /* ]]; then
    [[ -f "${url}" ]] || fail "download failed, local file missing: ${url}"
    cp -- "${url}" "${out}" || fail "download copy failed: ${url}"
    return 0
  fi
  if wget -q --no-proxy --no-check-certificate --tries=2 --timeout=15 -O "${out}" "${url}"; then
    return 0
  fi
  fail "download failed: ${url}"
}

# Must not run inside $(); fail() has to exit this shell.
require_digest() {
  local path="$1" expected="$2" label="$3" actual="$4"
  [[ -f "${path}" ]] || fail "missing ${label}: ${path}"
  [[ -n "${expected}" ]] || fail "stack-paired missing ${label} sha256"
  if [[ "${actual}" != "${expected}" ]]; then
    fail "stack-paired ${label} sha256 mismatch expected=${expected} actual=${actual}"
  fi
}

checkout_pin() {
  [[ "${PIN_SHA}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "stack-paired requires PADDLEFLEET_PIN_SHA (40 hex), got '${PIN_SHA}'"
  PIN_SHA="$(printf '%s' "${PIN_SHA}" | tr 'A-F' 'a-f')"
  rm -rf "${DEST}/PaddleFleet"
  log "clone ${GIT_URL}"
  if ! git clone --quiet "${GIT_URL}" "${DEST}/PaddleFleet" >/dev/null 2>"${DEST}/.git-clone.err"; then
    fail "git clone failed: $(tr '\n' ' ' <"${DEST}/.git-clone.err")"
  fi
  git -C "${DEST}/PaddleFleet" config advice.detachedHead false || true
  log "checkout ${PIN_SHA}"
  if ! git -C "${DEST}/PaddleFleet" checkout --quiet --force "${PIN_SHA}" >/dev/null 2>"${DEST}/.git-co.err"; then
    fail "git checkout failed for ${PIN_SHA}: $(tr '\n' ' ' <"${DEST}/.git-co.err")"
  fi
  ACTUAL_SHA="$(git -C "${DEST}/PaddleFleet" rev-parse HEAD)"
  if [[ "${ACTUAL_SHA}" != "${PIN_SHA}" ]]; then
    SOURCE_VERIFIED=false
    fail "stack-paired source SHA mismatch expected=${PIN_SHA} actual=${ACTUAL_SHA}"
  fi
  SOURCE_VERIFIED=true
  log "source commit verified ${ACTUAL_SHA}"
}

# Sets DEST_PATH and DEST_SHA in the caller. Must run in this shell so
# fail() writes the receipt (never wrap this in $()).
acquire_explicit() {
  local url="$1" expected="$2" dest_name="$3" label="$4"
  unpaired_url "${url}" && fail "stack-paired rejects unpaired ${label} URL: ${url}"
  [[ -n "${expected}" ]] || fail "stack-paired ${label} URL requires matching sha256"
  download "${url}" "${DEST}/${dest_name}"
  DEST_PATH="${DEST}/${dest_name}"
  DEST_SHA="$(sha256_file "${DEST_PATH}")"
  # Record path/digest before require_digest so a mismatch receipt still has them.
  if [[ "${label}" == paddlefleet\ wheel ]]; then
    PADDLEFLEET_WHEEL_PATH="${DEST_PATH}"
    ACTUAL_WHEEL_SHA="${DEST_SHA}"
    WHEEL_ORIGIN="ci_metadata"
  else
    PADDLEFLEET_OPS_WHEEL_PATH="${DEST_PATH}"
    ACTUAL_OPS_SHA="${DEST_SHA}"
    OPS_ORIGIN="ci_metadata"
  fi
  require_digest "${DEST_PATH}" "${expected}" "${label}" "${DEST_SHA}"
  LOADED_FROM="${LOADED_FROM:+${LOADED_FROM};}${DEST_PATH} from ${url}"
}

run_build() {
  local cmd="$1" glob="$2" label="$3"
  mkdir -p "${DEST}/dist"
  log "build ${label}: ${cmd}"
  if ! (cd "${DEST}/PaddleFleet" && bash -lc "${cmd}"); then
    fail "stack-paired build failed for ${label}"
  fi
  local built
  built="$(ls -1 ${glob} 2>/dev/null | head -n 1 || true)"
  [[ -n "${built}" && -f "${built}" ]] || fail "stack-paired build produced no ${label} (glob ${glob})"
  local dest_name
  dest_name="$(basename "${built}")"
  cp -f -- "${built}" "${DEST}/${dest_name}"
  DEST_PATH="${DEST}/${dest_name}"
  DEST_SHA="$(sha256_file "${DEST_PATH}")"
  LOADED_FROM="${LOADED_FROM:+${LOADED_FROM};}built ${DEST_PATH} from ${ACTUAL_SHA}"
}

acquire_wheel() {
  if [[ -n "${WHEEL_URL}" ]]; then
    acquire_explicit "${WHEEL_URL}" "${WHEEL_SHA}" "paddlefleet.whl" "paddlefleet wheel"
    PADDLEFLEET_WHEEL_PATH="${DEST_PATH}"
    ACTUAL_WHEEL_SHA="${DEST_SHA}"
    WHEEL_DIGEST_VERIFIED=true
    WHEEL_ORIGIN="ci_metadata"
    WHEEL_BUILT_FROM_COMMIT=""
  else
    local cmd="${BUILD_CMD:-uv build --wheel --package paddlefleet --out-dir '${DEST}/dist' --clear}"
    run_build "${cmd}" "${DEST}/dist/paddlefleet-*.whl" "paddlefleet wheel"
    PADDLEFLEET_WHEEL_PATH="${DEST_PATH}"
    ACTUAL_WHEEL_SHA="${DEST_SHA}"
    WHEEL_DIGEST_VERIFIED=true
    WHEEL_ORIGIN="build"
    WHEEL_BUILT_FROM_COMMIT="${ACTUAL_SHA}"
  fi
}

acquire_ops() {
  if [[ -n "${OPS_URL}" ]]; then
    acquire_explicit "${OPS_URL}" "${OPS_SHA}" "paddlefleet_ops.whl" "paddlefleet_ops wheel"
    PADDLEFLEET_OPS_WHEEL_PATH="${DEST_PATH}"
    ACTUAL_OPS_SHA="${DEST_SHA}"
    OPS_DIGEST_VERIFIED=true
    OPS_ORIGIN="ci_metadata"
    OPS_BUILT_FROM_COMMIT=""
  else
    local cmd="${BUILD_OPS_CMD:-uv build --wheel --package paddlefleet-ops --out-dir '${DEST}/dist' --no-build-isolation}"
    run_build "${cmd}" "${DEST}/dist/paddlefleet_ops-*.whl" "paddlefleet_ops wheel"
    PADDLEFLEET_OPS_WHEEL_PATH="${DEST_PATH}"
    ACTUAL_OPS_SHA="${DEST_SHA}"
    OPS_DIGEST_VERIFIED=true
    OPS_ORIGIN="build"
    OPS_BUILT_FROM_COMMIT="${ACTUAL_SHA}"
  fi
}

fetch_default() {
  log "mode=develop (historical unpaired CodeSync tarball)"
  download "${DEFAULT_TAR_URL}" "${DEST}/PaddleFleet.tar"
  rm -rf "${DEST}/PaddleFleet"
  tar xf "${DEST}/PaddleFleet.tar" -C "${DEST}"
  rm -f "${DEST}/PaddleFleet.tar"
  if [[ -d "${DEST}/PaddleFleet/.git" ]]; then
    git -C "${DEST}/PaddleFleet" pull || log "git pull skipped"
    ACTUAL_SHA="$(git -C "${DEST}/PaddleFleet" rev-parse HEAD 2>/dev/null || true)"
  fi
  download "${DEFAULT_WHL_URL}" "${DEST}/paddlefleet-0.0.0-py3-none-linux_x86_64.whl"
  download "${DEFAULT_OPS_URL}" "${DEST}/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl"
  PADDLEFLEET_WHEEL_PATH="${DEST}/paddlefleet-0.0.0-py3-none-linux_x86_64.whl"
  PADDLEFLEET_OPS_WHEEL_PATH="${DEST}/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl"
  WHEEL_URL="${DEFAULT_WHL_URL}"
  OPS_URL="${DEFAULT_OPS_URL}"
  ACTUAL_WHEEL_SHA="$(sha256_file "${PADDLEFLEET_WHEEL_PATH}")"
  ACTUAL_OPS_SHA="$(sha256_file "${PADDLEFLEET_OPS_WHEEL_PATH}")"
  WHEEL_ORIGIN="develop_latest"
  OPS_ORIGIN="develop_latest"
  SOURCE_VERIFIED=false
  WHEEL_DIGEST_VERIFIED=false
  OPS_DIGEST_VERIFIED=false
  LOADED_FROM="develop_latest ${DEFAULT_WHL_URL} ${DEFAULT_OPS_URL}"
  export_receipt_env
  write_envfile
  write_receipt "ok" "develop tarball and develop/latest wheels; unpaired with a stack pin"
}

acquire_source_tree() {
  local src="${DEST}/PaddleFleet"
  local ops="${src}/packages/paddlefleet_ops"
  [[ -d "${src}" ]] || fail "stack-paired source tree missing: ${src}"
  [[ -f "${src}/pyproject.toml" ]] || fail "stack-paired source tree missing pyproject.toml: ${src}"
  [[ -d "${ops}" ]] || fail "stack-paired ops source tree missing: ${ops}"
  PADDLEFLEET_WHEEL_PATH="${src}"
  PADDLEFLEET_OPS_WHEEL_PATH="${ops}"
  WHEEL_ORIGIN="source_tree"
  OPS_ORIGIN="source_tree"
  WHEEL_BUILT_FROM_COMMIT="${ACTUAL_SHA}"
  OPS_BUILT_FROM_COMMIT="${ACTUAL_SHA}"
  WHEEL_DIGEST_VERIFIED=false
  OPS_DIGEST_VERIFIED=false
  LOADED_FROM="source_tree ${src} ${ops} from ${ACTUAL_SHA}"
  log "source-tree paths ${src} ${ops}"
}

fetch_stack_paired() {
  log "mode=stack-paired"
  checkout_pin
  if [[ -n "${WHEEL_URL}" || -n "${OPS_URL}" || -n "${BUILD_CMD}" || -n "${BUILD_OPS_CMD}" ]]; then
    acquire_wheel
    acquire_ops
  else
    # No CI wheel URL and no explicit build: export the checked-out trees.
    # A later docker exec must source paddlefleet_alignment_pin.env.
    acquire_source_tree
  fi
  export_receipt_env
  write_envfile
  local pair pairing_status
  pair="$(pairing_fields)"
  pair="${pair#*$'\t'}"
  pairing_status="${pair%%$'\t'*}"
  write_receipt "ok" "source_commit checked out; pairing.status=${pairing_status}; stack_paired_proven=false"
}

install_offline_stubs() {
  local bin="$1"
  mkdir -p "${bin}"
  cat >"${bin}/wget" <<'WGET'
#!/usr/bin/env bash
out=""
url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -O) out="$2"; shift 2 ;;
    --*) shift ;;
    *) url="$1"; shift ;;
  esac
done
if [[ -z "${url}" || -z "${out}" ]]; then
  echo "wget-stub: missing url/out" >&2
  exit 1
fi
if [[ "${url}" == http://* || "${url}" == https://* ]]; then
  echo "wget-stub: blocked network ${url}" >&2
  exit 1
fi
src="${url#file://}"
if [[ -f "${src}" ]]; then
  cp -- "${src}" "${out}"
  exit 0
fi
echo "wget-stub: not a local file ${url}" >&2
exit 1
WGET
  chmod +x "${bin}/wget"
}

run_self_test() {
  trap - ERR
  local root script
  root="$(mktemp -d)"
  script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
  trap 'rm -rf "${root}"' RETURN
  install_offline_stubs "${root}/bin"
  export PATH="${root}/bin:${PATH}"

  git init -q "${root}/upstream"
  git -C "${root}/upstream" config user.email test@example.com
  git -C "${root}/upstream" config user.name test
  echo source-a >"${root}/upstream/README"
  mkdir -p "${root}/upstream/packages/paddlefleet_ops"
  printf '%s\n' '[project]' 'name = "paddlefleet"' >"${root}/upstream/pyproject.toml"
  printf '%s\n' '[project]' 'name = "paddlefleet-ops"' >"${root}/upstream/packages/paddlefleet_ops/pyproject.toml"
  git -C "${root}/upstream" add README pyproject.toml packages
  git -C "${root}/upstream" commit -q -m a
  local sha_a sha_b
  sha_a="$(git -C "${root}/upstream" rev-parse HEAD)"
  echo source-b >"${root}/upstream/README"
  git -C "${root}/upstream" add README
  git -C "${root}/upstream" commit -q -m b
  sha_b="$(git -C "${root}/upstream" rev-parse HEAD)"

  mkdir -p "${root}/art"
  echo py-body >"${root}/art/py.whl"
  echo ops-body >"${root}/art/ops.whl"
  local py_sha ops_sha
  py_sha="$(sha256_file "${root}/art/py.whl")"
  ops_sha="$(sha256_file "${root}/art/ops.whl")"

  expect_fail() {
    local dest="$1"
    local needle="$2"
    shift 2
    mkdir -p "${dest}"
    if "$@"; then
      echo "self-test FAIL: expected failure (${needle})" >&2
      exit 1
    fi
    local rec="${dest}/paddlefleet_alignment_pin_receipt.json"
    [[ -f "${rec}" ]] || { echo "self-test FAIL: missing error receipt ${rec}" >&2; exit 1; }
    grep -q '"status": "error"' "${rec}"
    grep -q "${needle}" "${rec}"
    echo "[self-test] fail-closed ${dest}: ${needle}"
  }

  local run
  run() { env PATH="${root}/bin:${PATH}" "$@"; }

  expect_fail "${root}/m1" "PADDLEFLEET_PIN_SHA" \
    run ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA= \
    bash "${script}" --dest "${root}/m1"

  expect_fail "${root}/m2" "rejects unpaired" \
    run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
        PADDLEFLEET_PIN_SHA="${sha_b}" PADDLEFLEET_GIT_URL="${root}/upstream" \
        PADDLEFLEET_WHEEL_URL="${DEFAULT_WHL_URL}" \
        PADDLEFLEET_WHEEL_SHA256="${py_sha}" \
        PADDLEFLEET_OPS_WHEEL_URL="${root}/art/ops.whl" \
        PADDLEFLEET_OPS_WHEEL_SHA256="${ops_sha}" \
    bash "${script}" --dest "${root}/m2"

  expect_fail "${root}/m3" "git checkout failed" \
    run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
        PADDLEFLEET_PIN_SHA="0000000000000000000000000000000000000000" \
        PADDLEFLEET_GIT_URL="${root}/upstream" \
    bash "${script}" --dest "${root}/m3"

  # Real checksum mismatch after a successful local copy (not a wget miss).
  expect_fail "${root}/m4" "sha256 mismatch" \
    run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
        PADDLEFLEET_PIN_SHA="${sha_b}" PADDLEFLEET_GIT_URL="${root}/upstream" \
        PADDLEFLEET_WHEEL_URL="${root}/art/py.whl" \
        PADDLEFLEET_WHEEL_SHA256="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef" \
        PADDLEFLEET_OPS_WHEEL_URL="${root}/art/ops.whl" \
        PADDLEFLEET_OPS_WHEEL_SHA256="${ops_sha}" \
    bash "${script}" --dest "${root}/m4"
  python3 - "${root}/m4/paddlefleet_alignment_pin_receipt.json" "${py_sha}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["status"] == "error"
wheel = next(a for a in doc["artifacts"] if a["name"] == "paddlefleet")
assert wheel["actual_sha256"] == sys.argv[2]
assert wheel["digest_verified"] is False
assert wheel["actual_sha256"] != (wheel.get("expected_sha256") or "")
print("m4 checksum-mismatch receipt has actual digest, not a download miss")
PY

  expect_fail "${root}/m5" "produced no paddlefleet" \
    run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
        PADDLEFLEET_PIN_SHA="${sha_b}" PADDLEFLEET_GIT_URL="${root}/upstream" \
        PADDLEFLEET_BUILD_CMD="mkdir -p '${root}/m5/dist'" \
        PADDLEFLEET_BUILD_OPS_CMD="true" \
    bash "${script}" --dest "${root}/m5"

  expect_fail "${root}/m6" "download failed" \
    run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
        PADDLEFLEET_PIN_SHA="${sha_b}" PADDLEFLEET_GIT_URL="${root}/upstream" \
        PADDLEFLEET_WHEEL_URL="https://example.invalid/paddlefleet.whl" \
        PADDLEFLEET_WHEEL_SHA256="${py_sha}" \
        PADDLEFLEET_OPS_WHEEL_URL="${root}/art/ops.whl" \
        PADDLEFLEET_OPS_WHEEL_SHA256="${ops_sha}" \
    bash "${script}" --dest "${root}/m6"

  expect_fail "${root}/m7" "git clone failed" \
    run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
        PADDLEFLEET_PIN_SHA="${sha_b}" \
        PADDLEFLEET_GIT_URL="${root}/no-such-remote" \
    bash "${script}" --dest "${root}/m7"

  run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
      PADDLEFLEET_PIN_SHA="${sha_b}" PADDLEFLEET_GIT_URL="${root}/upstream" \
      PADDLEFLEET_WHEEL_URL="${root}/art/py.whl" \
      PADDLEFLEET_WHEEL_SHA256="${py_sha}" \
      PADDLEFLEET_OPS_WHEEL_URL="${root}/art/ops.whl" \
      PADDLEFLEET_OPS_WHEEL_SHA256="${ops_sha}" \
      bash "${script}" --dest "${root}/ok-url"
  python3 - "${root}/ok-url/paddlefleet_alignment_pin_receipt.json" "${sha_b}" "${py_sha}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
sha_b, py_sha = sys.argv[2], sys.argv[3]
assert doc["status"] == "ok"
assert doc["source"]["actual_commit"] == sha_b
assert doc["source"]["commit_verified"] is True
wheel = next(a for a in doc["artifacts"] if a["name"] == "paddlefleet")
assert wheel["actual_sha256"] == py_sha
assert wheel["actual_sha256"] != sha_b
assert wheel["digest_verified"] is True
assert wheel.get("built_from_commit") in (None, "")
assert "verified" not in wheel
assert doc["pairing"]["stack_paired_proven"] is False
assert doc["pairing"]["status"] == "unproven"
assert "MinimaxV2.5_EP2" in doc["cases_preserved"]
assert "GLM45Air_EP2" in doc["cases_preserved"]
print("ok-url receipt fields checked")
PY
  [[ "$(git -C "${root}/ok-url/PaddleFleet" rev-parse HEAD)" == "${sha_b}" ]]

  run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
      PADDLEFLEET_PIN_SHA="${sha_b}" PADDLEFLEET_GIT_URL="${root}/upstream" \
      PADDLEFLEET_BUILD_CMD="mkdir -p '${root}/ok-build/dist' && cp '${root}/art/py.whl' '${root}/ok-build/dist/paddlefleet-0.0.0-py3-none-any.whl'" \
      PADDLEFLEET_BUILD_OPS_CMD="mkdir -p '${root}/ok-build/dist' && cp '${root}/art/ops.whl' '${root}/ok-build/dist/paddlefleet_ops-0.0.0-py3-none-any.whl'" \
      bash "${script}" --dest "${root}/ok-build"
  python3 - "${root}/ok-build/paddlefleet_alignment_pin_receipt.json" "${sha_b}" "${py_sha}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
sha_b, py_sha = sys.argv[2], sys.argv[3]
assert doc["status"] == "ok"
wheel = next(a for a in doc["artifacts"] if a["name"] == "paddlefleet")
assert wheel["actual_sha256"] == py_sha
assert wheel["actual_sha256"] != sha_b
assert wheel["origin"] == "build"
assert wheel["built_from_commit"] == sha_b
assert doc["source"]["actual_commit"] == sha_b
assert doc["pairing"]["stack_paired_proven"] is False
assert doc["pairing"]["status"] == "built_from_checked_out_pin"
print("ok-build receipt fields checked")
PY
  grep -q "PADDLEFLEET_SOURCE_COMMIT=${sha_b}" "${root}/ok-build/paddlefleet_alignment_pin.env"

  run ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
      PADDLEFLEET_PIN_SHA="${sha_b}" PADDLEFLEET_GIT_URL="${root}/upstream" \
      bash "${script}" --dest "${root}/ok-source"
  python3 - "${root}/ok-source/paddlefleet_alignment_pin_receipt.json" "${sha_b}" "${root}/ok-source" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
sha_b, dest = sys.argv[2], sys.argv[3]
assert doc["status"] == "ok"
assert doc["source"]["actual_commit"] == sha_b
assert doc["source"]["commit_verified"] is True
wheel = next(a for a in doc["artifacts"] if a["name"] == "paddlefleet")
ops = next(a for a in doc["artifacts"] if a["name"] == "paddlefleet_ops")
assert wheel["path"] == f"{dest}/PaddleFleet"
assert ops["path"] == f"{dest}/PaddleFleet/packages/paddlefleet_ops"
assert wheel["origin"] == "source_tree"
assert ops["origin"] == "source_tree"
assert doc["pairing"]["status"] == "source_tree_from_checked_out_pin"
assert doc["pairing"]["stack_paired_proven"] is False
print("ok-source receipt fields checked")
PY
  grep -q "PADDLEFLEET_WHEEL_PATH=${root}/ok-source/PaddleFleet$" "${root}/ok-source/paddlefleet_alignment_pin.env"
  grep -q "PADDLEFLEET_OPS_WHEEL_PATH=${root}/ok-source/PaddleFleet/packages/paddlefleet_ops$" "${root}/ok-source/paddlefleet_alignment_pin.env"
  grep -q "PADDLEFLEET_WHEEL_ORIGIN=source_tree" "${root}/ok-source/paddlefleet_alignment_pin.env"
  grep -q "PADDLEFLEET_WHEEL_DIGEST_VERIFIED=false" "${root}/ok-source/paddlefleet_alignment_pin.env"
  grep -q "PADDLEFLEET_SOURCE_COMMIT=${sha_b}" "${root}/ok-source/paddlefleet_alignment_pin.env"

  grep -q 'CodeSync/develop/PaddleFleet.tar' "${script}"
  grep -q 'PaddleFleet/develop/latest/paddlefleet-0.0.0-py3-none-linux_x86_64.whl' "${script}"

  echo "select_paddlefleet_alignment_pin self-test OK"
}

if [[ "${RUN_SELF_TEST}" == 1 ]]; then
  run_self_test
  exit 0
fi

mkdir -p "${DEST}"
case "${MODE}" in
  stack-paired) fetch_stack_paired ;;
  develop) fetch_default ;;
  *) fail "unknown ALIGNMENT_PADDLEFLEET_MODE=${MODE} (develop|stack-paired)" ;;
esac
