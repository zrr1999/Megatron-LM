#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# End-to-end path check: selector source-mode -> new-shell consume ->
# setup_venvs *path* consumer. This proves docker-exec handoff of source
# paths. It is NOT a uv install / real setup_venvs / numerical CI run.
# Isolated selector --self-test is not enough for the step boundary.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELECTOR="${ROOT}/select_paddlefleet_alignment_pin.sh"
CONSUME="${ROOT}/consume_paddlefleet_alignment_pin.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

git init -q "${tmp}/upstream"
git -C "${tmp}/upstream" config user.email test@example.com
git -C "${tmp}/upstream" config user.name test
mkdir -p "${tmp}/upstream/packages/paddlefleet_ops"
printf '%s\n' '[project]' 'name = "paddlefleet"' >"${tmp}/upstream/pyproject.toml"
printf '%s\n' '[project]' 'name = "paddlefleet-ops"' >"${tmp}/upstream/packages/paddlefleet_ops/pyproject.toml"
echo src >"${tmp}/upstream/README"
git -C "${tmp}/upstream" add README pyproject.toml packages
git -C "${tmp}/upstream" commit -q -m pin
PIN="$(git -C "${tmp}/upstream" rev-parse HEAD)"

# Step A: Get Whl equivalent (selector).
ALIGNMENT_PADDLEFLEET_MODE=stack-paired \
  PADDLEFLEET_PIN_SHA="${PIN}" \
  PADDLEFLEET_GIT_URL="${tmp}/upstream" \
  bash "${SELECTOR}" --dest "${tmp}/ws"

test -f "${tmp}/ws/paddlefleet_alignment_pin.env"
test -d "${tmp}/ws/PaddleFleet"

# Step B: new docker exec — drop selector shell state, keep only files.
# Requested mode/pin stay on the caller; leftover develop env must not win.
unset PADDLEFLEET_WHEEL_PATH PADDLEFLEET_OPS_WHEEL_PATH PADDLEFLEET_SOURCE_COMMIT || true
ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA="${PIN}" \
  bash "${CONSUME}" --env "${tmp}/ws/paddlefleet_alignment_pin.env" --out "${tmp}/ws/consumed.env"

# Negative: leftover develop env + requested stack-paired must fail closed.
mkdir -p "${tmp}/dev"
echo dummy >"${tmp}/dev/paddlefleet-0.0.0-py3-none-linux_x86_64.whl"
echo dummy >"${tmp}/dev/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl"
cat >"${tmp}/dev.env" <<EOF
PADDLEFLEET_WHEEL_PATH=${tmp}/dev/paddlefleet-0.0.0-py3-none-linux_x86_64.whl
PADDLEFLEET_OPS_WHEEL_PATH=${tmp}/dev/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl
ALIGNMENT_PADDLEFLEET_MODE=develop
PADDLEFLEET_WHEEL_ORIGIN=develop_latest
PADDLEFLEET_OPS_ORIGIN=develop_latest
EOF
if ALIGNMENT_PADDLEFLEET_MODE=stack-paired PADDLEFLEET_PIN_SHA="${PIN}" \
    bash "${CONSUME}" --env "${tmp}/dev.env" --out "${tmp}/dev.consumed.env"; then
  echo "handoff FAIL: leftover develop env overrode stack-paired" >&2
  exit 1
fi

# Step C: path consumer only. Does not run uv or setup_venvs.sh.
stub_setup="${tmp}/setup_path_consumer.sh"
cat >"${stub_setup}" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
# Mirrors setup_venvs.sh reading PADDLEFLEET_WHEEL_PATH. Path presence only.
PADDLEFLEET_WHEEL="${PADDLEFLEET_WHEEL_PATH:?missing PADDLEFLEET_WHEEL_PATH}"
PADDLEFLEET_OPS_WHEEL="${PADDLEFLEET_OPS_WHEEL_PATH:?missing PADDLEFLEET_OPS_WHEEL_PATH}"
[[ -d "${PADDLEFLEET_WHEEL}" || -f "${PADDLEFLEET_WHEEL}" ]] || { echo "missing ${PADDLEFLEET_WHEEL}" >&2; exit 1; }
[[ -d "${PADDLEFLEET_OPS_WHEEL}" || -f "${PADDLEFLEET_OPS_WHEEL}" ]] || { echo "missing ${PADDLEFLEET_OPS_WHEEL}" >&2; exit 1; }
echo "PATH_CONSUMER paddlefleet=${PADDLEFLEET_WHEEL}"
echo "PATH_CONSUMER ops=${PADDLEFLEET_OPS_WHEEL}"
echo "PATH_CONSUMER not_uv_install=true"
STUB
chmod +x "${stub_setup}"

set -a
# shellcheck disable=SC1090
. "${tmp}/ws/consumed.env"
set +a
bash "${stub_setup}" | tee "${tmp}/setup.out"
grep -q "PATH_CONSUMER paddlefleet=${tmp}/ws/PaddleFleet" "${tmp}/setup.out"
grep -q "packages/paddlefleet_ops" "${tmp}/setup.out"
grep -q "not_uv_install=true" "${tmp}/setup.out"

echo "paddlefleet pin handoff PATH_PASS pin=${PIN} (not uv install, not CI)"
