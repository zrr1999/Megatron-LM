#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# End-to-end: selector source-mode -> new-shell consume -> setup_venvs consumer.
# This is the docker-exec step boundary. Isolated selector --self-test is not enough.

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
unset PADDLEFLEET_WHEEL_PATH PADDLEFLEET_OPS_WHEEL_PATH PADDLEFLEET_SOURCE_COMMIT || true
export ALIGNMENT_PADDLEFLEET_MODE=stack-paired
bash "${CONSUME}" --env "${tmp}/ws/paddlefleet_alignment_pin.env" --out "${tmp}/ws/consumed.env"

# Step C: setup_venvs consumer. Fail if it would install the 0.0.0 develop wheel.
stub_setup="${tmp}/setup_venvs.sh"
cat >"${stub_setup}" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
# Mirrors setup_venvs.sh: it only sees exported PADDLEFLEET_WHEEL_PATH.
PADDLEFLEET_WHEEL="${PADDLEFLEET_WHEEL_PATH:-/workspace/paddlefleet-0.0.0-py3-none-linux_x86_64.whl}"
PADDLEFLEET_OPS_WHEEL="${PADDLEFLEET_OPS_WHEEL_PATH:-/workspace/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl}"
case "${PADDLEFLEET_WHEEL}" in
  */paddlefleet-0.0.0-py3-none-linux_x86_64.whl)
    echo "STUB_SETUP used hardcoded 0.0.0 wheel" >&2
    exit 1
    ;;
esac
[[ -d "${PADDLEFLEET_WHEEL}" || -f "${PADDLEFLEET_WHEEL}" ]] || { echo "missing ${PADDLEFLEET_WHEEL}" >&2; exit 1; }
[[ -d "${PADDLEFLEET_OPS_WHEEL}" || -f "${PADDLEFLEET_OPS_WHEEL}" ]] || { echo "missing ${PADDLEFLEET_OPS_WHEEL}" >&2; exit 1; }
echo "STUB_SETUP paddlefleet=${PADDLEFLEET_WHEEL}"
echo "STUB_SETUP ops=${PADDLEFLEET_OPS_WHEEL}"
STUB
chmod +x "${stub_setup}"

set -a
# shellcheck disable=SC1090
. "${tmp}/ws/consumed.env"
set +a
bash "${stub_setup}" | tee "${tmp}/setup.out"
grep -q "STUB_SETUP paddlefleet=${tmp}/ws/PaddleFleet" "${tmp}/setup.out"
grep -q "packages/paddlefleet_ops" "${tmp}/setup.out"

# Negative: alignment step that ignores consumed.env and defaults 0.0.0 must fail the stub.
if PADDLEFLEET_WHEEL_PATH=/workspace/paddlefleet-0.0.0-py3-none-linux_x86_64.whl \
   PADDLEFLEET_OPS_WHEEL_PATH=/workspace/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl \
   bash "${stub_setup}"; then
  echo "handoff FAIL: stub accepted 0.0.0 default" >&2
  exit 1
fi

echo "paddlefleet pin handoff OK pin=${PIN}"
