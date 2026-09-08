#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON_BIN:-python3}" - "$ROOT" <<'PY'
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1])
workflow = (root / '.github/workflows/alignment_model_accuracy.yml').read_text()
start = workflow.index('          bash Megatron-LM/scripts/prepare_paddlefleet_ops_submodules.sh')
end = workflow.index('\n          model_acc_align_exit_code=', start)
commands = workflow[start:end]
assert commands.count('setup_venvs.sh') == 2
assert commands.count('run_alignment_test.sh') == 1
assert '          set -eo pipefail' in workflow[:start]
real_timeout = shutil.which('timeout')
assert real_timeout
with tempfile.TemporaryDirectory() as directory:
    work = pathlib.Path(directory)
    (work / 'bin').mkdir()
    (work / 'ops').mkdir()
    (work / 'Megatron-LM').symlink_to(root)
    fleet = work / 'PaddleFleet/scripts/alignment_model_accuracy'
    fleet.mkdir(parents=True)
    setup = fleet / 'setup_venvs.sh'
    setup.write_text('''#!/usr/bin/env bash
set -eu
if [[ ${1:-} != --prepare-ops-submodules ]]; then echo setup >> "$TRACE"; exit 0; fi
[[ $2 == "$OPS" ]]
echo prepare >> "$TRACE"
n=$(grep -c prepare "$TRACE")
case "$CASE" in
 success) exit 0;;
 transient) [[ $n -ge 2 ]];;
 persistent) exit 23;;
 timeout) /bin/sleep 5;;
esac
''')
    (fleet / 'run_alignment_test.sh').write_text('echo train >> "$TRACE"\n')
    sleeper = work / 'bin/sleep'
    sleeper.write_text('#!/usr/bin/env bash\necho "sleep:$1" >> "$TRACE"\n')
    timer = work / 'bin/timeout'
    timer.write_text('''#!/usr/bin/env bash
set -eu
[[ $1 == --kill-after=30s && $2 == 15m ]]
echo bound >> "$TRACE"
shift 2
if [[ $CASE == timeout ]]; then exec "$REAL_TIMEOUT" --kill-after=0.1s 0.1s "$@"; fi
exec "$@"
''')
    sleeper.chmod(0o755)
    timer.chmod(0o755)
    env = dict(os.environ, PATH=str(work / 'bin') + ':' + os.environ['PATH'],
               TRACE=str(work / 'trace'), OPS=str(work / 'ops'), REAL_TIMEOUT=real_timeout)
    for case, count, code in [('success', 1, 0), ('transient', 2, 0),
                              ('persistent', 3, 23), ('timeout', 3, 124),
                              ('wheel', 0, 0)]:
        trace = work / 'trace'
        trace.write_text('')
        env.update(CASE=case, PADDLEFLEET_OPS_WHEEL_PATH=env['OPS'] if case != 'wheel' else str(work / 'ops.whl'))
        result = subprocess.run(['bash', '-eo', 'pipefail', '-c', commands], cwd=work,
                                env=env, capture_output=True, text=True, timeout=10)
        lines = trace.read_text().splitlines()
        assert result.returncode == code, (case, result.returncode, result.stderr)
        assert lines.count('prepare') == count, (case, lines)
        assert lines.count('bound') == count, (case, lines)
        assert [x for x in lines if x.startswith('sleep:')] == ['sleep:15', 'sleep:30'][:max(0, count - 1)], (case, lines)
        assert ('setup' in lines) == (code == 0), (case, lines)
        assert ('train' in lines) == (code == 0), (case, lines)
        assert lines.count('setup') <= 1 and lines.count('train') <= 1
        print(f'PASS: extracted workflow {case}, attempts={count}, exit={code}')
print('All source-submodule preflight fixtures passed')
PY
