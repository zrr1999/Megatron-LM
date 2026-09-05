#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Syntax-check every workflow `run:` block, then execute the extracted
# Get Whl docker-exec body against a failing selector. Independent
# helper --self-test is not this check.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
YAML="${ROOT}/../.github/workflows/alignment_model_accuracy.yml"
SELECTOR_PATH="/workspace/Megatron-LM/scripts/select_paddlefleet_alignment_pin.sh"
REQUIRE_PATH="/workspace/Megatron-LM/scripts/require_paddlefleet_selector_ok.sh"
BUILD_MARKER="build megatron-core"

python3 - "${YAML}" "${ROOT}" "${SELECTOR_PATH}" "${REQUIRE_PATH}" "${BUILD_MARKER}" <<'PY'
import os, re, subprocess, sys, tempfile, textwrap, pathlib, json, stat

yaml_path, scripts_root, selector_path, require_path, build_marker = sys.argv[1:6]
text = pathlib.Path(yaml_path).read_text()
lines = text.splitlines(True)

blocks = []
i = 0
while i < len(lines):
    m = re.match(r"^(\s*)run:\s*\|\s*$", lines[i])
    if not m:
        i += 1
        continue
    indent = len(m.group(1))
    name = "unnamed"
    for j in range(i, -1, -1):
        nm = re.match(r"^\s+- name:\s*(.*)$", lines[j])
        if nm:
            name = nm.group(1).strip()
            break
    i += 1
    body = []
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            body.append(line)
            i += 1
            continue
        lead = len(line) - len(line.lstrip(" "))
        if lead <= indent and line.strip():
            break
        body.append(line[indent + 2 :] if lead >= indent + 2 else line.lstrip())
        i += 1
    blocks.append((name, "".join(body)))

if not blocks:
    raise SystemExit(f"no run: | blocks in {yaml_path}")

tmp = pathlib.Path(tempfile.mkdtemp(prefix="yaml-run-"))
print(f"extracted {len(blocks)} run blocks from {yaml_path}")
for n, body in blocks:
    p = tmp / (re.sub(r"[^A-Za-z0-9._-]+", "_", n) + ".sh")
    p.write_text("#!/usr/bin/env bash\n" + body)
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"bash -n FAIL {n}: {r.stderr}")
    print(f"bash -n OK run:{n}")

    for m in re.finditer(r"""/bin/bash -c\s+'""", body):
        start = m.end()
        end = body.find("'\n", start)
        if end < 0:
            end = body.rfind("'")
        inner = body[start:end]
        if "'" in inner:
            raise SystemExit(
                f"nested single quote inside docker exec -c in {n}: "
                f"{inner[inner.find(chr(39))-40:inner.find(chr(39))+40]!r}"
            )
        inner_p = tmp / (p.stem + ".docker-inner.sh")
        inner_p.write_text("#!/usr/bin/env bash\n" + inner + "\n")
        r = subprocess.run(["bash", "-n", str(inner_p)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"bash -n FAIL docker-inner {n}: {r.stderr}")
        print(f"bash -n OK docker-inner:{n} (no nested single quotes)")

getwhl = next((b for n, b in blocks if n == "Get Whl"), None)
if getwhl is None:
    raise SystemExit("Get Whl run block missing")
m = re.search(r"""/bin/bash -c\s+'""", getwhl)
if not m:
    raise SystemExit("Get Whl docker exec -c missing")
inner = getwhl[m.end():]
end = inner.rfind("'")
inner = inner[:end]

ws = tmp / "ws"
(ws / "Megatron-LM/scripts").mkdir(parents=True)
(ws / "upload").mkdir(parents=True)
selector = ws / "Megatron-LM/scripts/select_paddlefleet_alignment_pin.sh"
require_src = pathlib.Path(scripts_root) / "require_paddlefleet_selector_ok.sh"
require_dst = ws / "Megatron-LM/scripts/require_paddlefleet_selector_ok.sh"
require_dst.write_text(require_src.read_text())
require_dst.chmod(require_dst.stat().st_mode | stat.S_IXUSR)
selector.write_text(textwrap.dedent("""\
    #!/usr/bin/env bash
    set -euo pipefail
    dest="${2:-/workspace}"
    mkdir -p "${dest}"
    cat >"${dest}/paddlefleet_alignment_pin_receipt.json" <<'EOF'
    {"schema":"paddlefleet-alignment-pin/v1","status":"error","detail":"git clone failed: github.com:443","mode":"stack-paired"}
    EOF
    echo "[paddlefleet-pin] FAIL: git clone failed: github.com:443" >&2
    echo "::error:: git clone failed: github.com:443" >&2
    exit 1
    """))
selector.chmod(selector.stat().st_mode | stat.S_IXUSR)
(ws / "Megatron-LM/scripts/dependence").mkdir(parents=True, exist_ok=True)
(ws / "Megatron-LM/scripts/dependence/build.sh").write_text("#!/usr/bin/env bash\necho BUILD_RAN > /workspace/upload/BUILD_RAN\n")
(ws / "Megatron-LM/scripts/dependence/build.sh").chmod(0o755)

rewritten = inner.replace("/workspace", str(ws))
rewritten = rewritten.replace("conda activate py_$python_version", "true")
rewritten = rewritten.replace(". /opt/conda/etc/profile.d/conda.sh", "true")
rewritten = "#!/usr/bin/env bash\nexport ALIGNMENT_PADDLEFLEET_MODE=stack-paired\nexport python_version=3.12\n" + rewritten
bin = tmp / "bin"
bin.mkdir()
(bin / "wget").write_text("#!/usr/bin/env bash\necho WGET_RAN \"$@\" >> '%s/WGET_RAN'\nexit 0\n" % ws)
(bin / "python").write_text("#!/usr/bin/env bash\necho PY_RAN \"$@\" >> '%s/PY_RAN'\nexit 0\n" % ws)
(bin / "python3").write_text("#!/usr/bin/env bash\nexec /usr/bin/python3 \"$@\"\n")
(bin / "pip").write_text("#!/usr/bin/env bash\necho PIP_RAN \"$@\" >> '%s/PIP_RAN'\nexit 0\n" % ws)
(bin / "ldconfig").write_text("#!/usr/bin/env bash\nexit 0\n")
for f in bin.iterdir():
    f.chmod(0o755)

script = tmp / "getwhl.extracted.sh"
script.write_text(rewritten)
script.chmod(0o755)
env = os.environ.copy()
env["PATH"] = str(bin) + ":" + env.get("PATH", "")
env["ALIGNMENT_PADDLEFLEET_MODE"] = "stack-paired"
r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
log = (r.stdout or "") + (r.stderr or "")
print("extracted Get Whl rc=", r.returncode)
print(log[-2000:])
if r.returncode == 0:
    raise SystemExit("FAIL: extracted Get Whl continued after selector failure")
if (ws / "WGET_RAN").exists():
    raise SystemExit("FAIL: wget ran after selector failure")
if (ws / "upload/BUILD_RAN").exists():
    raise SystemExit("FAIL: build.sh ran after selector failure")
if "selector failed; stop Get Whl" not in log and "git clone failed" not in log:
    raise SystemExit("FAIL: extracted Get Whl did not surface selector failure")
print("extracted Get Whl fixture: selector fail stopped remaining wheels and build")
print("workflow shell checks OK")
PY
echo "alignment workflow shell PATH_PASS (extracted YAML, not helper-only)"
