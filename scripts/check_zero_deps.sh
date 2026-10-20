#!/usr/bin/env bash
# Verify the core distribution declares and pulls no runtime dependencies.
# Runs inside the dev container (see ./dev zero-deps).
set -euo pipefail

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> build rampgate wheel"
uv build --package rampgate --wheel --out-dir "$WORK/dist" -q
WHEEL="$(ls "$WORK"/dist/rampgate-*.whl)"

echo "==> check METADATA has no Requires-Dist"
uv run --no-project --python 3.13 python - "$WHEEL" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
    requires = [l for l in zf.read(meta).decode().splitlines() if l.startswith("Requires-Dist:")]
if requires:
    sys.exit(f"core declares runtime dependencies: {requires}")
print("    no Requires-Dist")
EOF

echo "==> pip install ./packages/rampgate into a clean venv"
uv venv --python 3.13 --seed -q "$WORK/venv"
"$WORK/venv/bin/pip" install -q ./packages/rampgate
installed="$("$WORK/venv/bin/pip" list --format=freeze | cut -d= -f1 | sort | tr '\n' ' ')"
expected="pip rampgate "
if [[ "$installed" != "$expected" ]]; then
    echo "unexpected packages installed: $installed (expected: $expected)" >&2
    exit 1
fi
echo "    installed: $installed"
echo "==> zero-dependency check OK"
