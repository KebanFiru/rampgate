#!/usr/bin/env bash
# Before a release: build all four distributions, install them from the wheels into
# clean virtualenvs, and run the README quickstart. Runs inside the dev container
# (./dev release-check). The adapters' third-party dependencies come from PyPI.
set -euo pipefail

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> build sdists and wheels"
for p in rampgate rampgate-nvml rampgate-psutil rampgate-docker; do
    uv build --package "$p" --out-dir "$WORK/dist" -q
done
ls "$WORK/dist"

echo "==> check metadata (twine check)"
UV_TOOL_DIR="$WORK/tools" UV_TOOL_BIN_DIR="$WORK/bin" uvx -q twine check --strict "$WORK"/dist/*

echo "==> clean venv: pip install rampgate (wheel only), then the README quickstart"
uv venv --python 3.13 --seed -q "$WORK/core"
"$WORK/core/bin/pip" install -q --no-index "$WORK"/dist/rampgate-*-py3-none-any.whl
"$WORK/core/bin/python" examples/quickstart.py > "$WORK/quickstart.out"
tail -1 "$WORK/quickstart.out"
grep -q "^OOM kills: " "$WORK/quickstart.out"
installed="$("$WORK/core/bin/pip" list --format=freeze | cut -d= -f1 | sort | tr '\n' ' ')"
[[ "$installed" == "pip rampgate " ]] || { echo "unexpected: $installed" >&2; exit 1; }

echo "==> clean venv: the three adapters from their wheels"
uv venv --python 3.13 --seed -q "$WORK/adapters"
"$WORK/adapters/bin/pip" install -q --find-links "$WORK/dist" \
    rampgate-nvml==0.1.0 rampgate-psutil==0.1.0 rampgate-docker==0.1.0
"$WORK/adapters/bin/python" -c "
import rampgate, rampgate_docker, rampgate_nvml, rampgate_psutil
for m in (rampgate, rampgate_docker, rampgate_nvml, rampgate_psutil):
    print('   ', m.__name__, m.__version__)
    assert m.__version__ == '0.1.0', m
"
echo "==> release check OK"
