#!/usr/bin/env bash
# Integration tests (Docker daemon, GPU), or a command, inside ./dev integration.
set -euo pipefail

export UV_PROJECT_ENVIRONMENT=/tmp/venv-3.13
uv sync --all-packages --python 3.13 --locked -q

if [[ $# -gt 0 ]]; then
    exec uv run --no-sync "$@"
fi
echo "==> integration: docker + nvml live tests"
uv run --no-sync pytest -q -rs -p no:cacheprovider -m docker packages/rampgate-docker
uv run --no-sync pytest -q -rs -p no:cacheprovider packages/rampgate-nvml/tests/test_probe.py
echo "==> integration OK"
