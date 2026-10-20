#!/usr/bin/env bash
# CI for one Python version. Runs inside the dev container (see ./dev).
set -euo pipefail

PY="${1:?usage: ci.sh <python-version>}"
export UV_PROJECT_ENVIRONMENT="/tmp/venv-$PY"

echo "==> [py$PY] sync"
uv sync --all-packages --python "$PY" --locked -q

echo "==> [py$PY] ruff"
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .

echo "==> [py$PY] mypy --strict"
uv run --no-sync mypy

echo "==> [py$PY] pytest"
uv run --no-sync pytest -q -p no:cacheprovider \
    --cov --cov-report=term-missing packages

echo "==> [py$PY] OK"
