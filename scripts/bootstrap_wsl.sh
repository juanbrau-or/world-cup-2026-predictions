#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it before running this script." >&2
  exit 1
fi

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

[[ -f .env ]] || cp .env.example .env
uv sync --group dev
uv run wc2026 doctor
uv run ruff check .
uv run mypy src
uv run pytest
