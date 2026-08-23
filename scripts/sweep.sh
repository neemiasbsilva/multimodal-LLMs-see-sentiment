#!/usr/bin/env bash
# Thin wrapper so a sweep inherits the GPU environment. All arguments are
# forwarded: scripts/sweep.sh --track finetuning --backbone modernbert
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
exec uv run mllmsent sweep "$@"
