#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"

exec python -m exroma_bench.cli "$@"
