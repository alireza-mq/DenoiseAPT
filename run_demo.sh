#!/usr/bin/env bash
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8765}"
exec "$DEMO_ROOT/.venv/bin/python" "$DEMO_ROOT/server.py" --host 127.0.0.1 --port "$PORT"

