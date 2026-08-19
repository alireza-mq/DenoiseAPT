#!/usr/bin/env bash
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$DEMO_ROOT/.venv/bin/python" ]]; then
  python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' || {
    printf 'Python 3.10--3.12 is required.\n' >&2
    exit 1
  }
  python3 -m venv "$DEMO_ROOT/.venv"
fi
"$DEMO_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$DEMO_ROOT/.venv/bin/python" -m pip install -r "$DEMO_ROOT/requirements.txt"
"$DEMO_ROOT/.venv/bin/python" -m pip install -e "$DEMO_ROOT"
if [[ "${DOWNLOAD_DATASET:-0}" == "1" ]]; then
  "$DEMO_ROOT/.venv/bin/python" "$DEMO_ROOT/scripts/download_data.py"
fi
"$DEMO_ROOT/.venv/bin/python" "$DEMO_ROOT/scripts/prepare_demo_case.py" --ensure-demo-case
printf 'DenoiseAPT setup is complete. Run ./run_demo.sh\n'
