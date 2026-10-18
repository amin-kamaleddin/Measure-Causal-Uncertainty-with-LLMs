#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[dev]'
python -m sharecausal.cli doctor
pytest -q
