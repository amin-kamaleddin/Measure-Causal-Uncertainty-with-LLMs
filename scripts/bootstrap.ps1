$ErrorActionPreference = 'Stop'
python -m pip install -e '.[dev]'
python -m sharecausal.cli doctor
pytest -q
