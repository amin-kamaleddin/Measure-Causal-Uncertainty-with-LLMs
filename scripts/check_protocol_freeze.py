"""Verify frozen v1 algorithms and any explicitly recorded test-only erratum."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'docs/freeze_manifest_v1.json').read_text())
erratum_path = ROOT / 'docs/test_only_erratum_v1.json'
erratum = json.loads(erratum_path.read_text()) if erratum_path.exists() else None
accepted_errata = []
for filename, expected in manifest['file_sha256'].items():
    got = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    if got == expected:
        continue
    if (erratum is not None
        and filename == erratum['changed_frozen_file'] == 'tests/test_nonoracle.py'
        and expected == erratum['frozen_sha256']
        and got == erratum['patched_sha256']
        and erratum['causal_analysis_code_changed'] is False):
        accepted_errata.append(filename)
        continue
    raise SystemExit(f'FROZEN FILE CHANGED: {filename}; amend protocol and use new version')
print(f"Frozen protocol v1 OK: {len(manifest['file_sha256'])} checked; "
      f"{len(accepted_errata)} documented test-only erratum accepted")
