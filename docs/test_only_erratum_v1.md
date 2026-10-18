# Mac test portability erratum (v1.0.1) — 2026-10-16

**Status: tests only.** This correction leaves the internally frozen v1
scientific protocol, synthetic SCMs, estimation algorithms, analysis decisions,
configuration, random seeds and benchmark outputs unchanged. It is not a new
preregistration and does not retrospectively alter the frozen benchmark claims.
The Git history records the original test, and the original
`docs/freeze_manifest_v1.json` remains unchanged.

The original `test_unobserved_outcome_is_not_leaked_into_prediction` used
`assert x == y` on nested model dictionaries containing floating-point AIPW
results. Linear algebra implementations can exhibit tiny rounding variations
between separately evaluated, but observationally identical, DataFrames.
The corrected test confirms that observed-case input rows are *exactly*
identical, requires exact equality for structural decisions/reasons and all
nonfloat diagnostics, and allows only relative/absolute `1e-8` error in
floating-point fields. It still fails on substantive leakage and changed
assertion decisions. This is a cross-platform hypothesis; the Mac failure's
magnitude has not been measured here because the original error output was
truncated. If the revised test fails, investigate the displayed differing field
rather than raising tolerances without justification.

The original `test_no_raw_data_in_repository` traversed the whole `.venv`
directory and falsely flagged the `.dta` example data distributed with
`statsmodels`. The new test inspects tracked and unignored Git publication
candidates; force-tracked raw data will still cause failure. Where Git
metadata is absent, the fallback ignores virtual environments and locally
protected private-data directories. The release audit and human review remain
required to detect other kinds of sensitive research outputs.

`docs/test_only_erratum_v1.json` records the original and revised test hash.
`scripts/check_protocol_freeze.py` still enforces all untouched frozen hashes
exactly and reports the single permitted test-only change explicitly.

Validation to run on macOS:

```
python -m pytest -q
python scripts/check_protocol_freeze.py
```

Prefer the project's previously validated Python 3.13 environment for strict
reproducibility if a newer development version of Python behaves differently.
Do not re-run or overwrite historical frozen benchmark results merely to resolve
these tests. Live OpenRouter calls remain disabled.
