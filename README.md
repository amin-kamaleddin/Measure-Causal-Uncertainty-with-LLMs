# Identification-aware causal uncertainty

Exploratory executable evaluation of real LLM graph proposals, evidence screening, restricted backdoor identification, conditional estimation and abstention. Includes synthetic generators/tests and locally licensed SHARE analysis code. No SHARE data, empirical result tables, participant caches, credentials, private provider transcripts or protected scoring stores are included.

Install: `python -m pip install -e '.[dev]'`
Check: `python scripts/check_protocol_freeze.py`
Test: `pytest -q`
Final experiment CLI: `PYTHONPATH=src python scripts/synthetic_final_v1.py --help`

The final batch has96 primary requests plus32 counterargument requests across24 public cards and four exact models. Paid execution requires explicit authorization, a private output directory outside this repository and a runtime key. Do not rerun completed experiments: use their sealed receipts. The public aggregate release reports all actual failures and costs. No confirmatory superiority, structural coverage or causal SHARE effect is established.

Synthetic authoring source intentionally contains generator definitions for reproducibility. Protected live truth files and raw responses are excluded. Local source hashes are prospective commitments, not external preregistration. Single-team scenario definitions are not independently validated mechanism families. Earlier negative comparisons and precision diagnostics remain limitations.

See `docs/final_analysis_plan_exploratory.md`, frozen protocols and `outputs/synthetic/final_reviewed/` for reviewed synthetic aggregates when present. SHARE analyses require separate individual data registration and human disclosure review; the private empirical manuscript is deliberately omitted.

This is a clean-history release snapshot. No earlier repository history or private manuscript artifacts are imported. GitHub upload status is recorded in RELEASE_STATUS.md; local existence is not evidence of a remote publication.

Empirical reproduction requires individually licensed SHARE files. The memory-only reader uses pandas; pyreadstat is optional. The observed local versions are in observed_environment.json; only synthetic tests ran in the release snapshot. No remote CI result is claimed.
