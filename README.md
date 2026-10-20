# Identification-aware causal uncertainty

An executed exploratory study of evidence-grounded real LLM graphs, graph uncertainty, restricted causal identification, conditional estimation and abstention. Includes synthetic generators, tested provider infrastructure and local licensed-data analysis code. SHARE data, empirical findings, credentials, private transcripts and live scoring stores are excluded.

## Final evidence-grounded experiment

[Complete reviewed synthetic results](outputs/synthetic/evidence_grounded_v31_reviewed/README.md): 120 public cards, 36 development and 84 evaluation cards, 20 declared families (14 evaluation units). Exactly four selected models, two prompt variants and two repetitions give 1,920 primary slots and 120 critics. The original provider-slug error and all other failures are preserved. Across the original and repaired runs: 2,528 unique requests, USD 2.3540124765 in reported generation charges, zero unknown costs. The corrected schedule reused 1,468 receipts without duplicate paid calls. Whole software ceiling USD 100.

The full policy and strong equally informed evidence-only baseline asserted on exactly the same 16/84 cards, with no recorded unjustified assertions. Majority voting had two practical-null assertions (zero literal sign reversals) in 16 assertions. Neither primary policy attained the planned 30–50% assertion range. No confirmatory improvement, calibrated structural coverage or unconditional clinical intervention effect is established. The critic veto is redundant with the compiler under this fixed rule. Repeats are nested, not independent observations. Single-team authorship, known prose grammar, familiar causal motifs and sparse 14-family uncertainty limit generalization.

Earlier 48- and 128-call pilots, frozen protocols and the severe prior undercoverage finding remain separate and unchanged. [Frozen plan](docs/final_study_v3_locked_plan.md), [routing amendment](docs/final_study_v31_routing_amendment.md) and [prospective scientific scope review](docs/final_study_v3_scientific_scope_review.md) define the claims this evidence supports.

## Reproduction

```sh
python -m pip install -e '.[dev]'
python scripts/check_protocol_freeze.py
pytest -q
PYTHONPATH=src python scripts/final_study_v3.py --help
PYTHONPATH=src python scripts/final_study_v31.py --help
```

Local release verification (2026-10-20): 450 tests passed in 361.73 seconds; all 14 protected protocol entries passed, with one previously documented test-only erratum. Tests use synthetic fixtures and make no paid requests. Remote CI also passed447tests on Ubuntu/Python3.11 in218.98seconds,plus the frozen checker and all synthetic smoke checks: [verified workflow](https://github.com/amin-kamaleddin/LLM-Causal-Uncertainty/actions/runs/39183570679). Publication and historical CI failures are recorded in [RELEASE_STATUS.md](RELEASE_STATUS.md).

Authoring source is intentionally available for reproducibility. A new run must use a new private directory outside Git and its own prospective source commitment. Do not overwrite frozen results or impersonate the previous executed source revision. Paid calls require explicit authorization, exact models/providers, a runtime-only local key and bounded spending. Existing completed runs must resume from hash-bound receipts. Raw truth, observations, generation identifiers and provider bodies are protected; public exports use an explicit aggregate allowlist.

The clinical manuscript, figures and aggregates remain local pending author, scientific and SHARE disclosure/license review. Individual SHARE registration is required for empirical reproduction. New source code supplies missing caseness, country-omission and selection diagnostics without releasing participant data. No journal submission or SHARE publication notification has been sent.

This release has a clean research-code history, preserves the supplied GitHub repository initialization, and excludes original private research history. Bibliographic metadata are in references.bib. The unmodified controlling [MASTER PROMPT v2](docs/MASTER_PROMPT_v2.md), [reproduction instructions](docs/reproduction_and_private_artifacts.md) and [source inventory](docs/reproducibility_inventory.json) record what is public and what requires authorized private inputs. The clinical figure renderer reads counts from private results and has been checked against the original figures. Private manuscript builders containing empirical findings and historical draft handoffs remain outside this release.
