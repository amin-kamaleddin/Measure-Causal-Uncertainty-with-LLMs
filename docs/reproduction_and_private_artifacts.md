# Reproducing the research

The public release contains the analysis implementation, tests, configurations,
frozen scientific specifications and reviewed synthetic results. It deliberately
does not contain licensed SHARE inputs, clinical findings/manuscripts, respondent
records, protected live scoring stores, raw provider transcripts or credentials.
The unmodified controlling brief is [MASTER_PROMPT_v2.md](MASTER_PROMPT_v2.md).
Its original default budget and optional external-processing discussion are
superseded by the executed plan and the researcher's narrower authorization:
local-only SHARE processing, no participant data in provider calls, and an
explicit USD 100 software ceiling for the completed final experiment. The brief
does not authorize any new paid call or data transfer.

The 2026-10-20 source comparison is in
[reproducibility_inventory.json](reproducibility_inventory.json). Every applicable
runtime source, configuration and test from the original working repository is
represented in the release. Of 224 paired files, 221 are byte-identical. The
three reviewed differences are the already documented historical reporting fix,
its regression test, and the data-driven clinical figure renderer described below.
Do not merge the original private research history into the public repository.

## Public installation and verification

```sh
git clone https://github.com/amin-kamaleddin/LLM-Causal-Uncertainty.git
cd LLM-Causal-Uncertainty
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python scripts/check_protocol_freeze.py
python -m pytest -q
```

The executed environment snapshot for the final experiment is
`outputs/synthetic/evidence_grounded_v31_reviewed/executed_environment.json`.
Tested environments and historical CI results are recorded in RELEASE_STATUS.md.
The final experiment's frozen decisions, failures, cost accounting, method
comparisons and uncertainty diagnostics are reported in that reviewed directory.
Zero-assertion risks are undefined; uncertainty is exploratory and no superiority
over the strong evidence baseline is established.

## Reproduce clinical figures from authorized private outputs

The public renderer reads computed aggregate outputs; it has no provider client,
does not read SHARE archives and does not contain embedded clinical findings.
Participant-flow counts come from the disjoint selection table in `results.json`.
It rejects inconsistent totals and output locations inside the repository.

```sh
python scripts/render_final_study_clinical_figures.py \
  --package /absolute/path/to/private/computed_publication_package \
  --empirical-results /absolute/path/to/private/empirical_v3_results \
  --output-package /absolute/path/to/private/new_figure_reproduction
```

Required private inputs are:

- `--empirical-results/results.json`, from the saved longitudinal v3 analysis;
- `--package/share_binary/summary.json`, the verified caseness analysis;
- `--package/share_update/unknown_survival_tipping.csv`, the selection scenarios.

This produces five figure groups in PDF, SVG and PNG, plus file and input
provenance. The 2026-10-20 reproduction produced all five PNG files byte-for-byte
identical to the preserved originals, without re-estimating any model or making
provider calls. The comparison hashes and rendered clinical files remain private.
PDF/SVG metadata can vary with rendering time; visual/PNG equivalence was checked
under the recorded local environment rather than assumed across all platforms.

## Licensed analysis and private manuscript reproduction

The public source includes `scripts/real_share_followup_v3.py`,
`scripts/final_study_share_update.py` and `scripts/final_study_share_binary.py`.
Use their `--help` commands and the versioned configuration for argument details.
Empirical v3 reconstruction needs the individually licensed Wave 7/8/9 release
9.0.0 archives, the existing private saved graph records and comparison/prior-run
artifacts. The update script also verifies the original empirical manifest.
Existing completed analyses should be reused; this publication audit did not
rerun them. Their exact executed commands are in the researcher's private
`EXECUTION_EVIDENCE.md` and original execution receipts.

The original manuscript builders and historical clinical renderer embed
unreviewed SHARE findings. Their exact bytes remain in a protected local source
backup, with source hashes listed in `reproducibility_inventory.json`. This is an
explicit private dependency of exact manuscript-wording reproduction, not a
claim that GitHub alone regenerates the complete empirical manuscript. Authorized
replication needs those scripts, the private aggregate package and bibliography.
Restore hash-verified private builders into a private checkout with the original
directory layout before running them: some historical builders resolve inputs
relative to their own source file. Their three unpublished filenames are ignored
by this public release to prevent accidental staging. Keep the original clinical
renderer backup separately from its public replacement; do not overwrite either
version while trying to reproduce a historical source commitment.
The new public renderer replaces the latest clinical plotting dependency without
publishing the underlying findings. No frozen source or analysis rule was changed.

For recovery, the researcher must retain the original private checkout/executed
source snapshots, private publication-source backup, final manuscript package and
its checksum manifest, licensed data/provenance, and sealed provider/prediction
stores with their receipts. Credentials belong in a separate secret store.
Private artifacts need a separate secure backup; public Git cannot replace them.

## Historical model responses and frozen runs

Re-querying a changing hosted model does not reproduce a historical response.
Exact auditing uses the protected saved receipts, prompt/source hashes, predictions
and pre-truth seal. The public export contains reviewed aggregates and public
fictional evidence, not the protected transcript or truth stores. Reproducing the
historical receipt audit therefore requires authorized access to those private
artifacts; reproducing a fresh experiment is a different claim.

Do not rerun paid experiments to check publication completeness. A completed-run
resume requires the original frozen checkout and intact private artifacts. Its
source/HEAD checks intentionally reject a different checkout; do not bypass them
or relabel a new run as the original. A newly authorized live experiment must use
a new private output location, its own source commitment and explicit budget.

## Sharing boundary

The release rules in AGENTS.md exclude SHARE data **and aggregates**, private
manuscripts/figures, credentials, raw responses and protected scoring stores.
Their absence from public Git is intentional. Human disclosure/license and
scientific review remain required before distributing the clinical manuscript.
Historical draft handoffs are retained locally rather than bulk-uploaded without
review. Negative results and all frozen synthetic comparisons remain preserved.
