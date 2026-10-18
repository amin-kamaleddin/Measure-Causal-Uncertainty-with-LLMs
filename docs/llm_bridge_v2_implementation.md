# Step 6B offline synthetic bridge v2 — 2026-10-16

Implementation revision: **v2.1-independent-review**, on
`feature/llm-synthetic-bridge-v2`. The answer schema remains synthetic-graph-v2.0;
the correction changes decision/audit semantics, not that answer contract.
This is a local integration
checkpoint with invented deterministic fixtures, not a real LLM comparison,
confirmatory evaluation, or demonstration of methodological superiority.
The controlling brief remains `MASTER_PROMPT_v2.md`; the v1 protocol and its
negative findings remain unchanged.

## Architecture and public-information boundary

`llm/synthetic_schema.py` defines `synthetic-graph-v2.0`, a separate strict
Pydantic answer schema. The existing SHARE `GraphAnswer` is unchanged. The
public card matches v1 exactly: X1/X2 precede A, M follows A, Y follows M, and C
follows Y. `observed` records whether Y was collected. Optional U is latent,
has no incoming arrows, and cannot be marked observed or adjusted for. Directed
edge order, rationale, uncertainty, hypothetical evidence, contested assumptions
and explicit abstention are retained. Duplicate edges, cycles, backward arrows,
invalid roles, incomplete answers and post-treatment adjustment are rejected.
Only the population total effect of A on Y is supported.

`llm/synthetic_adapter.py` accepts recorded JSON, public metadata and provenance;
it receives no data or generator object. Provenance commits to the prompt hash,
fixture identity, dummy requested family, variant, repetition, timezone-bearing
timestamp and schema version. Rejections carry error types and approved reason
codes, response digest and available provenance; invalid raw content is never
copied to logs. Validation locations are the fixed scopes `response` or
`provenance`, never Pydantic's arbitrary field-name locations. Serialization
failures become source rejections with validated provenance when available;
one malformed record cannot suppress a valid record sharing its identity.
A source abstention contains no graph and contributes no vote.
Admission establishes schema and temporal consistency, not scientific truth.
Only `offline_fixture` provenance is currently allowed. Accepted audits explicitly
say `admission_basis=format_and_temporal_constraints_only` and
`evidence_support_status=not_adjudicated_offline_hypothesis`. Free-text citations
are not verified scientific evidence. The future evidence-support policy is
specified separately in `v2_prospective_experiment_design.md` (draft only).

`benchmark/bridge_v2.py` adapts records before examining outcomes, then applies
frozen backdoor identification to each accepted graph. A graph copy restricts
eligible measured controls to X1/X2 while retaining every original edge and
latent node. This matters because the frozen identifier's descendant exclusion
alone does not establish pre-exposure timing. It enumerates minimal-cardinality
eligible sets and deterministically chooses the first, rather than choosing by
estimated effects. Suggested adjustments are advisory. Identification also tests
for **any directed path**, including mediated paths, from A to Y. No such path
entails a structural zero under the admitted causal graph. The graph remains
accepted, its `total_effect_status` is `graph_entailed_structural_zero`, and its
individual and majority causal decisions abstain from directional claims without
fitting a nonzero AIPW effect. `structural_effect=0` is a graph constraint, not a
sample estimate or sampling confidence interval. With a causal path, uncontrolled
U common causes and selection warnings can fail measured-variable backdoor
identification. Backdoor failure is not a claim that no other identification
strategy could ever work; the identification scope is explicit in every result.

Graph identification, numerical estimation success, and model compatibility are
separate. All current graph claims report `model_compatibility=not_tested`; a
strong association cannot be used to reject a structural exclusion. No graph
compatibility test is implemented. Estimation failure leaves the mathematical
backdoor status intact and separately records `estimation_failed` and abstention.

Graph-conditional estimation reuses the frozen cross-fitted AIPW estimator (or
the frozen difference in means for an empty adjustment set). Numeric/binary
input checks precede fitting. Missing, invalid or insufficient observed inputs
cause estimation abstention. Rows with `observed==0` are excluded before any
numeric validation or fit; neither their Y nor descendants enter prediction.
Any incomplete outcome follow-up blocks directional assertions for all arms.
Observed-case fits may still be shown as conditional diagnostics; they do not
identify the population total effect under unresolved selection.

The no-LLM evidence graph assumes both X1 and X2 are confounders. It never
replaces a missing source proposal. Individual source arms retain rejection or
abstention. Strict directed-edge majority uses accepted nonabstaining fixtures
only, then repeats temporal validation and identification. Edge frequencies
are descriptive agreement, never probabilities. The screened robust arm
requires all accepted source graphs to constrain the same stipulated estimand
via a structural zero or a successful identified fit. A structural-zero graph
contributes [0,0] to the conditional constraint set, not an invented sampling CI.
The scenario grid includes the unshifted zero, so any such graph blocks strictly
positive/negative robust assertions. Separate `structural_zero_graphs` metadata
records that reason. A single admitted graph lacking identification or a valid
conditional estimate also blocks the assertion.
Rejected graphs are explicitly audited and excluded; the result is conditional
on that admission policy and the supported graph set being adequate.

The robust arm combines graph-conditional sampling intervals shifted by
−0.25, 0, +0.25 outcome units into an assumption-robustness envelope. These
arbitrary stress parameters are not calibrated hidden-confounding bounds. The
envelope is not a simultaneous 95% confidence set. No representative causal
point estimate is assigned to the robust arm. Assertions require the complete
envelope to exclude zero. An empty accepted set always abstains.

## Physical separation and runtime safeguards

`bridge_v2_smoke_data.py` creates three independent synthetic linear examples
(n=320 each; fixed seeds 60601–60603): measured, latent, and selected outcomes.
They have no SHARE-derived parameters. They do not use the v1 frozen split or
claim to be independent mechanism-family holdouts.

`bridge_v2_offline.py` is orchestration only. It passes the observable DataFrame
and fixed fixture records to the decision engine. **All three predictions** are
serialized, written to `prediction_N.json`, and SHA-256 hashed before any task's
truth or scoring label is read. `prediction_batch_manifest.json` commits to all
prediction hashes; its hash is recorded as `prediction_batch_sha256`. The runner
verifies all file hashes, fixes an immutable tuple of JSON snapshots, then scores.
`bridge_v2_scoring.py` consumes a
fixed JSON snapshot and truth only to score existing decisions. It cannot
update predictions. Tests change wrapper labels/truth while holding observables
fixed, verify identical decisions/proposals, and check that scoring changes.
Extra DataFrame columns or attributes are rejected. Missing-outcome comparisons
require exact structural equality and at most 1e-8 relative/absolute floating
roundoff, following the existing macOS erratum.

The standalone command has no live mode, no key loading, no .env loading, no
OpenRouter import, discovery or HTTP client. A synchronous process-local socket
veto blocks DNS and outbound connections/datagrams during the run, regardless
of environment approval/key variables. This guard is not a general sandbox
against arbitrary third-party code; the bridge has no subprocess transport or
network implementation. Tests stub HTTP/OpenRouter paths and prove zero calls,
including with deliberately set dummy approval/key environment variables.
Private path components are rejected before listing or reading fixtures;
symlink fixtures are rejected. No real archives, respondent caches or private
SHARE results are needed or accessed. The command refuses nonempty output
directories unless `--new-run` explicitly requests a unique child directory;
prior results cannot be overwritten. Generated smoke directories
are ignored by Git. No staging, commit, push or remote CI claim accompanies
this checkpoint.

## Four fixture records

All requested families are dummy placeholders; timestamps are fixed fixture
metadata rather than actual provider-call times. Every record uses the same
base public prompt and evidence hash.

| Fixture | Expected disposition | Meaning |
| --- | --- | --- |
| dummy_measured_x1 | Accepted; adjust X1 | Disputes X2→Y, different from the two-control evidence baseline |
| dummy_latent_u | Accepted; no measured backdoor set | Admits U→A and U→Y; vetoes robust identification |
| dummy_invalid_time | Rejected | M→A violates stipulated timing |
| dummy_explicit_abstention | Source abstention | No arrows, reason retained, no vote |

## Reproduction and outputs

The current shell has no bare `python`; use the repository's existing Python
3.15 virtual environment and `PYTHONPATH` explicitly. No installation or network
access was needed. The actual validation commands and results are recorded in
`RUN_STATUS.md`.

```bash
.venv/bin/python scripts/check_protocol_freeze.py
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest -q
PYTHONPATH="$PWD/src" .venv/bin/python -m sharecausal.cli llm-compare --dry-run
PYTHONPATH="$PWD/src" .venv/bin/python scripts/llm_bridge_v2_offline.py \
  --fixture-dir tests/fixtures/llm_bridge_v2 \
  --results-dir outputs/synthetic/llm_bridge_v2_offline_smoke \
  --new-run
git diff --check
git status --short
```

`--new-run` makes the documented command repeatable: each execution creates a
fresh `run-*` child under the chosen results directory and reports its exact
path in stdout JSON and `smoke_report.json`. Existing root artifacts and earlier
children remain intact. Without this option, the destination must be new/empty;
an occupied destination returns exit 2 with error code `results_not_empty` and
an explicit corrective hint. No result is silently reused or overwritten.

Outputs in the reported run directory are `smoke_report.json`,
`graph_provenance.json`, `prediction_batch_manifest.json`, `run_level.csv`,
`summary.csv` and three fixed prediction JSONs (eight artifacts). The report
includes the implementation revision, zero live/no-SHARE counters, source/fixture hashes,
public prompt/card, prediction hashes, package versions, individual audits,
identification sets, assertion rates and descriptive scores. Undefined selective
risk is JSON null when no assertion is made. These three examples are only an
executable engineering smoke test; their small denominators and constructed
fixture set do not support comparative inference.

## Differences from historical v1 and remaining scientific work

V1 embeds fixed graph objects inside a frozen analysis function. V2 consumes
validated recorded responses through a separate schema/adapter, checks full
provenance, derives controls only from public pre-exposure roles, records
explicit source abstention, and separates fixed prediction artifacts from
post-decision scores. V2 conservatively suppresses all directional assertions
under missing outcomes and reports no robust point estimate. Its voting policy
screens invalid records before voting. These are versioned design differences;
v2 smoke scores must not be pooled with historical v1 results. Neither v1's
locked seeds, thresholds, outputs nor its negative results were modified.

Before any real model experiment, separately approve and freeze a v2 evaluation
protocol: independently reviewed graph admission/evidence, genuinely independent
mechanism-family holdouts, estimator/positivity failure rules, missingness and
selection handling, bias grid, development-only calibration, repetition policy,
matched assertion-rate comparisons and uncertainty across independent repeats.
Calibration must accept only development `method`, `decision`, and
`assertion_margin`, may suppress but never create assertions, and must not read
truth labels. No calibration or heldout tuning occurs in this smoke command.

For future 3–4 real LLM arms, deliver identical serialized public cards, evidence,
answer schema and variant schedules from `prompt_payload`. Never pass generator
labels/edges/effects, potential outcomes, observed outcome statistics or SHARE
estimates into proposal generation/admission. Record exact prompts, hashes,
provider/model versions, sampling settings, errors, tokens and billed costs.
Prospective responses can use the same answer schema and graph-evaluation logic;
adding live provenance and a transport is a separately authorized change, not an
environment flag enabling this command. Provider discovery, privacy controls,
endpoint schema support, spend controls and separate prospective evaluation are
still required before any live experiment.

Remaining limitations include omitted true graphs, correlated/hallucinated
source assumptions, partial backdoor identification, nuisance misspecification,
weak overlap and normal interval approximations. Cross-fitting and graph voting
do not repair hidden confounding or selection. Always-abstaining arms have
undefined risk and an assertion cost. Novelty, real LLM performance, calibrated
structural coverage, transport to epidemiology and superiority remain unresolved.

## CLI failure diagnosis and subprocess regression

The original documented command was reproduced after the initial smoke files
already existed. Calling `run_offline_smoke` directly exposed the real exception:
`ValueError: Results directory must be new or empty; never overwrite prior runs`.
The runner had correctly protected prior results, but the entry point obscured
the reason behind a generic configuration/IO message. The previous positive
CLI test invoked `main()` in-process; its subprocess tests covered failures
only. That gap did not exercise the successful documented command end to end.

The corrected documentation uses explicit `--new-run` rather than removing
prior outputs or relaxing the collision guard. A repeated command computes and
writes a fresh run, with no graph or score fallback. New subprocess tests invoke
the real script from the repository root, using the repository JSON fixtures,
and parse all output files, provenance, hashes and expected scientific decisions.
They cover both a fresh destination and repeated runs under an occupied base,
and verify earlier artifacts remain byte-for-byte unchanged.

`bridge_v2_errors.py` supplies allowlisted diagnostic messages/codes. Failures
include the stage, a safe exception category, an errno name when relevant, and
an actionable hint. Exit 2 means configuration/dependency/IO failure; exit 1
means an unexpected execution failure. Validation, prediction writing, scoring,
provenance and final artifact writing have distinct stages. Raw exception text,
tracebacks, filenames, data values, environment values and parser argument
values are omitted from stderr; the real exception remains chained in Python
for controlled local diagnosis. Tests deliberately inject secret-bearing
exception text/paths and verify redaction. A failed/partial run returns nonzero
and never emits a fabricated success report or switches to a baseline graph.
That earlier CLI-only fix changed no identification, estimator, fixture, seed,
or frozen v1 file. The subsequent v2.1 scientific repair is described above and
below; v1 remains unchanged.

## Independent review correction — v2.1, 2026-10-16

The independent review is `docs/independent_review_v2.md`; its task specification
is `docs/codex_review_fix_v2.md`. Both were read completely and remain unchanged.
The supplied `tests/test_review_regressions.py` already existed locally and was
run without edits before any source correction: **3 failed**, reproducing all
three blocking findings. The no-directed-path source and robust arm asserted
positive; in-memory NaN escaped `json.dumps`; and an untrusted extra-field name
appeared in the rejection reason. The supplied tests are preserved byte-for-byte
(SHA-256 `51bfe8a06cb30e37c725d6b292d551ff46d2c33b7a997232f2af45614396c7cd`).
Original failure output, source hashes and copies of the review/task/tests are
retained in ignored local `outputs/synthetic/llm_bridge_v2_review_audit/`, never
staged or published. They contain invented test markers, not real credentials.

The corrected graph-conditional logic adds `has_directed_causal_path`,
`structural_zero`, `total_effect_status`, and explicit identification scope to
the v2 identification result. It preserves the frozen backdoor result separately.
A missing *direct* edge is not enough to infer zero: A→M→Y still has a causal
path and is estimated as a total effect with baseline controls. A→C←Y has no
such path. A structural zero is a causal-model implication under exclusion,
consistency and the stipulated SCM; it is not an empirical discovery, a null-test
result, a sign conclusion or assurance of compatibility with these observations.
The sample estimator is not used to override it. Even when measured backdoor
adjustment fails because U is admitted, absence of every directed path still
implies a graph-conditional zero. Those two facts are represented separately.

Structural-zero sources remain accepted and vote under the same admission rule.
The robust set includes their zero point constraint; the fixed sensitivity grid
contains zero shift, so its envelope always contains zero and cannot assert a
strict direction. Stress shifts around that constraint are departures from the
exact model, not confidence bounds for its structural zero. Structural effects
are labeled separately from sampling estimates/intervals. The scorer exports
those graph-status and compatibility fields as diagnostics without changing a
prediction; no positive/negative assertion is created from a zero constraint.

Response serialization now runs inside a source-level rejection boundary. NaN,
Infinity, unsupported JSON objects/keys, cycles, excessive nesting and invalid
text encoding do not kill a valid batch. Validated source identity is retained
where safe; no digest is invented when canonical serialization is impossible.
Invalid duplicate-identity records cannot overwrite a valid method result;
rejected-record aliases are disjoint from admitted dummy source IDs. Duplicate
admitted identities remain an explicit configuration error. Missing/invalid
responses are never replaced by the no-LLM graph.

Error audits use fixed `response`/`provenance` scopes and allowlisted types, with
at most eight coarse entries. They never include Pydantic `loc`, arbitrary JSON
key names, input values, raw exception text, or context except fixed approved
validator messages. This is validation-log redaction, not a universal secret
scrubber for intentionally retained, valid free-text evidence. The offline inputs
are synthetic; future real-provider recordings need separate data/privacy review.

All batch predictions and their hashes now precede the first scoring-truth read.
A full-batch manifest is written, hashed, and checked against fixed serialized
snapshots before any score. Tests guard truth properties to assert that all
prediction files and the manifest exist first. This strengthens local isolation;
independent third-party sealing remains unimplemented and is not claimed.

Broader tests add structural-zero variants, indirect paths, U-with-zero, strong
opposing associations, sensitivity grids including radius zero, mixed malformed
batches, duplicate identities, hostile nested/provenance/envelope field names,
non-JSON fuzz cases, and real subprocess artifacts with injected markers.
Existing positive CLI tests now require the extra batch manifest and verify its
hash in addition to every original artifact assertion. No supplied regression
or frozen test was weakened. Actual full-suite and smoke outcomes, exact commands,
source hashes and generated directories are recorded in `RUN_STATUS.md`.

The prospective design in `docs/v2_prospective_experiment_design.md` is a
**separate unapproved draft**, not a live adapter or experimental authorization.
It addresses evidence-bearing public tasks, the impossibility of inferring
hidden mechanism differences from identical cards, equal model inputs, blinded
support adjudication, future typed provider provenance, complete-batch sealing,
development-only calibration, independent family holdouts, matched assertion
rates, correlated errors, budgets and software locks. None of these future model
experiments, endpoint checks or spending was executed by this repair. V1 and its
observed negative findings remain unchanged. Scientific merge/live authorization
still requires review; no commit, merge or push has occurred.

The outer fixture-file decoder also treats JSON value-limit and recursion errors
as source rejections; these must not escape before the response adapter. A
subprocess regression supplies excessively deep JSON and an oversized integer
alongside valid fixtures, requiring completion and preservation of valid graph
predictions. Runtime graph-identification metadata likewise keeps numerical
estimation failures out of the `nonidentified_graphs` list; their separate
`estimation_failure_graphs` veto does not rewrite the backdoor criterion.
