"""Reviewed allowlist export: code/public synthetic packets/aggregate results only."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
from sharecausal.preflight_v23.sealing import verify,digest
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--package',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run=a.run.resolve();package=a.package.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
seal=json.loads((run/'prediction_seal.json').read_text());sealhash=(run/'prediction_seal.sha256').read_text().strip();verify(run,seal,'prediction_seal.json',sealhash);integrity=json.loads((run/'scoring_integrity.json').read_text());assert integrity['seal_verified_before_truth_open']
allow=['method_comparison.csv','family_paired_contrasts.csv','risk_coverage.csv','evidence_edge_accuracy.csv','causal_estimation_diagnostics.csv','development_target_coverage.csv','decision_dispositions.csv','provider_comparison.csv']
def csvcopy(source,target):
 with source.open(newline='') as f:rows=list(csv.reader(f))
 with target.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
 with target.open(newline='') as f:assert list(csv.reader(f))==rows
for name in allow:csvcopy(run/name,out/name)
for name in ('actual_unique_provider_requests.csv','disjoint_decision_dispositions.csv','prompt_repetition_stability.csv','causal_decision_repetition_stability.csv','critic_fidelity.csv','wrong_direction_vs_practical_null.csv'):csvcopy(package/'tables'/name,out/name)
for name in ('public_cards.json','catalog.json','calibration.json','leakage_audit.json','prior_packet_text_audit.json','executed_environment.json'):
 shutil.copyfile(run/name,out/name)
summary=json.loads((run/'summary.json').read_text());account=json.loads((package/'actual_execution_accounting.json').read_text());summary.update(actual_unique_request_accounting=account,scoring_integrity=integrity,software_cap_usd=100,primary_calls_in_corrected_schedule=1920,critic_calls_in_corrected_schedule=120,reused_primary_receipts=1440,reused_critic_receipts=28,new_requests_in_repaired_run=572,complete_predictions_sealed_before_truth=True,repeat_prediction_batch_sha256=seal['repeat_condition_predictions.json'],executed_source_sha256=json.loads((run/'commitment.json').read_text())['source_sha256'],calibration_reused_unchanged=True,raw_receipts_and_truth_exported=False,clinical_results_exported=False,review_scope='code and explicitly synthetic aggregates only; no claim of confirmatory superiority')
(out/'completed_run.json').write_text(json.dumps(summary,indent=2)+'\n')
for source in (package/'figures').glob('figure[345]_*'):
 if source.suffix not in ('.png','.pdf','.svg'):continue
 target=out/source.name
 if source.suffix=='.svg':target.write_text('\n'.join(line.rstrip() for line in source.read_text().splitlines())+'\n')
 else:shutil.copyfile(source,target)
(out/'README.md').write_text('''# Evidence-grounded real-model study: reviewed synthetic release

120 new fictional evidence cards: 36 development cards in 6 declared families and 84 evaluation cards in 14 families, six nested variants per family. This realizes 20 declared families, not the originally suggested 40 independent mechanisms. Familiar causal motifs and a prior equivalence witness are reused in new combinations. Text similarity is not proof of mechanism independence. Single-team authorship, common prose grammar and no external expert adjudication limit generalization.

Exactly four models, two variants and two repetitions: 1,920 primary slots plus 120 Gemini critics. The primary comparison uses the first base response. Other prompts, sample repeats and graphs do not increase independent family count. All 240 primary-sample predictions, 720 alternate-condition estimates, original-attempt predictions, receipts and commitments were sealed before guarded truth scoring. Full hashes and actual aggregate accounting are in completed_run.json. The original calibration was preserved through the provider-routing amendment.

2,528 unique requests include 480 zero-cost investigator provider-slug rejections, 480 corrected DeepSeek requests and 128 unique critic prompts. The corrected analytical schedule has 2,040 receipts; 1,468 existing receipts were reused. Exact models/providers, failures and actual provider-reported charges are retained. Typed valid is not graph admission. Technical failures are not scientific abstentions. No response IDs, raw transcripts, credential, live truth, observations or SHARE findings are included.

The full policy and strong equally informed evidence-only baseline asserted on the same 16/84 evaluation cards, with zero recorded unjustified assertions. Neither reached the 30–50% comparison range. Majority voting made two practical-null assertions in 16 assertions; no method made a literal sign reversal. The frozen wrong_sign fields deliberately count wrong direction OR absolute true effect <=0.10; their values are unchanged. The added decomposition distinguishes these components without changing a score or decision. The naive all-baseline arm made six proved-nonidentification errors and seven invalid-target assertions; its six known-world composite errors were practical-null cases. Unknown certificates remain unknown and produce risk bounds. Zero-assertion risk is undefined. The locked critic veto is redundant with the compiler; no incremental critic effect is claimed. The family bootstrap is descriptive, often degenerate, and does not establish sparse-event precision or structural coverage. Historical failures and unfavorable findings remain preserved. No confirmatory methodological superiority is established.

The scripts and authoring source permit new local reproduction. Live runs require an explicit local key, budget and private directory outside Git. Their prospective source commitments are tied to the executed working-repository revision; a new checkout/run must generate a new commitment rather than overwrite or impersonate an old one. The one historical post-score reporting correction in the release is documented separately and did not affect this study's executed engine/scorer.
''')
manifest={str(f.relative_to(out)):digest(f) for f in out.iterdir() if f.is_file() and f.name!='release_manifest.json'};(out/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(files=len(manifest),actual_requests=account['actual_unique_requests'],cost_usd=account['reported_generation_cost_usd'],prediction_seal_sha256=sealhash,output=str(out))))
