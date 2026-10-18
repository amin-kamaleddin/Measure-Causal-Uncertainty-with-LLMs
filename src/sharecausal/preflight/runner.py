"""Offline preflight orchestration. Prediction sealing precedes private-file IO."""
from contextlib import contextmanager, ExitStack
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
from unittest.mock import patch
from typing import Literal

from pydantic import Field, model_validator
import yaml

from . import VERSION
from .contracts import Strict, PublicWorldCardV2, canonical, packet_hash, public_exports, prompt_payload, sha
from .comparison import public_comparisons, calibrate_public_margins
from .budget import FakePrice, SimulatedBudget, cost_envelope
from .design import PrecisionConfig, SplitUnit, lint_split, precision_diagnostics
from .provenance import MockTransport, validate_record


class Config(Strict):
    schema_version: Literal['preflight-config-2.2']
    protocol_status: Literal['candidate_not_frozen']
    authorized_openrouter_usd: int = Field(ge=0, le=0)
    public_packets: str
    private_truth: str
    split_manifest: str
    fake_pricing: str
    decisions: str
    models: list[str] = Field(min_length=3, max_length=4)
    variants: list[str]
    repetitions: int = Field(ge=1, le=10)
    max_retries: int = Field(ge=0, le=3)
    input_token_reservation: int = Field(gt=0)
    output_token_reservation: int = Field(gt=0)
    hypothetical_global_cap_usd: float = Field(ge=0)
    hypothetical_per_model_cap_usd: float = Field(ge=0)
    support_policies: list[str]
    primary_support_policy: Literal['DECISION_REQUIRED']
    candidate_assertion_target: float = Field(gt=0, lt=1)
    candidate_matching_tolerance: float = Field(gt=0, lt=1)
    candidate_practical_zero: float = Field(ge=0)
    precision: PrecisionConfig

    @model_validator(mode='after')
    def choices(self):
        if len(set(self.models)) != len(self.models) or len({m.split('/')[0] for m in self.models}) != len(self.models):
            raise ValueError('distinct_mock_families_required')
        if self.variants != ['base', 'challenge']:
            raise ValueError('candidate_variant_schedule_required')
        if self.support_policies != ['supported_only', 'retain_alternatives']:
            raise ValueError('both_candidate_support_policies_required')
        return self


class Decision(Strict):
    id: str
    status: Literal['DECISION_REQUIRED']
    question: str
    options: list[str] = Field(min_length=2)
    recommended: str
    evidence: str

    @model_validator(mode='after')
    def recommendation(self):
        if self.recommended not in self.options:
            raise ValueError('recommended_option_missing')
        return self


@contextmanager
def network_veto():
    def denied(*args, **kwargs):
        raise RuntimeError('network_forbidden_in_offline_preflight')
    with ExitStack() as stack:
        for name in ('socket.create_connection', 'socket.getaddrinfo', 'socket.socket.connect',
                     'socket.socket.connect_ex', 'socket.socket.sendto', 'socket.socket.sendmsg'):
            try:
                stack.enter_context(patch(name, denied))
            except AttributeError:
                if not name.endswith('sendmsg'):
                    raise
        yield


def safe_path(path):
    lexical = Path(path).expanduser().absolute()
    def check(p):
        if any(part.lower() in {'private', 'share-research', 'results_private', '.aws', '.ssh', '.config', '.codex'}
               or part.lower().startswith('.env') for part in p.parts[2:]):
            raise ValueError('private_or_credential_path_forbidden')
        if any(word in p.name.lower() for word in ('credentials', 'api_key', 'apikey', 'secrets')):
            raise ValueError('credential_filename_forbidden')
        if p.suffix.lower() in {'.zip', '.dta', '.sav', '.parquet', '.feather'}:
            raise ValueError('non_synthetic_file_type_forbidden')
    check(lexical)
    if lexical.is_symlink():
        raise ValueError('input_output_symlink_forbidden')
    resolved = lexical.resolve(); check(resolved)
    return resolved


def load_json(path):
    return json.loads(safe_path(path).read_text())


def write_json(path, data):
    path.write_bytes(canonical(data))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['no_rows'])
        writer.writeheader(); writer.writerows(rows)


def run_preflight(config_path, results_dir, *, new_run=False):
    with network_veto():
        config_file = safe_path(config_path)
        cfg = Config.model_validate(yaml.safe_load(config_file.read_text()))
        cards = [PublicWorldCardV2.model_validate(x) for x in load_json(cfg.public_packets)]
        if not cards or len({packet_hash(c) for c in cards}) != len(cards):
            raise ValueError('unique_nonempty_public_cards_required')
        cards = sorted(cards, key=packet_hash)
        by_hash = {packet_hash(c): c for c in cards}
        prices = [FakePrice.model_validate(x) for x in load_json(cfg.fake_pricing)]
        if {p.model_id for p in prices} != set(cfg.models) or len(prices) != len(cfg.models):
            raise ValueError('fake_pricing_catalog_mismatch')
        for p in prices: p.validate_known()
        decisions = [Decision.model_validate(x) for x in load_json(cfg.decisions)]
        if not decisions or len({d.id for d in decisions}) != len(decisions):
            raise ValueError('decision_register_required')
        output = safe_path(results_dir)
        if output.exists() and (not output.is_dir() or (not new_run and any(output.iterdir()))):
            raise ValueError('results_directory_not_empty')
        output.mkdir(parents=True, exist_ok=True)
        if new_run: output = Path(tempfile.mkdtemp(prefix='run-', dir=output))
        public_dir = output / 'public'; public_dir.mkdir()
        exports = public_exports(cards)
        for filename, raw in exports.items(): (public_dir / filename).write_bytes(raw)
        public_manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in exports.items()}
        write_json(output / 'public_manifest.json', public_manifest)
        transport = MockTransport()
        budget = SimulatedBudget(cfg.hypothetical_global_cap_usd, cfg.hypothetical_per_model_cap_usd)
        comparisons, recorded = {}, []
        for card in cards:
            selected = {}
            for model in cfg.models:
                price = next(p for p in prices if p.model_id == model)
                for variant in cfg.variants:
                    for repetition in range(cfg.repetitions):
                        answer, prov = transport.record(card, model, variant, repetition, price, budget,
                            cfg.input_token_reservation, cfg.output_token_reservation, cfg.max_retries)
                        status, graph = validate_record(card, answer, prov.model_dump(), 'retain_alternatives', variant)
                        if status['reason'] == 'provenance_invalid' or not status['format_valid']:
                            raise ValueError('mock_provenance_contract_failed')
                        recorded.append({'card_hash': packet_hash(card), 'variant': variant, 'repetition': repetition,
                                         'answer': answer, 'provenance': prov.model_dump()})
                        if variant == 'base' and repetition == 0:
                            selected[model] = [answer]
            comparisons[packet_hash(card)] = public_comparisons(card, cfg.models, selected, cfg.support_policies)
        # Hypothetical scoring probes are fixed BEFORE reading any truth, not model performance.
        probes = {name: {h: decision for h in sorted(by_hash)} for name, decision in (
            ('always_abstain_contract_probe', 'abstain'), ('positive_contract_probe', 'positive'))}
        write_json(output / 'comparison_batch.json', comparisons)
        write_json(output / 'mock_provider_records.json', recorded)
        write_json(output / 'decision_contract_probes.json', probes)
        seal_names = ['public_manifest.json', 'comparison_batch.json', 'mock_provider_records.json', 'decision_contract_probes.json']
        seal = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in seal_names}
        seal.update({'public/'+name: value for name, value in public_manifest.items()})
        write_json(output / 'prediction_seal.json', seal)
        sealed_hash = hashlib.sha256((output/'prediction_seal.json').read_bytes()).hexdigest()
        for name, expected in seal.items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest() != expected:
                raise ValueError('prediction_seal_integrity_failed')
        # Truth and split metadata first opened here, after every public-only decision is fixed.
        from .certificates import PrivateWorldTruthV2, certify
        from .scoring import score_contract
        worlds = [PrivateWorldTruthV2.model_validate(x) for x in load_json(cfg.private_truth)]
        if len({w.world_id for w in worlds}) != len(worlds) or {w.public_card_hash for w in worlds} != set(by_hash):
            raise ValueError('truth_world_card_coverage_invalid')
        units = [SplitUnit.model_validate(x) for x in load_json(cfg.split_manifest)]
        if {u.card_hash for u in units} != set(by_hash): raise ValueError('split_card_coverage_invalid')
        for w in worlds:
            u = next(u for u in units if u.card_hash == w.public_card_hash)
            if w.mechanism_family != u.mechanism_family or w.narrative_style != u.narrative_style or w.generator_seed not in u.seeds:
                raise ValueError('split_truth_binding_invalid')
        split = lint_split(units)
        certifications = {w.world_id: certify(w, by_hash[w.public_card_hash], worlds) for w in worlds}
        indexed_worlds = {w.world_id: w for w in worlds}
        witness_pairs = [w.certificate.witness_world_ids for w in worlds if w.certificate.kind == 'binary_pair']
        fixed_probes = json.loads((output/'decision_contract_probes.json').read_bytes())
        scores = {name: score_contract(canonical(probe), worlds, by_hash, cfg.candidate_practical_zero)
                  for name, probe in fixed_probes.items()}
        precision = precision_diagnostics(cfg.precision)
        costs = [cost_envelope(prices, n, len(cfg.variants), cfg.repetitions, cfg.max_retries,
                   cfg.input_token_reservation, cfg.output_token_reservation,
                   cfg.hypothetical_per_model_cap_usd, cfg.hypothetical_global_cap_usd)
                 for n in cfg.precision.card_counts]
        calibration = calibrate_public_margins([
            {'method': 'graph_only_preflight', 'decision': 'abstain', 'margin': 0.0} for _ in cards],
            cfg.candidate_assertion_target)
        firewall = {'passed': True, 'public_serializer': 'explicit_field_allowlist',
                    'truth_opened_after_prediction_seal': True, 'public_payloads': len(exports),
                    'public_filename_policy': 'content_addressed_sorted_no_world_ids_or_splits',
                    'truth_worlds': len(worlds), 'ambiguous_pair_public_hashes_equal':
                        bool(witness_pairs) and all(all(k in indexed_worlds for k in ids)
                                                   and len({indexed_worlds[k].public_card_hash for k in ids}) == 1
                                                   for ids in witness_pairs),
                    'limitations': 'Semantic packet authorship and independent sealing still require review.'}
        write_json(output/'identification_audit.json', certifications)
        write_json(output/'scoring_contract_diagnostics.json', scores)
        write_json(output/'firewall_audit.json', firewall)
        write_json(output/'split_audit.json', split)
        write_json(output/'decision_register.json', [d.model_dump() for d in decisions])
        write_csv(output/'precision_diagnostics.csv', precision)
        write_csv(output/'cost_envelope.csv', [{k:v for k,v in row.items() if k != 'per_model_hypothetical_usd'} for row in costs])
        report = {'version': VERSION, 'fixture_mode': True, 'real_llm_calls': 0,
                  'openrouter_cost_usd': 0, 'share_records_used': 0, 'authorized_openrouter_usd': 0,
                  'protocol_status': 'candidate_not_frozen', 'ready_to_freeze': False,
                  'diagnostics_only': True, 'decision_required_count': len(decisions),
                  'public_cards': len(cards), 'world_probes': len(worlds), 'mock_transport_calls': transport.calls,
                  'simulated_usage_cost_usd_not_spending': float(sum(budget.spent.values())),
                  'prediction_seal_sha256': sealed_hash, 'certifications': certifications,
                  'split_audit': split, 'firewall_audit': firewall,
                  'precision': precision, 'cost_envelopes': costs, 'calibration_diagnostic': calibration,
                  'coverage_comparison_status': 'unmatched_graph_only_no_effect_estimation',
                  'comparison_rules': {'preselected_single_model': cfg.models[0],
                     'ballot_variant': 'base', 'ballot_repetition': 0,
                     'majority_denominator': 'all_planned_models',
                     'multiple_graphs_per_model': 'abstain_without_extra_votes',
                     'cycle_policy': 'abstain_no_repair', 'primary_support_policy': 'DECISION_REQUIRED',
                     'human_adjudication': 'not_performed'},
                  'scoring_contract_probe_summary': {name: {k:v for k,v in s.items() if k not in ('tasks','world_rows')}
                                                     for name,s in scores.items()},
                  'warning': 'Synthetic fixtures and candidate simulations only; no real LLM performance, power guarantee or superiority.',
                  'environment': {'python': platform.python_version(),
                     'packages': {p: importlib.metadata.version(p) for p in ('pydantic','numpy','scipy','PyYAML')}}}
        write_json(output/'preflight_report.json', report)
        summary = ['# Offline prospective-study preflight', '', '**Candidate, not frozen. Authorized spending: USD 0.**',
                   '', f'Public cards: {len(cards)}; private world probes: {len(worlds)}; pending PI decisions: {len(decisions)}.',
                   '', 'Precision assumes independent cards clustered within families. Seeds/prompts are nested probes.',
                   'Prices/capabilities are LOCAL FAKE fixtures. These results are not LLM performance or validated power.',
                   'Uncertified worlds remain in denominators and are excluded from binary nonidentification error labels.',
                   '', '| Candidate cards | Planned calls | Maximum attempts | Fake cost USD | Within hypothetical caps |',
                   '| --- | --- | --- | --- | --- |']
        summary += [f"| {c['independent_cards']} | {c['planned_calls']} | {c['maximum_attempts']} | {c['hypothetical_total_usd']:.4f} | {c['within_hypothetical_caps']} |" for c in costs]
        summary += ['', '## DECISION_REQUIRED', '']+[f'- **{d.id}**: {d.question} Recommended option for review: {d.recommended}. Evidence: {d.evidence}' for d in decisions]
        (output/'diagnostics.md').write_text('\n'.join(summary)+'\n')
        root = Path(__file__).resolve().parents[3]
        inputs = [config_file]+[safe_path(getattr(cfg,k)) for k in ('public_packets','private_truth','split_manifest','fake_pricing','decisions')]
        sources = sorted(Path(__file__).parent.glob('*.py'))+[root/'scripts/v2_preflight_offline.py',root/'docs/freeze_manifest_v1.json']
        relative = lambda p: str(p.relative_to(root)) if p.is_relative_to(root) else p.name
        manifest = {'inputs': {relative(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    'sources': {relative(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    'outputs': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(output.rglob('*')) if p.is_file()}}
        write_json(output/'content_manifest.json', manifest)
        return report, output, hashlib.sha256((output/'content_manifest.json').read_bytes()).hexdigest()
