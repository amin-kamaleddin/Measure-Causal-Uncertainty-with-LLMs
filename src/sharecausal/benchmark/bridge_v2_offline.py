"""Offline-only orchestration, filesystem policy, and a process-local network veto."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
from unittest.mock import patch

import pandas as pd

from sharecausal.llm.synthetic_schema import PUBLIC_CARD, digest, prompt_payload
from .bridge_v2 import IMPLEMENTATION_VERSION, analyze_observables
from .bridge_v2_errors import OfflineConfigurationError, error_stage
from .bridge_v2_scoring import score_fixed_predictions, summarize_scores
from .bridge_v2_smoke_data import smoke_tasks


def _deny_network(*args, **kwargs):
    raise RuntimeError('V2 offline bridge forbids all network requests')


@contextmanager
def offline_network_veto():
    """Process-global for this synchronous command; no key/env can enable live IO."""
    with ExitStack() as stack:
        for target in ('socket.create_connection', 'socket.getaddrinfo',
                       'socket.socket.connect', 'socket.socket.connect_ex',
                       'socket.socket.sendto', 'socket.socket.sendmsg'):
            # sendmsg is absent on some supported platforms.
            attr = target.rsplit('.', 1)[1]
            try:
                stack.enter_context(patch(target, _deny_network))
            except AttributeError:
                if attr != 'sendmsg':
                    raise
        yield


def safe_path(path: str | Path) -> Path:
    """Reject private locations without opening/listing their contents."""
    def reject_private(candidate):
        if any(p.lower() in {'private', 'share-research', 'results_private'}
               for p in candidate.parts[2:]):
            # macOS resolves /tmp to /private/tmp; only root /private is exempt.
            raise OfflineConfigurationError('private_path', 'path_policy')

    lexical = Path(path).expanduser().absolute()
    # Veto explicit private paths before resolve() can inspect their metadata.
    reject_private(lexical)
    resolved = lexical.resolve()
    reject_private(resolved)
    return resolved


def load_fixture_records(fixture_dir: str | Path) -> list[dict]:
    path = safe_path(fixture_dir)
    if not path.is_dir():
        raise OfflineConfigurationError('fixture_directory_missing', 'fixtures')
    files = sorted(path.glob('*.json'))
    if not files:
        raise OfflineConfigurationError('fixture_directory_empty', 'fixtures')
    records = []
    for file in files:
        if file.is_symlink():
            raise OfflineConfigurationError('fixture_symlink', 'fixtures')
        try:
            record = json.loads(file.read_text())
        except (ValueError, UnicodeError, RecursionError):
            # Preserve missing/invalid response as a rejection; no raw bytes in audit.
            record = None
        records.append(record)
    return records


def prepare_results_dir(results_dir: str | Path, *, new_run: bool = False) -> Path:
    output = safe_path(results_dir)
    if output.exists():
        if not output.is_dir():
            raise OfflineConfigurationError('results_not_directory', 'results_directory')
        if not new_run and any(output.iterdir()):
            raise OfflineConfigurationError('results_not_empty', 'results_directory')
    output.mkdir(parents=True, exist_ok=True)
    if new_run:
        # Explicit opt-in: an atomic unique child, never reuse/overwrite a prior run.
        return Path(tempfile.mkdtemp(prefix='run-', dir=output))
    return output


def run_offline_smoke(fixture_dir: str | Path, results_dir: str | Path, *,
                      new_run: bool = False) -> dict:
    with error_stage('network_guard'), offline_network_veto():
        with error_stage('fixtures'):
            records = load_fixture_records(fixture_dir)
        with error_stage('results_directory'):
            output = prepare_results_dir(results_dir, new_run=new_run)
        with error_stage('synthetic_generation'):
            tasks = smoke_tasks()
        rows, predictions, audit = [], [], []
        for index, task in enumerate(tasks):
            # Generator objects and labels never cross into this function call.
            with error_stage('prediction'):
                prediction = analyze_observables(task.frame, records, card=PUBLIC_CARD, seed=601)
                fixed = json.dumps(prediction, sort_keys=True, indent=2, allow_nan=False)
            filename = f'prediction_{index}.json'
            with error_stage('prediction_write'):
                (output / filename).write_text(fixed)
            sha = hashlib.sha256(fixed.encode()).hexdigest()
            predictions.append({'prediction_file': filename, 'prediction_sha256': sha})
            audit.append({'task_index': index, 'accepted': prediction['accepted_count'],
                          'rejected': prediction['rejected_count'],
                          'source_abstained': prediction['source_abstention_count'],
                          'n_observed': prediction['n_observed'],
                          'robust_reason': prediction['methods']['robust_screened']['reason'],
                          'proposals': prediction['graph_audit']})
        # Commit the entire batch before reading any task's truth or scoring label.
        with error_stage('prediction_write'):
            batch_manifest = {'implementation_version': IMPLEMENTATION_VERSION,
                              'predictions': predictions}
            batch_bytes = json.dumps(batch_manifest, sort_keys=True, indent=2,
                                     allow_nan=False).encode()
            batch_sha256 = hashlib.sha256(batch_bytes).hexdigest()
            (output / 'prediction_batch_manifest.json').write_bytes(batch_bytes)
            fixed_batch = tuple((output / p['prediction_file']).read_text() for p in predictions)
            if any(hashlib.sha256(f.encode()).hexdigest() != p['prediction_sha256']
                   for f, p in zip(fixed_batch, predictions)):
                raise ValueError('Fixed prediction batch integrity failure')
        with error_stage('scoring'):
            for task, fixed in zip(tasks, fixed_batch):
                scored = score_fixed_predictions(fixed, truth=task.true_ate, identified=task.identified)
                rows.extend(dict(task=task.label, **r) for r in scored)
        with error_stage('scoring'):
            summary = summarize_scores(rows)
        with error_stage('provenance'):
            root = Path(__file__).resolve().parents[3]
            source_files = [*sorted((root / 'src/sharecausal').glob('**/*v2*.py')),
                            root / 'src/sharecausal/llm/synthetic_schema.py',
                            root / 'src/sharecausal/llm/synthetic_adapter.py',
                            root / 'scripts/llm_bridge_v2_offline.py',
                            root / 'docs/freeze_manifest_v1.json', root / 'docs/test_only_erratum_v1.json']
            report = {'kind': 'v2_offline_synthetic_integration_smoke', 'fixture_mode': True,
                      'implementation_version': IMPLEMENTATION_VERSION,
                      'results_dir': str(output),
                      'real_llm_calls': 0, 'openrouter_cost_usd': 0, 'share_records_used': 0,
                      'network_policy': 'process_local_socket_veto', 'n_tasks': len(predictions),
                      'n_method_rows': len(rows), 'failures': [], 'summary': summary,
                      'predictions': predictions, 'graph_screen_audit': audit,
                      'prediction_batch_manifest': 'prediction_batch_manifest.json',
                      'prediction_batch_sha256': batch_sha256,
                      'public_prompt': prompt_payload(PUBLIC_CARD),
                      'public_card_sha256': digest(PUBLIC_CARD),
                      'source_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                        for f in source_files},
                      'fixture_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                         for f in sorted(safe_path(fixture_dir).glob('*.json'))},
                      'environment': {'python': platform.python_version(),
                                      'python_implementation': platform.python_implementation(),
                                      'platform': platform.platform(), 'machine': platform.machine(),
                                      'packages': {p: importlib.metadata.version(p) for p in
                                                   ('numpy', 'pandas', 'scipy', 'scikit-learn', 'networkx', 'pydantic')}},
                      'warning': 'Deterministic dummy fixtures, not real LLM performance. '
                                 'Integration smoke only; no superiority, calibration or confirmatory claim.'}
        with error_stage('artifact_write'):
            (output / 'smoke_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            (output / 'graph_provenance.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
            pd.DataFrame(rows).to_csv(output / 'run_level.csv', index=False)
            pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
        return report
