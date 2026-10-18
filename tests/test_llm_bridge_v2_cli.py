"""Real subprocess success/collision regressions for the documented offline CLI."""
from __future__ import annotations

import csv
import errno
import hashlib
from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/llm_bridge_v2'
ENTRY = ROOT / 'scripts/llm_bridge_v2_offline.py'
SECRET = 'TEST_SECRET_MUST_NOT_APPEAR_IN_DIAGNOSTICS'


def cli(output, *extra):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src')
    env['OPENROUTER_API_KEY'] = SECRET
    env['SHARECAUSAL_LIVE_LLM_APPROVED'] = '1'
    return subprocess.run(
        [sys.executable, 'scripts/llm_bridge_v2_offline.py',
         '--fixture-dir', 'tests/fixtures/llm_bridge_v2',
         '--results-dir', str(output), *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=45,
    )


def hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir() if p.is_file()}


def verify_artifacts(completed, base, *, new_run):
    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr
    assert SECRET not in completed.stdout
    stdout = json.loads(completed.stdout)
    output = Path(stdout['results_dir'])
    assert output.is_dir()
    assert (output.parent == base.resolve() if new_run else output == base.resolve())
    assert set(hashes(output)) == {
        'smoke_report.json', 'graph_provenance.json', 'prediction_0.json',
        'prediction_1.json', 'prediction_2.json', 'run_level.csv', 'summary.csv',
        'prediction_batch_manifest.json',
    }
    report = json.loads((output / 'smoke_report.json').read_text())
    assert report['results_dir'] == str(output)
    for payload in (stdout, report):
        assert payload['fixture_mode'] is True
        assert payload['real_llm_calls'] == payload['openrouter_cost_usd'] == payload['share_records_used'] == 0
        assert payload['n_tasks'] == 3 and payload['n_method_rows'] == 21
    assert report['failures'] == []
    batch_file = output / report['prediction_batch_manifest']
    assert hashlib.sha256(batch_file.read_bytes()).hexdigest() == report['prediction_batch_sha256']
    assert json.loads(batch_file.read_text())['predictions'] == report['predictions']
    assert report['fixture_sha256'] == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FIXTURES.glob('*.json')
    }
    audit = json.loads((output / 'graph_provenance.json').read_text())
    assert audit == report['graph_screen_audit'] and len(audit) == 3
    for task in audit:
        assert (task['accepted'], task['rejected'], task['source_abstained']) == (2, 1, 1)
        by_source = {p['provenance']['fixture_id']: p for p in task['proposals']}
        assert by_source['dummy_measured_x1']['status'] == 'accepted'
        assert by_source['dummy_latent_u']['status'] == 'accepted'
        assert not by_source['dummy_latent_u']['identification']['identified_by_backdoor']
        assert by_source['dummy_invalid_time']['status'] == 'rejected'
        assert 'Invalid temporal arrow' in by_source['dummy_invalid_time']['reason']
        assert by_source['dummy_explicit_abstention']['status'] == 'source_abstained'
    for index, item in enumerate(report['predictions']):
        file = output / item['prediction_file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == item['prediction_sha256']
        prediction = json.loads(file.read_text())
        methods = prediction['methods']
        assert methods['robust_screened']['decision'] == 'abstain'
        assert 'not identified' in methods['robust_screened']['reason']
        for source in ('dummy_latent_u', 'dummy_invalid_time', 'dummy_explicit_abstention'):
            assert methods[source]['decision'] == 'abstain'
        expected = 'positive' if index < 2 else 'abstain'
        for method in ('dummy_measured_x1', 'no_llm_evidence_graph', 'naive_majority'):
            assert methods[method]['decision'] == expected
    with (output / 'run_level.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (output / 'summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(rows) == 21 and len(summary) == 7
    robust = next(row for row in summary if row['method'] == 'robust_screened')
    assert robust['n_asserted'] == '0' and robust['false_sign_risk'] == ''
    assert all(SECRET not in p.read_text() for p in output.iterdir())
    return output


@pytest.mark.parametrize('new_run', [False, True])
def test_documented_cli_success_in_real_subprocess(tmp_path, new_run):
    """Use repository fixtures, root cwd, real interpreter and actual disk artifacts."""
    base = tmp_path / 'smoke'
    result = cli(base, *(['--new-run'] if new_run else []))
    verify_artifacts(result, base, new_run=new_run)


def test_documented_repeat_creates_new_run_and_preserves_existing_results(tmp_path):
    base = tmp_path / 'smoke'
    first = verify_artifacts(cli(base), base, new_run=False)
    original = hashes(first)
    # Exact old invocation now explains the observed collision, without overwriting.
    refused = cli(base)
    assert refused.returncode == 2 and not refused.stdout
    error = json.loads(refused.stderr)
    assert error['error_code'] == 'results_not_empty'
    assert error['stage'] == 'results_directory'
    assert '--new-run' in error['hint']
    assert error['live_fallback'] is False
    assert SECRET not in refused.stderr
    assert hashes(first) == original
    second = verify_artifacts(cli(base, '--new-run'), base, new_run=True)
    second_hashes = hashes(second)
    third = verify_artifacts(cli(base, '--new-run'), base, new_run=True)
    assert second != third
    assert hashes(first) == original and hashes(second) == second_hashes


@pytest.mark.parametrize('failure,code,status', [
    (PermissionError(errno.EACCES, SECRET, '/sensitive/' + SECRET), 'filesystem_io', 2),
    (ValueError(SECRET), 'invalid_configuration_or_value', 2),
    (RuntimeError(SECRET), 'unexpected_failure', 1),
])
def test_exception_diagnostics_do_not_echo_messages_or_paths(monkeypatch, capsys, tmp_path, failure, code, status):
    from sharecausal.benchmark import bridge_v2_offline as runner
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(runner, 'run_offline_smoke', fail)
    entry = runpy.run_path(str(ENTRY))
    assert entry['main'](['--fixture-dir', str(FIXTURES), '--results-dir', str(tmp_path)]) == status
    captured = capsys.readouterr()
    assert not captured.out and SECRET not in captured.err
    payload = json.loads(captured.err)
    assert payload['error_code'] == code
    assert payload['hint'] and payload['stage'] == 'startup'
    assert payload['live_fallback'] is False
    if isinstance(failure, OSError):
        assert payload['errno'] == 'EACCES'


def test_real_runner_error_reports_stage_and_redacts_underlying_exception(monkeypatch, capsys, tmp_path):
    from sharecausal.benchmark import bridge_v2_offline as runner
    def fail(*args):
        raise PackageNotFoundError(SECRET)
    monkeypatch.setattr(runner.importlib.metadata, 'version', fail)
    entry = runpy.run_path(str(ENTRY))
    assert entry['main'](['--fixture-dir', str(FIXTURES), '--results-dir', str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and SECRET not in captured.err
    payload = json.loads(captured.err)
    assert payload['stage'] == 'provenance'
    assert payload['error_code'] == 'dependency_unavailable'
    assert (tmp_path / 'prediction_0.json').is_file()
    assert not (tmp_path / 'smoke_report.json').exists()  # no fabricated successful report


def test_invalid_cli_arguments_do_not_echo_secret_values(tmp_path):
    result = cli(tmp_path, '--unsupported-option=' + SECRET)
    assert result.returncode == 2 and not result.stdout
    assert SECRET not in result.stderr
    assert json.loads(result.stderr)['error_code'] == 'invalid_arguments'
