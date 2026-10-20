"""Only invented aggregates: verify safe, data-driven publication rendering."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def inputs(tmp_path):
    package = tmp_path / 'private_inputs'
    empirical = tmp_path / 'empirical'
    (package / 'share_binary').mkdir(parents=True)
    (package / 'share_update').mkdir()
    empirical.mkdir()
    fit = dict(effect=-.2, ci_lower=-.3, ci_upper=-.1,
               t_ci_lower=-.32, t_ci_upper=-.08, n=700)
    result = dict(
        flow=dict(valid_wave8_activity=1000),
        selection_counts=[dict(country='ALL', A=arm, n_eligible=500,
            known_dead_undefined_Y=25, recorded_alive_observed_Y=375,
            recorded_alive_noninterview=20,
            recorded_alive_interview_missing_Y=30, unknown_vital_status=50)
            for arm in (0, 1)],
        estimates={name:fit for name in ('aipw_demographic_country',
            'same_sample_without_w8_depression','aipw_with_w8_depression')},
        prior_wave_depression={'without':fit, 'with_D7':fit})
    (empirical/'results.json').write_text(json.dumps(result))
    (package/'share_binary/summary.json').write_text(json.dumps(
        dict(estimate=dict(fit, effect=-.02, t_ci_lower=-.03, t_ci_upper=-.01))))
    (package/'share_update/unknown_survival_tipping.csv').write_text(
        'hypothetical_unknown_alive_fraction,delta_unexposed,delta_exposed,difference\n'
        '0,0,-2,-.2\n0,0,0,-.1\n0,0,2,0\n1,0,-2,-.3\n1,0,0,0\n1,0,2,.3\n')
    return package, empirical


def command(package, empirical, output):
    env = dict(os.environ)
    env.pop('OPENROUTER_API_KEY', None)
    return subprocess.run([sys.executable,
        str(REPO/'scripts/render_final_study_clinical_figures.py'),
        '--package',str(package),'--empirical-results',str(empirical),
        '--output-package',str(output)],cwd=REPO,env=env,
        capture_output=True,text=True)


def test_figure_counts_come_from_invented_aggregates(tmp_path):
    package, empirical = inputs(tmp_path)
    output=tmp_path/'private_render'
    run=command(package,empirical,output)
    assert run.returncode==0,run.stderr
    provenance=json.loads((output/'clinical_figure_provenance.json').read_text())
    assert provenance['flow_counts']['n_eligible']==1000
    assert provenance['flow_counts']['known_alive']==850
    assert provenance['provider_calls']==0
    svg=(output/'figures/figure2_share_timing_flow.svg').read_text()
    assert 'Valid Wave 8 participation: 1,000' in svg
    assert 'Main complete covariate sample: 700.' in svg
    manifest=json.loads((output/'clinical_figure_manifest.json').read_text())
    assert len(manifest)==15
    assert all(hashlib.sha256((output/item['file']).read_bytes()).hexdigest()
               ==item['sha256'] for item in manifest)


def test_nonexhaustive_followup_states_fail_before_figures(tmp_path):
    package, empirical = inputs(tmp_path)
    path=empirical/'results.json';result=json.loads(path.read_text())
    result['selection_counts'][0]['n_eligible']+=1
    path.write_text(json.dumps(result))
    output=tmp_path/'private_render'
    run=command(package,empirical,output)
    assert run.returncode!=0
    assert 'followup_states_not_exhaustive' in run.stderr
    assert not list(output.rglob('*.png'))


def test_clinical_output_inside_repository_is_rejected(tmp_path):
    package, empirical = inputs(tmp_path)
    output=REPO/'outputs/private/publication_boundary_test'
    run=command(package,empirical,output)
    assert run.returncode!=0
    assert 'clinical_output_must_be_outside_repository' in run.stderr
    assert not output.exists()
