from pathlib import Path
import os
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_approved_config_has_disjoint_seeds_and_disabled_external():
    cfg = yaml.safe_load((ROOT/'configs/approved_protocol_v1.yaml').read_text())
    assert cfg['research_design']['share_application'] == 'descriptive_observational'
    assert cfg['privacy']['openrouter_live_authorized'] is False
    assert cfg['privacy']['mode'] == 'local_only'
    ev = cfg['synthetic_evaluation']
    assert set(ev['development_seeds']).isdisjoint(ev['heldout_seeds'])
    assert ev['no_oracle_graph_support'] is True
    assert ev['proposal_sources'] == 'offline_fixed_fixtures_not_real_llms'


def test_live_llm_requires_independent_approval_before_key_or_any_network():
    env = os.environ.copy()
    env.pop('SHARECAUSAL_LIVE_LLM_APPROVED', None)
    env.pop('OPENROUTER_API_KEY', None)
    env['PYTHONPATH'] = str(ROOT/'src')
    x = subprocess.run([sys.executable, '-m', 'sharecausal.cli', 'llm-compare', '--live'],
                       text=True, capture_output=True, env=env, cwd=ROOT, timeout=20)
    assert x.returncode != 0
    assert 'Live OpenRouter calls are disabled' in x.stderr
