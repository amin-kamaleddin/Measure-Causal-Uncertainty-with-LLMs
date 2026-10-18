"""Only invented frames/ZIPs. Never discover real archives or private records."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import pytest
from sharecausal.data.archives import NAME
from sharecausal.data.variables import MAP
from sharecausal.empirical_v3.io import digest, private_destination
from sharecausal.empirical_v3.graphs import backdoor, scenarios, audit_records
from sharecausal.empirical_v3.statistics import influence_interval, paired, country_estimates
from sharecausal.empirical_v3.selection import classify, counts, mnar, response_weighting
from sharecausal.empirical_v3.runner import load_config

REPO=Path(__file__).resolve().parents[1]
CONFIG=REPO/'configs/real_share_followup_v3.yaml'


@pytest.fixture
def frame():
    rng=np.random.default_rng(7);n=240
    a=np.arange(n)%2
    d=pd.DataFrame(dict(A=a,Y=2-.4*a+rng.normal(0,.2,n),age=rng.normal(65,5,n),
        female=np.arange(n)%3==0,education=rng.integers(1,6,n),country=np.arange(n)%5,
        D8=rng.integers(0,5,n),D7=np.nan,poor=(np.arange(n)%3==0).astype(float),
        SHARELIFE=1,psu=['cluster_'+str(i%11) for i in range(n)],interview9=1,deceased9=0))
    d.loc[:4,['interview9','deceased9','Y']]=[2,1,np.nan]
    d.loc[5:14,['interview9','Y']]=[0,np.nan]
    d.loc[15:19,['interview9','deceased9','Y']]=[np.nan,np.nan,np.nan]
    return d


def test_status_partition_and_unknown_survival(frame):
    status=classify(frame)
    assert (status=='known_dead_undefined_Y').sum()==5
    assert (status=='recorded_alive_noninterview').sum()==10
    assert (status=='unknown_vital_status').sum()==5
    for row in counts(frame):
        assert sum(v for k,v in row.items() if k not in ['country','A','n_eligible'])==row['n_eligible']


@pytest.mark.parametrize('bad',['death_with_Y','death_main_interview','eol_alive'])
def test_contradictory_status_fails(frame,bad):
    if bad=='death_with_Y':frame.loc[0,'Y']=2
    if bad=='death_main_interview':frame.loc[0,'interview9']=1
    if bad=='eol_alive':frame.loc[0,'deceased9']=0
    with pytest.raises(ValueError):classify(frame)


def test_mnar_excludes_death_unknown_and_bounds(frame):
    rows,s=mnar(frame,[-1,0,1])
    assert len(rows)==9 and s['excluded_deaths']==5 and s['excluded_unknown_vital']==5
    assert all(r['n_recorded_alive']==230 and r['n_missing']==10 for r in rows)
    assert all(s['worst_case_difference_lower']<=r['mean_difference']<=s['worst_case_difference_upper'] for r in rows)


def test_survivor_weights_do_not_relabel_death(frame):
    r=response_weighting(frame,load_config(CONFIG))
    assert r['n_recorded_alive_complete_covariates']==230
    assert r['death_records_used']==r['unknown_vital_records_used']==0
    assert r['n_observed']+r['n_missing']==230
    assert r['max_weight']<=20


def test_paired_country_influence_matches_manual():
    p0=np.array([1,2,3,4,5,6.]);p1=p0+np.array([0,1,0,1,0,1.])
    c=np.array([1,1,2,2,3,3])
    r=influence_interval(p1-p0,c)
    assert r['effect']==.5 and r['se']==0
    with pytest.raises(ValueError):influence_interval(p1,[1,1,1,2,2,2])


def test_matched_d8_sample_and_shared_folds(frame):
    d=frame.dropna(subset=['Y']).copy();d.loc[30,'D8']=np.nan
    e0,e1,diff=paired(d,load_config(CONFIG),('D8',))
    assert e0['n']==e1['n']==diff['n']==len(d)-1
    assert diff['effect']==pytest.approx(e1['effect']-e0['effect'])
    assert diff['same_rows'] and diff['same_crossfit_folds']


def test_string_psu_cluster_regression(frame):
    c=load_config(CONFIG);c['country_min_n']=30;c['country_min_arm']=9
    rows=country_estimates(frame.dropna(subset=['Y']),c)
    assert len(rows)==5 and all(r['uncertainty']=='PSU_cluster_t' for r in rows)


def test_country_support_not_silently_pooled(frame):
    rows=country_estimates(frame.dropna(subset=['Y']),load_config(CONFIG))
    assert all(r['status']=='INSUFFICIENT_SUPPORT' and r['effect'] is None for r in rows)


def test_theory_d8_pre_post_and_latent_backdoor():
    ss=scenarios();assert len(ss)==8
    for s in ss:
        edges=set(s['edges'])
        assert not ({('A','D8'),('D8','A')}<=edges)
        assert ('Y','V') not in edges
        assert s['causal_verdict']=='ABSTAIN'
        if ('U','A') in edges:assert backdoor(edges)==[]
        elif ('A','D8') in edges:assert backdoor(edges)==[['B','H']]
        else:assert backdoor(edges)==[['B','D8','H']]


def test_cycle_fails_and_empty_backdoor_is_distinct():
    with pytest.raises(ValueError):backdoor([('A','Y'),('Y','A')])
    assert backdoor([('A','Y')])==[[]]


def test_outputs_private_empty_only(tmp_path):
    with pytest.raises(ValueError):private_destination(tmp_path/'public')
    root=private_destination(tmp_path/'private'/'run');(root/'existing').write_text('preserve')
    with pytest.raises(ValueError):private_destination(root)


@pytest.mark.parametrize('field,value',[('processing_mode','authorized_external'),('folds',1),
    ('propensity_clip',0),('mnar_shifts',[float('nan')]),('seed',True)])
def test_config_fail_closed(tmp_path,field,value):
    import yaml
    c=load_config(CONFIG);c[field]=value;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):load_config(p)


def command(script,*args):
    env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None);env.pop('SHARECAUSAL_LIVE_LLM_APPROVED',None)
    env['PYTHONPATH']=str(REPO/'src');env['MPLCONFIGDIR']=str(REPO/'outputs/private/.test_plot_cache')
    return subprocess.run([sys.executable,str(REPO/'scripts'/script),*map(str,args)],
                          cwd=REPO,env=env,capture_output=True,text=True,timeout=180)


def test_network_and_credentials_guard_subprocess(tmp_path):
    code='''from sharecausal.empirical_v3.io import install_offline_guard
import socket
install_offline_guard()
for action in [lambda:socket.getaddrinfo("openrouter.ai",443),lambda:socket.socket().connect(("127.0.0.1",9)),lambda:open(".env")]:
 try:action()
 except PermissionError:pass
 else:raise AssertionError("Guard failed")
print("offline_guard_passed")
'''
    r=subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONPATH':str(REPO/'src')},capture_output=True,text=True)
    assert r.returncode==0 and r.stdout.strip()=='offline_guard_passed'


def make_inputs(root):
    """Independent synthetic archive fixture, including exact release status labels."""
    n=240;rng=np.random.default_rng(21);ids=[f'INVENTED_SENTINEL_{i}' for i in range(n)]
    root.mkdir()
    labels={'interview':{0:'No interview',1:'Main interview',2:'End-of-Life interview'},
            'deceased':{0:'Alive',1:'Deceased'}}
    for w in [7,8,9]:
        with zipfile.ZipFile(root/NAME.format(wave=w),'w') as z:
            for (wave,module),columns in MAP.items():
                if wave!=w:continue
                d=pd.DataFrame({c:ids if c=='mergeid' else np.zeros(n) for c in columns})
                if 'country' in d:d['country']=11+np.arange(n)%5
                if module=='cv_r':
                    d['interview']=1;d['int_year']=2016+(w-7)*3;d['int_month']=3
                if w==8:
                    if module=='cv_r':d['age_int']=60+np.arange(n)%20;d['gender']=1+np.arange(n)%3%2
                    if module=='ac':d['ac035d5']=np.arange(n)%2
                    if module=='gv_health':d['eurod']=rng.integers(0,8,n)
                    if module=='gv_isced':d['isced1997_r']=rng.integers(1,6,n)
                    if module=='gv_weights':d['psu']=['PSU_'+str(i%13) for i in range(n)]
                if w==7:
                    if module=='cc':d['cc733_']=2+np.arange(n)%3%2
                    if module=='gv_weights':d['SHARELIFE']=1
                    if module=='gv_health':d['eurod']=-9
                if w==9:
                    if module=='cv_r':
                        d.loc[:4,['deceased','interview']]=[1,2]
                        d.loc[5:14,'interview']=0
                    if module=='gv_health':d['eurod']=rng.integers(0,8,n);d.loc[:14,'eurod']=-9
                buff=io.BytesIO();d.to_stata(buff,write_index=False,version=118,
                      value_labels={k:v for k,v in labels.items() if k in d})
                z.writestr(f'sharew{w}_rel9-0-0_{module}.dta',buff.getvalue())
    records=[];models=[];provenance=[]
    for i in range(4):
        edges=[] if i==3 else [dict(source=a,target=b,causal_role='hypothesis',temporal_rationale='Invented',
                                  evidence_reference='unverified hypothesis',uncertainty='high') for a,b in [('A','Y'),('U','A'),('U','Y')]]
        item=dict(model_requested=f'test-only-{i}',input_sha256='a'*64,schema_sha256='b'*64,
                  graph=dict(edges=edges,abstain=i!=2,abstention_reason='Test',contested_assumptions=[]))
        p=root/f'record{i}.json';p.write_text(json.dumps(item));records.append(p)
        models.append(dict(model=item['model_requested'],source_sha256=digest(p)))
        provenance.extend(dict(model=item['model_requested'],arrow=e['source']+'->'+e['target'],
                               evidence_reference=e['evidence_reference']) for e in edges)
    (root/'comparison_summary.json').write_text(json.dumps(dict(real_llm_response_records=4,n_unique_proposed_edges=3,models=models)))
    pd.DataFrame({'arrow':['A->Y','U->A','U->Y']}).to_csv(root/'edge_proposals.csv',index=False)
    pd.DataFrame(provenance).to_csv(root/'edge_provenance_PRIVATE.csv',index=False)
    (root/'prior.json').write_text('{"estimates":{}}')
    return records


def test_actual_cli_with_invented_zip_integration(tmp_path):
    inp=tmp_path/'input';records=make_inputs(inp);out=tmp_path/'private'/'run'
    args=['--data-dir',inp,'--comparison-dir',inp,'--prior-results',inp/'prior.json',
          '--config',CONFIG,'--results-dir',out]
    for p in records:args+=['--record',p]
    r=command('real_share_followup_v3.py',*args)
    assert r.returncode==0,(r.stdout,r.stderr)
    result=json.loads((out/'results.json').read_text())
    assert result['share_records_used']==240 and result['estimates']['aipw_demographic_country']['n']==225
    assert result['real_llm_calls']==result['network_calls']==result['openrouter_cost_usd']==result['respondent_uploads']==0
    assert result['causal_decision']=='ABSTAIN'
    assert len(list(out.glob('*.pdf')))>=8 and len(list(out.glob('*.csv')))>=10
    assert not any(p.suffix in ['.parquet','.dta','.zip'] for p in out.iterdir())
    for p in out.glob('*.json'):
        text=p.read_text();json.loads(text)
        assert 'INVENTED_SENTINEL_' not in text and '"mergeid"' not in text
        assert p.stat().st_mode & 0o077 == 0
    assert all(digest(out/k)==v for k,v in json.loads((out/'artifact_manifest.json').read_text()).items())
    again=command('real_share_followup_v3.py',*args)
    assert again.returncode==2 and 'FAILED' in again.stderr


def test_cli_error_redacts_hostile_validation(tmp_path):
    inp=tmp_path/'input';records=make_inputs(inp)
    records[0].write_text('{"model_requested":"sk-SECRET_SENTINEL","graph":{"edges":[{"source":"PRIVATE_SENTINEL"}]}}')
    args=['--data-dir',inp,'--comparison-dir',inp,'--prior-results',inp/'prior.json',
          '--results-dir',tmp_path/'private'/'failed']
    for p in records:args+=['--record',p]
    r=command('real_share_followup_v3.py',*args)
    assert r.returncode==2
    assert 'SECRET_SENTINEL' not in r.stdout+r.stderr and 'PRIVATE_SENTINEL' not in r.stdout+r.stderr
    error=json.loads(r.stderr.splitlines()[-1]);assert error['stage']=='saved_graph_audit'
    assert error['code_locations']
