import json
import subprocess
import sys

def _run(*args):
    return subprocess.run([sys.executable,'-m','sharecausal.cli',*args],capture_output=True,text=True)

def test_doctor_cli():
    result=_run('doctor')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='ready'

def test_mock_cli():
    r=_run('llm-compare','--dry-run')
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['actual_llm_results']==0

def test_inspect_synthetic_stata(mock_archives):
    r=_run('inspect','--data-dir',str(mock_archives))
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['metadata_verified']['w8.ac']['rows']==40
