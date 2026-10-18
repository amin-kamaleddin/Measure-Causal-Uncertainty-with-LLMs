"""Private exact command/exit ledger. Never captures environment or headers."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess


def record(out,command,code,started,ended,**extra):
    path=Path(out)/'commands_and_exit_codes.json'
    rows=json.loads(path.read_text()) if path.exists() else []
    rows.append(dict(argv=command,exit_code=code,started_utc=started,ended_utc=ended,**extra))
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(rows,indent=2));pending.chmod(0o600);pending.replace(path)


def execute(command,cwd,out,timeout=None,env=None):
    start=datetime.now(timezone.utc).isoformat()
    if env is None:
        env=dict(os.environ)
        env.pop('OPENROUTER_API_KEY',None)
        env.pop('SHARECAUSAL_LIVE_LLM_APPROVED',None)
    result=subprocess.run(command,cwd=cwd,timeout=timeout,env=env)
    if Path(out).exists():
        record(out,command,result.returncode,start,datetime.now(timezone.utc).isoformat(),cwd=str(cwd))
    return result
