"""Process-local audit policy; no credential read, socket or worker subprocess."""
import os
from pathlib import Path
import sys

KEY_NAMES=frozenset(('OPENROUTER_API_KEY','SHARECAUSAL_LIVE_LLM_APPROVED'))


def forbidden(path, *, runtime_source=False):
    p=Path(path).absolute()
    return (any(x.lower() in {'share-research','private','results_private','.aws','.ssh','.config','.codex'}
                or x.lower().startswith('.env') for x in p.parts[2:])
            or p.suffix.lower() in {'.zip','.dta','.sav','.parquet','.feather'}
            or (not runtime_source and any(s in p.name.lower() for s in ('credentials','api_key','apikey','secrets'))))


class ProcessGuard:
    """A role gets explicit data files; runtime/library source remains importable.

    Truth is denied in every role except scorer, after an explicit seal check.
    This is a tested local Python IO barrier, not an OS security sandbox or human custody.
    """
    def __init__(self,role,root,*,inputs=(),output=None,truth=None,generation=None):
        self.role=role;self.root=Path(root).resolve();self.truth_ready=False
        self.truth=Path(truth).resolve() if truth else None
        self.generation=Path(generation).resolve() if generation else None
        self.inputs={Path(p).resolve() for p in inputs}
        self.output=Path(output).resolve() if output else None
        self.runtime=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),self.root/'src',Path('/usr/share')]
        self.counts={'credential_or_data_denials':0,'truth_denials':0,'network_denials':0,'subprocess_denials':0}

    def install(self):
        original=os._Environ.__getitem__
        def no_key_read(env,key):
            if key in KEY_NAMES:raise PermissionError('credential_access_forbidden')
            return original(env,key)
        os._Environ.__getitem__=no_key_read
        sys.addaudithook(self.audit)
        return self

    def role_output_file(self,path):
        if not self.output or not path.is_relative_to(self.output):return False
        if self.role=='orchestrator':return True
        allowed={'generate':{'sample_manifest.json','generator_receipt.json'},
                 'predict':{'mock_records.json','prediction_batch.json','prediction_guard_receipt.json'},
                 'score':{'scored_results.json','scorer_receipt.json'}}
        if path.parent==self.output and path.name in allowed.get(self.role,set()):return True
        return self.role=='generate' and path.parent==self.output/'observed' and path.name.startswith('sample-') and path.suffix=='.json'

    def audit(self,event,args):
        if event.startswith('socket.'):
            self.counts['network_denials']+=1;raise PermissionError('network_forbidden')
        if event=='subprocess.Popen' and self.role!='orchestrator':
            self.counts['subprocess_denials']+=1;raise PermissionError('worker_subprocess_forbidden')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).absolute()
        runtime_source=path.suffix in ('.py','.pyc','.so') and any(path.is_relative_to(p) for p in self.runtime[:2])
        if forbidden(path,runtime_source=runtime_source):
            self.counts['credential_or_data_denials']+=1;raise PermissionError('credential_or_data_path_forbidden')
        # Lexical checks above happen before any resolution into private directories.
        path=path.resolve()
        runtime_source=path.suffix in ('.py','.pyc','.so') and any(path.is_relative_to(p) for p in self.runtime[:2])
        if forbidden(path,runtime_source=runtime_source):raise PermissionError('resolved_private_path_forbidden')
        mode=args[1] if len(args)>1 else 'r';flags=args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR))
        if self.truth and path==self.truth:
            if writing or self.role!='score' or not self.truth_ready:
                self.counts['truth_denials']+=1;raise PermissionError('scoring_truth_before_seal_forbidden')
            return
        if self.generation and path==self.generation and (writing or self.role!='generate'):
            raise PermissionError('generator_plan_not_analysis_input')
        if writing:
            if not self.role_output_file(path):raise PermissionError('write_outside_role_output')
        elif (path not in self.inputs and not self.role_output_file(path)
              and not any(path.is_relative_to(p) for p in self.runtime)):
            raise PermissionError('undeclared_input_read_forbidden')
