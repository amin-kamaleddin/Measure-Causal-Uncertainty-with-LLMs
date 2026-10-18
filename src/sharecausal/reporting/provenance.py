"""Private provenance manifest only, never commit input archive hashes with row data."""
from __future__ import annotations
import datetime
import hashlib
import json
import platform
import importlib.metadata
from pathlib import Path
from sharecausal.data.archives import archive_inventory

def environment():
    deps=['numpy','pandas','scipy','scikit-learn','networkx','statsmodels','matplotlib','pydantic','httpx','PyYAML','pytest']
    out={}
    for x in deps:
        try:out[x]=importlib.metadata.version(x)
        except importlib.metadata.PackageNotFoundError:out[x]='unavailable'
    return {'python':platform.python_version(),'platform':platform.platform(), 'dependencies':out}

def manifest(paths,config:dict,seed,method):
    enc=json.dumps(config,sort_keys=True).encode()
    return {'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'data_provenance':archive_inventory(paths,hash_input=True),
            'config_sha256':hashlib.sha256(enc).hexdigest(),
            'config':config,'seed':seed,'method':method,'environment':environment(),
            'private':'SHARE-derived outputs and input hashes are not for public repository'}
