"""Private, path-limited cache of PUBLIC model prompts and raw answers."""
import json
from pathlib import Path

def cache_path(root, key):
    if len(key)!=64 or any(ch not in '0123456789abcdef' for ch in key):raise ValueError('Unsafe cache digest')
    return Path(root)/f'{key}.json'

def save(root,key,obj):
    p=cache_path(root,key);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True));p.chmod(0o600)

def load(root,key):
    p=cache_path(root,key)
    return json.loads(p.read_text()) if p.exists() else None
