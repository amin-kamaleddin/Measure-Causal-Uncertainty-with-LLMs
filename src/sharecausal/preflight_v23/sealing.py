"""Exclusive immutable local commitments, checked before and after scoring IO."""
import hashlib
from pathlib import Path
from sharecausal.preflight.contracts import canonical,sha


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path,data):
    with Path(path).open('xb') as stream:stream.write(canonical(data))


def seal(output,names,seal_name):
    if len(set(names))!=len(names):raise ValueError('duplicate_seal_member')
    manifest={name:digest(output/name) for name in sorted(names)}
    write_once(output/seal_name,manifest)
    return manifest,digest(output/seal_name)


def verify(output,manifest,seal_name,expected_hash):
    if digest(output/seal_name)!=expected_hash:raise ValueError('immutable_seal_modified')
    for name,expected in manifest.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or digest(output/p)!=expected:
            raise ValueError('sealed_member_modified')


def definition_commitment(root,config,plan,families,decisions):
    """Definitions/code only. Never opens or hashes private truth before prediction."""
    sources=sorted((root/'src/sharecausal/preflight_v23').glob('*.py'))
    sources+=sorted((root/'src/sharecausal/preflight').glob('*.py'))
    sources+=[root/'scripts/v23_preflight_offline.py']
    sources+=[root/'scripts/build_v23_fixtures.py']
    return {'schema_version':'design-commitment-2.3','candidate_config':config,
            'rubric':plan,'family_definitions':families,'decisions':decisions,
            'source_sha256':{str(p.relative_to(root)):digest(p) for p in sources},
            'truth_custody':'single_team_local_integrity_only_DECISION_REQUIRED'}
