from pathlib import Path
import os
import subprocess
import pytest
from sharecausal.llm.prompts import PUBLIC_CONTEXT
from sharecausal.llm.compare import dry_run
from sharecausal.llm.cache import cache_path

def test_prompt_is_public_and_not_observed_data():
    assert 'NO empirical coefficients' in PUBLIC_CONTEXT
    assert 'mergeid' not in PUBLIC_CONTEXT
    assert dry_run()['cost_usd']==0

def test_cache_only_digest():
    with pytest.raises(ValueError):cache_path('/tmp','../../secret')

def _published_candidates(root: Path) -> list[str]:
    """List tracked and unignored files; exclude ignored local environments.

    Git sees forced-tracked raw files even when the .gitignore contains their
    extension. A clean source distribution (without .git) uses an equivalent
    conservative walk that skips environments and approved private locations.
    """
    try:
        scan = subprocess.run(
            ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
            cwd=root, capture_output=True, check=False,
        )
    except FileNotFoundError:
        scan = None
    if scan is not None and scan.returncode == 0:
        return [os.fsdecode(p) for p in scan.stdout.split(b'\0') if p]

    skip_directories = {
        '.git', '.venv', 'venv', 'env', '.tox', '.nox', 'site-packages',
        '__pycache__', '.pytest_cache', 'build', 'dist',
    }
    skip_private_prefixes = {
        ('data', 'private'), ('data', 'raw'), ('outputs', 'private'),
        ('private',), ('results_private',),
    }
    paths = []
    for folder, dirs, files in os.walk(root):
        relative_parts = Path(folder).relative_to(root).parts
        dirs[:] = [name for name in dirs
                   if name not in skip_directories and
                   not any((relative_parts + (name,))[:len(prefix)] == prefix
                           for prefix in skip_private_prefixes)]
        for name in files:
            paths.append(str((Path(folder) / name).relative_to(root)))
    return paths


def test_no_raw_data_in_repository():
    """Prevent Git releases of respondent archives, not .venv examples.

    statsmodels ships harmless .dta fixtures in site-packages. A local Python
    environment is not a repository publication candidate; however a file
    force-added to Git *is* checked regardless of extension ignore patterns.
    """
    root = Path(__file__).parents[1]
    ignore = (root / '.gitignore').read_text()
    assert '.venv/' in ignore
    assert '*.dta' in ignore and '*.zip' in ignore
    sensitive_extensions = ('.dta', '.sav', '.parquet', '.feather',
                            '.pkl', '.pickle', '.zip', '.dta.gz')
    unsafe = [name for name in _published_candidates(root)
              if name.lower().endswith(sensitive_extensions)
              or Path(name).name.lower().startswith('sharew')]
    assert not unsafe, f'Potentially publishable sensitive-data files: {unsafe}'
