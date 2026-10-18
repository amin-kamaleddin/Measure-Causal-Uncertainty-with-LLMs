"""Memory-only selective Stata readers; no respondent file written to disk."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import re
import zipfile
import pandas as pd

WAVES = (7, 8, 9)
NAME = 'sharew{wave}_rel9-0-0_ALL_datasets_stata.zip'


def locate_archives(data_dir: str | Path) -> dict[int, Path]:
    root = Path(data_dir)
    paths = {w: root / NAME.format(wave=w) for w in WAVES}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required SHARE release 9.0.0 archives missing: ' + ', '.join(missing))
    return paths


def module_filename(wave: int, module: str) -> str:
    if wave not in WAVES or not re.fullmatch(r'[a-z_0-9]+', module):
        raise ValueError('Invalid wave/module')
    return f'sharew{wave}_rel9-0-0_{module}.dta'


def read_module(archive: str | Path, wave: int, module: str,
                columns: list[str] | None = None) -> pd.DataFrame:
    """Use pandas' Stata parser on ZIP bytes, preserving numeric coding and NaN.

    `pyreadstat` is optional (unavailable on some disconnected machines); no
    temporary Stata extracts and no dependency on Stata software.
    """
    member = module_filename(wave, module)
    with zipfile.ZipFile(archive) as z:
        if member not in z.namelist():
            raise KeyError(f'Required module {member} absent from {archive}')
        raw = io.BytesIO(z.read(member))
    with pd.io.stata.StataReader(raw, convert_categoricals=False) as reader:
        vars_available = set(reader.variable_labels())
        requested = columns if columns is not None else list(vars_available)
        absent = set(requested) - vars_available
        if absent:
            raise ValueError(f'{member} is missing required columns: {sorted(absent)}')
        data = reader.read(columns=requested)
    if 'mergeid' in data and data.mergeid.duplicated().any():
        ndup = int(data.mergeid.duplicated().sum())
        raise ValueError(f'{member}: {ndup} duplicate mergeid values; cannot 1:1 join')
    return data


def metadata(archive: str | Path, wave: int, module: str) -> dict:
    member = module_filename(wave, module)
    with zipfile.ZipFile(archive) as z:
        if member not in z.namelist():
            raise KeyError(member)
        b = io.BytesIO(z.read(member))
    with pd.io.stata.StataReader(b, convert_categoricals=False) as reader:
        labels = reader.variable_labels()  # initializes parser
        codelabels = reader.value_labels()
        var_to_label = dict(zip(reader._varlist, reader._lbllist))
        return {'wave': wave, 'module': module, 'rows': int(reader._nobs),
                'columns': list(labels), 'variable_labels': labels,
                'value_labels': {x: {str(k): v for k, v in codelabels.get(var_to_label.get(x, ''), {}).items()}
                                 for x in labels}}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def archive_inventory(paths: dict[int, Path], hash_input: bool = False) -> dict:
    inventory = {}
    for w, path in paths.items():
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            modules = [n.rsplit('_', 1)[-1] for n in names if n.endswith('.dta')]
            entry = {'archive': path.name, 'wave': w, 'members': len(names),
                     'total_uncompressed_bytes': sum(x.file_size for x in z.infolist()),
                     'stata_modules': len(modules)}
            if hash_input:
                entry['sha256'] = sha256_file(path)
            inventory[str(w)] = entry
    return inventory
