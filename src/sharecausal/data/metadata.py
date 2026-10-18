"""Nonidentifying metadata summary for offline audit."""
from .archives import metadata
from .variables import MAP

def core_metadata(paths):
    out = {}
    for (w, mod), cols in MAP.items():
        md = metadata(paths[w], w, mod)
        absent = set(cols) - set(md['columns'])
        if absent:
            raise ValueError(f'{w}/{mod} columns not found: {sorted(absent)}')
        out[f'w{w}.{mod}'] = {'rows': md['rows'], 'columns_verified': cols,
                              'labels': {c: md['variable_labels'][c] for c in cols},
                              'codes': {c: md['value_labels'][c] for c in cols if md['value_labels'][c]}}
    return out
