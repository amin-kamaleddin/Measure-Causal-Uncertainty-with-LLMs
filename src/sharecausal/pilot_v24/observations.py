"""Analysis-side allowlist and masking. No generator or scorer imports."""
import numpy as np
import pandas as pd
from sharecausal.preflight.contracts import sha


def clean_frame(task,frame):
    columns=sorted(v.name for v in task.card.variables if v.observed)+['observed']
    if not isinstance(frame,pd.DataFrame) or set(frame.columns)!=set(columns) or frame.attrs:
        raise ValueError('only_observed_columns_without_metadata_required')
    if not frame.observed.isin([0,1]).all():raise ValueError('invalid_recording_indicator')
    df=frame[columns].copy()
    # Drop unrecorded outcomes AND all post-treatment measurements before any value validation/hash/fit.
    post=[v.name for v in task.card.variables if v.observed and v.time>next(v.time for v in task.card.variables if v.name=='A')]
    for name in post:df.loc[df.observed==0,name]=None
    for name in columns:
        values=df.loc[df.observed==1,name] if name in post else df[name]
        arr=np.asarray(values,dtype=float)
        if not np.isfinite(arr).all() or not np.isin(arr,[0,1]).all():raise ValueError('observed_binary_numeric_values_required')
    return df


def frame_payload(task,frame):
    df=clean_frame(task,frame)
    return {'columns':list(df.columns),'rows':[[None if pd.isna(x) else int(x) for x in row]
            for row in df.itertuples(index=False,name=None)]}


def frame_hash(task,frame):return sha(frame_payload(task,frame))


def read_sample(task,data):
    if set(data)!={'schema_version','task_hash','sample_id','columns','rows'} or data['schema_version']!='observed-sample-2.4':
        raise ValueError('sample_schema_invalid')
    if len(set(data['columns']))!=len(data['columns']):raise ValueError('duplicate_observable_column')
    if any(len(row)!=len(data['columns']) for row in data['rows']):raise ValueError('sample_row_shape_invalid')
    return clean_frame(task,pd.DataFrame(data['rows'],columns=data['columns']))
