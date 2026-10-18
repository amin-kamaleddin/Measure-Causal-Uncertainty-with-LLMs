"""Stratified estimates; smaller samples mean limited power and overlap."""
import numpy as np
from .estimators import aipw,frame_matrix,_interval

def estimate_subgroups(df, features=('age','female','education'), min_n=50, folds=3,seed=11):
    out={}
    d=df.dropna(subset=['Y','A','poor',*features,'country'])
    collected={}
    for k in [0,1]:
        sub=d.loc[d.poor==k]
        if len(sub)<min_n or min(sub.A.value_counts().reindex([0,1],fill_value=0))<folds*3:
            out[str(k)]={'status':'insufficient_sample','n':len(sub)};continue
        x=frame_matrix(sub, continuous=features)
        try:
            fit,score=aipw(sub.Y,sub.A,x,folds,seed,cluster=sub.country,return_scores=True)
            out[str(k)]={'status':'estimated',**fit.as_dict()}
            collected[k]=(sub.index,score)
        except ValueError as e:out[str(k)]={'status':'not_estimable','n':len(sub),'reason':str(e)}
    if all(out[str(k)].get('status')=='estimated' for k in [0,1]):
        delta=out['1']['effect']-out['0']['effect']
        # IF of subgroup contrast on the union of rows; clusters may occur in both groups.
        score_full=np.zeros(len(d)); positions={idx:j for j,idx in enumerate(d.index)}
        for k,(indices,score) in collected.items():
            n_sub=len(indices); mean=out[str(k)]['effect']
            influence=(1 if k==1 else -1)*(len(d)/n_sub)*(score-mean)
            for idx,value in zip(indices,influence):score_full[positions[idx]]=value
        _,se,_,_=_interval(score_full,cluster=d.country)
        out['poor_minus_other_effect']=delta
        out['interaction_difference_se_approx_country_cluster']=se
        out['interaction_difference_95_interval']=[delta-1.959963984540054*se,delta+1.959963984540054*se]
        out['interaction_warning']='Exploratory, country-cluster approximate, not causal effect modification'
    return out
