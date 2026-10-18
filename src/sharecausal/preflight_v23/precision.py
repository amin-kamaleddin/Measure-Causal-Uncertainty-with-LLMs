"""Predeclared coverage stress grid; finite-family bounded intervals vs t comparator.

For independent family means D_g in [-1,1] and fixed weights w_g=n_g/N,
Hoeffding gives P(|sum w_g(D_g-E D_g)| >= h) <= 2 exp(-h²/(2 sum w_g²)).
No within-family independence, normality or positive event count is required.
It is conservative and may be far too wide. Family independence is an assumption.
"""
import math
from typing import Literal
import numpy as np
from scipy.special import ndtr
from scipy.stats import t
from pydantic import Field,model_validator
from sharecausal.preflight.contracts import Strict,sha


class PrecisionPlan(Strict):
    card_counts:list[int]
    family_counts:list[int]
    differences_per_card:list[float]
    conditional_error_a:list[float]
    latent_family_correlations:list[float]
    source_b_dropout:list[float]
    couplings:list[Literal['shared_uniform','conditionally_independent']]
    repetitions:int=Field(ge=200,le=20000)
    seed:int=Field(ge=0)
    nominal_coverage:float=Field(gt=0,lt=1)
    minimum_coverage:float=Field(gt=0,lt=1)
    maximum_MC_half_width:float=Field(gt=0,lt=1)
    precision_half_width:float=Field(gt=0,lt=1)
    assertion_rate:float=Field(gt=0,lt=1)
    rare_event_stress:bool

    @model_validator(mode='after')
    def valid_grid(self):
        fields=('card_counts','family_counts','differences_per_card','conditional_error_a',
                'latent_family_correlations','source_b_dropout','couplings')
        if any(not getattr(self,k) or len(set(getattr(self,k)))!=len(getattr(self,k)) for k in fields):
            raise ValueError('nonempty_unique_precision_grids_required')
        if any(g<3 or n<g or n%g for n in self.card_counts for g in self.family_counts):
            raise ValueError('balanced_card_family_design_required')
        if (any(not 0<=r<1 for r in self.latent_family_correlations+self.source_b_dropout)
            or any(not 0<p<1 for p in self.conditional_error_a)
            or any(d<=0 for d in self.differences_per_card)
            or self.minimum_coverage>self.nominal_coverage):raise ValueError('invalid_precision_probabilities')
        for p in self.conditional_error_a:
            for d in self.differences_per_card:
                for dropout in self.source_b_dropout:
                    q=(self.assertion_rate*p-d)/(self.assertion_rate*(1-dropout))
                    if not 0<=q<=1:raise ValueError('infeasible_error_contrast_dropout_scenario')
        return self


def wilson(successes,trials,z=1.959963984540054):
    if trials<1 or not 0<=successes<=trials:raise ValueError('invalid_binomial_counts')
    p=successes/trials;den=1+z*z/trials
    center=(p+z*z/(2*trials))/den
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return max(0.,center-half),min(1.,center+half)


def coverage_gate(successes,trials,minimum,maximum_MC_half_width):
    lo,hi=wilson(successes,trials)
    if hi<minimum:status='FAIL_UNDERCOVERAGE'
    elif lo<minimum or (hi-lo)/2>maximum_MC_half_width:status='FAIL_MC_UNSTABLE'
    else:status='PASS_SCENARIO_COVERAGE'
    return {'coverage_MC_lower':lo,'coverage_MC_upper':hi,
            'coverage_MC_half_width':(hi-lo)/2,'coverage_gate':status}


def bounded_half_width(weights,nominal=.95):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or len(w)<2 or np.any(w<=0) or not np.isclose(w.sum(),1):
        raise ValueError('independent_family_weights_required')
    return math.sqrt(2*float(np.square(w).sum())*math.log(2/(1-nominal)))


def family_intervals(values,nominal=.95):
    """Rows are simulations, columns families. Equal family sizes required here."""
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[1]<3 or not np.isfinite(x).all() or np.any(np.abs(x)>1):
        raise ValueError('bounded_independent_family_means_required')
    g=x.shape[1];mean=x.mean(axis=1)
    h_t=float(t.ppf((1+nominal)/2,g-1))*x.std(axis=1,ddof=1)/math.sqrt(g)
    h_b=np.repeat(bounded_half_width(np.repeat(1/g,g),nominal),len(x))
    return mean,{'family_t_comparator':h_t,'bounded_family_hoeffding':h_b}


def simulate_case(n,g,difference,p_a,rho,dropout,coupling,plan,label='planning'):
    # Seed derives from scenario content, never truth or heldout model results.
    key=[n,g,difference,p_a,rho,dropout,coupling,label]
    rng=np.random.default_rng(plan.seed+int(sha(key)[:12],16))
    reps=plan.repetitions;size=n//g;shape=(reps,g,size)
    family_z=rng.normal(size=(reps,g,1))
    z=math.sqrt(rho)*family_z+math.sqrt(1-rho)*rng.normal(size=shape)
    u_a=ndtr(z)
    u_b=u_a if coupling=='shared_uniform' else ndtr(math.sqrt(rho)*family_z+
            math.sqrt(1-rho)*rng.normal(size=shape))
    # Entire family loses source B, independently of errors. All planned cards remain.
    present_b=rng.random((reps,g,1))>=dropout
    asserted=rng.random(shape)<plan.assertion_rate
    p_b=(plan.assertion_rate*p_a-difference)/(plan.assertion_rate*(1-dropout))
    if not 0<=p_b<=1:raise ValueError('invalid_simulation_marginals')
    a=(u_a<p_a)&asserted;b=(u_b<p_b)&asserted&present_b
    delta=a.astype(float)-b.astype(float)
    estimates,intervals=family_intervals(delta.mean(axis=2),plan.nominal_coverage)
    rows=[]
    for method,half in intervals.items():
        lower=np.maximum(-1,estimates-half);upper=np.minimum(1,estimates+half)
        covered=(lower<=difference)&(difference<=upper);successes=int(covered.sum())
        gate=coverage_gate(successes,reps,plan.minimum_coverage,plan.maximum_MC_half_width)
        mean_half=float(np.mean((upper-lower)/2))
        rows.append({'scenario_id':sha(key)[:16],'scenario_kind':label,'cards':n,'families':g,
            'difference_per_eligible_card':difference,'conditional_error_a':p_a,'conditional_error_b':p_b,
            'latent_Gaussian_family_correlation_not_binary_ICC':rho,'source_b_family_dropout':dropout,
            'paired_error_coupling':coupling,'procedure':method,'MC_repetitions':reps,
            'mean_estimate':float(estimates.mean()),'MC_sd':float(estimates.std(ddof=1)),
            'coverage_successes':successes,'observed_coverage':successes/reps,**gate,
            'mean_half_width':mean_half,'half_width_90th_percentile':float(np.quantile((upper-lower)/2,.9)),
            'zero_event_fraction':float(np.mean(~(a|b).any(axis=(1,2)))),
            'zero_width_fraction':float(np.mean(half==0)),
            'assertion_rate_a':float(asserted.mean()),'assertion_rate_b':float((asserted&present_b).mean()),
            'coverage_matched_population':abs(plan.assertion_rate-.5)<=.1 and abs(plan.assertion_rate*(1-dropout)-.5)<=.1,
            'excludes_zero_fraction_NOT_validated_power':float(np.mean((lower>0)|(upper<0))),
            'precision_gate':bool(gate['coverage_gate']=='PASS_SCENARIO_COVERAGE' and
                                  np.quantile((upper-lower)/2,.9)<=plan.precision_half_width),
            'status':'candidate_scenario_not_LLM_performance'})
    return rows


def diagnostics(plan):
    rows=[]
    for n in plan.card_counts:
        for g in plan.family_counts:
            for d in plan.differences_per_card:
                for p in plan.conditional_error_a:
                    for rho in plan.latent_family_correlations:
                        for drop in plan.source_b_dropout:
                            for coupling in plan.couplings:
                                rows.extend(simulate_case(n,g,d,p,rho,drop,coupling,plan))
            if plan.rare_event_stress:
                rows.extend(simulate_case(n,g,.002,.005,.6,0.,'shared_uniform',plan,'rare_event_negative_control'))
    return rows
