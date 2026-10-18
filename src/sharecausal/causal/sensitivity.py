"""Sensitivity parameters are user-chosen shifts, not empirically estimated bias."""
from .robust_conclusions import ConditionalInterval, assess

def shifted_intervals(estimate:dict, shifts:list[float]):
    return [{'bias_shift':float(s),'effect':estimate['effect']+float(s),
            'lower':estimate['ci_lower']+float(s),'upper':estimate['ci_upper']+float(s)}
            for s in shifts]

def default_structural_verdict(primary:dict, alt:dict|None, shifts):
    included=[ConditionalInterval('B_pre_only',primary['effect'],primary['ci_lower'],primary['ci_upper'],True),
              ConditionalInterval('latent_U',0,0,0,False)]
    if alt:
        included.append(ConditionalInterval('W8_D8_pre_assumption',alt['effect'],alt['ci_lower'],alt['ci_upper'],True))
    return assess(included,shifts)
