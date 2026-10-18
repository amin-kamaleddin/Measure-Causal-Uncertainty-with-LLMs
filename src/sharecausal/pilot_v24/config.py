"""Strict, candidate-only offline configuration."""
from typing import Literal
from pydantic import Field,model_validator
from sharecausal.preflight.contracts import Strict


class Estimation(Strict):
    folds:Literal[3]
    estimator_seed:int=Field(ge=0)
    propensity_clip:float=Field(gt=0,lt=.5)
    maximum_extreme_propensity_fraction:float=Field(ge=0,lt=1)
    bias_shifts:list[float]

    @model_validator(mode='after')
    def shifts(self):
        if not self.bias_shifts or 0. not in self.bias_shifts or len(set(self.bias_shifts))!=len(self.bias_shifts):
            raise ValueError('prespecified_unique_bias_shifts_including_zero_required')
        return self


class Config(Strict):
    schema_version:Literal['pilot-config-2.4']
    protocol_status:Literal['candidate_not_frozen']
    study_kind:Literal['exposed_development_exploratory_pilot']
    policy_approval:Literal['DECISION_REQUIRED']
    authorized_openrouter_usd:int=Field(ge=0,le=0)
    public_tasks:str
    generation_plan:str
    scoring_truth:str
    families:str
    decisions:str
    mock_rules:str
    precision_reference_config:str
    models:list[str]
    variants:list[str]
    source_repetitions:int=Field(ge=1,le=10)
    sample_size:int=Field(ge=30,le=100000)
    sample_repetitions:int=Field(ge=1,le=10)
    sampling_seed:int=Field(ge=0)
    primary_sample_repetition:Literal[0]
    estimation:Estimation
    fixed_margin_thresholds:list[float]
    practical_zero_sensitivity:list[float]
    assertion_target:float=Field(gt=0,lt=1)
    matching_tolerance:float=Field(gt=0,lt=1)

    @model_validator(mode='after')
    def declared(self):
        if self.models!=[f'mock-family-{x}/fixture' for x in 'abcd'] or self.variants!=['base','challenge']:
            raise ValueError('fixed_mock_family_schedule_required')
        if self.fixed_margin_thresholds!=[0.,.1,.25,.5] or self.practical_zero_sensitivity!=[0.,.05,.1]:
            raise ValueError('candidate_fixed_diagnostic_grids_required')
        return self
