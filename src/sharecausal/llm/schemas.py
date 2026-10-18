"""Structured LLM graph elicitation with hostile-output validation."""
from __future__ import annotations
from pydantic import BaseModel,Field,ConfigDict,field_validator
import networkx as nx

KNOWN={'B','A','D8','Y','H','U','S'}
class Arrow(BaseModel):
    model_config=ConfigDict(extra='forbid')
    source:str
    target:str
    causal_role:str
    temporal_rationale:str
    evidence_reference:str
    uncertainty:str
    @field_validator('source','target')
    @classmethod
    def approved_variables(cls,value):
        if value not in KNOWN:raise ValueError(f'Unrecognized node: {value}')
        return value

class GraphAnswer(BaseModel):
    model_config=ConfigDict(extra='forbid')
    edges:list[Arrow]=Field(default_factory=list,max_length=40)
    abstain:bool
    abstention_reason:str
    contested_assumptions:list[str]

    @field_validator('edges')
    @classmethod
    def validate_graph(cls,arrows):
        if len({(a.source,a.target) for a in arrows})!=len(arrows):
            raise ValueError('Duplicate directed edge')
        g=nx.DiGraph((a.source,a.target) for a in arrows)
        if not nx.is_directed_acyclic_graph(g):raise ValueError('Cyclic graph')
        if any((a.source=='Y' and a.target in ('A','B','H','D8','U')) or
               (a.source in ('A','Y','D8','S') and a.target in ('B','H'))
               for a in arrows):
            raise ValueError('Backwards time arrow or implausible post-exposure cause of baseline')
        return arrows
