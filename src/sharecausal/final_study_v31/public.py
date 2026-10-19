"""Public-only provider contract. No authoring, sampling, or truth imports."""
import json
import re
from typing import Literal
from pydantic import Field, model_validator
from sharecausal.preflight.contracts import Strict, sha, canonical
from sharecausal.synthetic_final_v1.public import MODELS, settings as old_settings

class Document(Strict):
    evidence_id: str
    text: str

class Card(Strict):
    task_id: str
    title: str
    variables: dict[str, dict]
    query: str
    documents: list[Document]

    @model_validator(mode='after')
    def binding(self):
        value=self.model_dump(); value.pop('task_id')
        if self.task_id != sha(value): raise ValueError('card_hash_mismatch')
        if len({d.evidence_id for d in self.documents})!=len(self.documents):
            raise ValueError('duplicate_evidence')
        return self

class Edge(Strict):
    source: str
    target: str
    evidence_id: str
    quote: str
    support: Literal['entailed','credible_ambiguous']
    role: Literal['direct','confounding','mediation','selection','uncertain']

class Graph(Strict):
    edges: list[Edge] = Field(max_length=20)
    explanation: str

class Answer(Strict):
    status: Literal['graphs','abstain']
    graphs: list[Graph] = Field(max_length=6)
    reason: str

    @model_validator(mode='after')
    def disposition(self):
        if (self.status=='graphs') != bool(self.graphs): raise ValueError('inconsistent_answer')
        return self

class Review(Strict):
    graph_hash: str
    verdict: Literal['retain','reject','uncertain']
    evidence_id: str
    quote: str
    reason: str

class Critic(Strict):
    reviews: list[Review]

def schema(critic=False):
    value=(Critic if critic else Answer).model_json_schema()
    def walk(x):
        if isinstance(x,dict):
            for k in ('minItems','maxItems'): x.pop(k,None)
            if x.get('type')=='object':
                x['required']=list(x.get('properties',{}));x['additionalProperties']=False
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(value);return value

def export(card):
    if type(card) is not Card: raise ValueError('typed_public_card_required')
    value=Card.model_validate(card.model_dump()).model_dump()
    text=canonical(value).lower()
    if any(x in text for x in (b'true_ate',b'world_id',b'generator_seed',b'family_id',b'potential_outcome',b'mergeid',b'openrouter_api_key')) or re.search(rb'sk-[a-z0-9]{8,}',text):
        raise ValueError('private_marker_in_payload')
    return value

def messages(card,variant,repetition,graphs=None):
    critic=variant=='critic'
    instruction=('Audit every anonymized graph against the same fictional evidence. No source identities are supplied. '
        'Reject a graph only for a demonstrable contradiction or unsupported edge; uncertain hypotheses should remain uncertain. '
        'Cite an exact evidence quote for each verdict. Return all graph hashes once.' if critic else
        'Propose complete causal graph alternatives supported by these fictional documents, or explicitly abstain. '
        'Every edge requires an existing evidence ID and an exact substring quote. Distinguish entailed arrows from credible ambiguous routes. '
        'Use only listed variable symbols; preserve chronology and all established arrows. Missing arrows in a complete graph assert their absence. '
        'Do not infer an effect sign or fabricate citations. Use separate graphs to retain genuine ambiguity. '
        + ('Act as a skeptic: search for hidden common causes, selection, incompatible timing, and alternative explanations.' if variant=='counterargument' else 'Read the complete evidence before proposing graphs.'))
    packet=dict(public_task=export(card),condition=variant,repetition=repetition,answer_schema=schema(critic))
    if critic:
        packet['anonymized_graphs']=[dict(graph_hash=sha(g),graph=dict(edges=g['edges'])) for g in graphs]
    return [dict(role='system',content=instruction),dict(role='user',content=canonical(packet).decode())]

def settings(model):
    result=old_settings(model)
    # Same named providers as the completed development experiment, no fallback.
    result['provider']['only']=[dict(zip(MODELS,('google-ai-studio','openai','digitalocean','open-inference')))[model['id']]]
    return result

def decode(body,critic=False):
    from sharecausal.pilot_v24.capture import _unique_object,_reject_constant
    choices=body.get('choices')
    choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {}
    message=choice.get('message')
    message=message if isinstance(message,dict) else {}
    if message.get('refusal'):return 'refusal',None
    value=message.get('content')
    if not isinstance(value,str) or not value.strip():return 'null_content',None
    try:
        parsed=(Critic if critic else Answer).model_validate(json.loads(value,object_pairs_hook=_unique_object,parse_constant=_reject_constant))
        return ('explicit_abstention' if not critic and parsed.status=='abstain' else 'valid'),parsed.model_dump()
    except (ValueError,TypeError,RecursionError):return 'schema_failure',None
