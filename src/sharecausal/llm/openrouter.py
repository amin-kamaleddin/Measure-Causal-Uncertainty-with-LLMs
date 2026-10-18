"""OpenRouter budget-guarded API. No external calls without explicit --live."""
from __future__ import annotations
import hashlib
import json
import os
import time
from datetime import datetime,timezone
import httpx
from .schemas import GraphAnswer
from .prompts import messages
from .budget import Budget

BASE='https://openrouter.ai/api/v1'


def _strict_graph_schema():
    """Make a provider-compatible strict schema without loosening Pydantic validation.

    Pydantic defaults make edges optional in exported JSON Schema; strict
    structured-output endpoints require all declared object properties.
    Array-length limits are still checked by GraphAnswer after decoding.
    """
    schema = GraphAnswer.model_json_schema()

    def normalize(part):
        if isinstance(part, dict):
            part.pop('maxItems', None)
            part.pop('minItems', None)
            if part.get('type') == 'object':
                part['required'] = list(part.get('properties', {}))
                part['additionalProperties'] = False
            for value in part.values():
                normalize(value)
        elif isinstance(part, list):
            for value in part:
                normalize(value)

    normalize(schema)
    return schema


def _safe_http_error(response, key, model_id):
    """Report only whitelisted API diagnostics, never headers/raw metadata."""
    try:
        body = response.json()
        error = body.get('error', {}) if isinstance(body, dict) else {}
        if not isinstance(error, dict):
            error = {}
    except ValueError:
        error = {}
    meta = error.get('metadata') or {}
    if not isinstance(meta, dict):
        meta = {}

    def clean(value, limit=600):
        return str(value or 'unknown').replace(key, '[REDACTED]')[:limit]

    return RuntimeError(
        'OpenRouter HTTP {status} model={model} code={code} type={etype} '
        'provider_code={pcode} message={message}'.format(
            status=response.status_code,
            model=clean(model_id, 120),
            code=clean(error.get('code'), 60),
            etype=clean(meta.get('error_type'), 80),
            pcode=clean(meta.get('provider_code'), 80),
            message=clean(error.get('message')),
        )
    )


def available_models(key=None,transport=None):
    client=transport or httpx.Client(timeout=25.0)
    with client if transport is None else _noop_context(client):
        response=client.get(BASE+'/models',headers={'Authorization':f'Bearer {key}'} if key else {})
        response.raise_for_status()
        models=response.json().get('data',[])
        if not isinstance(models,list):raise ValueError('Malformed model discovery response')
        return models

class _noop_context:
    def __init__(self,obj):self.obj=obj
    def __enter__(self):return self.obj
    def __exit__(self,*args):return False


def choose_distinct_models(models,preferred=('openai','anthropic','google','meta-llama'),max_models=4,model_ids=None):
    """Choose currently listed 3-4 text/JSON models from distinct families.

    Explicit model IDs are validated against the live catalog, not assumed valid.
    Defaults choose the least expensive currently suitable model per family,
    which is a COST baseline, not a prevalidated comparable-capability study.
    """
    if model_ids:
        catalog={m.get('id'):m for m in models}
        if len(model_ids)<3 or len(model_ids)>4 or len(set(model_ids))!=len(model_ids):
            raise ValueError('Specify 3 or 4 different OpenRouter model IDs')
        if len({x.split('/')[0] for x in model_ids})!=len(model_ids):
            raise ValueError('One chosen model per family; no duplicate families')
        missing=[x for x in model_ids if x not in catalog]
        if missing:raise ValueError('Model IDs absent from current OpenRouter catalog: '+', '.join(missing))
        chosen=[catalog[x] for x in model_ids]
        for m in chosen:
            if 'response_format' not in (m.get('supported_parameters') or []):
                raise ValueError('Model does not advertise required JSON response format: '+m['id'])
            if 'text' not in (m.get('architecture') or {}).get('output_modalities',['text']):
                raise ValueError('Model does not support text responses: '+m['id'])
        return chosen
    chosen=[]
    for author in preferred:
        family=[m for m in models if m.get('id','').startswith(author+'/')
                and 'response_format' in (m.get('supported_parameters') or [])
                and 'text' in (m.get('architecture') or {}).get('output_modalities',['text'])]
        if not family:continue
        def price(m):
            try:return float(m['pricing']['prompt'])+float(m['pricing']['completion'])
            except (ValueError,KeyError,TypeError):return float('inf')
        family=[x for x in family if 0<price(x)<float('inf')]
        if family:chosen.append(min(family,key=price))
        if len(chosen)==max_models:break
    if len(chosen)<3:
        raise RuntimeError(f'Only {len(chosen)} distinct JSON-capable families; need >=3 for valid live comparison')
    return chosen


def upper_cost_estimate(model, prompt_chars, max_completion_tokens=1200):
    """Conservative char-to-token input upper bound; includes 2x safety reserve.

    Costs stored as USD per token in /models metadata. Reasoning/endpoint
    surcharges and cost uncertainty may make this insufficient; stop on discrepancy.
    """
    price=model['pricing'];p=float(price['prompt']);c=float(price['completion'])
    return 2.0*(prompt_chars*p+max_completion_tokens*c)


def complete(model,variant,budget:Budget,key:str,client=None,max_tokens=1200,seed=None,reasoning_effort=None):
    if not key:raise RuntimeError('OPENROUTER_API_KEY is required for --live')
    if reasoning_effort not in (None, 'low', 'medium', 'high'):
        raise ValueError('reasoning_effort must be low, medium, high, or None')
    supported=set(model.get('supported_parameters') or [])
    if reasoning_effort is not None and 'reasoning' not in supported:
        raise ValueError('Requested reasoning effort is not advertised by this model')
    chat=messages(variant)
    reserve=budget.reserve(upper_cost_estimate(model,sum(len(x['content']) for x in chat),max_tokens))
    strict_schema = _strict_graph_schema()
    payload={'model':model['id'],'messages':chat,'max_tokens':max_tokens,
             'response_format':{'type':'json_schema','json_schema':{'name':'share_graph',
                     'strict':True,'schema':strict_schema}},
             'provider':{'require_parameters':True,'allow_fallbacks':False},'usage':{'include':True}}
    # Requiring all parameters would block models that do not implement temperature.
    if 'temperature' in supported:payload['temperature']=0
    if seed is not None and 'seed' in supported:payload['seed']=seed
    if reasoning_effort is not None:
        payload['reasoning']={'effort':reasoning_effort}
    owned=client is None
    client=client or httpx.Client(timeout=75.)
    try:
        for attempt in range(3):
            try:
                r=client.post(BASE+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload)
                if r.status_code in (429,500,502,503,504) and attempt<2:
                    time.sleep(attempt+1);continue
                if r.is_error:
                    raise _safe_http_error(r, key, model['id'])
                answer=r.json()
                if isinstance(answer,dict) and answer.get('error'):
                    # Some provider failures arrive with HTTP 200; fail closed.
                    raise _safe_http_error(r, key, model['id'])
                break
            except (httpx.TransportError,httpx.TimeoutException):
                if attempt==2:raise
                time.sleep(attempt+1)
        usage=answer.get('usage') or {}
        if usage.get('cost') is None:
            # Fail closed; do not continue unaccounted calls.
            raise RuntimeError('Provider response missing billed cost; stop experiment')
        budget.reconcile(reserve,float(usage['cost']))
        reserve=0.
        # A 200 response may contain no visible completion: for reasoning models,
        # finish_reason=length often means the token cap was consumed by reasoning.
        # This is a *billed failed result*, never an empty graph or an abstention.
        choices=answer.get('choices')
        choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {}
        message=choice.get('message')
        raw=message.get('content') if isinstance(message,dict) else None
        if not isinstance(raw,(str,bytes,bytearray)) or not raw.strip():
            details=usage.get('completion_tokens_details') or {}
            if not isinstance(details,dict):details={}
            def safe_label(value):
                # Never emit raw provider response text, reasoning, or headers.
                return ''.join(c for c in str(value or 'unknown') if c.isascii() and (c.isalnum() or c in '_.-'))[:80]
            def safe_count(value):
                return value if isinstance(value,int) and not isinstance(value,bool) and 0<=value<10**9 else 'unknown'
            raise RuntimeError(
                'OpenRouter HTTP 200 without usable JSON content; '
                f'model={safe_label(model["id"])} '
                f'response_id={safe_label(answer.get("id"))} '
                f'finish_reason={safe_label(choice.get("finish_reason"))} '
                f'completion_tokens={safe_count(usage.get("completion_tokens"))} '
                f'reasoning_tokens={safe_count(details.get("reasoning_tokens"))} '
                f'billed_cost_usd={float(usage["cost"]):.8g}. '
                'This may indicate a reasoning-token budget exhaustion, but the '
                'finish_reason must be checked. No automatic retry was made.'
            )
        parsed=GraphAnswer.model_validate_json(raw)
        return {'model':answer.get('model',model['id']), 'model_requested':model['id'],
                'provider_response_id':answer.get('id'),'timestamp':datetime.now(timezone.utc).isoformat(),
                'prompt_variant':variant,'seed_requested':seed,'seed_sent':payload.get('seed'),
                'temperature_sent':payload.get('temperature'),
                'reasoning_effort_sent':reasoning_effort,
                'max_tokens_sent':max_tokens,
                'finish_reason':choice.get('finish_reason'),
                'input_sha256':hashlib.sha256(json.dumps(chat,sort_keys=True).encode()).hexdigest(),
                'schema_sha256':hashlib.sha256(json.dumps(strict_schema,sort_keys=True).encode()).hexdigest(),
                'usage':usage,'graph':parsed.model_dump()}
    finally:
        if reserve>0:budget.cancel(reserve)
        if owned:client.close()
