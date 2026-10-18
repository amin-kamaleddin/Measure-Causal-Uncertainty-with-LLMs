"""Public-only request contract. No generator or scorer imports."""
from pydantic import Field
from sharecausal.preflight.contracts import Strict, canonical, sha
from sharecausal.preflight_v23.packets import prompt
from sharecausal.preflight_v23.admission import Proposal

MODELS = ('google/gemini-3.5-flash-lite', 'openai/gpt-6-luna',
          'meta-llama/llama-4-maverick', 'deepseek/deepseek-v4.1-flash')
SLOTS = tuple(f'mock-family-{s}/fixture' for s in 'abcd')
ESTIMATION = dict(folds=3, estimator_seed=71813, propensity_clip=.01,
                  maximum_extreme_propensity_fraction=.20, bias_shifts=[-.1, 0., .1])
SCORING = dict(sample_repetitions=1, primary_sample_repetition=0,
               fixed_margin_thresholds=[0., .1, .25, .5],
               practical_zero_sensitivity=[0., .05, .1], assertion_target=.5,
               matching_tolerance=.1)


class Answer(Strict):
    proposals: list[Proposal] = Field(min_length=1, max_length=10)


def schema():
    value = Answer.model_json_schema()
    def normalize(x):
        if isinstance(x, dict):
            x.pop('minItems', None); x.pop('maxItems', None)
            if x.get('type') == 'object':
                x['required'] = list(x.get('properties', {})); x['additionalProperties'] = False
            for v in x.values(): normalize(v)
        elif isinstance(x, list):
            for v in x: normalize(v)
    normalize(value)
    return value


def messages(task, variant="base"):
    return [dict(role='system', content=(
        'You are assessing fictional synthetic scenario evidence, not real studies. '
        'Return only a JSON object with proposals: an array of complete causal graph '
        'hypotheses in the supplied schema. Use only allowed nodes and exact evidence IDs. '
        'Distinguish stipulated arrows from ambiguous hypotheses. Retain supported arrows '
        'in each complete graph. Give supported alternatives when facts cannot distinguish '
        'them, or explicitly abstain with a reason. Absence of a path in a hypothesis is '
        'not an unconditional zero claim. Adjustment may contain observed baseline nodes '
        'only; never adjust for mediators or post-outcome selection. No empirical effects '
        'or hidden mechanisms are provided. Do not invent evidence or infer effect signs.')),
        dict(role='user', content=canonical(dict(packet=prompt(task, "challenge" if variant=="counterargument" else "base"), answer_schema=schema())).decode())]


def settings(model):
    supported = set(model.get('supported_parameters') or [])
    if 'response_format' not in supported: raise ValueError('response_format_unavailable')
    result = dict(max_tokens=8192 if model["id"] == MODELS[3] else 4096,
                  provider=dict(require_parameters=True, allow_fallbacks=False,
                      max_price=dict(prompt=2*float(model['pricing']['prompt'])*1e6,
                                     completion=2*float(model['pricing']['completion'])*1e6)))
    if 'temperature' in supported and model['id'] != MODELS[1]: result['temperature'] = 0
    if 'reasoning' in supported: result['reasoning'] = dict(effort='low')
    return result


def request_key(task_hash, model_id, chat, effective, variant="base"):
    return sha(dict(task_sha256=task_hash, model=model_id, messages=chat,
                    schema=schema(), settings=effective, variant=variant, repetition=0))
