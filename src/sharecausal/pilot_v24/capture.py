"""Deterministic public-only source capture, redacted parsing and fixed ballots."""
import json
from collections import Counter
from sharecausal.preflight.contracts import sha,canonical
from sharecausal.preflight.graph_utils import is_acyclic
from sharecausal.preflight.comparison import Edge
from sharecausal.preflight_v23.admission import Proposal,no_llm,proposal,audit
from sharecausal.preflight_v23.packets import prompt,task_hash
from sharecausal.preflight_v23.transport import MockReceipt


def _reject_constant(_):raise ValueError('nonfinite_JSON')

def _unique_object(pairs):
    data={}
    for k,v in pairs:
        if k in data:raise ValueError('duplicate_JSON_field')
        data[k]=v
    return data


def parse_wire(task,wire):
    try:
        decoded=json.loads(wire,parse_constant=_reject_constant,object_pairs_hook=_unique_object)
        if not isinstance(decoded,list) or len(decoded)>10:raise ValueError('response_list_required')
    except (ValueError,TypeError,RecursionError):
        return {'status':'invalid_response','proposals':[],'rejected_proposals':1}
    valid=[];bad=0
    for value in decoded:
        try:
            p=Proposal.model_validate(value)
            if not audit(task,p.model_dump(),'permissive_sensitivity')[0]['format_valid']:raise ValueError('invalid_time')
            valid.append(p.model_dump())
        except (ValueError,TypeError):bad+=1
    status='invalid_response' if bad else 'absent' if not valid else 'source_abstained' if all(
        p['graph']['abstain'] for p in valid) else 'multiple_graphs' if len(valid)>1 else 'graph_response'
    # A partly malformed source is a failed source ballot; it cannot hide extra votes.
    return {'status':status,'proposals':valid if not bad else [],'rejected_proposals':bad}


def capture(task,model,variant,repetition,rules):
    baseline=no_llm(task)
    if model.endswith('a/fixture'):answers=[baseline[0].model_dump()]
    elif model.endswith('b/fixture'):answers=[baseline[-1].model_dump()]
    elif model.endswith('c/fixture'):
        note=next(e for e in task.card.evidence if e.kind=='general')
        sure=[Edge(source=e.source,target=e.target,evidence_id=e.evidence_id) for e in task.card.evidence
              if e.kind=='edge' and e.relation=='supported']
        answers=[proposal(sure+[Edge(source='U',target='Y',evidence_id=note.evidence_id)]).model_dump()]
    elif model.endswith('d/fixture'):
        answers=([p.model_dump() for p in baseline] if len(baseline)>1 else
            [baseline[0].model_dump()] if task.card.structural_exclusion or any(v.name=='M' for v in task.card.variables)
            else [proposal(abstain=True).model_dump()])
    else:raise ValueError('unknown_mock_source')
    if rules['absent_d_on_unknown_recording'] and model.endswith('d/fixture') and task.card.selection=='selection_unknown':answers=[]
    wire=canonical(answers).decode()
    if rules['malformed_c_challenge'] and variant=='challenge' and model.endswith('c/fixture'):wire='{invalid offline mock JSON'
    parsed=parse_wire(task,wire)
    fields={'source_type':'offline_fixture','transport':'local_only_no_sender','model_id':model,
        'variant':variant,'repetition':repetition,'prompt_sha256':sha(prompt(task,variant)),
        'task_sha256':task_hash(task),'response_sha256':sha(wire),'schema_sha256':sha(Proposal.model_json_schema())}
    receipt=MockReceipt(**fields,record_sha256=sha(fields))
    return {'receipt':receipt.model_dump(),'parsed':parsed,'parsed_sha256':sha(parsed)}


def validate_record(task,record,model,variant,repetition):
    try:
        if set(record)!={'receipt','parsed','parsed_sha256'}:raise ValueError('record_fields')
        r=MockReceipt.model_validate(record['receipt'])
        if (r.record_sha256!=sha(r.model_dump(exclude={'record_sha256'}))
            or r.model_id!=model or r.variant!=variant or r.repetition!=repetition
            or r.prompt_sha256!=sha(prompt(task,variant)) or r.task_sha256!=task_hash(task)
            or r.schema_sha256!=sha(Proposal.model_json_schema()) or record['parsed_sha256']!=sha(record['parsed'])):
            raise ValueError('record_binding')
        parsed=record['parsed']
        if set(parsed)!={'status','proposals','rejected_proposals'} or parsed['status'] not in (
            'invalid_response','absent','source_abstained','multiple_graphs','graph_response'):
            raise ValueError('parsed_schema')
        if type(parsed['rejected_proposals']) is not int or parsed['rejected_proposals']<0:
            raise ValueError('rejection_count')
        if not isinstance(parsed['proposals'],list):raise ValueError('proposal_list')
        for raw in parsed['proposals']:
            if not audit(task,raw,'permissive_sensitivity')[0]['format_valid']:raise ValueError('proposal_invalid')
    except (ValueError,TypeError,AttributeError,KeyError):return False
    return True


def vote(task,by_model,models):
    if len(set(models))!=len(models) or not models:raise ValueError('unique_planned_sources_required')
    counts=Counter();empties=0;source_audit={}
    for model in models:
        record=by_model.get(model,{'status':'absent','proposals':[]})
        rows=record['proposals']
        if record['status']!='graph_response' or len(rows)!=1:
            source_audit[model]=record['status'];continue
        a,p=audit(task,rows[0],'permissive_sensitivity')
        if not a['format_valid'] or p.graph.abstain:source_audit[model]='invalid_or_abstained';continue
        source_audit[model]='one_format_valid_ballot';empties+=not p.graph.edges
        counts.update((e.source,e.target) for e in p.graph.edges)
    edges=sorted(e for e,c in counts.items() if c>len(models)/2)
    if not edges and empties<=len(models)/2:reason='no_strict_majority'
    elif not is_acyclic(edges):reason='cyclic_majority_no_repair'
    else:reason='strict_format_valid_majority'
    if reason!='strict_format_valid_majority':return None,{'reason':reason,'planned_denominator':len(models),'sources':source_audit}
    # Cite only public evidence; voted ungrounded edges remain visibly unrelated.
    cited=[]
    for a,b in edges:
        matches=[e for e in task.card.evidence if e.kind=='edge' and (e.source,e.target)==(a,b)]
        evidence=sorted(matches,key=lambda e:(e.relation!='supported',e.evidence_id))[0] if matches else task.card.evidence[0]
        cited.append(Edge(source=a,target=b,evidence_id=evidence.evidence_id))
    return proposal(cited).model_dump(),{'reason':reason,'planned_denominator':len(models),'sources':source_audit}
