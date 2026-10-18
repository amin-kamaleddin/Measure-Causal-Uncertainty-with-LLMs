"""Post-seal rational binary-DAG certificates. No label-based non-ID inference."""
from fractions import Fraction as F
from itertools import product
from typing import Literal
from pydantic import Field, model_validator
from sharecausal.preflight.contracts import Strict
from sharecausal.preflight.graph_utils import has_path
from .packets import task_hash
from .admission import supported_exclusion


class BinaryNode(Strict):
    name: str = Field(pattern=r'^[A-Z]$')
    parents: list[str]
    probability_one: list[str] = Field(min_length=1)

    @model_validator(mode='after')
    def table(self):
        if len(set(self.parents))!=len(self.parents) or len(self.probability_one)!=2**len(self.parents):
            raise ValueError('invalid_binary_CPT_shape')
        if any(not 0<=F(p)<=1 for p in self.probability_one):raise ValueError('invalid_CPT_probability')
        for position in range(len(self.parents)):
            bit = 2**(len(self.parents)-position-1)
            if not any(F(self.probability_one[i]) != F(self.probability_one[i+bit])
                       for i in range(len(self.probability_one)) if not i & bit):
                raise ValueError('declared_causal_parent_has_no_functional_influence')
        return self


class BinaryDAG(Strict):
    nodes: list[BinaryNode] = Field(min_length=2,max_length=6)

    @model_validator(mode='after')
    def topological(self):
        seen=set()
        for node in self.nodes:
            if node.name in seen or not set(node.parents)<=seen:raise ValueError('CPT_topological_order_required')
            seen.add(node.name)
        if not {'A','Y'}<=seen:raise ValueError('A_Y_required')
        return self


def law(scm, observed, intervention=None):
    """Independent exogenous threshold SCM, summing all unobserved binary nodes."""
    names=[n.name for n in scm.nodes];result={}
    if not set(observed)<=set(names):raise ValueError('unknown_observable')
    for values in product((0,1),repeat=len(names)):
        state=dict(zip(names,values));p=F(1)
        for node in scm.nodes:
            if intervention is not None and node.name=='A':
                p*=int(state['A']==intervention);continue
            index=0
            for parent in node.parents:index=2*index+state[parent]
            q=F(node.probability_one[index]);p*=q if state[node.name] else 1-q
        key=tuple(state[x] for x in observed);result[key]=result.get(key,F(0))+p
    return result


def effect(scm):
    return law(scm,['Y'],1)[(1,)]-law(scm,['Y'],0)[(1,)]


def edges(scm):return [(p,n.name) for n in scm.nodes for p in n.parents]


class Proof(Strict):
    kind: Literal['randomized_complete','public_exclusion','observational_pair','none']
    witness_ids: list[str]


class Truth(Strict):
    schema_version: Literal['truth-2.3']
    world_id: str = Field(pattern=r'^world-[a-z0-9-]+$')
    public_task_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    scm: BinaryDAG
    population_effect_exact: str
    family_id: str
    proof: Proof

    @model_validator(mode='after')
    def consistent(self):
        if F(self.population_effect_exact)!=effect(self.scm):raise ValueError('false_population_effect')
        return self


def compatible(truth,task):
    card=task.card;names={n.name for n in truth.scm.nodes};pairs=set(edges(truth.scm))
    if names!={v.name for v in card.variables}:return False
    timing={v.name:v.time for v in card.variables}
    if any(timing[a]>timing[b] for a,b in pairs):return False
    if card.assignment=='randomized_documented' and any(b=='A' for a,b in pairs):return False
    if supported_exclusion(task) and has_path(pairs):return False
    for e in card.evidence:
        if e.kind=='edge':
            if e.relation=='supported' and (e.source,e.target) not in pairs:return False
            if e.relation=='contradicted' and (e.source,e.target) in pairs:return False
    # Positive observed assignment probabilities are required by the public stipulation.
    return all(p>0 for p in law(truth.scm,['A']).values())


def certify(truth,task,worlds):
    if truth.public_task_hash!=task_hash(task):raise ValueError('certificate_task_hash_mismatch')
    if not compatible(truth,task):raise ValueError('SCM_incompatible_with_public_information')
    result={'status':'uncertified','reason':'no_implemented_proof',
            'scope':'public_packet_plus_full_observed_law','supported_structural_zero':False,
            'graph_conditional_zero':not has_path(edges(truth.scm))}
    card=task.card
    if truth.proof.kind=='public_exclusion' and supported_exclusion(task) and not has_path(edges(truth.scm)):
        return {**result,'status':'identified','reason':'public_global_exclusion',
                'functional':'0','supported_structural_zero':True}
    complete=card.selection=='complete_outcomes'
    assignment=any(e.kind=='assignment' and e.relation=='supported' and
                   (e.source,e.target)==('A','Y') for e in card.evidence)
    if (truth.proof.kind=='randomized_complete' and complete and assignment and
            card.assignment=='randomized_documented' and not any(b=='A' for a,b in edges(truth.scm))):
        return {**result,'status':'identified','reason':'independent_assignment_complete_Y_consistency_positivity',
                'functional':'E[Y|A=1]-E[Y|A=0]'}
    if truth.proof.kind!='observational_pair':return result
    if {v.name for v in card.variables if v.observed}!={'A','Y'} or not complete:
        return {**result,'reason':'witness_observable_or_selection_scope_unproved'}
    if len(set(truth.proof.witness_ids))!=2:return {**result,'reason':'two_distinct_witnesses_required'}
    index={w.world_id:w for w in worlds}
    if len(index)!=len(worlds):raise ValueError('duplicate_world_ID')
    pair=[index.get(k) for k in truth.proof.witness_ids]
    if any(w is None for w in pair) or truth.world_id not in truth.proof.witness_ids:
        return {**result,'reason':'missing_witness'}
    if not all(w.public_task_hash==task_hash(task) and compatible(w,task) for w in pair):
        return {**result,'reason':'witness_incompatible_with_public_information'}
    joints=[law(w.scm,['A','Y']) for w in pair];effects=[effect(w.scm) for w in pair]
    if joints[0]!=joints[1] or effects[0]==effects[1]:return {**result,'reason':'observational_equivalence_not_proved'}
    return {**result,'status':'provably_nonidentified','reason':'exact_equal_full_observed_law_distinct_ATE',
            'witness_ids':truth.proof.witness_ids,'witness_effects_exact':[str(e) for e in effects],
            'observed_joint_exact':{','.join(map(str,k)):str(v) for k,v in sorted(joints[0].items())}}
