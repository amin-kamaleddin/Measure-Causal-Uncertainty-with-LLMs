"""Scoring-only truth and restricted, executable identification certificates.

These are proofs for the specified binary fixture class, not a general ID solver.
Latent labels, selection warnings and backdoor failure never certify full non-ID.
"""
from fractions import Fraction as F
from typing import Literal

from pydantic import Field, model_validator
from .contracts import Strict, PublicWorldCardV2, packet_hash


class BinarySCM(Strict):
    kind: Literal['randomized', 'latent_copy']
    p_root: float = Field(ge=0, le=1)
    p_y0: float = Field(ge=0, le=1)
    p_y1: float = Field(ge=0, le=1)
    p_a_u0: float = Field(ge=0, le=1)
    p_a_u1: float = Field(ge=0, le=1)


class Certificate(Strict):
    kind: Literal['randomized_complete', 'public_exclusion', 'binary_pair', 'none']
    witness_world_ids: list[str]
    public_card_hash: str = Field(pattern=r'^[a-f0-9]{64}$')


class PrivateWorldTruthV2(Strict):
    schema_version: Literal['private-truth-2.2']
    world_id: str = Field(pattern=r'^world-[a-z0-9]+$')
    public_card_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    true_scm: BinarySCM
    population_effect: float
    potential_outcomes: list[list[float]] = Field(min_length=1)
    finite_sample_effect: float
    mechanism_family: str
    narrative_style: str
    generator_seed: int
    backdoor_failure: bool
    selection_warning: bool
    structural_zero_graph: bool
    certificate: Certificate

    @model_validator(mode='after')
    def internally_consistent(self):
        if any(len(p) != 2 for p in self.potential_outcomes):
            raise ValueError('invalid_potential_outcome_pair')
        if any(x not in (0, 1) for p in self.potential_outcomes for x in p):
            raise ValueError('binary_potential_outcomes_required')
        if self.true_scm.kind == 'latent_copy' and any(p[0] != p[1] for p in self.potential_outcomes):
            raise ValueError('latent_copy_potential_outcomes_incompatible')
        avg = sum(p[1] - p[0] for p in self.potential_outcomes) / len(self.potential_outcomes)
        if abs(avg - self.finite_sample_effect) > 1e-12:
            raise ValueError('finite_effect_mismatch')
        if abs(float(intervention_effect(self.true_scm)) - self.population_effect) > 1e-12:
            raise ValueError('population_effect_mismatch')
        if self.certificate.public_card_hash != self.public_card_hash:
            raise ValueError('certificate_binding_mismatch')
        return self


def observational_joint(scm: BinarySCM) -> dict[tuple[int, int], F]:
    p = F(str(scm.p_root))
    if scm.kind == 'randomized':
        q0, q1 = F(str(scm.p_y0)), F(str(scm.p_y1))
        return {(0, 0): (1-p)*(1-q0), (0, 1): (1-p)*q0,
                (1, 0): p*(1-q1), (1, 1): p*q1}
    a0, a1 = F(str(scm.p_a_u0)), F(str(scm.p_a_u1))
    return {(0, 0): (1-p)*(1-a0), (1, 0): (1-p)*a0,
            (0, 1): p*(1-a1), (1, 1): p*a1}


def intervention_effect(scm: BinarySCM) -> F:
    return F(str(scm.p_y1)) - F(str(scm.p_y0)) if scm.kind == 'randomized' else F(0)


def _true_edges(scm):
    if scm.kind == 'latent_copy':
        return {('U', 'A'), ('U', 'Y')}
    return {('A', 'Y')} if intervention_effect(scm) else set()


def evidence_compatible(scm, card):
    if not 0 < scm.p_root < 1 or (scm.kind == 'latent_copy' and
            not (0 < scm.p_a_u0 < 1 and 0 < scm.p_a_u1 < 1)):
        return False
    if card.assignment == 'randomized_documented' and scm.kind != 'randomized':
        return False
    if card.structural_exclusion and intervention_effect(scm) != 0:
        return False
    edges = _true_edges(scm)
    for item in card.evidence:
        if item.kind != 'edge':
            continue
        edge = (item.source, item.target)
        if item.relation == 'supported' and edge not in edges:
            return False
        if item.relation == 'contradicted' and edge in edges:
            return False
    return True


def certify(truth: PrivateWorldTruthV2, card: PublicWorldCardV2, worlds) -> dict:
    result = {'status': 'uncertified', 'reason': 'no_implemented_certificate',
              'scope': 'public_packet_and_full_observed_distribution',
              'backdoor_failure': truth.backdoor_failure,
              'selection_warning': truth.selection_warning,
              'private_structural_zero_graph': truth.structural_zero_graph,
              'supported_structural_zero': False}
    if truth.public_card_hash != packet_hash(card):
        raise ValueError('truth_card_binding_mismatch')
    if {v.name for v in card.variables if v.observed} != {'A', 'Y'}:
        return {**result, 'reason': 'certificate_observable_scope_not_implemented'}
    if not evidence_compatible(truth.true_scm, card):
        raise ValueError('truth_incompatible_with_public_packet')
    proof = truth.certificate
    complete = card.selection == 'complete_outcomes' and not truth.selection_warning
    if proof.kind == 'randomized_complete':
        documented = any(e.kind == 'assignment' and e.relation == 'supported' for e in card.evidence)
        if (complete and documented and card.assignment == 'randomized_documented'
                and truth.true_scm.kind == 'randomized' and 0 < truth.true_scm.p_root < 1):
            return {**result, 'status': 'identified', 'reason': 'public_randomization_consistency_positivity_complete_Y',
                    'functional': 'sum_y y*(P(Y=y|A=1)-P(Y=y|A=0))'}
    elif proof.kind == 'public_exclusion':
        documented = any(e.kind == 'exclusion' and e.relation == 'supported' for e in card.evidence)
        if complete and documented and card.structural_exclusion and intervention_effect(truth.true_scm) == 0:
            return {**result, 'status': 'identified', 'reason': 'stipulated_public_structural_exclusion',
                    'functional': '0', 'supported_structural_zero': True}
    elif proof.kind == 'binary_pair':
        if len(set(proof.witness_world_ids)) != 2:
            return {**result, 'reason': 'two_distinct_witnesses_required'}
        indexed = {w.world_id: w for w in worlds}
        pair = [indexed.get(k) for k in proof.witness_world_ids]
        if any(w is None for w in pair) or truth.world_id not in proof.witness_world_ids:
            return {**result, 'reason': 'missing_witness_world'}
        left, right = pair
        compatible = (complete and card.assignment == 'assignment_unknown'
                      and all(w.public_card_hash == packet_hash(card) for w in pair)
                      and all(evidence_compatible(w.true_scm, card) and not w.selection_warning for w in pair))
        if not compatible:
            return {**result, 'reason': 'witness_not_compatible_with_supplied_information'}
        joints = [observational_joint(w.true_scm) for w in pair]
        effects = [intervention_effect(w.true_scm) for w in pair]
        if compatible and joints[0] == joints[1] and effects[0] != effects[1]:
            return {**result, 'status': 'provably_nonidentified',
                    'reason': 'compatible_SCMs_same_full_observed_law_distinct_intervention_effects',
                    'witness_world_ids': proof.witness_world_ids,
                    'observed_joint_exact': {f'{a},{y}': str(p) for (a, y), p in sorted(joints[0].items())},
                    'witness_effects_exact': [str(e) for e in effects]}
        return {**result, 'reason': 'witness_not_established'}
    return result
