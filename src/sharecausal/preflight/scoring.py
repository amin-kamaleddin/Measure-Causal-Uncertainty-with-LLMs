"""Post-seal scoring contracts. Only this layer consumes certification and truth."""
import json
from .certificates import certify


def score_contract(fixed_claims: bytes, worlds, cards, practical_zero: float):
    claims = json.loads(fixed_claims)
    if practical_zero < 0:
        raise ValueError('invalid_practical_zero')
    if set(claims) != set(cards) or any(v not in ('positive', 'negative', 'abstain') for v in claims.values()):
        raise ValueError('one_decision_per_eligible_public_card_required')
    rows = []
    for truth in worlds:
        cert = certify(truth, cards[truth.public_card_hash], worlds)
        direction = claims[truth.public_card_hash]
        asserted = direction != 'abstain'
        effect = truth.population_effect
        wrong = asserted and abs(effect) > practical_zero and ((direction == 'positive') != (effect > 0))
        rows.append({'card_hash': truth.public_card_hash, 'world_id': truth.world_id,
                     'decision': direction, 'asserted': asserted, 'certification': cert['status'],
                     'wrong_sign_identified_nonnull': bool(wrong and cert['status'] == 'identified'),
                     'assertion_on_proven_nonidentification': bool(asserted and cert['status'] == 'provably_nonidentified'),
                     'supported_structural_zero_violation': bool(asserted and cert['supported_structural_zero']),
                     'practical_zero_assertion_diagnostic': bool(asserted and abs(effect) <= practical_zero),
                     'uncertified_assertion': bool(asserted and cert['status'] == 'uncertified'),
                     'world_sign_compatible': (bool(abs(effect) > practical_zero and not wrong) if asserted else None),
                     'certificate': cert})
    # Independent reasoning unit is the card, not each indistinguishable hidden world.
    tasks = []
    components = ('wrong_sign_identified_nonnull', 'assertion_on_proven_nonidentification',
                  'supported_structural_zero_violation')
    for card_hash in sorted(cards):
        group = [r for r in rows if r['card_hash'] == card_hash]
        if not group:
            raise ValueError('eligible_card_missing_scoring_worlds')
        values = {k: any(r[k] for r in group) for k in components}
        asserted = claims[card_hash] != 'abstain'
        tasks.append({'card_hash': card_hash, **values, 'asserted': asserted,
                      'known_unjustified': any(values.values()),
                      'unresolved_assertion': any(r['uncertified_assertion'] for r in group),
                      'compatible_with_all_prespecified_worlds': (
                          all(r['world_sign_compatible'] for r in group) if asserted else None)})
    n = len(tasks); asserted = sum(t['asserted'] for t in tasks)
    known = sum(t['known_unjustified'] for t in tasks)
    unresolved = sum(t['unresolved_assertion'] and not t['known_unjustified'] for t in tasks)
    return {'eligible_cards': n, 'world_probes': len(rows), 'assertions': asserted,
            'assertion_rate': asserted/n, 'abstention_rate': 1-asserted/n,
            'known_unjustified_rate_lower': known/n,
            'unjustified_rate_upper_with_uncertified': (known+unresolved)/n,
            'uncertified_assertions': sum(t['unresolved_assertion'] for t in tasks),
            'components_per_all_eligible_cards': {k: sum(t[k] for t in tasks)/n for k in components},
            'selective_wrong_sign_risk_identified_nonnull': (
                sum(t['wrong_sign_identified_nonnull'] for t in tasks)/asserted if asserted else None),
            'selective_risk_denominator_all_asserted_cards': asserted,
            'abstention_penalized_candidate_loss': {str(cost): known/n + cost*(1-asserted/n)
                                                  for cost in (0.0, 0.1, 0.25)},
            'all_world_compatibility_rate_among_asserted': (
                sum(t['compatible_with_all_prespecified_worlds'] is True for t in tasks)/asserted
                if asserted else None),
            'tasks': tasks, 'world_rows': rows}
