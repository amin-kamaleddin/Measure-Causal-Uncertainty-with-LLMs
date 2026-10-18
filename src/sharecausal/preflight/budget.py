"""Fake local pricing and simulated reservations, never actual spending."""
from dataclasses import dataclass
from decimal import Decimal

from pydantic import Field
from .contracts import Strict


class FakePrice(Strict):
    model_id: str = Field(pattern=r'^mock-family-[a-z]/fixture$')
    fixture_only: bool
    input_usd_per_million: float | None
    output_usd_per_million: float | None
    supports_json_schema: bool | None
    max_context_tokens: int = Field(gt=0)

    def validate_known(self):
        if (self.fixture_only is not True or self.input_usd_per_million is None
                or self.output_usd_per_million is None or self.supports_json_schema is not True
                or min(self.input_usd_per_million, self.output_usd_per_million) < 0):
            raise ValueError('unknown_fake_price_or_capability')

    def cost(self, input_tokens, output_tokens):
        self.validate_known()
        if min(input_tokens, output_tokens) < 0 or input_tokens+output_tokens > self.max_context_tokens:
            raise ValueError('token_reservation_invalid')
        return ((Decimal(str(self.input_usd_per_million))*input_tokens
                 + Decimal(str(self.output_usd_per_million))*output_tokens) / 1_000_000)


def cost_envelope(prices, cards, variants, repetitions, max_retries, input_tokens, output_tokens,
                  per_model_cap, global_cap):
    if min(cards, variants, repetitions, input_tokens, output_tokens) < 1 or max_retries < 0:
        raise ValueError('invalid_cost_design')
    if not prices or len({p.model_id for p in prices}) != len(prices):
        raise ValueError('unique_fake_models_required')
    per_model_calls = cards*variants*repetitions
    reservations = {p.model_id: p.cost(input_tokens, output_tokens)*per_model_calls*(1+max_retries)
                    for p in prices}
    total = sum(reservations.values(), Decimal(0))
    return {'independent_cards': cards, 'planned_calls': len(prices)*per_model_calls,
            'bounded_retry_calls': len(prices)*per_model_calls*max_retries,
            'maximum_attempts': len(prices)*per_model_calls*(1+max_retries),
            'reserved_input_tokens_per_attempt': input_tokens, 'reserved_output_tokens_per_attempt': output_tokens,
            'per_model_hypothetical_usd': {m: float(v) for m, v in reservations.items()},
            'hypothetical_total_usd': float(total),
            'within_hypothetical_caps': total <= Decimal(str(global_cap)) and
                                         all(v <= Decimal(str(per_model_cap)) for v in reservations.values()),
            'live_authorized_usd': 0, 'live_feasible_under_current_authorization': False,
            'pricing_status': 'local_fake_fixture_not_OpenRouter_prices'}


@dataclass
class Reservation:
    model: str
    amount: Decimal
    active: bool = True


class SimulatedBudget:
    def __init__(self, global_cap, per_model_cap):
        self.global_cap, self.per_model_cap = Decimal(str(global_cap)), Decimal(str(per_model_cap))
        if min(self.global_cap, self.per_model_cap) < 0:
            raise ValueError('invalid_budget_cap')
        self.spent = {}; self.reservations = []; self.stopped = False

    def reserve(self, model, amount):
        amount = Decimal(str(amount))
        if not amount.is_finite() or amount < 0 or self.stopped:
            raise ValueError('reservation_not_allowed')
        held = sum((r.amount for r in self.reservations if r.active), Decimal(0))
        model_held = sum((r.amount for r in self.reservations if r.active and r.model == model), Decimal(0))
        if (sum(self.spent.values(), Decimal(0))+held+amount > self.global_cap or
                self.spent.get(model, Decimal(0))+model_held+amount > self.per_model_cap):
            self.stopped = True
            raise ValueError('hypothetical_budget_ceiling')
        result = Reservation(model, amount); self.reservations.append(result); return result

    def reconcile(self, reservation, actual):
        if not any(r is reservation for r in self.reservations) or not reservation.active:
            raise ValueError('invalid_reservation_receipt')
        actual = Decimal(str(actual))
        if not actual.is_finite() or actual < 0:
            self.stopped = True; raise ValueError('invalid_simulated_billed_cost')
        reservation.active = False
        self.spent[reservation.model] = self.spent.get(reservation.model, Decimal(0)) + actual
        if actual > reservation.amount:
            self.stopped = True
            raise ValueError('simulated_cost_exceeded_reservation')
