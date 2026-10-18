"""Strict upper-envelope reservation; no unbounded fallback."""
from __future__ import annotations
from decimal import Decimal

class Budget:
    def __init__(self,ceiling_usd=25.0):
        if ceiling_usd<0:raise ValueError('Budget cannot be negative')
        self.limit=Decimal(str(ceiling_usd));self.spent=Decimal('0');self.reserved=Decimal('0')
    def reserve(self,estimated_usd):
        cost=Decimal(str(estimated_usd))
        if cost<0:raise ValueError('Negative cost')
        if self.spent+self.reserved+cost>self.limit:
            raise RuntimeError('OpenRouter cost ceiling would be exceeded; request blocked')
        self.reserved+=cost
        return float(cost)
    def reconcile(self,held,actual):
        cost=Decimal(str(actual))
        if cost<0:raise ValueError('Negative cost')
        self.reserved-=Decimal(str(held));self.spent+=cost
        if self.spent>self.limit:
            raise RuntimeError('Billed usage exceeded ceiling; stop all calls')
    def cancel(self,held):
        self.reserved-=Decimal(str(held))
