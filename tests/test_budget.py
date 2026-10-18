import pytest
from sharecausal.llm.budget import Budget

def test_ceiling_and_reconcile():
    b=Budget(.10)
    held=b.reserve(.08)
    with pytest.raises(RuntimeError):b.reserve(.04)
    b.reconcile(held,.05)
    assert str(b.spent)=='0.05'
    with pytest.raises(RuntimeError):b.reserve(.06)

def test_zero_dollar_denies_nonfree_calls():
    b=Budget(0)
    with pytest.raises(RuntimeError):b.reserve(.001)
    assert b.reserve(0)==0
