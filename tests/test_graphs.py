import pytest
from sharecausal.causal.graphs import SPECS,GraphSpec,validate_temporal_graph,edge_frequencies
from sharecausal.causal.identification import identify,backdoor_sets
from sharecausal.causal.robust_conclusions import assess,ConditionalInterval

def test_identification_varies_with_graph():
    results={x.name:identify(x) for x in SPECS}
    assert results['pre_depression']['identified_by_backdoor']
    assert ['B','D8'] in results['pre_depression']['adjustment_sets']
    assert results['depression_mediator']['adjustment_sets']==[['B']]
    assert not results['unmeasured_frailty']['identified_by_backdoor']
    assert not results['observed_survivors']['identified_by_backdoor']

def test_cycle_rejected():
    g=GraphSpec('bad',(('A','Y'),('Y','A')),frozenset({'A','Y'}),'test')
    with pytest.raises(ValueError,match='cycle'):g.networkx()

def test_temporal_violation():
    g=GraphSpec('bad',(('Y','A'),),frozenset({'A','Y'}),'test')
    with pytest.raises(ValueError,match='Backward'):validate_temporal_graph(g,{'A':8,'Y':9})

def test_identification_first_abstention():
    safe=ConditionalInterval('g0',-.7,-.95,-.4,True)
    bad=ConditionalInterval('g1',0,0,0,False)
    assert assess([safe,bad])['decision']=='abstain'
    assert assess([safe],shifts=[-.1,0,.1])['decision']=='negative'
    assert assess([safe],shifts=[-1.,0,1.])['decision']=='abstain'
    assert assess([safe],expected_graphs=['g0','g1'])['decision']=='abstain'
