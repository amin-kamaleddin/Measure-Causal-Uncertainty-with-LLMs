from sharecausal.benchmark.scm import generate,SCENARIOS
from sharecausal.benchmark.runner import benchmark
from sharecausal.benchmark.ablations import graph_accuracy,majority_graph
from sharecausal.causal.graphs import GraphSpec

def test_coupled_ground_truth():
    for mechanism in SCENARIOS:
        s=generate(mechanism,n=1000,seed=99)
        assert -2<s.true_ate<2
        assert s.frame.shape[0]==1000
        if mechanism=='selection':assert s.frame.observed.mean()<1
        if mechanism=='hidden':assert ('U','A') in s.true_edges

def test_runner_mock_not_llm():
    x=benchmark(quick=True,seeds=[19],n=600)
    assert len(x['rows'])==6*4
    assert not x['failed_runs']
    assert x['summary']['oracle_graph_set_robust']['assertion_rate']<=1
    assert 'NOT performance of LLM' in x['warning']

def test_graph_majority_not_calibrated():
    g=GraphSpec('a',(('X','A'),),frozenset({'X','A'}),'')
    m=majority_graph([g,g,g]);assert m.edges==g.edges
    assert graph_accuracy(m.edges,g.edges)['directed_shd']==0


def test_benchmark_writes_aggregate_and_run_level_csv(tmp_path):
    from sharecausal.benchmark.runner import benchmark
    import pandas as pd
    d=tmp_path/'synthetic'
    out=benchmark(quick=True,seeds=[11],n=100,results_dir=d)
    assert (d/'benchmark.json').exists()
    assert len(pd.read_csv(d/'benchmark_run_level.csv'))==len(out['rows'])
    assert (d/'benchmark_summary.csv').exists()
    assert (d/'fixed_graph_quality.csv').exists()
