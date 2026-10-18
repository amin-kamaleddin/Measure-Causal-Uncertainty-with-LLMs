import pytest
from sharecausal.llm.schemas import Arrow,GraphAnswer
from sharecausal.llm.compare import dry_run,mock_graphs
from sharecausal.llm.openrouter import upper_cost_estimate,choose_distinct_models

def test_offline_mock_has_no_live_results():
    dry=dry_run();assert dry['live_requests']==0
    assert dry['actual_llm_results']==0
    assert dry['mock_valid_graphs']==2
    assert len(mock_graphs())==2

def test_bad_graph_rejected():
    def arrow(a,b):return {'source':a,'target':b,'causal_role':'','temporal_rationale':'',
            'evidence_reference':'','uncertainty':''}
    with pytest.raises(ValueError):GraphAnswer(edges=[arrow('A','Y'),arrow('Y','A')],abstain=False,
          abstention_reason='',contested_assumptions=[])

def test_model_selection_distinct_families():
    catalog=[{'id':f'{family}/modelx', 'supported_parameters':['response_format'],
              'architecture':{'output_modalities':['text']},
              'pricing':{'prompt':'0.0000002','completion':'0.000001'}} for family in
              ['openai','anthropic','google','meta-llama']]
    assert len(choose_distinct_models(catalog))==4
    assert upper_cost_estimate(catalog[0],400,100)<.01


def test_graph_comparison_counts_are_not_probabilities():
    from sharecausal.llm.compare import summarize_graphs,mock_graphs
    g=mock_graphs()
    records=[{'model_requested':'mock/a','graph':g[0].model_dump()},
             {'model_requested':'mock/b','graph':g[1].model_dump()}]
    out=summarize_graphs(records)
    assert out['n_models']==2
    assert out['n_graphs']==2
    assert out['n_abstained']==1
    assert out['edge_frequencies_not_probabilities']['B->A']==1.0
    assert out['mean_pairwise_jaccard']<1.0
    assert not out['naive_majority_arrows_without_identification_validation'] or all(
        isinstance(x,list) for x in out['naive_majority_arrows_without_identification_validation'])


def test_catalog_selection_validates_family_and_json():
    import pytest
    from sharecausal.llm.openrouter import choose_distinct_models
    m=[{'id':f'{family}/test','supported_parameters':['response_format'],
        'architecture':{'output_modalities':['text']},'pricing':{'prompt':'0.000001','completion':'0.000002'}}
        for family in ('openai','anthropic','google')]
    assert len(choose_distinct_models(m,max_models=3,model_ids=[x['id'] for x in m]))==3
    with pytest.raises(ValueError):
        choose_distinct_models(m,model_ids=['openai/test','openai/other','google/test'])
    with pytest.raises(ValueError):
        choose_distinct_models(m,model_ids=['openai/test','anthropic/test','nonexistent/model'])


def test_reject_wave9_to_wave8_time_travel():
    import pytest
    from pydantic import ValidationError
    from sharecausal.llm.schemas import GraphAnswer
    p={'edges':[{'source':'Y','target':'D8','causal_role':'unverified',
                 'temporal_rationale':'no','evidence_reference':'hypothesis','uncertainty':'high'}],
       'abstain':True,'abstention_reason':'backwards','contested_assumptions':[]}
    with pytest.raises(ValidationError):GraphAnswer.model_validate(p)
