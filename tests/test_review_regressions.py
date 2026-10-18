"""Independent scientific/security regression tests for the Step 6B bridge.

Place this file in tests/test_review_regressions.py inside the research repo.
No SHARE data or OpenRouter key is used. The three tests FAIL against the
uploaded v2 code, as verified 2026-10-16, and should PASS after a correct fix.
"""
import json
import math
from pathlib import Path

import networkx as nx
from sharecausal.benchmark.bridge_v2 import analyze_observables
from sharecausal.benchmark.bridge_v2_smoke_data import smoke_tasks
from sharecausal.llm.synthetic_adapter import adapt_record
from sharecausal.llm.synthetic_schema import PUBLIC_CARD

FIXTURES = Path(__file__).parent / 'fixtures' / 'llm_bridge_v2'


def measured_fixture():
    return json.loads((FIXTURES / 'dummy_measured_x1.json').read_text())


def test_no_causal_path_must_not_assert_nonzero_total_effect():
    record = measured_fixture()
    record['response']['edges'] = [
        e for e in record['response']['edges']
        if (e['source'], e['target']) != ('A', 'Y')
    ]
    graph = adapt_record(record, PUBLIC_CARD).graph
    assert graph is not None
    assert not nx.has_path(graph.networkx(), 'A', 'Y')

    out = analyze_observables(smoke_tasks()[0].frame, [record], card=PUBLIC_CARD)
    # A supported DAG without a causal path implies zero total effect.
    # No sample-based positive or negative causal assertion can be valid under it.
    assert out['methods']['dummy_measured_x1']['decision'] == 'abstain'
    assert out['methods']['robust_screened']['decision'] == 'abstain'


def test_nonfinite_record_field_is_rejected_not_crash():
    record = measured_fixture()
    record['response']['edges'][0]['uncertainty'] = math.nan
    result = adapt_record(record, PUBLIC_CARD)
    assert result.graph is None
    assert result.audit['status'] == 'rejected'


def test_validation_location_does_not_expose_untrusted_key():
    record = measured_fixture()
    secret_marker = 'API_TOKEN_MARKER_DO_NOT_EXPOSE_42'
    record['response'][secret_marker] = 'invalid extra field'
    result = adapt_record(record, PUBLIC_CARD)
    assert result.graph is None
    assert result.audit['status'] == 'rejected'
    assert secret_marker not in json.dumps(result.audit)
