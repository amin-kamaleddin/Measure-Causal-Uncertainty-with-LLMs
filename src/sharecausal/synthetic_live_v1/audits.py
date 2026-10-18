"""Pre-estimation public graph and edge audit, including invalid raw answers."""
import csv
import json
from sharecausal.preflight_v23.admission import audit,support
from sharecausal.pilot_v24.graphs import identify_graph


def graph_audits(tasks,requests,out):
    graphs=[]; edges=[]
    for item in requests:
        model=item['model']; h=item['task_sha256']; task=tasks[h]; raws=[]
        if item.get('record_file'):
            record=json.loads((out/item['record_file']).read_text())
            if record['proposals_wire']: raws=json.loads(record['proposals_wire'])
            elif record.get('raw_file'):
                body=json.loads((out/record['raw_file']).read_text())
                try:
                    raw=json.loads(body['choices'][0]['message']['content'])
                    if isinstance(raw,dict) and isinstance(raw.get('proposals'),list): raws=raw['proposals'][:10]
                except (TypeError,ValueError,KeyError,IndexError):pass
        if not raws:
            graphs.append(dict(model=model,task_sha256=h,hypothesis=None,format_valid=False,
                admitted=False,reason=item['status'],explicit_epistemic_abstention=False))
        for i,raw in enumerate(raws):
            admission,p=audit(task,raw)
            ident=identify_graph(task,[(e.source,e.target) for e in p.graph.edges]) if admission['admitted'] else {}
            graphs.append(dict(model=model,task_sha256=h,hypothesis=i,**admission,
                identification=ident.get('status'),adjustment=ident.get('adjustment'),
                explicit_zero_claim=p.structural_zero_claim if p else None,
                explicit_epistemic_abstention=p.graph.abstain if p else False))
            if p:
                for edge in p.graph.edges:
                    edges.append(dict(model=model,task_sha256=h,hypothesis=i,
                        source=edge.source,target=edge.target,evidence_id=edge.evidence_id,
                        support_tier=support(task,edge),graph_admitted=admission['admitted']))
            elif isinstance(raw,dict) and isinstance(raw.get('graph'),dict):
                # Report only allowlisted node labels for invalid graphs, never raw field names/text.
                known={v.name for v in task.card.variables}
                for edge in raw['graph'].get('edges',[])[:40] if isinstance(raw['graph'].get('edges'),list) else []:
                    if isinstance(edge,dict) and edge.get('source') in known and edge.get('target') in known:
                        edges.append(dict(model=model,task_sha256=h,hypothesis=i,source=edge['source'],
                            target=edge['target'],support_tier='invalid_graph_claim_not_admitted',graph_admitted=False))
    return dict(graphs=graphs,edges=edges)


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
