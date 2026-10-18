"""Executable CLI. No implicit external calls or SHARE result publication."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from .reporting.provenance import environment


def main(argv=None):
    parser=argparse.ArgumentParser(prog='sharecausal')
    sub=parser.add_subparsers(dest='cmd',required=True)
    sub.add_parser('doctor')
    p=sub.add_parser('inspect');p.add_argument('--data-dir',required=True)
    p=sub.add_parser('analyze');p.add_argument('--data-dir',required=True)
    p.add_argument('--config');p.add_argument('--results-dir',default='outputs/private')
    p=sub.add_parser('benchmark');p.add_argument('--quick',action='store_true');p.add_argument('--results-dir',default='outputs/synthetic')
    p.add_argument('--seeds',type=int,default=4);p.add_argument('--n',type=int,default=700)
    p=sub.add_parser('benchmark-nonoracle')
    p.add_argument('--frozen',action='store_true',help='Execute internally locked protocol v1 development/holdout split')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--seeds',type=int,default=None)
    p.add_argument('--n',type=int,default=None)
    p.add_argument('--bias-radius',type=float,default=0.25)
    p.add_argument('--results-dir',default='outputs/synthetic/nonoracle')
    p=sub.add_parser('llm-compare')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run',action='store_true');group.add_argument('--live',action='store_true')
    p.add_argument('--max-cost-usd',type=float,default=25.0)
    p.add_argument('--models-config',default='configs/models.example.yaml')
    p.add_argument('--results-dir',default='outputs/private/llm')
    p=sub.add_parser('report');p.add_argument('--results-dir',required=True)
    args=parser.parse_args(argv)
    if args.cmd=='doctor':
        import networkx,pydantic,sklearn,pandas
        print(json.dumps({'status':'ready','env':environment(),'no_share_or_llm_requests':True},indent=2))
    elif args.cmd=='inspect':
        from .data.archives import locate_archives,archive_inventory
        from .data.metadata import core_metadata
        paths=locate_archives(args.data_dir)
        md=core_metadata(paths)
        print(json.dumps({'inventory':archive_inventory(paths),'metadata_verified':
                 {k:{'rows':v['rows'],'columns_verified':v['columns_verified']} for k,v in md.items()}},indent=2))
    elif args.cmd=='analyze':
        from .analysis import run
        result=run(args.data_dir,args.results_dir,args.config)
        print(json.dumps({'status':'completed_private','flow':result['flow'],
             'primary':result['estimates']['aipw_demographic_country'],
             'structural_decision':result['structural_robustness']['decision'],
             'result_dir':str(Path(args.results_dir).resolve())},indent=2))
    elif args.cmd=='benchmark':
        from .benchmark.runner import benchmark
        from .reporting.plots import benchmark_plots,graph_quality_plot
        if args.seeds<1:parser.error('--seeds must be positive')
        seeds=[11,29,47,67][:args.seeds] if args.quick else list(range(100,100+args.seeds))
        res=benchmark(quick=args.quick,seeds=seeds,n=args.n,results_dir=args.results_dir)
        benchmark_plots(res,args.results_dir)
        graph_quality_plot(res['fixed_graph_baseline_quality'],args.results_dir)
        print(json.dumps({'mode':'synthetic','summary':res['summary'],'failures':res['failed_runs'],
           'result_dir':str(Path(args.results_dir).resolve())},indent=2))
    elif args.cmd=='benchmark-nonoracle':
        from .benchmark.nonoracle import benchmark_nonoracle
        if args.seeds is not None and args.seeds<1:parser.error('--seeds must be positive')
        if args.frozen:
            if args.quick or args.seeds is not None or args.n is not None or args.bias_radius != 0.25:
                parser.error('--frozen cannot be combined with --quick, --seeds, --n or --bias-radius')
            from .benchmark.nonoracle import benchmark_locked_split
            from hashlib import sha256
            import yaml
            cfgpath=Path(__file__).resolve().parents[2] / 'configs' / 'approved_protocol_v1.yaml'
            source=cfgpath.read_bytes()
            cfg=yaml.safe_load(source)['synthetic_evaluation']
            result=benchmark_locked_split(
                development_seeds=tuple(cfg['development_seeds']),
                heldout_seeds=tuple(cfg['heldout_seeds']), n=cfg['sample_size'],
                target_rate=cfg['target_assertion_rate'], bias_radius=cfg['bias_shift_radius'],
                results_dir=args.results_dir,config_sha256=sha256(source).hexdigest())
            print(json.dumps({'mode':result['kind'],
                'heldout_comparison':result['heldout_target_coverage_comparison'],
                'failed_runs':result['heldout_failures'], 'real_llm_calls':0,
                'result_dir':str(Path(args.results_dir).resolve())},indent=2))
        else:
            seeds=tuple(range(1001,1001+(args.seeds if args.seeds is not None else (2 if args.quick else 12))))
            n=args.n if args.n is not None else (350 if args.quick else 1200)
            result=benchmark_nonoracle(seeds=seeds,n=n,results_dir=args.results_dir,
                                       bias_radius=args.bias_radius)
            print(json.dumps({'mode':result['kind'], 'summary':result['summary'],
                              'failed_runs':result['failures'], 'real_llm_calls':0,
                              'result_dir':str(Path(args.results_dir).resolve())},indent=2))
    elif args.cmd=='llm-compare':
        from .llm.compare import dry_run
        if args.dry_run:
            print(json.dumps(dry_run(),indent=2))
        else:
            if os.environ.get('SHARECAUSAL_LIVE_LLM_APPROVED') != '1':
                raise RuntimeError('Live OpenRouter calls are disabled until separate PI authorization; set SHARECAUSAL_LIVE_LLM_APPROVED=1 only after that approval')
            from .llm.openrouter import available_models,choose_distinct_models,complete
            from .llm.budget import Budget
            from .llm.cache import save
            from .llm.prompts import hash_messages
            import hashlib
            key=os.environ.get('OPENROUTER_API_KEY','')
            if not key:raise RuntimeError('Missing OPENROUTER_API_KEY in environment; no calls were made')
            budget=Budget(args.max_cost_usd)
            import yaml
            cfg_path=Path(args.models_config)
            cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
            models=choose_distinct_models(available_models(key),
                preferred=tuple(cfg.get('preferred_families',['openai','anthropic','google','meta-llama'])),
                max_models=int(cfg.get('max_models',4)),
                model_ids=cfg.get('model_ids'))
            dest=Path(args.results_dir).resolve()
            if '/private' not in str(dest).replace('\\','/'):
                raise ValueError('Live LLM provenance must be in a path containing private')
            records=[]
            for model in models:
                for variant in cfg.get('prompt_variants',['base','counterargument']):
                    for repeat in range(int(cfg.get('replications',2))):
                        entry=complete(model,variant,budget,key,seed=repeat)
                        cachekey=hashlib.sha256((model['id']+hash_messages(variant)+str(repeat)).encode()).hexdigest()
                        save(dest,cachekey,entry)
                        records.append(entry)
                        print(json.dumps({'model':model['id'],'variant':variant,'repeat':repeat,
                               'cost_spent_usd':float(budget.spent),'abstain':entry['graph']['abstain']}))
                        if budget.spent>=budget.limit:raise RuntimeError('Budget exhausted: stopped')
            from .llm.compare import summarize_graphs
            (dest/'comparison_summary.json').write_text(json.dumps(summarize_graphs(records),indent=2))
    elif args.cmd=='report':
        p=Path(args.results_dir)/'results.json'
        if not p.exists():raise FileNotFoundError('Missing private results.json; run analyze first')
        data=json.loads(p.read_text())
        from .reporting.manuscript import manuscript
        print(json.dumps({'manuscript':str(manuscript(data,args.results_dir)),
                'causal_decision':data['structural_robustness']['decision']}))
    return 0

if __name__=='__main__':
    sys.exit(main())
