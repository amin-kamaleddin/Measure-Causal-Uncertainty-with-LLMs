#!/usr/bin/env python3
"""Offline replay of task-bound synthetic provider responses; no sending interface."""
import argparse
import json
from pathlib import Path
from sharecausal.empirical_v3.io import install_offline_guard, write_json


def main():
    install_offline_guard()
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='role',required=True)
    a=sub.add_parser('predict');a.add_argument('--tasks',type=Path,required=True)
    a.add_argument('--records',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('score');a.add_argument('--predictions',type=Path,required=True)
    a.add_argument('--truth',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.role=='predict':
            from sharecausal.provider_replay_v1.prediction import seal
            from sharecausal.preflight_v23.packets import PublicTask
            result=seal([PublicTask.model_validate(t) for t in json.loads(args.tasks.read_text())],
                         json.loads(args.records.read_text()),args.out)
        else:
            from sharecausal.provider_replay_v1.scoring import score
            if args.out.exists():raise ValueError('Refuse overwrite')
            result=score(args.predictions,args.truth);write_json(args.out,result)
    except Exception:
        print(json.dumps({'status':'FAILED','role':args.role,'action':'Check schema, task/prompt binding and prediction seal; raw errors suppressed.'}))
        return 2
    print(json.dumps({'status':'SUCCESS','role':args.role,'real_llm_calls':0,'cost_usd':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
