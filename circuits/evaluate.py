"""Frozen hard-model evaluator: no gradients, no checkpoint shopping by test.

Reports abstentions as *errors*, all four cells, unique event conditions,
mean output event/packet cost and manifest checksums. Run only on a frozen
export; when training fails, a random export receives the same honest audit.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from model import ROOT,infer,load_model
from task import freeze_manifests,read_manifest,CONFIG,SPLITS


def ensure_manifests():
    index=ROOT/'results'/'manifest_index.json'
    if not index.exists() or any(not (ROOT/'results'/f'manifest_{split}.jsonl').exists() for split in SPLITS):
        freeze_manifests()
    saved=json.loads(index.read_text(encoding='utf8'))
    for split,record in saved['splits'].items():
        path=ROOT/record['path'];raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=record['sha256']:raise ValueError(f'Frozen {split} manifest mutated {digest} != {record["sha256"]}')
        if len(raw.splitlines())!=record['count']:raise ValueError(f'Frozen {split} count mismatch')
    return saved


def hard_statistics(model,records,*,report_unique=True):
    ts=time.monotonic()
    tally=[]
    cells=collections.defaultdict(list)
    unique=collections.defaultdict(list)
    counts=collections.Counter()
    for r in records:
        result=infer(model,r,trace=False)
        tally.append(int(result['correct']))
        cells[f"{r['order']}_{r['context']}"].append(int(result['correct']))
        unique[(r['onset'],r['gap'],r['order'],r['context'],r['noise_a'],r['noise_b'])].append(int(result['correct']))
        counts['abstentions']+=result['abstained']
        counts['hard_events']+=result['exit_events']
        counts['output_release_events']+=len(result['release_events'])
        counts['beyond_horizon_deliveries']+=result['beyond_horizon_deliveries']
        for key,value in enumerate(result['counts']):counts[f'output_{key}_release_total']+=value
    sizes={k:len(v) for k,v in cells.items()}
    accuracy={k:sum(v)/len(v) for k,v in cells.items()}
    out={'n':len(records),'accuracy':sum(tally)/len(tally),'abstention_fraction':counts['abstentions']/len(tally),
         'cell_accuracy':accuracy,'cell_n':sizes,'worst_cell_accuracy':min(accuracy.values()),
         'unique_exact_conditions':len(unique),
         'unique_condition_average_accuracy':statistics.fmean(statistics.fmean(v) for v in unique.values()),
         'mean_hard_yard_events':counts['hard_events']/len(tally),
         'mean_output_releases':counts['output_release_events']/len(tally),
         'mean_out_of_horizon_data_deliveries':counts['beyond_horizon_deliveries']/len(tally),
         'total_output_release_counts':[counts[f'output_{k}_release_total'] for k in range(2)],
         'wall_seconds':time.monotonic()-ts}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='results/untrained_seed11.json')
    p.add_argument('--split',choices=['all',*SPLITS],default='all')
    p.add_argument('--limit',type=int,default=0,help='DEVELOPMENT FAST CHECK only; published comparisons use full frozen 2048')
    p.add_argument('--out',default=None)
    args=p.parse_args()
    model=load_model(ROOT/args.model)
    indexed=ensure_manifests()
    results={'model':args.model,'model_seed':model['model_seed'],'origin':model.get('origin'),
             'architecture':model.get('architecture',{}),'manifest_sha256':{s:r['sha256'] for s,r in indexed['splits'].items()},
             'evaluation_limit':args.limit,'note':'An untrained export is not evidence that learning succeeded.', 'splits':{}}
    for split in SPLITS if args.split=='all' else [args.split]:
        data=read_manifest(split)
        if args.limit: data=data[:args.limit]
        t=hard_statistics(model,data)
        results['splits'][split]=t
        print(f'{split:>14}: accuracy={t["accuracy"]:.3%}, worst cell={t["worst_cell_accuracy"]:.3%}, abstentions={t["abstention_fraction"]:.3%}, n={t["n"]} ({t["wall_seconds"]:.1f}s)',flush=True)
    if args.out:
        path=ROOT/args.out;path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8');print(path)
    return results

if __name__=='__main__':main()
