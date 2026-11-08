"""Supervised BPTT trainer for hard-event railway; explicit CPU work.

Full run defaults to the 1,500 steps / 32 batch / 5 seeds in DESIGN.md.
Use --smoke to validate gradients and checkpoint/resume without assuming that
these tiny smoke runs learn the contextual order task.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import ROOT,save_model
from model_torch import SurrogateRail
from task import examples,read_manifest

CFG=json.loads((ROOT/'config.json').read_text(encoding='utf8'))['train']

def build_batch(rng,bs):
    x=examples('train',bs,seed=rng.randrange(1<<32))
    return (torch.tensor([z['a_tick'] for z in x],dtype=torch.long),
            torch.tensor([z['b_tick'] for z in x],dtype=torch.long),
            torch.tensor([z['context'] for z in x],dtype=torch.long),
            torch.tensor([z['label'] for z in x],dtype=torch.long),x)

@torch.no_grad()
def evaluate_training_forward(m,items,limit=32):
    # Use the hard PyTorch forward, no scripted correction or dense classifier.
    m.eval();correct=0;ties=0;ce=0.0
    for pos in range(0,min(len(items),limit),4):
        data=items[pos:pos+4]
        A=torch.tensor([z['a_tick'] for z in data],dtype=torch.long)
        B=torch.tensor([z['b_tick'] for z in data],dtype=torch.long)
        c=torch.tensor([z['context'] for z in data],dtype=torch.long)
        y=torch.tensor([z['label'] for z in data],dtype=torch.long)
        C=m(A,B,c)
        ties+=int((C[:,0]==C[:,1]).sum())
        correct+=int(((C.argmax(-1)==y)&(C[:,0]!=C[:,1])).sum())
        ce+=float(F.cross_entropy(C,y,reduction='sum'))
    m.train();return {'accuracy':correct/min(limit,len(items)),'ties':ties,'cross_entropy':ce/min(limit,len(items))}

def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed+10000)
    device='cpu'
    m=SurrogateRail(args.seed,fixed_phase=args.fixed_phase,fixed_delay=args.fixed_delay,
                    fixed_topology=args.fixed_topology,spiking=args.spiking,init_threshold=args.init_threshold).to(device)
    optim=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=float(CFG['learning_rate']))
    outdir=ROOT/'checkpoints';outdir.mkdir(exist_ok=True)
    ckpt=outdir/f'{args.name}_seed{args.seed}.pt'
    step=0;history=[]
    if args.resume:
        saved=torch.load(ckpt,map_location=device,weights_only=False)
        m.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer'])
        rng.setstate(saved['data_rng']);torch.set_rng_state(saved['torch_rng'])
        step=saved['step'];history=saved['history']
        print(f'RESUMED {ckpt} at update {step}',flush=True)
    prev=m.hard_mask().clone();changes=0
    start=time.monotonic()
    for update in range(step,args.updates):
        A,B,c,y,records=build_batch(rng,args.batch)
        optim.zero_grad(set_to_none=True)
        counts,stats=m(A,B,c,return_metrics=True,temperature_fire=float(CFG['temperature_fire']),
                       temperature_phase=float(CFG['temperature_phase']),temperature_delay=float(CFG['temperature_delay']))
        cross=F.cross_entropy(counts,y)
        loss=cross+float(CFG['penalty_hidden_events'])*stats['hidden_events']/args.batch + \
             float(CFG['penalty_deliveries'])*stats['deliveries']/args.batch + \
             float(CFG['penalty_context'])*m.context.square().sum()
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss step {update}')
        loss.backward()
        grad_values={n:float(p.grad.abs().sum()) if p.grad is not None else 0 for n,p in m.named_parameters()}
        if not all(math.isfinite(v) for v in grad_values.values()):raise ValueError('Nonfinite gradients '+str(grad_values))
        torch.nn.utils.clip_grad_norm_(m.parameters(),float(CFG['gradient_clip']))
        optim.step()
        current=m.hard_mask();changes+=int((prev!=current).sum().item());prev=current.clone()
        hits=int(((counts.argmax(-1)==y)&(counts[:,0]!=counts[:,1])).sum())
        entry={'update':update+1,'loss':float(loss.detach()),'CE':float(cross.detach()),
               'train_accuracy':hits/args.batch,'abstentions':int((counts[:,0]==counts[:,1]).sum()),
               'hard_event_count':float(stats['hidden_events'].detach()),
               'hard_delivery_count':float(stats['deliveries'].detach()),
               'mask_hamming_cumulative':changes,'grad_l1':grad_values}
        history.append(entry)
        if (update+1)%args.log_every==0 or update==step:
            print(f'update {update+1:>4} CE={entry["CE"]:.4f} acc={hits/args.batch:.3f} abst={entry["abstentions"]}/{args.batch} event={entry["hard_event_count"]:.0f} |grad|='+str({k:round(v,3) for k,v in grad_values.items()}),flush=True)
    elapsed=time.monotonic()-start
    torch.save({'model':m.state_dict(),'optimizer':optim.state_dict(), 'data_rng':rng.getstate(),
                'torch_rng':torch.get_rng_state(),'step':args.updates,'history':history,
                'args':vars(args),'seconds_this_call':elapsed},ckpt)
    export=m.hard_export()
    export['training']={'updates':args.updates,'seed':args.seed,'experiment':args.name,'seconds_this_call':elapsed,
                        'batch_size':args.batch,'checkpoint':str(ckpt.relative_to(ROOT)),
                        'note':'Surrogate-trained; do NOT treat as validated without held-out evidence'}
    from model import validate_model
    validate_model(export)
    path=ROOT/'results'/f'{args.name}_seed{args.seed}_hard.json'
    save_model(export,path)
    receipt=ROOT/'results'/f'{args.name}_seed{args.seed}_training.json'
    receipt.write_text(json.dumps({'history':history,'elapsed_seconds':elapsed,'export':str(path.relative_to(ROOT)),
                                   'optimizer':'Adam','seed':args.seed,
                                   'limit':'Smoke-run checkpoint is plumbing validation, not success on the design targets.'},indent=2)+'\n')
    print(f'WROTE {ckpt}, {path}, {receipt}; elapsed {elapsed:.2f}s')
    return m,history

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--updates',type=int,default=1)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--log-every',type=int,default=1)
    ap.add_argument('--name',default='smoke')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--fixed-phase',action='store_true')
    ap.add_argument('--fixed-delay',action='store_true')
    ap.add_argument('--fixed-topology',action='store_true')
    ap.add_argument('--spiking',action='store_true')
    ap.add_argument('--init-threshold',type=float,default=0.35)
    args=ap.parse_args();train(args)
