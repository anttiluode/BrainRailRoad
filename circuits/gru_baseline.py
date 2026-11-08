"""Independent, conventional GRU sanity baseline for contextual A/B order.

Receives exactly four time channels per tick (A, B, context0, context1).
Returns one final binary decision at a declared readout tick. No phase window or event-count
abstention, which makes this a deliberately strong non-spiking reference.
Not a drop-in identical-work comparator to the hard train network.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from model import ROOT
from task import examples, read_manifest

class GRUBaseline(nn.Module):
    def __init__(self, hidden=15):
        super().__init__()
        self.gru=nn.GRU(4,hidden,batch_first=True)
        self.out=nn.Linear(hidden,2)
    def forward(self, features):
        _,last=self.gru(features)
        return self.out(last[-1])

def tensorize(batch, readout_tick=319):
    B=len(batch);x=torch.zeros(B,readout_tick+1,4)
    y=torch.tensor([r['label'] for r in batch],dtype=torch.long)
    for i,r in enumerate(batch):
        if int(r['a_tick']) <= readout_tick:x[i,int(r['a_tick']),0]=1.
        if int(r['b_tick']) <= readout_tick:x[i,int(r['b_tick']),1]=1.
        x[i,:,2+int(r['context'])]=1.
    return x,y

@torch.no_grad()
def evaluate(model, samples, block=64, readout_tick=319):
    model.eval();correct=0;count=0;loss=0.;cells={f'{o}_{c}':[0,0] for o in (0,1) for c in (0,1)}
    for start in range(0,len(samples),block):
        part=samples[start:start+block];x,y=tensorize(part,readout_tick)
        out=model(x);pred=out.argmax(dim=1);hits=(pred==y)
        correct+=int(hits.sum());count+=len(part)
        loss+=float(F.cross_entropy(out,y,reduction='sum'))
        for row,hit in zip(part,hits.tolist()):
            k=f'{row["order"]}_{row["context"]}'
            cells[k][0]+=int(hit);cells[k][1]+=1
    model.train()
    return {'accuracy':correct/count,'ce':loss/count,'cell_accuracy':{k:a/b for k,(a,b) in cells.items()}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--hidden',type=int,default=15);ap.add_argument('--updates',type=int,default=700)
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--val-every',type=int,default=100)
    ap.add_argument('--checkpoints',action='store_true')
    ap.add_argument('--eval-frozen',action='store_true',help='Load best-validation saved checkpoint and evaluate the untouched frozen manifests')
    ap.add_argument('--readout-tick',type=int,default=319)
    args=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    if args.eval_frozen:
        path=ROOT/'checkpoints'/f'gru_h{args.hidden}_t{args.readout_tick}_seed{args.seed}.pt'
        frozen=torch.load(path,map_location='cpu',weights_only=True)
        m=GRUBaseline(args.hidden);m.load_state_dict(frozen['model'])
        report={'architecture':'GRU on A,B and tonic context, readout at tick '+str(args.readout_tick),
                'seed':args.seed,'hidden':args.hidden,'parameters':sum(p.numel() for p in m.parameters()),
                'checkpoint':path.name,'splits':{}}
        from evaluate import ensure_manifests
        ensure_manifests()
        for split in ('validation','interpolation','noisy','extrapolation'):
            records=read_manifest(split)
            report['splits'][split]=evaluate(m,records,readout_tick=args.readout_tick)
            print(split,report['splits'][split],flush=True)
        output=ROOT/'results'/f'gru_h{args.hidden}_t{args.readout_tick}_seed{args.seed}_frozen.json'
        output.write_text(json.dumps(report,indent=2)+'\n')
        print('Saved',output,flush=True)
        return report
    m=GRUBaseline(args.hidden);opt=torch.optim.Adam(m.parameters(),lr=.003)
    rng=random.Random(args.seed+10000)
    val=read_manifest('validation')[:512] if (ROOT/'results/manifest_validation.jsonl').exists() else examples('validation',512)
    param=sum(p.numel() for p in m.parameters())
    print('GRU parameters',param,'seed',args.seed,'hidden',args.hidden,flush=True)
    best=None;best_weights=None;history=[];start=time.monotonic()
    for i in range(1,args.updates+1):
        batch=examples('train',args.batch,seed=rng.randrange(1<<32));x,y=tensorize(batch,args.readout_tick)
        opt.zero_grad(set_to_none=True)
        logits=m(x);loss=F.cross_entropy(logits,y);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        if i==1 or i%args.val_every==0:
            metrics=evaluate(m,val,readout_tick=args.readout_tick);metrics.update(step=i,train_ce=float(loss.detach()))
            history.append(metrics)
            rank=(metrics['accuracy'],-metrics['ce'])
            if best is None or rank>best:
                best=rank;best_weights={n:p.detach().cpu().clone() for n,p in m.state_dict().items()}
            print(f"step={i} val_acc={metrics['accuracy']:.3f} val_CE={metrics['ce']:.3f} train_CE={metrics['train_ce']:.3f} elapsed={time.monotonic()-start:.1f}s",flush=True)
    m.load_state_dict(best_weights)
    result={'experiment':'gru-context-order','seed':args.seed,'hidden':args.hidden,'parameters':param,'readout_tick':args.readout_tick,'updates':args.updates,'validation_512':history,'best_validation':max(history,key=lambda a:(a['accuracy'],-a['ce'])),'elapsed_seconds':time.monotonic()-start}
    print('BEST (validation only)',result['best_validation'],flush=True)
    if args.checkpoints:
        ckpt=ROOT/'checkpoints'/f'gru_h{args.hidden}_t{args.readout_tick}_seed{args.seed}.pt';ckpt.parent.mkdir(exist_ok=True)
        torch.save({'model':best_weights,'seed':args.seed,'hidden':args.hidden},ckpt)
        (ROOT/'results'/f'gru_h{args.hidden}_t{args.readout_tick}_seed{args.seed}_dev.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':main()
