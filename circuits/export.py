"""Export the hard operational state of a PyTorch checkpoint, never proxies."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from model import ROOT,save_model


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_checkpoint(checkpoint_path:Path,target:Path):
    import torch
    from model_torch import SurrogateRail
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    args=saved['args']
    model=SurrogateRail(args['seed'],fixed_phase=args.get('fixed_phase',False),
             fixed_delay=args.get('fixed_delay',False),fixed_topology=args.get('fixed_topology',False),
             spiking=args.get('spiking',False),init_threshold=args.get('init_threshold',.35))
    model.load_state_dict(saved['model'])
    operational=model.hard_export()
    try:
        commit=subprocess.check_output(['git','-C',str(ROOT.parent),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:commit='unknown (source archive or non-git working tree)'
    operational['provenance']={'checkpoint_sha256':digest(checkpoint_path),
        'configuration_sha256':digest(ROOT/'config.json'),
        'design_sha256':digest(ROOT/'DESIGN.md'),
        'git_commit':commit,'updates':saved['step'],
        'train_args':args,
        'note':'Hard export retains exactly 36 active rails and integer delay indices. Surrogate tensors not exported.'}
    operational['training']={'seed':args['seed'],'updates':saved['step'], 'name':args['name'],
                             'batch_size':args['batch'],'origin':'supervised straight-through surrogate gradient'}
    target.parent.mkdir(exist_ok=True,parents=True)
    save_model(operational,target)
    return operational


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();m=export_checkpoint(args.checkpoint,args.out)
    print(json.dumps({'path':str(args.out),'schema':m['schema_version'],
        'live_rails':sum(e['active'] for e in m['edges']),
        'checkpoint_sha':m['provenance']['checkpoint_sha256']},indent=2))

if __name__=='__main__':main()
