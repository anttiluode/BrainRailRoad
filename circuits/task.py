"""Balanced contextual-order task, separately frozen data and model RNGs."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from model import ROOT, trial

CONFIG=json.loads((ROOT/"config.json").read_text(encoding="utf8"))["task"]
SPLITS={
    "validation":(CONFIG["train_gaps"],int(CONFIG["validation_size"])),
    "interpolation":(CONFIG["test_gaps"],int(CONFIG["test_size"])),
    "noisy":(CONFIG["test_gaps"],int(CONFIG["test_size"])),
    "extrapolation":(CONFIG["extrapolation_gaps"],int(CONFIG["test_size"]))
}

def examples(split:str="train",n:int=32,seed:int|None=None)->list[dict]:
    if split not in ("train",*SPLITS):raise ValueError(split)
    if n%4:raise ValueError("n must be divisible by 4 for exact order/context balance")
    rng=random.Random(int(CONFIG["seeds"].get("training" if split=="train" else split) if seed is None else seed))
    gaps=CONFIG["train_gaps"] if split=="train" else SPLITS[split][0]
    cells=[(o,c) for o in (0,1) for c in (0,1)]*(n//4)
    rng.shuffle(cells)
    result=[]
    for order,context in cells:
        onset=rng.randint(*CONFIG["onsets"]);gap=rng.choice(gaps)
        na=rng.randint(CONFIG["jitter_min"],CONFIG["jitter_max"]) if split=="noisy" else 0
        nb=rng.randint(CONFIG["jitter_min"],CONFIG["jitter_max"]) if split=="noisy" else 0
        result.append(trial(order,context,onset,gap,na,nb))
    return result


def freeze_manifests(folder:Path=ROOT/"results"):
    folder.mkdir(exist_ok=True,parents=True)
    hashes={}
    for split,(_,n) in SPLITS.items():
        samples=examples(split,n)
        contents="".join(json.dumps(s,sort_keys=True,separators=(",",":"))+"\n" for s in samples)
        path=folder/("manifest_"+split+".jsonl")
        path.write_text(contents,encoding="utf8")
        hashes[split]={"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(contents.encode()).hexdigest(),"count":n,
                      "gaps":SPLITS[split][0],"seed":CONFIG["seeds"][split]}
    info={"splits":hashes,"train":{"seed":CONFIG["seeds"]["training"],"gaps":CONFIG["train_gaps"],"generated_online":True},
          "warning":"The clean support is finite; repeats are not independent test conditions. Evaluate per unique (onset,gap,order,context) and across model seeds."}
    (folder/"manifest_index.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    return info


def read_manifest(split,root=ROOT):
    with (root/"results"/("manifest_"+split+".jsonl")).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--freeze",action="store_true");args=ap.parse_args()
    if args.freeze:
        print(json.dumps(freeze_manifests(),indent=2))
    else:
        print(examples("train",4))
