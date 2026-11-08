"""Float64 hard-event railway reference. NO surrogate or silent analog readout.

Canonical IDs: 0=A, 1=B, 2..7=H1, 8..13=H2, 14..15=outputs.
All releases and deliveries are discrete logical packets. Stored phase/port
metadata is diagnostic only and is never fed to a downstream yard.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf8"))["model"]
P = int(CFG["period"])
TAU = tuple(CFG["address_delays"])
DELAYS = tuple(CFG["candidate_delays"])
YARDS = tuple(range(2, 16))
HIDDEN = tuple(range(2, 14))
OUTPUTS = (14, 15)
GROUPS = ("source_h1", "h1_h2", "h2_h2", "h2_output")
QUOTAS = {"source_h1": 1, "h1_h2": 3, "h2_h2": 1, "h2_output": 3}
ANGLE = (0.0, math.pi/2, math.pi, 3*math.pi/2)
VEC = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def candidate_edges() -> list[dict[str, Any]]:
    """Generate the same 90 directed candidate slots in every language.

    Round-robin receiver ports by canonical source order *within each
    incoming family*, not by visual position or by semantic class label.
    """
    edges: list[dict[str, Any]] = []
    specs = [
        ("source_h1", lambda d: (0,1), range(2,8)),
        ("h1_h2", lambda d: range(2,8), range(8,14)),
        ("h2_h2", lambda d: (s for s in range(8,14) if s!=d), range(8,14)),
        ("h2_output", lambda d: range(8,14), range(14,16)),
    ]
    for family, sources, targets in specs:
        for dst in targets:
            for port, src in enumerate(sources(dst)):
                edges.append({"id": len(edges), "family": family, "src": src,
                              "dst": dst, "port": port % 4})
    assert len(edges)==90
    return edges


def make_model(seed: int=11) -> dict[str, Any]:
    rng = random.Random(seed)
    edges = candidate_edges()
    for e in edges:
        e.update(weight=rng.uniform(0.4,1.2),delay=DELAYS[rng.randrange(len(DELAYS))],
                 score=rng.gauss(0.0,0.05),active=False,broken=False)
    for family in GROUPS:
        for dst in sorted({e["dst"] for e in edges if e["family"]==family}):
            es = [e for e in edges if e["family"]==family and e["dst"]==dst]
            if family=="source_h1":
                # Source balance is an initialization constraint, not target coding.
                chosen = next(e for e in es if e["src"] == ((dst-2)%2))
                chosen["score"] += 0.3
            else:
                picks = rng.sample(es, QUOTAS[family])
                for e in picks: e["score"]+=0.3
    apply_topology(edges)
    return {"schema_version":1,"model_seed":seed,"period":P,"ticks":int(CFG["ticks"]),
            "kappa":float(CFG["kappa"]),"leak":float(CFG["leak"]),
            "basket_leak":float(CFG["basket_leak"]),"chandelier_leak":float(CFG["chandelier_leak"]),
            "basket_strength":float(CFG["basket_strength"]),"veto_threshold":float(CFG["veto_threshold"]),
            "epsilon":float(CFG["epsilon"]),"context_scale":float(CFG["context_scale"]),
            "address_delays":list(TAU),"candidate_delays":list(DELAYS),
            "score_window":[128,320],
            "thresholds":[0.35 for _ in YARDS],
            "context":[[rng.gauss(0.0,0.05),rng.gauss(0.0,0.05)] for _ in HIDDEN],
            "edges":edges, "origin":"random-untrained", "training":None}


def apply_topology(edges: list[dict[str,Any]]) -> None:
    """Hard independent top-k masks per (destination, incoming family)."""
    for e in edges: e["active"] = False
    for family in GROUPS:
        for dst in sorted({e["dst"] for e in edges if e["family"] == family}):
            eligible=[e for e in edges if e["family"]==family and e["dst"]==dst and not e.get("broken",False)]
            if len(eligible)<QUOTAS[family]:
                raise ValueError(f"Cannot fill quota {family} to {dst}")
            for e in sorted(eligible,key=lambda r:(-r["score"],r["id"]))[:QUOTAS[family]]:
                e["active"]=True


def validate_model(model:dict[str,Any], *, enforce_quotas=True)->None:
    if int(model["schema_version"])!=1: raise ValueError("schema_version")
    es=model["edges"]
    if len(es)!=90: raise ValueError("Expected 90 candidates")
    canonical=candidate_edges()
    for i,(e,c) in enumerate(zip(es,canonical)):
        if any(e[k]!=c[k] for k in ("id","family","src","dst","port")): raise ValueError(f"Rail {i} is noncanonical")
        if e["delay"] not in DELAYS or not (-2.0<=e["weight"]<=2.0 if model.get("architecture",{}).get("spiking") else 0.05<=e["weight"]<=2.0): raise ValueError(f"invalid rail {i}")
        if e["active"] and e.get("broken",False): raise ValueError("Active broken rail")
    if enforce_quotas:
        if sum(e["active"] for e in es)!=36: raise ValueError("Expected exactly 36 active rails")
        for family in GROUPS:
            for dst in {e["dst"] for e in es if e["family"]==family}:
                if sum(e["active"] for e in es if e["family"]==family and e["dst"]==dst)!=QUOTAS[family]:
                    raise ValueError(f"quota {family}:{dst}")
    if len(model["thresholds"])!=14 or any(not 0.15<=v<=1.5 for v in model["thresholds"]): raise ValueError("thresholds")
    if len(model["context"])!=12 or any(len(r)!=2 for r in model["context"]): raise ValueError("context")


def trial(order:int,context:int,onset:int,gap:int,noise_a:int=0,noise_b:int=0)->dict[str,Any]:
    if order not in (0,1) or context not in (0,1): raise ValueError("order/context must be bits")
    a=onset+(0 if order==0 else gap)+noise_a
    b=onset+(gap if order==0 else 0)+noise_b
    return {"order":order,"context":context,"onset":onset,"gap":gap,
            "noise_a":noise_a,"noise_b":noise_b,"a_tick":a,"b_tick":b,"label":order^context}


def infer(model:dict[str,Any], example:dict[str,Any], *, trace=False, broken_edges=(),
          basket=None, chandelier=None, phase_override=None, ticks=None) -> dict[str,Any]:
    """Reference semantics §4, including output release rather than AIS time.

    Source events occur at their task timestamps, with no AIS phase or source
    gate. Output yards have no learned final transmission delay. Basket and
    chandelier inputs map yard integer IDs to {tick: positive magnitude}.
    """
    if model.get("architecture",{}).get("spiking"):
        return infer_spiking(model, example, trace=trace, broken_edges=broken_edges, ticks=ticks)
    if ticks is None: ticks=int(model["ticks"])
    edges=model["edges"]
    intact=[bool(e["active"]) and not e.get("broken",False) and e["id"] not in broken_edges for e in edges]
    by_source=[[] for _ in range(16)]
    for i,e in enumerate(edges):
        if intact[i]: by_source[e["src"]].append(i)
    arrivals:dict[int,list[tuple[int,float]]] = defaultdict(list)
    releases:dict[int,list[tuple[int,int,int]]] = defaultdict(list)
    events=[]; deliveries=[]; output_releases=[]; scheduled_late=0
    # Source bypass: no source AIS or source phase selection.
    for src,emitted in ((0,int(example["a_tick"])),(1,int(example["b_tick"]))):
        for eid in by_source[src]:
            e=edges[eid]; at=emitted+e["delay"]
            arrivals[at].append((eid,e["weight"]))
            if at>=ticks: scheduled_late+=1
            if trace: deliveries.append({"edge":eid,"source":src,"dest":e["dst"],"tick":at,"origin":"sensory"})
    L=[0.]*16; zx=[0.]*16;zy=[0.]*16;b=[0.]*16;c=[0.]*16
    # Static clock reference, no PLL in this main-task implementation.
    for t in range(ticks):
        arrival_by_edge=defaultdict(float)
        for eid,value in arrivals.pop(t,[]): arrival_by_edge[eid]+=value
        incoming=defaultdict(list)
        for eid,value in arrival_by_edge.items(): incoming[edges[eid]["dst"]].append((eid,value))
        theta=2.0*math.pi*(t%P)/P
        ct=math.cos(theta);st=math.sin(theta)
        exit_tick=(t>0 and t%P==0)
        for y in YARDS:
            b[y]=model["basket_leak"]*b[y]+float((basket or {}).get(y,{}).get(t,0))
            c[y]=model["chandelier_leak"]*c[y]+float((chandelier or {}).get(y,{}).get(t,0))
            basal=math.fsum(math.exp(model["kappa"]*(math.cos(theta-ANGLE[edges[eid]["port"]])-1))*v for eid,v in incoming.get(y,[]))
            shunt=math.exp(-model["basket_strength"]*b[y])
            lstar=(model["leak"]*L[y]+basal)*shunt
            xstar=(model["leak"]*zx[y]+basal*ct)*shunt
            ystar=(model["leak"]*zy[y]+basal*st)*shunt
            L[y],zx[y],zy[y]=lstar,xstar,ystar
            if not exit_tick: continue
            gain=math.exp(model["context_scale"]*math.tanh(model["context"][y-2][int(example["context"])])) if y in HIDDEN else 1.0
            radius=math.sqrt(xstar*xstar+ystar*ystar+model["epsilon"]**2)
            h=model["thresholds"][y-2]
            fire=(gain*radius>=h and lstar>model["epsilon"] and c[y]<model["veto_threshold"])
            if not fire: continue
            scores=(xstar/radius,ystar/radius,-xstar/radius,-ystar/radius)
            address=max(range(4),key=lambda j:(scores[j],-j))
            if model.get("architecture",{}).get("fixed_phase"):
                address=0
            if phase_override is not None:
                if isinstance(phase_override,int):address=phase_override%4
                else:address=int(phase_override.get(address,address))%4
            release=t+model["address_delays"][address]
            keep=max(0.0,1.0-h/(gain*radius))
            L[y]*=keep;zx[y]*=keep;zy[y]*=keep
            ev={"yard":y,"exit_tick":t,"release_tick":release,"phase":address,
                "vector":[xstar,ystar],"radius":radius,"load":lstar,"gain":gain}
            events.append(ev)
            releases[release].append((y,address,t))
            for eid in by_source[y]:
                e=edges[eid];arr=release+e["delay"]
                arrivals[arr].append((eid,e["weight"]))
                if arr>=ticks:scheduled_late+=1
                if trace:deliveries.append({"edge":eid,"source":y,"dest":e["dst"],"tick":arr,"origin":"yard"})
        for y,k,et in releases.pop(t,[]):
            if y in OUTPUTS:
                output_releases.append({"yard":y,"tick":t,"phase":k,"exit_tick":et})
    start,stop=model["score_window"]
    counts=[sum(start<=r["tick"]<stop and r["yard"]==y for r in output_releases) for y in OUTPUTS]
    decision=None if counts[0]==counts[1] else int(counts[1]>counts[0])
    result={"counts":counts,"prediction":decision,"label":int(example["label"]),
            "correct":decision==example["label"],"abstained":decision is None,
            "active_rails":sum(int(v) for v in intact),"exit_events":len(events),
            "internal_deliveries":len(deliveries) if trace else None,
            "beyond_horizon_deliveries":scheduled_late,"release_events":output_releases}
    if trace:
        result["events"]=events;result["deliveries"]=deliveries
    return result


def infer_spiking(model:dict[str,Any],example:dict[str,Any],*,trace=False,broken_edges=(),ticks=None)->dict[str,Any]:
    """Strong signed-weight comparator from DESIGN.md Section 7.

    No phase capture, no vector, no exit clock. Subtractive reset at any tick
    and two-tick absolute refractory, with released packets delayed by rail.
    """
    if ticks is None:ticks=int(model["ticks"])
    edges=model["edges"]
    alive=[bool(e["active"]) and not e.get("broken",False) and e["id"] not in broken_edges for e in edges]
    by_source=[[] for _ in range(16)]
    for i,e in enumerate(edges):
        if alive[i]:by_source[e["src"]].append(i)
    arrivals=defaultdict(list);events=[];deliveries=[];output_releases=[];scheduled_late=0
    for src,emitted in ((0,int(example["a_tick"])),(1,int(example["b_tick"]))):
        for eid in by_source[src]:
            e=edges[eid];at=emitted+e["delay"]
            arrivals[at].append((eid,e["weight"]))
            if at>=ticks:scheduled_late+=1
            if trace:deliveries.append({"edge":eid,"source":src,"dest":e["dst"],"tick":at,"origin":"sensory"})
    value=[0.]*16;lastfire=[-10000]*16
    for t in range(ticks):
        incoming=defaultdict(list)
        for eid,x in arrivals.pop(t,[]):incoming[edges[eid]["dst"]].append(x)
        for y in YARDS:
            candidate=model["leak"]*value[y]+math.fsum(incoming.get(y,[]))
            gain=math.exp(model["context_scale"]*math.tanh(model["context"][y-2][int(example["context"])])) if y in HIDDEN else 1.0
            h=model["thresholds"][y-2]
            fire=gain*candidate>=h and t-lastfire[y]>=2
            if not fire:
                value[y]=candidate;continue
            value[y]=candidate-h/gain
            lastfire[y]=t
            events.append({"yard":y,"exit_tick":t,"release_tick":t,"phase":0,"vector":[0,0],"radius":candidate,"load":candidate,"gain":gain})
            if y in OUTPUTS:output_releases.append({"yard":y,"tick":t,"phase":0,"exit_tick":t})
            for eid in by_source[y]:
                e=edges[eid];at=t+e["delay"]
                arrivals[at].append((eid,e["weight"]))
                if at>=ticks:scheduled_late+=1
                if trace:deliveries.append({"edge":eid,"source":y,"dest":e["dst"],"tick":at,"origin":"yard"})
    start,stop=model["score_window"]
    counts=[sum(start<=r["tick"]<stop and r["yard"]==y for r in output_releases) for y in OUTPUTS]
    decision=None if counts[0]==counts[1] else int(counts[1]>counts[0])
    result={"counts":counts,"prediction":decision,"label":int(example["label"]),
            "correct":decision==example["label"],"abstained":decision is None,
            "active_rails":sum(int(v) for v in alive),"exit_events":len(events),
            "internal_deliveries":len(deliveries) if trace else None,
            "beyond_horizon_deliveries":scheduled_late,"release_events":output_releases}
    if trace:result.update(events=events,deliveries=deliveries)
    return result


def save_model(model,path):
    validate_model(model)
    Path(path).write_text(json.dumps(model,indent=2,sort_keys=True)+"\n",encoding="utf8")


def load_model(path):
    model=json.loads(Path(path).read_text(encoding="utf8"));validate_model(model);return model


if __name__ == "__main__":
    m=make_model();validate_model(m)
    v=trial(0,1,80,10)
    print(json.dumps({"model":"ok","rail_count":sum(e["active"] for e in m["edges"]),"example":v,"result":infer(m,v,trace=True)["counts"]},indent=2))
