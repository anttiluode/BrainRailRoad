"""Generate reference receipts for packet-by-packet Node comparison."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import make_model,infer,trial,save_model
from task import examples
m=make_model(11)
ex=[trial(o,c,78,8+(i%9)*2,na,nb) for i,(o,c,na,nb) in enumerate([
    (0,0,0,0),(0,1,0,0),(1,0,0,0),(1,1,0,0),
    (0,1,-1,2),(1,1,2,-1),(0,0,1,1),(1,0,0,-2),
    (1,0,-1,2),(0,1,2,0),(0,0,-2,1),(1,1,0,2)])]
# Include an active-edge damage condition without modifying the frozen model.
fixtures=[]
for i,x in enumerate(ex):
    kw={"trace":True}
    if i in (5,7):kw["broken_edges"]={next(e["id"] for e in m["edges"] if e["active"] and e["family"]=="h1_h2")}
    if i in (8,9): kw["basket"]={8:{96:1.,128:.5}}
    if i in (10,11):kw["chandelier"]={14:{160:2.}}
    record=infer(m,x,**kw)
    fixture={"example":x,"result":record,
             "options":{ "brokenEdges":list(kw.get("broken_edges",[])),
               "basket":kw.get("basket",{}),"chandelier":kw.get("chandelier",{})}}
    fixtures.append(fixture)
folder=ROOT/"results"
save_model(m,folder/"untrained_seed11.json")
(folder/"parity_cases.json").write_text(json.dumps(fixtures,indent=2)+"\n",encoding="utf8")
print(f"Wrote {len(fixtures)} exact reference cases, {sum(len(f['result']['events']) for f in fixtures)} internal firings")
