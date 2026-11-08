"""Receipt: the trained stimulus-reset export, float64 reference, on the 512 rescue validation trials."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import infer,validate_model
from rescue import make_examples
path=ROOT/'results/rescue/order_random_reset_nopen_u600_stage3_final_hard.json'
m=json.loads(path.read_text());validate_model(m)
val=make_examples('order_random',512,77002)
cases=[]
for i,x in enumerate(val):
    r=infer(m,x,trace=i<24)
    c={'example':x,'counts':r['counts'],'prediction':r['prediction'],'exit_events':r['exit_events'],'clock_origin':r['clock_origin']}
    if i<24:c['events']=[[e['yard'],e['exit_tick'],e['release_tick'],e['phase']] for e in r['events']];c['deliveries']=r['deliveries']
    cases.append(c)
(ROOT/'results/rescue/parity_reset_trained.json').write_text(json.dumps(cases)+'\n')
print(f'Wrote {len(cases)} trained stimulus-reset receipts ({sum(len(c.get("events",[])) for c in cases)} traced events)')
