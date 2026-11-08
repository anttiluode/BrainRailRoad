"""Save compact cross-language parity for fixed-phase & signed spiking baseline."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_torch import SurrogateRail
from model import infer,trial,save_model
cases=[]
for kind in ('fixed_phase','spiking'):
    train=SurrogateRail(11,**{kind:True})
    model=train.hard_export()
    save_model(model,ROOT/'results'/f'{kind}_reference.json')
    for order in (0,1):
        for context in (0,1):
            x=trial(order,context,95,8)
            y=infer(model,x,trace=True)
            cases.append({'model':kind,'example':x,'counts':y['counts'],
                          'prediction':y['prediction'],'exit_events':y['exit_events'],
                          'deliveries':y['internal_deliveries'],
                          'first_releases':y['release_events'][:8]})
(ROOT/'results'/'ablation_parity_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
print(f'Generated {len(cases)} baseline parity checks')
