"""Causal/unit tests against the exact reference; no PyTorch needed."""
import sys
import unittest
import copy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import (make_model,validate_model,infer,trial,candidate_edges,
                   phase_address,apply_topology,DELAYS,QUOTAS)
from task import examples,SPLITS


def forced_model():
    m=make_model(11)
    for e in m['edges']:
        e['weight']=2.0
        e['delay']=1
    m['thresholds']=[0.15]*14
    return m


class ExactYardTests(unittest.TestCase):
    def test_graph_and_quotas(self):
        m=make_model(11)
        self.assertEqual(len(candidate_edges()),90)
        self.assertEqual(sum(e['active'] for e in m['edges']),36)
        validate_model(m)
        for family,quota in QUOTAS.items():
            for dst in sorted({e['dst'] for e in m['edges'] if e['family']==family}):
                self.assertEqual(sum(e['active'] for e in m['edges'] if e['family']==family and e['dst']==dst),quota)

    def test_phase_quadrants_and_lower_tie(self):
        for x,y,k in ((1,0,0),(0,1,1),(-1,0,2),(0,-1,3),(0,0,0),(1,1,0),(-1,-1,2)):
            self.assertEqual(phase_address(x,y),k)

    def test_data_balance_and_label_blindness(self):
        data=examples('train',64)
        counts={(o,c):sum(x['order']==o and x['context']==c for x in data) for o in (0,1) for c in (0,1)}
        self.assertEqual(set(counts.values()),{16})
        self.assertTrue(all(x['label']==(x['order']^x['context']) for x in data))
        self.assertTrue(all(64<=x['onset']<=95 and x['gap'] in {6,8,10,12,14,16,18} for x in data))
        a=trial(0,0,78,10);b=trial(0,1,78,10)
        self.assertEqual((a['a_tick'],a['b_tick']),(b['a_tick'],b['b_tick']))
        self.assertNotEqual(a['label'],b['label'])

    def test_source_has_no_ais_and_travel_is_causal(self):
        m=forced_model();v=trial(0,0,95,6)
        r=infer(m,v,trace=True)
        starts={0:v['a_tick'],1:v['b_tick']}
        for d in r['deliveries']:
            e=m['edges'][d['edge']]
            if d['source'] in starts:self.assertEqual(d['tick'],starts[d['source']]+e['delay'])
            else:
                self.assertTrue(any(ev['yard']==d['source'] and d['tick']==ev['release_tick']+e['delay'] for ev in r['events']))
        self.assertTrue(all(x['exit_tick']>0 and x['exit_tick']%32==0 for x in r['events']))
        self.assertTrue(all(x['tick']==x['exit_tick']+8*x['phase'] for x in r['release_events']))

    def test_disabled_rails_never_deliver(self):
        m=forced_model();v=trial(0,0,95,6)
        r=infer(m,v,trace=True)
        self.assertEqual(sum(e['active'] for e in m['edges']),36)
        self.assertTrue(all(m['edges'][d['edge']]['active'] for d in r['deliveries']))
        active=next(e['id'] for e in m['edges'] if e['active'] and e['family']=='source_h1')
        damaged=infer(m,v,trace=True,broken_edges={active})
        self.assertNotIn(active,[x['edge'] for x in damaged['deliveries']])
        self.assertEqual(damaged['active_rails'],35)

    def test_basket_and_chandelier_are_distinct(self):
        m=forced_model();v=trial(0,0,95,6)
        src=next(e for e in m['edges'] if e['active'] and e['family']=='source_h1' and e['src']==0)
        y=src['dst']
        no=infer(m,v,trace=True)
        self.assertTrue(any(ev['yard']==y and ev['exit_tick']==96 for ev in no['events']))
        bs=infer(m,v,trace=True,basket={y:{96:8}})
        self.assertFalse(any(ev['yard']==y and ev['exit_tick']==96 for ev in bs['events']))
        ch=infer(m,v,trace=True,chandelier={y:{96:2}})
        self.assertFalse(any(ev['yard']==y and ev['exit_tick']==96 for ev in ch['events']))
        # Chandelier kept the charge, so it can fire at a later exit.
        self.assertTrue(any(ev['yard']==y and ev['exit_tick']==128 for ev in ch['events']))

    def test_context_alone_cannot_fire(self):
        m=forced_model();v=trial(0,1,95,6)
        for e in m['edges']:
            if e['family']=='source_h1':e['active']=False
        r=infer(m,v,trace=True)
        self.assertFalse(r['events'])
        self.assertEqual(r['counts'],[0,0])
        self.assertIsNone(r['prediction'])

    def test_fixed_model_hard_event_output_repeatable(self):
        m=forced_model();v=trial(1,1,95,6)
        before=infer(m,v,trace=True)
        after=infer(copy.deepcopy(m),v,trace=True)
        self.assertEqual(before,after)
        self.assertEqual(before['correct'],before['prediction']==v['label'])

    def test_score_window_reads_output_releases_not_ais(self):
        m=forced_model();v=trial(0,0,95,6)
        r=infer(m,v,trace=True)
        self.assertEqual(r['counts'],[sum(x['yard']==y and 128<=x['tick']<320 for x in r['release_events']) for y in (14,15)])


if __name__=='__main__':unittest.main()
