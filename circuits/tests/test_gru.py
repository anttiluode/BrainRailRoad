"""Conventional task-control checks, optional when torch isn't installed."""
import importlib.util
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch optional')
class GRUBaselineChecks(unittest.TestCase):
    def test_only_sensory_and_tonic_context_channels(self):
        from task import examples
        from gru_baseline import tensorize
        records=examples('train',8,seed=503)
        x,y=tensorize(records,127)
        self.assertEqual(tuple(x.shape),(8,128,4))
        for i,r in enumerate(records):
            self.assertEqual(x[i,:,0].sum().item(),1)
            self.assertEqual(x[i,:,1].sum().item(),1)
            self.assertEqual(x[i,r['a_tick'],0].item(),1)
            self.assertEqual(x[i,r['b_tick'],1].item(),1)
            self.assertEqual(x[i,:,2+r['context']].sum().item(),128)
            self.assertEqual(int(y[i]),r['order']^r['context'])

    def test_conventional_gru_param_count(self):
        import torch
        from gru_baseline import GRUBaseline
        g=GRUBaseline(hidden=15)
        self.assertEqual(sum(p.numel() for p in g.parameters()),977)
        self.assertEqual(tuple(g(torch.zeros(4,128,4)).shape),(4,2))

if __name__=='__main__':unittest.main()
