"""Optional CPU torch tests; parity and ST checks do not rely on GPU."""
import importlib.util
import sys
import math
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional CPU torch not installed')
class TorchCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def batch(self):
        import torch
        from task import examples
        samples=examples('train',8,seed=1001)
        return samples,tuple(torch.tensor([x[k] for x in samples],dtype=torch.long) for k in ('a_tick','b_tick','context','label'))

    def test_torch_hard_counts_match_float64_reference(self):
        import torch
        from model_torch import SurrogateRail
        from model import infer
        samples,(a,b,c,y)=self.batch()
        for seed in (11,23):
            m=SurrogateRail(seed)
            with torch.no_grad():counts=m(a,b,c).tolist()
            py=[infer(m.hard_export(),x)['counts'] for x in samples]
            self.assertEqual(counts,py)

    def test_forward_value_cannot_change_with_temperature(self):
        import torch
        from model_torch import SurrogateRail
        samples,(a,b,c,y)=self.batch()
        m=SurrogateRail(11)
        with torch.no_grad():
            normal=m(a,b,c,temperature_fire=.25,temperature_phase=.5,temperature_delay=1.)
            hot=m(a,b,c,temperature_fire=3.,temperature_phase=4.,temperature_delay=9.)
        self.assertTrue(torch.equal(normal,hot))

    def test_nonzero_finite_surrogate_gradients(self):
        import torch
        from torch.nn import functional as F
        from model_torch import SurrogateRail
        samples,(a,b,c,y)=self.batch()
        m=SurrogateRail(11)
        loss=F.cross_entropy(m(a,b,c),y)
        loss.backward()
        for name in ('raw_weight','raw_threshold','context','score','delay_logits'):
            g=getattr(m,name).grad
            self.assertIsNotNone(g,name)
            self.assertTrue(bool(torch.isfinite(g).all()),name)
            self.assertGreater(float(g.abs().sum()),0,name)

    def test_resume_exactly_matches_continuous_optimizer(self):
        import copy
        import torch
        from torch.nn import functional as F
        from model_torch import SurrogateRail
        samples,(a,b,c,y)=self.batch()
        continuous=SurrogateRail(11)
        optimizer=torch.optim.Adam(continuous.parameters(),lr=.003)
        for i in range(1):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(continuous(a,b,c),y).backward()
            optimizer.step()
        checkpoint={'model':copy.deepcopy(continuous.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict())}
        resumed=SurrogateRail(11)
        resumed.load_state_dict(checkpoint['model'])
        optim2=torch.optim.Adam(resumed.parameters(),lr=.003)
        optim2.load_state_dict(checkpoint['optimizer'])
        optimizer.zero_grad(set_to_none=True);F.cross_entropy(continuous(a,b,c),y).backward();optimizer.step()
        optim2.zero_grad(set_to_none=True);F.cross_entropy(resumed(a,b,c),y).backward();optim2.step()
        for (name,p),(name2,q) in zip(continuous.named_parameters(),resumed.named_parameters()):
            self.assertEqual(name,name2)
            self.assertTrue(torch.equal(p,q),name)

    def test_spiking_signed_baseline_torch_hard_parity(self):
        import torch
        from model_torch import SurrogateRail
        from model import infer,validate_model
        samples,(a,b,c,y)=self.batch()
        m=SurrogateRail(11,spiking=True)
        with torch.no_grad():counts=m(a,b,c).tolist()
        model=m.hard_export()
        validate_model(model)
        self.assertEqual(counts,[infer(model,x)['counts'] for x in samples])

if __name__=='__main__':unittest.main()
