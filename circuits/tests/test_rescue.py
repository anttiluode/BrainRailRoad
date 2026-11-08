"""Checks for the optional stimulus-locked clock reset (rescue experiment)."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import make_model, infer, trial, validate_model, clock_origin  # noqa: E402


class StimulusReset(unittest.TestCase):
    def test_default_clock_is_absolute_and_unchanged(self):
        m = make_model(11)
        self.assertNotIn('clock', m)
        r = infer(m, trial(0, 0, 80, 10))
        self.assertNotIn('clock_origin', r)
        self.assertNotIn('pilot_deliveries', r)

    def test_reset_origin_is_label_neutral(self):
        m = make_model(11); m['clock'] = {'mode': 'stimulus_reset'}
        for onset in (64, 77, 95):
            for gap in (6, 18):
                origins = {clock_origin(m, trial(o, c, onset, gap)) for o in (0, 1) for c in (0, 1)}
                self.assertEqual(origins, {onset})

    def test_unknown_clock_rejected(self):
        m = make_model(11); m['clock'] = {'mode': 'psychic'}
        with self.assertRaises(ValueError):
            validate_model(m)

    def test_no_exit_at_or_before_origin_and_pilot_is_counted(self):
        m = make_model(11)
        for e in m['edges']: e['weight'] = 2.0; e['delay'] = 1
        m['thresholds'] = [0.15] * 14
        m['clock'] = {'mode': 'stimulus_reset'}
        r = infer(m, trial(1, 0, 83, 12), trace=True)
        self.assertEqual(r['pilot_deliveries'], 14)
        self.assertTrue(r['events'])
        for ev in r['events']:
            self.assertGreater(ev['exit_tick'], 83)
            self.assertEqual((ev['exit_tick'] - 83) % 32, 0)

    def test_torch_reset_hard_counts_match_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed')
        from model_torch import SurrogateRail
        from rescue import make_examples, tensors
        m = SurrogateRail(11, clock_reset=True)
        with torch.no_grad():
            m.raw_threshold.fill_(-4.0)  # thresholds near 0.15 so the network is busy
            m.raw_weight.fill_(2.0)
        items = make_examples('xor', 16, 5) + make_examples('order_random', 8, 6)
        A, B, c, _ = tensors(items)
        with torch.no_grad():
            counts = m(A, B, c).round().long().tolist()
        export = m.hard_export()
        self.assertEqual(export['clock']['mode'], 'stimulus_reset')
        busy = 0
        for x, tc in zip(items, counts):
            ref = infer(export, x)
            self.assertEqual(tc, ref['counts'])
            busy += sum(ref['counts'])
        self.assertGreater(busy, 0)


if __name__ == '__main__':
    unittest.main()
