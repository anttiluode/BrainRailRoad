"""Empty-yard gradient probe for the rescue experiment. Measures, changes nothing.

Question: does the hard empty-yard factor in the surrogate pulse,
    fire = ST_fire * 1[L* > eps],
cut off gradient that learning would need?

For a saved state (or the untrained init) and a fixed batch, split by class, we
compute the CE gradient under (a) the design surrogate and (b) the same
surrogate with that factor removed (`surrogate_through_empty=True`, a
diagnostic flag that is never used for training). The hard forward is identical
in both, because the hard firing decision already requires L* > eps.
We also count, per output yard and class, how many exit slots are
empty / loaded-but-subthreshold / firing.
"""
from __future__ import annotations
import argparse
import json
import math
import torch
from torch.nn import functional as F
from model_torch import SurrogateRail
from rescue import make_examples, tensors, OUT

GROUPS = ('raw_weight', 'raw_threshold', 'context', 'score', 'delay_logits')
O1 = 13  # yard index of output O1 inside the 14-yard tensors (O0 = 12)


def output_slot_census(m, items):
    """Exit-slot states of the two output yards, read from the float64 reference."""
    rec = {12: [0, 0, 0], 13: [0, 0, 0]}  # empty, loaded-subthreshold, firing
    from model import infer
    model = m.hard_export()
    for x in items:
        r = infer(model, x, trace=True)
        fired = {(e['yard'], e['exit_tick']) for e in r['events']}
        # reference exit ticks for this trial
        origin = r.get('clock_origin', 0)
        exits = [t for t in range(1, 320) if t > origin and (t - origin) % 32 == 0]
        loaded = _loaded_exits(model, x)
        for y in (14, 15):
            for t in exits:
                if (y, t) in fired: rec[y - 2][2] += 1
                elif loaded.get((y, t), False): rec[y - 2][1] += 1
                else: rec[y - 2][0] += 1
    return {('O0' if k == 12 else 'O1'): dict(zip(('empty', 'loaded_subthreshold', 'firing'), v)) for k, v in rec.items()}


def _loaded_exits(model, x):
    """Replay the reference loop just far enough to read L* at exits (no behaviour change)."""
    import model as ref
    from collections import defaultdict
    loaded = {}
    # L* is not exported by infer, so replicate the output yards' load update from
    # the traced deliveries (outputs have no basket input in this task).
    edges = model['edges']
    r = ref.infer(model, x, trace=True)
    origin = r.get('clock_origin', 0)
    L = {14: 0.0, 15: 0.0}
    fired = {(e['yard'], e['exit_tick']): e for e in r['events']}
    for t in range(320):
        for y in (14, 15):
            th = 2 * math.pi * ((t - origin) % 32) / 32
            v = 0.0
            for d in r['deliveries']:
                if d['dest'] == y and d['tick'] == t:
                    port = edges[d['edge']]['port']
                    v += math.exp(model['kappa'] * (math.cos(th - ref.ANGLE[port]) - 1)) * edges[d['edge']]['weight']
            L[y] = model['leak'] * L[y] + v
            if t > origin and (t - origin) % 32 == 0:
                loaded[(y, t)] = L[y] > model['epsilon']
                e = fired.get((y, t))
                if e: L[y] *= max(0.0, 1 - model['thresholds'][y - 2] / (e['gain'] * e['radius']))
    return loaded


def grads(m, items, through_empty):
    m.surrogate_through_empty = through_empty
    m.zero_grad(set_to_none=True)
    A, B, c, y = tensors(items)
    C = m(A, B, c)
    F.cross_entropy(C, y).backward()
    out = {n: float(p.grad.norm()) if p.grad is not None else 0.0 for n, p in m.named_parameters() if n in GROUPS}
    out['O1_threshold'] = float(m.raw_threshold.grad[O1].abs())
    into_o1 = [i for i in range(90) if int(m.dst[i]) == O1]
    out['O1_incoming_weight_and_score'] = float(m.raw_weight.grad[into_o1].norm() + m.score.grad[into_o1].norm())
    out['hard_counts_mean'] = [float(v) for v in C.detach().mean(0)]
    m.surrogate_through_empty = False
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', default='', help='rescue *_final.pt; omit for untrained init')
    ap.add_argument('--clock-reset', action='store_true', help='only used with no --state')
    ap.add_argument('--task', default='order_random')
    ap.add_argument('--n', type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.state:
        saved = torch.load(args.state, weights_only=False)
        m = SurrogateRail(saved['seed'], clock_reset=saved['clock_reset']); m.load_state_dict(saved['state'])
    else:
        m = SurrogateRail(11, clock_reset=args.clock_reset)
    items = make_examples(args.task, args.n, 77003)
    report = {'state': args.state or 'untrained', 'task': args.task, 'clock_reset': m.clock_reset, 'by_class': {}}
    for cls in (0, 1):
        sub = [x for x in items if x['label'] == cls]
        report['by_class'][cls] = {'design_surrogate': grads(m, sub, False),
                                   'mask_removed_diagnostic': grads(m, sub, True),
                                   'output_exit_slots': output_slot_census(m, sub)}
    print(json.dumps(report, indent=1))
    name = 'probe_' + (args.state.split('/')[-1].replace('_final.pt', '') if args.state else f"untrained_{'reset' if m.clock_reset else 'abs'}")
    (OUT / f'{name}.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
