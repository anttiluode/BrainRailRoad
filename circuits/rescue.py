"""Rescue experiment (8 Nov 2026): is the silence attractor a clock problem?

One small CPU question, not a sweep. Same model seed, same training batches,
four configurations:

    absolute clock  x  {design loss, activity penalties off}
    stimulus reset  x  {design loss, activity penalties off}

Tasks (all hard-event, label never enters the model):
    order_fixed   A-before-B vs B-before-A, onset fixed at 80, context held at 0
    order_random  same, onset uniform 64..95 (one full clock period)
    xor           the original design task, y = order XOR context

The reported numbers are the *exported hard model* run through the float64
reference `model.infer`, never surrogate values. A torch-vs-reference count
parity check is run on every evaluated model.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import ROOT, infer, trial, validate_model
from model_torch import SurrogateRail

CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf8'))
TRAIN = CFG['train']
GAPS = CFG['task']['train_gaps']
OUT = ROOT / 'results' / 'rescue'
TASKS = ('order_fixed', 'order_random', 'xor')


def make_examples(task: str, n: int, seed: int) -> list[dict]:
    """Balanced examples. For order tasks the label is the order bit and context is 0."""
    rng = random.Random(seed)
    if task == 'xor':
        cells = [(o, c) for o in (0, 1) for c in (0, 1)] * (n // 4)
    else:
        cells = [(o, 0) for o in (0, 1)] * (n // 2)
    rng.shuffle(cells)
    out = []
    for order, context in cells:
        onset = 80 if task == 'order_fixed' else rng.randint(64, 95)
        x = trial(order, context, onset, rng.choice(GAPS))
        if task != 'xor':
            x['label'] = order  # order-only sanity task: no context XOR
        out.append(x)
    return out


def tensors(batch):
    return (torch.tensor([z['a_tick'] for z in batch], dtype=torch.long),
            torch.tensor([z['b_tick'] for z in batch], dtype=torch.long),
            torch.tensor([z['context'] for z in batch], dtype=torch.long),
            torch.tensor([z['label'] for z in batch], dtype=torch.long))


def evaluate_reference(model: dict, items: list[dict]) -> dict:
    """Actual hard-event model through the float64 reference. No surrogate anywhere."""
    correct = ties = pred0 = pred1 = 0
    out_counts = [0, 0]
    exits = 0
    cells = {}
    for x in items:
        r = infer(model, x)
        correct += int(r['correct'])
        ties += int(r['abstained'])
        pred0 += int(r['prediction'] == 0)
        pred1 += int(r['prediction'] == 1)
        out_counts[0] += r['counts'][0]
        out_counts[1] += r['counts'][1]
        exits += r['exit_events']
        key = f"order{x['order']}_ctx{x['context']}"
        c = cells.setdefault(key, [0, 0])
        c[0] += int(r['correct']); c[1] += 1
    n = len(items)
    acc = correct / n
    # one-sided exact binomial tail P(X >= correct | p=0.5)
    tail = sum(math.comb(n, k) for k in range(correct, n + 1)) / 2 ** n
    return {'n': n, 'accuracy': acc, 'abstentions': ties, 'predicted_O0': pred0, 'predicted_O1': pred1,
            'mean_output_releases': [out_counts[0] / n, out_counts[1] / n],
            'mean_exit_events_all_yards': exits / n,
            'pilot_deliveries_per_trial': 14 if (model.get('clock') or {}).get('mode') == 'stimulus_reset' else 0,
            'binomial_p_vs_chance': tail,
            'cells': {k: v[0] / v[1] for k, v in sorted(cells.items())}}


@torch.no_grad()
def torch_hard_counts(m, items):
    A, B, c, _ = tensors(items)
    return m(A, B, c).round().long().tolist()


def parity_check(m, model, items) -> int:
    """Number of trials where torch hard counts differ from the float64 reference."""
    tc = torch_hard_counts(m, items)
    return sum(tc[i] != infer(model, x)['counts'] for i, x in enumerate(items))


def run(task, clock_reset, penalties, updates, batch, seed, data_seed, dev, val, log_every=25, dev_every=50):
    torch.manual_seed(seed)
    m = SurrogateRail(seed, clock_reset=clock_reset)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=float(TRAIN['learning_rate']))
    rng = random.Random(data_seed)  # identical batch stream for every configuration
    hist = []
    best = {'dev_accuracy': -1.0, 'update': 0, 'state': copy.deepcopy(m.state_dict())}
    t0 = time.monotonic()

    def dev_eval(update):
        tc = torch_hard_counts(m, dev)
        hits = sum((c[1] > c[0]) == bool(x['label']) and c[0] != c[1] for c, x in zip(tc, dev))
        ties = sum(c[0] == c[1] for c in tc)
        acc = hits / len(dev)
        if acc > best['dev_accuracy']:
            best.update(dev_accuracy=acc, update=update, state=copy.deepcopy(m.state_dict()))
        return acc, ties

    acc0, ties0 = dev_eval(0)
    hist.append({'update': 0, 'dev_accuracy': acc0, 'dev_abstentions': ties0})
    for u in range(1, updates + 1):
        batch_items = make_examples(task, batch, rng.randrange(1 << 31))
        A, B, c, y = tensors(batch_items)
        opt.zero_grad(set_to_none=True)
        counts, stats = m(A, B, c, return_metrics=True, temperature_fire=float(TRAIN['temperature_fire']),
                          temperature_phase=float(TRAIN['temperature_phase']), temperature_delay=float(TRAIN['temperature_delay']))
        ce = F.cross_entropy(counts, y)
        loss = ce
        if penalties:
            loss = loss + float(TRAIN['penalty_hidden_events']) * stats['hidden_events'] / batch \
                + float(TRAIN['penalty_deliveries']) * stats['deliveries'] / batch \
                + float(TRAIN['penalty_context']) * m.context.square().sum()
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), float(TRAIN['gradient_clip'])))
        opt.step()
        hard = counts.detach()
        entry = {'update': u, 'CE': float(ce.detach()), 'train_acc': float(((hard.argmax(-1) == y) & (hard[:, 0] != hard[:, 1])).float().mean()),
                 'train_abstentions': int((hard[:, 0] == hard[:, 1]).sum()),
                 'hidden_events_per_trial': float(stats['hidden_events'].detach()) / batch,
                 'output_releases_per_trial': float(hard.sum()) / batch, 'grad_norm_preclip': gnorm}
        if u % dev_every == 0:
            entry['dev_accuracy'], entry['dev_abstentions'] = dev_eval(u)
        hist.append(entry)
        if u % log_every == 0:
            print(f"  u={u:4d} CE={entry['CE']:.3f} acc={entry['train_acc']:.2f} abst={entry['train_abstentions']:2d}/{batch} "
                  f"hid/trial={entry['hidden_events_per_trial']:.1f} out/trial={entry['output_releases_per_trial']:.2f} |g|={gnorm:.3f}"
                  + (f" dev={entry['dev_accuracy']:.3f}" if 'dev_accuracy' in entry else ''), flush=True)
    seconds = time.monotonic() - t0
    final_state = copy.deepcopy(m.state_dict())
    final_model = m.hard_export(); validate_model(final_model)
    final = evaluate_reference(final_model, val)
    final['torch_reference_count_mismatches'] = parity_check(m, final_model, val[:128])
    m.load_state_dict(best['state'])
    best_model = m.hard_export(); validate_model(best_model)
    best_eval = evaluate_reference(best_model, val)
    best_eval['torch_reference_count_mismatches'] = parity_check(m, best_model, val[:128])
    return {'final': final, 'best_on_dev': {'update': best['update'], 'dev_accuracy': best['dev_accuracy'], **best_eval},
            'history': hist, 'seconds': seconds}, final_model, best_model, final_state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=TASKS, required=True)
    ap.add_argument('--configs', default='abs_pen,abs_nopen,reset_pen,reset_nopen')
    ap.add_argument('--updates', type=int, default=300)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--data-seed', type=int, default=90811)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = make_examples(args.task, 128, 77001)          # checkpoint selection only
    val = make_examples(args.task, 512, 77002)          # reported numbers only
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.configs.split(','):
        clock_reset = name.startswith('reset'); penalties = name.endswith('_pen')
        print(f"== {args.task} / {name}: clock={'stimulus_reset' if clock_reset else 'absolute'} penalties={'on' if penalties else 'off'}", flush=True)
        res, fm, bm, state = run(args.task, clock_reset, penalties, args.updates, args.batch, args.seed, args.data_seed, dev, val)
        stem = f"{args.task}_{name}_u{args.updates}{args.tag}"
        (OUT / f"{stem}.json").write_text(json.dumps({'task': args.task, 'config': name, 'clock_reset': clock_reset,
                                                      'penalties': penalties, 'updates': args.updates, 'batch': args.batch,
                                                      'model_seed': args.seed, 'data_seed': args.data_seed, **res}, indent=1) + '\n')
        fm['training'] = {'experiment': 'rescue.py', 'task': args.task, 'config': name, 'updates': args.updates,
                          'selection': 'final update', 'note': 'development run, not the five-seed protocol'}
        torch.save({'state': state, 'clock_reset': clock_reset, 'seed': args.seed}, OUT / f"{stem}_final.pt")  # gitignored
        (OUT / f"{stem}_final_hard.json").write_text(json.dumps(fm, indent=1, sort_keys=True) + '\n')
        f, b = res['final'], res['best_on_dev']
        print(f"   FINAL hard ref: acc={f['accuracy']:.3f} abst={f['abstentions']}/{f['n']} O0/O1={f['predicted_O0']}/{f['predicted_O1']} "
              f"p={f['binomial_p_vs_chance']:.2g} mismatch={f['torch_reference_count_mismatches']}", flush=True)
        print(f"   BEST(dev@{b['update']}) hard ref: acc={b['accuracy']:.3f} abst={b['abstentions']}/{b['n']} "
              f"O0/O1={b['predicted_O0']}/{b['predicted_O1']} p={b['binomial_p_vs_chance']:.2g}  {res['seconds']:.0f}s", flush=True)
        summary[name] = {'final': {k: v for k, v in f.items()}, 'best_on_dev': {k: v for k, v in b.items()}, 'seconds': res['seconds']}
    (OUT / f"summary_{args.task}_u{args.updates}{args.tag}.json").write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
