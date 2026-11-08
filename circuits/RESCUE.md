# Rescue experiment — is the silence attractor a clock problem?

**8 November 2026. One small CPU experiment, model seed 11, one data stream. Not the five-seed protocol.**

**Verdict: the success criterion is not met.** No configuration exceeds chance on validation under the design's scoring rule (ties are errors). The experiment did answer the question it was built for, and it located the bottleneck precisely:

1. **The absolute clock is the cause of total silence when onset is random.** With the original clock the network falls completely silent and its gradient goes to zero. With a stimulus-locked reset it does not.
2. **The reset does not get the railway above chance.** It converts the random-onset task into the fixed-onset task. On that task the original railway already learns the order bit perfectly, but only as a *one-sided code*: one order fires an output, the other order is completely silent. Silence is a tie, and a tie scores as an error, so accuracy sits at exactly 50%.
3. **Silent trials produce exactly zero gradient.** That is why the one-sided code is a fixed point. The empty-yard mask is not what cuts the gradient. Removing it changes only threshold gradients, which are spurious. The cut is the zero vector itself: the derivative of the integrated vector's magnitude is zero at the origin.

## What was changed (all optional, default off)

| Change | Where | Default |
|---|---|---|
| `clock.mode = "stimulus_reset"`. An explicit pilot fires at the first source emission r = min(a_tick, b_tick) and is delivered to all 14 yards at tick r. Every yard's phase origin becomes r, and exits move to r+32, r+64, … | `model.py`, `circuit_core.js`, `model_torch.py` (`clock_reset=True`), exported in the model JSON | `absolute` (original semantics) |
| `surrogate_through_empty`, a diagnostic flag that drops the empty-yard factor from the surrogate pulse. **Used only in the probe, never in training.** | `model_torch.py` | off |
| Browser lab shows the clock mode and the pilot tick. It also flags order-only exports, whose target is the order bit rather than the XOR label. | `index.html` | — |

The pilot is **label-neutral**: r is the onset, identical for A-then-B and B-then-A (tested in `tests/test_rescue.py`). It carries no load, no vector and no label, and it is counted as 14 pilot deliveries per trial. It does carry the *time* of the first stimulus to every yard at zero delay. That is engineered infrastructure, and it is counted as such.

The original hard semantics are untouched when `clock` is absent. All original tests and the original 16 parity fixtures pass unchanged.

## Protocol

- Training: `rescue.py`. Design trainer, Adam 0.003, batch 32, the design temperatures, model seed 11, data seed 90811. All four configurations see an **identical batch stream**.
- "Penalties off" removes the hidden-event, delivery and context penalties. The cross-entropy on output release counts is unchanged.
- Order-only tasks: label = order bit, context held at 0, gaps {6, 8, …, 18}.
- Checkpoint selection uses a 128-trial dev set (seed 77001). **All reported numbers are on a separate 512-trial validation set (seed 77002).** They come from the *exported hard model* run through the float64 reference `model.infer`, never from surrogate values.
- On every evaluated model, the torch hard counts were checked against the reference on 128 trials: **0 mismatches in every run**.
- Total compute: about 13 CPU-minutes on 2 cores.

## Results (hard reference, 512 validation trials, chance = 50%)

### Step 1 — fixed onset (tick 80), order only, original clock, 200 updates

| Config | Accuracy | Abstentions | Predicted O0 / O1 | Mean output releases O0, O1 |
|---|---:|---:|---:|---:|
| absolute, penalties on | 50.0% | 256 | 256 / 0 | 2.24, 0.00 |
| absolute, penalties off | 50.0% | 256 | 256 / 0 | 1.98, 0.00 |

Every A-then-B trial fires O0 and is correct. Every B-then-A trial is fully silent at the outputs (0 deliveries into O0 or O1), which counts as a tie and an error. **The order information is carried perfectly by the hard events**, but in a code the scoring rule cannot accept.

### Step 2 — random onset (64–95, one full period), order only, 200 updates

| Config | Accuracy | Abstentions | Predicted O0 / O1 | Output releases O0, O1 | Exits/trial | Final grad norm |
|---|---:|---:|---:|---:|---:|---:|
| absolute, penalties on | 0.0% | 512 | 0 / 0 | 0, 0 | 1.5 | 0.015 |
| absolute, penalties off | 0.0% | 512 | 0 / 0 | 0, 0 | 1.4 | 0.000 |
| reset, penalties on | 19.7% | 411 | 0 / 101 | 0.21, 0.87 | 3.0 | 0.155 |
| **reset, penalties off** | 50.0% | 256 | 0 / 256 | 0.00, 1.86 | 8.3 | 0.014 |

The absolute-clock models had the best dev score at update 0, before training: 13.3%, with O1 firing about equally for both classes. Training then drove both into total silence. This is the predicted collapse: with no class information at the outputs, random firing costs more cross-entropy than silence.

### Step 3 — extend only the clear improvement: reset, penalties off, 600 updates

| Updates | Accuracy | Abstentions | Predicted O0 / O1 | Output releases O0, O1 | Exits/trial | Grad norm |
|---:|---:|---:|---:|---:|---:|---:|
| 600 | 50.0% | 256 | 0 / 256 | 0.00, 2.50 | 13.2 | **0.000** |

Activity per class in the final model:

| Class | H1 events | H2 events | Output events |
|---|---:|---:|---:|
| A-then-B (class 0) | 1.16 | 0 | 0 |
| B-then-A (class 1) | 5.02 | 15.14 | 5.0 |

The training loss settles at 0.350 ≈ ln 2 / 2. Class 1 contributes ≈ 0 because its counts are [0, 5]. Class 0 contributes exactly ln 2 from the [0, 0] tie, and **zero gradient**. The extension changed nothing; it is a fixed point.

Python/JavaScript parity for this trained reset export:
- all 512 validation decisions and counts are identical;
- the first 24 trials have exact event and delivery traces (`tests/rescue_parity_*`);
- 12 new reset fixtures on the untrained and strong models are also exact.

## The bottleneck: a silent yard is invisible to the surrogate

Prediction uses C_k = Σ_exits fire_k · phase-slot. In the trainer a yard's surrogate pulse is

```
fire = [ H(a r − h) − σ((a r − h)/T).detach() + σ((a r − h)/T) ] · 1[L* > ε]      r = sqrt(z_x² + z_y² + ε²)
```

On a trial where an output yard received nothing, z = 0 and L* = 0. There are two separate reasons its gradient is zero:

1. **The empty-yard factor 1[L* > ε] multiplies the pulse by 0.** Gradient is zero for everything.
2. **Even without that factor, ∂r/∂z = z/r = 0 at z = 0.** The surrogate's only route to upstream rails runs through r, so a packet that *would* have arrived gets no credit. This includes zero-valued alternative-delay slots and dormant rails, which do enter z with gradient attached. Only the threshold h and the context gain a still see a gradient.

Measured with `rescue_probe.py`: class-split CE gradients on 32 trials per class, design surrogate vs. mask removed.

| State | Class (outputs) | Grad norm, weights / scores / delays / context: design | same, mask removed | Threshold grad norm: design → mask removed |
|---|---|---|---|---|
| untrained, reset clock | class 0 (both outputs empty in all 224 exit slots) | 0 / 0 / 0 / 0 | identical: 0 / 0 / 0 / 0 | 0 → 0.459 |
| trained 600 updates, reset | class 0 (both outputs empty in all 224 slots) | 0 / 0 / 0 / 0 | identical: 0 / 0 / 0 / 0 | 0 → 0.434 |
| untrained, absolute clock | class 0 (O1 fires in 11 of 288 slots) | 0.120 / 0.057 / 0.039 / 0.097 | identical (context to ~1e-9) | 0.203 → 0.563 |

**Finding.** The hard empty-yard mask does not cut any *useful* gradient. Removing it adds gradient only to thresholds of yards that received nothing. That gradient is spurious, because no threshold can make an empty yard fire (the hard rule needs L* > ε). Weight, topology and delay gradients are bit-identical with and without the mask. Context gradients differ only in the ninth significant figure, through the gain a in σ((a·ε − h)/T) at empty yards. **Recommendation: leave the mask as it is.**

The real cut is the zero-vector degeneracy of r. It means a fully silent trial teaches nothing. Once one class is silent end-to-end, which here means the whole H2 layer, that class can never be pulled back. This also explains the absolute-clock collapse: once all outputs are silent for every trial, the gradient is exactly zero everywhere.

## Permissible conclusions

- The stimulus-locked reset **removes the total-silence collapse** that the absolute clock produces under random onset. This is one seed, so treat it as evidence, not proof.
- With either clock, the hard railway **discriminates order perfectly in its event pattern** (fire vs. silence). It does **not** produce a two-sided output code, so under the design's rule its validation accuracy is at chance (50.0%, binomial p = 0.52). **Success is not claimed.**
- The context-XOR task was not run. It would hit the same one-sided fixed point before XOR even matters.

## Next experiment (not done — a surrogate-only change, flagged for decision)

Give silent yards a credit route that does not pass through the degenerate norm. The scalar load is linear in arrivals (∂L*/∂I_e = g_e ≠ 0), so a zero-valued alternative-delay or dormant-rail slot *does* carry gradient into L*. A candidate rule:

```
soft pulse uses  a·(r + β·L*) − h   instead of  a·r − h      (hard forward unchanged)
```

This needs the empty-yard factor removed in the *surrogate only*. It would be a new, explicitly labelled training estimator, not a change to the machine. It should be tested the same way: one seed, the same four-way table, hard-reference numbers only.

## Reproduce

```
cd circuits
python rescue.py --task order_fixed  --configs abs_pen,abs_nopen --updates 200 --tag _stage1
python rescue.py --task order_random --updates 200 --tag _stage2
python rescue.py --task order_random --configs reset_nopen --updates 600 --tag _stage3
python rescue_probe.py --clock-reset ; python rescue_probe.py
python rescue_probe.py --state results/rescue/order_random_reset_nopen_u600_stage3_final.pt
python tests/rescue_parity_python.py && node tests/rescue_parity_node.cjs
```

Raw per-run histories, hard exports and probe outputs are in `results/rescue/`. The `.pt` torch states are gitignored and can be regenerated deterministically with the commands above.
