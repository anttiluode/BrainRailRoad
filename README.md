# BrainRailRoad — phase-addressed circuits, not scripted train cartoons

A testable CPU/Node implementation of the [Turntable Circuits design](circuits/DESIGN.md), commissioned on **8 November 2026**.

**Research status: implementation in progress.** The *hard-event forward model* and source-/packet-level Python↔Node parity are implemented. A PyTorch straight-through surrogate BPTT trainer works and produces nonzero finite gradients. The first development training attempts **failed**: the classifier can collapse to output ties (abstentions). An embedded reference model in the browser is explicitly **untrained**, and the design's 95%/85% accuracy targets are **not** claimed. The full five-seed experiment, pilot robustness, matched baselines, and teacher-assisted lesion recovery remain future work.

## Run the railway

The browser lab is at [`circuits/index.html`](circuits/index.html). Download it **with** [`circuits/circuit_core.js`](circuits/circuit_core.js) and open the HTML locally. There are no external imports, remote inference services or cloud requirements. The lab embeds its operational model directly; importing another exported model JSON changes actual rail weights, delays, thresholds and masks, not just animation.

```bash
cd circuits
python task.py --freeze                        # frozen balanced validation + test manifests
python tests/parity_python.py                  # make hard float64 model and fixtures
node tests/parity_node.cjs                     # exact packets / release ticks / 1e-10 states
python -m unittest discover -s tests -v       # Python core and training tests

# One actual surrogate-BPTT gradient / optimizer update (a smoke test, not learning success)
python train.py --name smoke --seed 11 --batch 4 --updates 1

# An experimental training run (CPU, may fail to learn)
python train.py --name full --seed 11 --batch 32 --updates 1500 --threads 1
```

Python 3.10+, NumPy **not required**, and `torch` (CPU build) only for training. Node 18+ for JS parity. Configurations and the task split seeds live in [`circuits/config.json`](circuits/config.json).

## What exists

- **14 yards**: six in H1, six in H2, two outputs, plus A and B sensory emitters.
- **90 candidate directed rails**, with exact **36 active** from per-destination quotas (6 + 18 + 6 + 6). Every candidate's fixed receiver port angle, mutable weight and 8-choice transmission delay are exported.
- **Four output phases** `0,8,16,24` ticks from each yard's 32-tick exit clock. A hard unit packet is all a receiver obtains: no vector payload or contextual label.
- **Exact tick order** of the attached design, hard arrival capture, multiplicative apical/context gain, scalar and complex soma, chandelier veto, basket shunt, and thresholded, phase-addressed release.
- **Balanced task** `y = (A-before-B ? 0 : 1) XOR context`, with separately seeded data manifests and independent noise.
- **Real straight-through BPTT** that keeps the *forward values hard*, schedules all alternatives in the backward graph, and gives dormant candidates surrogate gradients. This is explicitly **supervised nonlocal search**, not a local biological learning law.
- A browser lab that replays **the real exported forward engine** with order/context, jitter, deliberate cuts, time stepping, a packet log, actual release counts, and model import/export.

## Hard-event semantics

Every hidden/output yard computes one integrated vector from delayed unit pulses, sampled through phase-selective port gates. At `t=32,64,...` the yard either emits **one** event or stays silent; the chosen phase address delays its actual release. Rail travel adds a separate odd-integer transmission delay. An output's own release, **not its AIS crossing**, counts toward its class result in `[128,320)`. A tie is an abstention and an error. Inference never evaluates dormant rails and never has access to the desired label. The source train is emitted immediately at its task timestamp, without an invented source AIS.

The Python reference is [`circuits/model.py`](circuits/model.py); the offline Node/browser mirror is [`circuits/circuit_core.js`](circuits/circuit_core.js). Trainable raw parameters and surrogate gradients are in [`circuits/model_torch.py`](circuits/model_torch.py).

## Initial dev failure — preserve, don't hide it

The first 100-update, 32-trial-batch seed-11 probe with the design's threshold initial value `0.35` fell to complete output abstention (`CE≈ln2=0.6931`). A second development probe using `0.20` initialization initially produced more events but also collapsed into abstention by update 200. A third `0.16` initialization likewise collapsed by update 150. In all three, **finite nonzero gradients** reached delay, phase-influencing weights, thresholds, context and rail scores. This demonstrates the training *plumbing*, **not** successful multi-stage learning. These probes motivated a future investigation of credit assignment / quiescent-output attractors before using held-out test results for model selection.

Development probes are not the five-seed protocol. No headline accuracy is reported from them. A complete experiment must run the exact stored manifests, five seeds, fixed-phase/fixed-delay/fixed-topology controls, conventional spiking and GRU comparisons, and lesion recovery under its supervised protocol. Those items are not silently marked as finished.

## Reproducibility / no hidden teacher

The default bundled model was drawn from `make_model(11)` and is identified as **random-untrained** in the export. The browser's answer panel shows what this model actually does, often `ABSTAIN` or an incorrect response. If a checkpoint is trained later, `train.py` writes a **hard** export in `circuits/results/`; the same JSON can be loaded in the lab without trainer code. Every operational active rail can be audited; context channels and pilot infrastructure are separately counted and cannot inject labels.

## Next experiment

First rescue learning without secretly introducing a differentiable hidden-state readout or post-hoc hard-coded routing. Diagnose the silence attractor; try and explicitly label any different initial thresholds, curriculum or additional training penalty. Then freeze validated checkpoints, run the designed baselines and damage protocol. After that, implement the precisely defined pilot-only clock correction and its separate time/jitter accounting. At each stage compare Python/JavaScript event traces, not merely final accuracy.

**Prior work and mathematical boundaries:** [TurntableNeuron](https://github.com/anttiluode/TurntableNeuron), [TurntableAISol](https://github.com/anttiluode/TurntableAISol), [KasvavatKiskotSol](https://github.com/anttiluode/KasvavatKiskotSol), [KasvavatKiskotClaude](https://github.com/anttiluode/KasvavatKiskotClaude). The design draws on learned delays in spiking neural networks and polychronization. This artifact does not establish a new neuroscience law, Feynman quantum computation, fully local autonomous plasticity or superiority over established models.
