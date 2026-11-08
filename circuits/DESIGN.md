# Turntable Circuits
## A phase-addressed railway that learns reusable temporal computation
### Design and implementation handoff for Sol — 8 November 2026

**Status:** proposed architecture and experimental protocol, not a paper reporting new results.  
**Design author:** Astra, for Antti and Sol.  
**Deliverable:** one working CPU-trainable model, one offline browser laboratory, and an honest comparison.  
**Recommended home:** an isolated `turntable-circuits` branch in `anttiluode/TurntableAISol`; preserve the existing lab and results.

## 1. The decision

Build a small network in which each yard integrates timed input into a vector, converts that vector into one of four departure phases, and communicates through a limited set of learned rails. Learn connection strength, conduction delay, and active connectivity together on a context-dependent order task.

Use **hard discrete events in the forward simulation, with surrogate gradients during supervised training**. Export the actual discrete model to JavaScript. Do not first build a continuous model and hope its discretization behaves similarly.

The main claim under investigation is:

> Can phase-addressed communication and learned sparse connectivity form a reusable temporal circuit that remains useful when timing varies or a connection fails?

Success means learning a task with a transparent mechanism. Superiority over established networks is a separate, measured claim.

### Why this version

| Approach | Advantage | Decision |
|---|---|---|
| Timing-gated ant-colony routing | Simple, exploratory, close to growing-rail prototypes | Keep as lineage; finding one rewarded route does not teach the proposed multistage computation |
| Local reward and eligibility traces throughout the network | Attractive eventual autonomous learner | Defer: hidden-layer credit assignment and structural exploration are unsolved in these repositories |
| Hard-event network with supervised surrogate-gradient training | Explicit learning signal, implementable controls, direct comparison with spiking baselines | **Build this first** |

This deliberately introduces an external teacher. It does not claim local biological learning. A successful trained circuit gives a concrete object on which to investigate local learning later.

## 2. What is inherited, and what changes

The source review used these repository snapshots:

| Source | Snapshot | Reuse |
|---|---|---|
| TurntableNeuron | `006206cd3775667d6225b3ca6b1dc48823c02d23` | Vector platform, rhythmic capture, timing-based addressing, pilot/data distinction |
| TurntableAISol | `af74a0cec819417c36375968f57b3f76889db270` | Layered yards; separate apical gain, basket shunt, and chandelier veto; visual language |
| KasvavatKiskotSol | `2febfe2f8601ec7456c8af49b0bba2d589ccb732` | A rail must actually be built to carry data; visible resource accounting |
| KasvavatKiskotClaude | `547de301f9bbd42451db2ec562fbddf2ce37c874` | Exploration/lock-in lessons and damaged-route experiments |
| GeometricNeuronOriginReview | `d30862c3460f9cef58b25e0d53388b030f9bca7b` | Reduce the model to its actual causal state; preserve scheduler semantics |
| TorsionFreeMirrorLock | `205a7d379274cb84739c26cc6935731ac75b33e6` | Separate research question; do not bolt hidden-fiber memory onto this version |

The major intentional simplification is **four discrete phase addresses**, instead of an arbitrary angle-to-delay conversion. It avoids the branch-cut problem of differentiating `atan2` near zero and gives an exact browser implementation.

This is a new model inspired by the earlier cores, not a drop-in patch claiming to preserve their numerical results. Keep their files and published receipts intact.

## 3. The machine

### 3.1 Fixed size for the first experiment

- Two sensory sources: A and B. Each emits one identical unit packet per trial. A task timestamp s means immediate source emission, with no source AIS or phase selection. An active source-to-H1 rail delivers it at s + d_e, with its usual weight applied once.
- Two hidden layers, H1 and H2, six yards each.
- Two output yards, O0 and O1.
- One externally supplied context bit, represented as two one-hot context channels.
- Shared nominal period P = 32 ticks.
- Four phase addresses, with departure delays 0, 8, 16, 24 ticks.
- 320 ticks per trial, including warm-up.
- No GPU requirement. Python/PyTorch for training, Node/JavaScript for exact inference and browser replay.

Candidate **directed data rails**:

| Family | Candidates | Active incoming quota | Active rails |
|---|---:|---|---:|
| A/B → H1 | 12 | 1 per H1 yard | 6 |
| H1 → H2 | 36 | 3 per H2 yard | 18 |
| H2 → H2, excluding self | 30 | 1 per H2 yard | 6 |
| H2 → O0/O1 | 12 | 3 per output | 6 |
| Total | **90** | | **36** |

The quota is an exact structural budget, not a drawing threshold on a dense working network. Dormant rails carry zero forward current. These quotas are a design constraint, not a discovered brain structure.

The context bus and pilot bus are separate engineered infrastructure. Count their parameters and fan-out separately; do not pretend they are free or included in the 36 data rails. There are 24 learned context gains: two context channels into each of 12 hidden yards. No context connection reaches an output yard.

No sensory-to-output skip connections. No target-label input. No hidden vector payload in a packet.

### 3.2 Rail state

Each candidate rail e has:

- source, destination, and receiver port angle, fixed by the graph generator;
- positive weight w_e in [0.05, 2];
- delay chosen from D = {1, 3, 5, 7, 9, 11, 13, 15} ticks;
- structural score s_e and hard active mask m_e;
- broken flag, initially false.

Receiver port angles are assigned round-robin over {0, π/2, π, 3π/2} in canonical source-ID order within each incoming family. Freeze and export this assignment. The assignment contains no class information.

Different source connections can share a port angle. A port is a timing preference, not a semantic label. Rail coordinates in the drawing do not determine delay: delay is an explicit learned state in this version. True geometric growth into empty space is deferred.

### 3.3 Yard state

Every hidden/output yard stores:

- scalar load L ≥ 0;
- vector z = (z_x, z_y);
- local clock phase θ;
- basket state b ≥ 0 and chandelier state c ≥ 0;
- delayed output queue entries.

Context produces an instantaneous hidden-yard gain a, not an extra basal current. The main task has b = c = 0; explicit intervention inputs exercise the two inhibitors later. Do not add trainable inhibitory feedback in the first version.

Use float64 in the reference and browser parity fixtures. CPU training may use float32 only after verifying that classification and event decisions agree away from threshold ties.

## 4. Exact forward semantics

These rules define the model, including the tick order. Sol may optimize their implementation but must preserve their meaning.

### 4.1 Initialization and clocks

At trial start clear L, z, inhibitor states and queues. Reset dynamic state, **not learned parameters or topology**. Use θ(t) = 2πt/P in the clean reference. All yards start with the same phase reference; their incoming ports provide different listening windows.

Exit opportunities occur at t = 32, 64, 96, ... . There is no exit at tick zero. Main-task clean training does not need a simulated PLL: it uses the explicitly supplied reference. The pilot intervention in Section 9 adds the clock mechanism and charges its cost.

### 4.2 Tick update order

For integer tick t:

1. Deliver every packet previously scheduled for t. Aggregate by receiving rail.
2. Read context and intervention inputs for t.
3. Apply leakage, basal capture, and basket shunt.
4. If this is the yard's exit opportunity, evaluate the AIS and form one departure.
5. Schedule newly formed departures; no packet may arrive in its birth tick.
6. Record output packets whose release time is t, then advance time. Outputs have no outgoing data rails: their phase-delayed releases themselves are the externally visible output channel.

Edges have delay at least one. A recurrent packet therefore cannot affect its own creation instant.

### 4.3 Capture and integration

For arriving rail e at yard j, let I_e(t) be packet count at that tick, weighted by w_e and multiplied by m_e and the intact-edge indicator. A logical packet has unit amplitude; multiple packets may add.

g_e(t) = exp[κ (cos(θ_j(t) − φ_e) − 1)], with κ = 2 initially.

B_j(t) = Σ_e g_e(t) I_e(t).

Use λ = 0.985 initially, shared and fixed:

L* = [λL + B] exp(−1.4b)  
z* = [λz + B(cos θ, sin θ)] exp(−1.4b).

All accepted contributions at a tick use the receiver's **current** phase. Across ticks their vectors can reinforce or cancel. This is not a hidden message vector sent through the rail.

Basket and chandelier inputs update before this operation:

b ← 0.82b + basket_input  
c ← 0.84c + chandelier_input.

Without interventions both stay zero.

For hidden yard j and one-hot context x:

a_j = exp[0.7 tanh(U_j · x)].

For outputs a_j = 1. Context can change the effect of existing input but cannot inject load or create a departure from an empty yard.

### 4.4 One event at an exit

Let r = sqrt(z_x*² + z_y*² + ε²), ε = 10^−8, and learned threshold h_j ∈ [0.15, 1.5].

At an exit, emit one logical packet iff:

a_j r ≥ h_j, L* > ε, and c < 0.45.

At other ticks emit nothing. This first version emits at most one packet per yard per rotation. Do not silently introduce an analog amplitude output or carriage-count channel.

Select departure address from the pre-reset vector:

q_k = z* · v_k / max(r, ε),  
v_k = (cos(2πk/4), sin(2πk/4)),  
k* = argmax_k q_k.

Ties choose the lowest k. Its release delay is τ = 8k*. No event means no packet at any phase, even if the address calculation exists internally.

If firing occurs, consume state proportionally:

keep = max(0, 1 − h_j / (a_j r));  
L ← keep L*; z ← keep z*.

Otherwise keep L*, z*. A chandelier veto leaves state available, subject to the normal leak; it does not secretly erase it.

For each active, intact outgoing data rail, schedule the packet at:

t_arrive = t_exit + τ + d_e.

The logical departure exists once; delivery copies along outgoing rails are separately counted for work and transport cost. Birth records retain the chosen address for debugging only. Receivers may not read that metadata.

### 4.5 What a relay means

An integrating yard **decodes incoming timing into state and encodes its own state into a new phase**. It is not a transparent repeater and is not required to preserve the input timestamp.

Three unit-level checks are mandatory:

- Inputs with distinct timings can produce distinct output addresses.
- Opposing accepted contributions can cancel and suppress departure.
- Changing only an output phase can change which downstream yard responds.

Report cases where different inputs intentionally collapse to the same output. Do not describe every relay as information preserving.

## 5. Learning without a secret second model

### 5.1 Hard forward, surrogate backward

Use backpropagation through time over the full 320-tick trial. The forward pass must use the exact hard threshold, hard address, hard rail delay, and hard topology described above.

For a hard variable H and differentiable proxy S, use:

ST(H, S) = stop_gradient(H − S) + S.

Its forward value is H. Its gradient is that of S. This is a biased optimization estimator, not a physical learning law or an exact derivative of the discrete simulator.

- Threshold proxy: sigmoid((a r − h)/T_fire), initially T_fire = 0.25.
- Phase proxy: softmax(q/T_phase), initially T_phase = 0.5.
- Delay proxy: softmax(delay_logits/T_delay), initially T_delay = 1.
- Topology proxy: sigmoid(s), replacing the gradient of the corresponding per-receiver hard top-k mask.

Keep temperature settings fixed during the first run. Annealing is a later development-set option, not an unreported rescue.

The empty-load condition and external chandelier veto are hard masks. Threshold/reset derivatives use the same straight-through firing variable; do not detach the entire reset. Clamp operations have their ordinary autograd semantics.

### 5.2 Delayed differentiable scheduling

For each potential departure, write into future queue tensors for all four phase slots and all eight delay slots, multiplying the ST firing, phase selector, delay selector, mask and weight. The **forward values** populate only the selected event time on active intact rails. Backward computation can evaluate alternatives.

Use a list of immutable per-time tensors or equivalent functional accumulation. Do not mutate a tensor that autograd needs from an earlier tick. Accumulate overlapping arrival times by addition. Queue entries are already weighted rail currents I_e; apply weight and mask exactly once, not again at reception.

In the trainer, evaluate these scheduled tensor expressions at every exit for every candidate rail even when their hard forward value is zero. A Python branch that skips a nonfiring yard, an inactive edge, or an unchosen phase would also remove its surrogate gradient and invalidate this training design. The sparse JavaScript inference engine may skip those zeros because it does not differentiate.

This gives a well-defined surrogate gradient for delays and phases without pretending that discrete indices are differentiable. Skip writes beyond the trial horizon and report how many actual packets were scheduled beyond it.

A dormant edge can receive a surrogate score gradient even though it carries no forward packet. **That is nonlocal computational search during training.** Count candidate evaluations, wall time and memory. Do not claim the trainer explores only the 36 built rails.

At inference, export only hard choices and implement a sparse event queue. No proxy or dormant-edge calculation remains.

### 5.3 Initial conditions and optimizer

Use five independent model seeds. Weights start uniformly in [0.4, 1.2]; thresholds at 0.35; context U near zero (normal SD 0.05); delay logits near zero (normal SD 0.1). Initialize topology scores so H1 splits evenly between sources and every H2/output has its quota; use seeded noise to break ties.

Use Adam, learning rate 0.003, batch size 32, gradient norm clip 1.0, initially 1,500 updates. Parameterize bounded quantities with sigmoid transforms; report both trainable raw values and operational hard choices.

Check actual forward firing in a short development run. If all paths remain silent, diagnose weights/thresholds and adjust initialization on development data; record it. Do not use target-coded clock offsets, hand-assigned routes, or labels injected as packets to make learning appear successful.

No result is assumed. If optimization fails, keep and report the failure before changing the model.

### 5.4 Loss and prediction

Let C0 and C1 be the hard output **release** counts during ticks 128 through 319 inclusive. Count release at t_exit + τ, not the earlier AIS decision. Each output schedules its phase-selected unit release into its own observation queue; there is no learned final transmission delay. Its training queue uses the same all-phase ST construction as internal transport. Prediction is argmax(C0,C1); a tie is an abstention and counts as an error.

Train with cross-entropy on logits [C0,C1], using the surrogate gradients through their events. Add:

- 0.001 × mean hidden-yard events per trial;
- 0.0001 × mean logical data-rail deliveries per trial;
- 0.0001 × sum of squared context weights.

These are starting settings. CE, penalties, total events and accuracy must be reported separately. Do not call output load, a dense hidden-state classifier, or labels at intermediate yards the main output. Such readouts would answer a different question.

Checkpoint on validation accuracy, breaking ties by lower validation loss, then fewer events. Never select the seed or checkpoint by test performance.

## 6. One task that actually requires composition

### 6.1 Contextual order

The source interface emits two identical unit packets, one on A and one on B. Only source and emission time differ; learned rail delays determine subsequent arrivals.

Define order bit o = 0 for A-before-B and 1 for B-before-A. Context bit c is balanced and independent. Desired output:

y = o XOR c.

| Sequence | Context 0 | Context 1 |
|---|---|---|
| A then B | O0 | O1 |
| B then A | O1 | O0 |

The evaluator computes y. It is never part of the model input.

Context is supplied throughout the trial as a tonic one-hot cue. Both classes have the same context frequencies, event counts, amplitude, trial length and onset/gap distributions. No reward or label arrives during inference.

First-emission tick is sampled from 64–95. The second emission follows after gap Δ. Both source emissions precede the scoring window; their delayed H1 arrivals need not do so.

### 6.2 Frozen splits

Use a data RNG independent of model initialization. Before training, generate and save manifests:

- Training: onsets 64–95; gaps {6, 8, 10, 12, 14, 16, 18}; balanced four (o,c) cases.
- Validation: same support, separate event/noise seeds; 2,048 trials.
- Clean interpolation test: gaps {7, 9, 11, 13, 15, 17}; 2,048 trials.
- Noisy test: same test gaps; independent integer jitter on each sensory event, uniform in {−2,−1,0,1,2}; 2,048 trials.
- Harder timing test: gaps {4,5,19,20,21,22}; separate 2,048 trials. This is extrapolation; report separately.
- Single-token and empty trials: diagnostic counterfactuals, not part of the binary accuracy denominator.

Jitter is sampled after choosing order and cannot reverse the event order for these main-test gaps. Preserve the intended label. Report performance for every context/order cell and the worst cell, not only pooled accuracy.

Clean support is finite: repeated draws may be identical. Separate RNG seeds alone do not make identical examples independent or establish generalization. The clean test specifically holds out gap values; compute accuracy per unique condition as well as over draws, and base uncertainty primarily on independent model seeds rather than duplicated trials.

The set is intentionally tiny and synthetic. It tests mechanism and trainability, not general intelligence or a real-world application. A later three-token task can test richer memory, but is not needed for the initial handoff.

## 7. Comparisons that prevent a misleading victory

Train all models on the same examples, splits and seed count.

1. **Fixed sparse Turntable:** identical model, same 36 rails, same trainable weights/delays/context, frozen topology. Measures the benefit of choosing connectivity.
2. **Fixed-delay Turntable:** learned weights/topology/context, seeded frozen delays. Measures delay learning.
3. **Fixed-phase Turntable:** all hidden departures use τ = 0; other mechanisms train normally. Measures whether phase coding earns its cost.
4. **Conventional spiking baseline:** same 12-hidden/2-output candidate graph, incoming quotas and learned topology/delays/context. Remove phase capture (g = 1), vector state and clocks. Every tick set v* = 0.985v + Σ_e I_e. Fire at most once at that tick iff a_j v* ≥ h_j and at least two ticks have elapsed since its last firing; initial last-fire time is negative infinity. Reset subtractively: v = v* − h_j/a_j on a spike, otherwise v = v*. Do not clamp negative v. Give this baseline signed weights in [−2,2] using 2 tanh(raw), initialized positive in [0.4,1.2]; document that extra representational freedom. Context gain and threshold bounds are identical to the main model; outputs have a = 1. A spike releases immediately (τ = 0), then travels by the learned rail delay. Outputs are scored in the same release window. Use the same ST threshold temperature, delay/mask estimators, optimizer, updates, data, seeds and validation selection. Charge its actual events and work. This is a strong conventional comparator, not a rate-matched claim; the separate fixed-phase Turntable isolates removal of phase coding more narrowly.
5. **Small GRU:** receives A, B, context0, context1 each tick and emits one final binary decision. Pick the nearest hidden size that brackets the proposed model's operational parameter count; report both neighbors if no close match exists.

Distinguish trainable search parameters (90 candidate masks and delay logits) from exported inference parameters. Match/report both, rather than calling the models matched because they have the same neuron count.

Report accuracy, worst-cell accuracy, uncertainty across five seeds, parameter counts, events/deliveries, actual train/inference wall time and peak memory. Dense surrogate training cost counts even when exported inference is sparse. CPU work is not a measurement of physical energy.

Do not require every mechanism to improve this task. If a simpler version ties, recommend the simpler version for this task and retain the richer mechanism as unproven.

## 8. Growth, damage and reuse

### 8.1 Structural learning

During optimization recompute hard top-k within each incoming quota after every parameter update. Mask changes are learned rail replacements. Record births/deaths and Hamming distance of successive masks.

This is selection among 90 predefined candidates. It is not free-space neurite growth, local resource transport, or a spontaneous path-finding wave.

After training, freeze topology for all ordinary test sets. Persistent routes must survive trial resets.

### 8.2 Damage protocol

For each trained seed choose one active H1→H2 rail using **highest traffic on validation trials**, with canonical-ID tie-break. Mark it broken. Do not choose the victim from test-set behavior.

Measure:

- Accuracy immediately after the cut, before any update.
- Recovery after 25, 100 and 250 supervised updates on fresh training examples.
- A matched undamaged continuation.
- A frozen-topology control allowed to adapt weights/delays but not reconnect.
- A rewiring model with the broken edge permanently excluded from top-k.

The broken rail never secretly regrows. Replacement may use another candidate. Reserve the same quota of intact active rails after the first topology update; state clearly that the system is permitted to redeploy the material from the broken rail.

Call this **teacher-assisted adaptation after damage**. It is not autonomous healing or a stored internal goal.

### 8.3 Reuse rather than one-route replacement

Mix both contexts throughout initial training and test. A context change must select behavior without parameter updates.

Also replay a fixed set of context-0 trials before and after a block of context-1 inference trials. With trial resets and learning disabled, predictions must match exactly. This verifies frozen reusable behavior; it is not evidence against catastrophic forgetting during continual learning.

Do not turn a goal-retargeting demonstration into a claim that the network stores multiple tasks.

## 9. Synchronization is infrastructure, not payload

The clean model assumes an explicit reference. Do not claim independent clocks have learned to communicate.

For an additional robustness experiment, implement pilot-only phase correction before and during a trial. The pilot cannot add basal load or alter its timing according to the class. It is an engineered input broadcast to all 14 yards, including outputs.

For this protocol only, simulate global ticks −672 through 351, providing room for warm-up and shifted scoring. At tick −672 initialize each yard's unwrapped phase α_j independently and uniformly in [0,2π), using an exported evaluation seed. At every subsequent tick first advance α_j by 2π/32. Flag an exit if the integer cycle index floor(α_j/2π) increased under that natural advance. Then, if a pilot arrives at this tick, apply α_j ← α_j − 0.4 sin(α_j). Use θ_j = wrap(α_j) for capture at this tick, and use the already flagged exit for its AIS. For K = 0.4 this correction stays within the current cycle; it does not itself create an exit. Record initial phases and exact numerical boundary conventions in parity fixtures.

Nominal pilot arrival ticks are 32n for integer n. Deliver the 20 warm-up pilots n = −20 through −1, then continue n = 0 through 10 during the data run. The clean convention absorbs a calibrated constant pilot transit time into the source reference, leaving residual delay δ = 0. This is a supplied calibrated reference, not a learned path-delay estimator.

At global tick zero clear data state and queues while preserving clock phase; suppress an exit at zero. Context is supplied from zero onward. There is no sensor input during warm-up. Compare identical initial phases with correction K = 0 versus K = 0.4. Do not enable frequency learning or unequal nominal periods in this version.

Use common residual shifts δ ∈ {−8,−4,0,4,8}. A shared shift moves every pilot arrival and both source emissions by δ; it does not alter internal rail delays. Score output releases in [128+δ, 320+δ), using the declared external source reference in the evaluator. Report a fixed global [128,320) window as a separate latency diagnostic, never mix it into synchronization accuracy. The extended simulation horizon covers every shifted window. Do not feed δ numerically to any yard.

For independent-data-jitter tests keep δ = 0 and add independently sampled uniform integer jitter in [−J,J] to each source emission, J ∈ {0,1,2,3}. For pilot-jitter tests keep source emissions clean and add one independent uniform integer jitter sample per pilot broadcast, J ∈ {0,1,2,4}, shared across recipients for that broadcast. Use the same frozen model and manifests for both clock conditions. Report each perturbation separately, with no retraining. Pilot delivery count, the 672-tick warm-up, and extra observation time are all charged separately from the main-task inference cost.

A pilot can fix a shared reference error. It cannot restore information erased by independent message noise. Keep those results separate.

## 10. The controls from AISol

Provide interactive and deterministic checks, using the same yard engine:

- Apical context alone cannot create a departure from zero state.
- Increasing basket input reduces L and vector magnitude.
- Chandelier input blocks an otherwise valid exit without an extra state reset.
- Removing the veto can allow a later departure from retained state.
- Scrambling phase addresses after training changes only the release rule, not wiring or weights.

The model does not need trainable basket/chandelier populations to demonstrate the main architecture. Their first role is clear causal control. This avoids adding loosely constrained machinery before the core computation works.

## 11. Success criteria and permissible conclusions

These are proposed engineering targets, not predicted results.

**Working integrated demonstration:**

- Median clean interpolation accuracy ≥95% across five model seeds.
- Median independent-jitter accuracy ≥85%; report each cell.
- All seeds, including failures, reported.
- A saved hard model gives identical decisions in Python and JavaScript on all 2,048 clean interpolation trials.
- Deterministic fixture events match exactly in time, chosen phase and active edge. Floating states agree within an explicit numerical tolerance.
- Only active intact rails carry forward packets; quotas and causal scheduling never fail.
- Browser runs the actual exported model, not a scripted animation.

**Evidence that the extra architecture helps:**

Compare the paired-seed results against learned-delay spiking and fixed-phase controls. Seek a consistent accuracy/robustness benefit or lower exported work at similar accuracy. Report paired differences with uncertainty; five seeds are exploratory, so avoid declaring a broad statistical breakthrough.

A clean task solved by all models establishes feasibility only. A failed hypothesis is a valid result. Do not increase task difficulty repeatedly until one favorite wins without labeling that as development.

## 12. Implementation package Sol should deliver

Recommended additive directory: `circuits/`.

| File | Responsibility |
|---|---|
| `DESIGN.md` | This specification, with any implemented deviations documented |
| `config.json` | Every default, split seed, quota, phase slot, optimizer setting |
| `task.py` | Balanced task generator and frozen split manifests |
| `model.py` | Exact hard forward and surrogate backward |
| `train.py` | CPU training, checkpoints, resumable RNG/optimizer state |
| `evaluate.py` | Frozen evaluation, baselines, damage and ablations |
| `export.py` | Hard model export with schema version and provenance |
| `circuit_core.js` | Dependency-free hard inference for Node and browser |
| `index.html` | Offline laboratory loading an embedded exported model |
| `tests/` | Scheduling, channels, learning plumbing and parity fixtures |
| `results/` | Per-seed raw results, summaries, manifests and hashes |

The HTML should embed the default model or load it through a file picker. Do not rely on `fetch('./model.json')` when opened as `file://`. Keep any required JS beside the HTML and provide a ZIP for offline use.

Export: schema version, all operational values, active edges/delays, port assignments, clock/reset semantics, score window, model seed, source commit, checkpoint hash and training configuration. Never export an optimizer proxy as though it were the operational model.

### Build order

1. Implement the exact hard reference, simple deterministic fixtures and the context/order generator.
2. Add surrogate learning and train the full model plus fixed-phase and scalar-spiking controls. Inspect failures before UI work.
3. Export to JavaScript, establish parity, then finish the browser lab and remaining comparisons.

Do not spend the first day building a large animated railway. A small working circuit with truthful replay is the deliverable.

### Browser interaction

Show the 14 yards, active rails, four phase slots, the two sensory trains, context, and output prediction. Dormant candidates may appear only in an optional overlay. Width means weight, color indicates event phase, and a small badge shows delay; avoid making visual track length imply an unmodeled physical quantity.

Controls: choose AB/BA, flip context, change gap, add independent jitter, freeze/step, cut a rail, restore saved model, and replay a recorded training checkpoint. Show expected versus actual output, abstentions, packet count and current active-rail count.

Training need not run in the browser. Label replayed training as replay; do not make animation imply live learning.

## 13. Verification that matters

Use purposeful tests, including:

- A one-packet route with exact known arrival ticks and no zero-delay feedback.
- Dormant/broken edges deliver nothing; recurrent loops cannot update within the birth tick.
- Address selection near each quadrant and deterministic tie behavior.
- A cancellation fixture and an apical-alone fixture.
- Hard forward output unchanged when surrogate temperatures change.
- Nonzero finite surrogate gradients for a hand-constructed threshold/phase/delay example; do not compare them with finite differences of the discontinuous hard simulator.
- Topology updates stay inside incoming quotas and never select broken edges.
- Context/order balance and absence of label leakage.
- Checkpoint resume reproduces subsequent training on a tiny deterministic run.
- Python/JavaScript packet-by-packet parity and full test-set decision parity.

If float32 rounding changes a hard decision, preserve and diagnose the offending case. The reference remains float64; do not conceal discrepancies with an accuracy-only comparison.

## 14. Limits and prior work

The proposed system is a phase-coded recurrent spiking architecture with learned sparse connectivity. It does not escape ordinary dynamical-system descriptions, and established neural models may reproduce its behavior.

Relevant prior work:

1. Eugene Izhikevich (2006), *Polychronization: Computation With Spikes*. Delays and plasticity form temporal firing groups. https://www.izhikevich.org/publications/spnet.htm
2. Hammouamri, Khalfaoui Hassani and Masquelier (ICLR 2024), *Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings*. https://proceedings.iclr.cc/paper_files/paper/2024/hash/4df1cc5a7528b7197ad8ae76ff30107a-Abstract-Conference.html
3. Mészáros, Knight and Nowotny (2024), *Learning Delays Through Gradients and Structure*. Delay learning and dynamic pruning/reconnection are already studied together. https://arxiv.org/abs/2407.18917
4. Vassallo and Taherinejad (2026), *Three factor delay learning rules for spiking neural networks*. Relevant future comparison for online/local credit. https://arxiv.org/abs/2601.00668
5. The six Antti repositories listed in Section 2; their previous measurements are background, not results of this design.

Do not claim novelty merely for phase coding, XOR by vector cancellation, learned delays, structural plasticity or contextual gain. Any contribution must identify the specific mechanism, its measured effect, its computational cost, and its limits.

The geometry here is candidate adjacency plus explicit delay. Feynman path sums, torsion-free groups, internal goals, free-space growth and biological realism are not requirements for this experiment.

## 15. Direct handoff to Sol

Build the model in this document as one coherent experiment. Start in TurntableAISol on an isolated branch, preserve the existing lab, and keep the new core under `circuits/`.

The essential requirement is that input timing changes a yard's internal vector, that vector selects an actual departure phase, and the next yard receives only the resulting timed unit packet. A fixed material budget selects which directed rails exist; delays and weights are trained; context changes gain without directly supplying an answer.

Use supervised surrogate training openly. Match the hard browser runtime to the hard training forward pass. Report all five seeds and comparisons, including failures. If training does not work, deliver the diagnosed failure and checkpoints instead of replacing learning with hand-coded routing.

The completed artifact should let Antti flip the context, swap the train order, disturb the timing and cut a rail—and see the actual learned computation respond.

**The railway is useful when it learns what should happen next, and its learned structure survives the passing train.**

## November 8 rescue experiment (stimulus-locked clock)

- Added an optional `clock.mode="stimulus_reset"`: an explicit pilot at r=min(a_tick,b_tick) reaches all 14 yards, and phase origin and exits move to r. Default remains the absolute clock; original fixtures and tests are unchanged. The reset is mirrored in `circuit_core.js`, has 12 new parity fixtures, and passes trained-export parity on 512 trials.
- Seed 11, identical batches, 200 updates. Order-only, fixed onset: 50.0% (one class fires O0, the other is silent). Order-only, random onset: absolute clock 0.0% (total silence) with or without penalties; reset 19.7% with penalties and 50.0% without (one-sided code). Reset without penalties at 600 updates: unchanged, gradient 0.000. Success criterion not met.
- Empty-yard mask probe: removing it changes only threshold gradients (spurious). The zero-gradient cause is ∂r/∂z=0 at z=0 on fully silent trials. The mask is left unchanged. See `RESCUE.md`.
