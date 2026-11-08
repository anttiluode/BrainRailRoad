# Implementation ledger (read this before interpreting results)

- The user's repository `anttiluode/BrainRailRoad` is used instead of the design's suggested isolated branch in `TurntableAISol`. No older repositories are touched.
- The exact reference is in `model.py`; the PyTorch ST variant is separately in `model_torch.py` to keep the independently inspectable float64 causal reference free of trainer logic.
- All 90 candidate edges exist at training time; 36 hard active edges alone transport actual events. The trainer computes and differentiates alternative phase/delay/topology schedules even when the hard event count is zero.
- The clean model uses its supplied 32-tick phase reference. The independent pilot PLL experiment from design §9 is **not yet implemented** and is not claimed.
- The fixed-phase and spiking ablations have initial training plumbing, but the complete baseline suite, exact spiking browser decoder and held-out benchmark have not yet been signed off.
- The browser's initial seed-11 model is **random untrained**. It should not be described as a working learned classifier.
- Three development-only threshold initializations (`0.35`, `0.20`, `0.16`) were tried on seed 11 with 100/200/150 optimizer steps respectively. All subsequently approached silence/ties, a negative result. Test manifests were not used to tune them.
- Training checkpoints are PyTorch-specific and not inference exports. Only the bounded hard JSON after training is suitable for the JavaScript engine; each export includes an origin tag.
- All timestamps and model-seed results are deterministic. The existing task's four scoring cells are A/B order × two contexts, with no target or label feed-in.
- A future successful experimental revision must update this ledger instead of replacing the failed history.

## November 8 GRU development control

- A four-input-channel, 977-trainable-parameter GRU was run as an independent task sanity check on five seeded models, 700 updates per seed. The first tick-319 readout probe remained near chance. Changing the only readout to tick 127—after every possible sensory emission, before the railway scoring window—allowed this baseline to learn. This is explicitly *not* a claim of parameter/work parity or a learned railway.
- Checkpoint selection used the first 512 validation examples; all four 2048-item frozen manifests were evaluated only afterward and are reported per seed and per order/context cell. No model was selected by test outcomes.
- An extra output coverage penalty tested on the hard railway (seed 11, initial threshold 0.16, 400 development updates) did not fix training: 128-example validation accuracy peaked around 34% early and was 18% at step 400, with frequent abstentions. It is not incorporated into the default trainer or any successful claim.
- The main design's five-seed, 1500-update hard-event experiment, clock-pilot intervention, lesion recovery and matched spiking/GRU comparisons have not been completed. The verified hard reference remains distinct from a successfully optimized circuit.
