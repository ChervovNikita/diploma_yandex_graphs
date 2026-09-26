# Optimizer aggregation diagnostic: compact supplement

Run from the supplement root:

```bash
python3 experiments_iclr/optimizer_history/verify_compact_optimizer_v2.py
```

Python 3 and NumPy suffice for this compact verifier. A successful run prints
`"status": "PASS"`, covering all 108 fixed validation cells (96 mechanism
and 12 norm-matched V2) and all 60 allowed held-out score cells (48 mechanism
and 12 V2). It checks source/freezes and graph fingerprints recorded in the
locks, reconstructs validation checkpoint/candidate selection from retained
1,000-epoch traces, binds each score to its independent audit and lock, and
recomputes official-split hard-class accuracies from retained decisions.

`MECHANISM_PROTOCOL.md` and `NORM_MATCHED_SYNC_V2_PROTOCOL.md` state the
prospective designs. `FROZEN_STUDY.json` lists SHA-256 hashes of the Cora
Planetoid public, WikiCS official split 0, Actor Geom-GCN split 0, and
Platonov filtered Chameleon split 0 raw inputs, plus hashes of the derived
features, labels, edges, and masks. `tuning.py` defines the exact loading and
edge transformation. Reproducing training requires recovering those exact
published data bytes under its `data/` paths and passing its `check-freeze`
gate; this package does not ship raw graph inputs. The training code needs
NumPy, PyTorch, PyTorch Geometric, and a CUDA-capable GPU. Its source and data
are hash-frozen; exact Python/CUDA/library versions are not pinned by the
compact package, so record the rerun environment. The protocol files give
the fixed 1,000-epoch run, validation-lock, then test-score sequence.

The mechanism study compares default TIED, UNTIED, and synchronized copies
under matched initial member functions and training conditions. SYNC also
has a six-candidate validation grid, so its tuned comparison is descriptive
and has unequal search effort. NORM-SYNC V1 stopped at a training-only Cora
finite-precision norm-gate failure before held-out scoring. The separately
frozen V2 keeps the original norm tolerance and uses a bounded FP32 rounding
correction; all 12 V2 cells were run and audited. The original failure record
and both protocols remain in this package. A post-score compact-verifier
schema repair accepted the mechanism auditor's intentionally absent top-level
`protocol` field while retaining freeze/lock and every per-score protocol
check. It changed no training, selection, predictions, or independent audit.

The compact package omits checkpoints, float logits, and raw graph files to
fit the submission limit. Its verifier therefore cannot independently rerun
model inference, inspect optimizer moments, or recompute cross-entropy from
float logits. Full checkpoint/logit replay and collapse checks were performed
in the independent author audit; hashes bind their outputs to this package.
The 3-seed results use one fixed split per graph and do not establish a
general accuracy advantage for optimizer synchronization or norm matching.
