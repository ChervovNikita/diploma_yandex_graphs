# Frozen-basis PPI predictive development study v1

The purpose is to test whether the fixed train-only spectral bases improve held-out graph predictions after ordinary AdamW training. The preceding D4/E4 selection and F2 projection audit measured a local gradient proxy, not predictive performance. F2 favored the selected basis under the 10% label intervention in the two nonzero-rank seeds, but the raw-gradient basis won in all three full-label seeds. That adverse evidence remains part of this study's provenance.

Each of the six existing seed/label cells has four arms: tied, selected, random and raw. All start from exactly the same recorded P-only epoch25 source tensors. Selected/random/raw use the existing basis bytes and exactly the existing selected ranks; tied uses rank zero. No new basis, rank, target, pooling rule or hyperparameter is selected here. The seed1/10% cell has rank zero for every arm and remains in all tables. It is an identity control, not a missing result.

The architecture is the original four-member GNNM SAGE model: three width128 residual blocks, hidden multiplier1, shared LayerNorm, original biases, dropout0.2, mean-member BCE loss and raw-logit averaging. Only the three neighbor maps receive frozen output bases with zero centered trainable coefficients. The source, root maps, normalization, FFNs and boundary projectors are all trainable. Every arm gets a fresh AdamW(lr0.001, weight_decay0, eps1e-8, betas0.9/0.999), with no imported moments or steps. This tests the stated adapter under AdamW; no projected-SGD trajectory identity is claimed.

## Equal representative budget and validation policy

- All twenty official training graphs, all observed labels from the original nested masks, seeds0/1/2, label fractions0.1/1.0, float32, no AMP.
- Exactly100 additional epochs, one complete graph per update, 2,000 updates per arm. The warm start has already cost250 updates on P10. The100-epoch cap is an exploratory equal budget, not evidence of convergence or an arm-specific stop. No early stopping or hyperparameter search is permitted.
- Graph order is a Python shuffle seeded19709+1000*seed+epoch. Before each graph update reset the torch CPU and visible CUDA RNG to67001+10000*seed+20*epoch+official_graph_index. This couples dropout draws across arms without reusing the P25 optimizer state.
- Every completed epoch scores the complete official validation split. Epoch0 is also scored and is eligible, so degradation relative to the common source is visible. Select maximum pooled micro-F1, then minimum pooled BCE, then earliest epoch. Retain all101 validation member-logit files and metric rows, every improving checkpoint, epoch100 state, optimizer states, masks, source hashes and RNG evidence. No test loader, test scoring or test-byte hashing is implemented. Existing public cache availability is not a cryptographic test seal.
- The primary development comparison is selected against each of tied/random/raw at10% labels. Full-label results are mandatory adverse controls and cannot be omitted or used to revise this rule. Report each cell, selected-checkpoint and fixed-epoch100 contrasts, including every loss/failure and the rank-zero cell.
- Continue this particular hypothesis to independent-task confirmation only if all24 fits and numerical replays pass; the mean selected-minus-control validation micro-F1 is at least0.005 for **each** of the three controls at10%; both nonzero-rank low-label seeds improve against each control; no low-label paired difference is below-0.002; and selected pooled BCE is no worse on average than each control. These are resource-allocation criteria, not a significance test or publication claim. A failure retains all records and triggers a new explicit prospective idea, not silent tuning of this study.
- A primary arm whose selected epoch is90 or later is flagged as budget-limited if its mean validation F1 over epochs96--100 exceeds epochs91--95 by more than0.0001, or its mean pooled BCE falls by more than0.00001. Any such flag blocks promotion: a truncated control trajectory is not evidence of success. A later budget extension must be registered separately and give all four arms the same extra budget.
- Report the full-label direction and magnitude separately regardless of the gate. Three optimizer seeds on one training/validation graph partition do not establish generalization across tasks or a well-powered test. A successful development gate still requires a separate new dataset and prospective confirmation before claiming method superiority.

## Engineering and cost gates

One CPU preflight covers all six cells/four arms on official training graph0, verifies bit-identical initial eval and dropout-enabled outputs, unchanged source tensors/RNG on construction, fresh empty AdamW states, finite one-step updates, frozen bases/contrast buffers, centered corrections, output dimensions and actual parameter/buffer/optimizer bytes. It scores no validation/test data. This is24 engineering updates, not a predictive fit.

GPU fitting requires the passed preflight, a runtime freeze matching exact sources and input evidence, the repository cache wrapper, and an explicitly named currently idle GPU checked before creating a CUDA context. Only the coordinator launches jobs. The script cannot download/process datasets and refuses existing output directories.

Each arm retains final fit/evaluation time, actual selected-model latency and memory, model/optimizer/basis bytes and the fixed full-width four-member propagation count. The three correction arms have matched capacity. No activation or propagation reduction is claimed. Charge the common P-only fits, D/E collection, F2 audit and their replay/supervision records separately; shared preprocessing must not disappear from total method cost or be charged four times as though independently repeated. Preparation freezes, preflights and repeated engineering checks are separate costs. Latency profiling uses training graph0, eval mode, two warmups and ten timed full member forwards; it is an engineering workload, not a benchmark-wide claim.

Each validation checkpoint also records equal-graph mean-member BCE, equal-graph pooled BCE and their difference. BCE convexity makes this Jensen gap nonnegative; it separates the member objective into pooled predictive loss and disagreement. The gap does not depend on the binary targets, does not establish beneficial diversity, and is not a novelty claim. Descent in member loss without pooled predictive improvement cannot pass the development gate.

## Coordinator execution

From the phase root, first transfer the exact new directory and prepared protocol. Source `protocols/repo_env.sh` and use `protocols/run_logged.py` for runtime commands.

```bash
python spectral_privacy/ppi_training_v1/prepare.py --output spectral_privacy/ppi_training_v1/PROSPECTIVE_PROTOCOL.json
python spectral_privacy/ppi_training_v1/runner.py preflight --protocol spectral_privacy/ppi_training_v1/PROSPECTIVE_PROTOCOL.json --output spectral_privacy/ppi_training_v1/preflight_run01
python spectral_privacy/ppi_training_v1/runner.py freeze-runtime --protocol spectral_privacy/ppi_training_v1/PROSPECTIVE_PROTOCOL.json --preflight spectral_privacy/ppi_training_v1/preflight_run01/PREFLIGHT.json --output spectral_privacy/ppi_training_v1/RUNTIME_PROTOCOL.json
```

For every seed0/1/2, label10/100, and tied/selected/random/raw arm, supply a fresh fit output directory and a single authorized visible idle GPU:

```bash
CUDA_VISIBLE_DEVICES=GPU_ID python spectral_privacy/ppi_training_v1/runner.py fit --protocol spectral_privacy/ppi_training_v1/RUNTIME_PROTOCOL.json --seed 0 --label-fraction 0.1 --arm selected --device cuda:0 --physical-gpu GPU_ID --output spectral_privacy/ppi_training_v1/fits/seed0_label10_selected_run01
python spectral_privacy/ppi_training_v1/runner.py replay --protocol spectral_privacy/ppi_training_v1/RUNTIME_PROTOCOL.json --run spectral_privacy/ppi_training_v1/fits/seed0_label10_selected_run01 --output spectral_privacy/ppi_training_v1/replays/seed0_label10_selected_run01.json
```

The replay recomputes every epoch metric from saved logits, verifies selection and basis/contract identity, then performs a fresh CPU forward from the selected checkpoint. It is a numerical checkpoint replay, not an independent fresh training reproduction. Its stored-logit metric tolerance is2e-5 and fresh forward tolerance is2e-4 absolute/relative; report observed differences and threshold changes explicitly.
