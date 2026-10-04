# HL-GNN DDI fresh-training source preparation

Date: 2026-10-04. Source preparation completed; **release disabled, dataset/runtime unqualified, no training or scoring**.

The runnable entrypoint is `train_ddi.py`; its `native/` modules retain the pinned author training recipe. `config.json` fixes three prospective independent seeds `[0, 1, 2]`, each with the entire author budget. The seeds are an explicit prospective replication choice, not an author seed recommendation or a tuning grid. No new GNNM variants or auxiliary losses are implemented.

Author source: [LARS-research/HL-GNN](https://github.com/LARS-research/HL-GNN/tree/0855b0de74a8f0586b8cc203e9ba4dbbb57243f4), commit `0855b0de74a8f0586b8cc203e9ba4dbbb57243f4`. Seven author README/source files were copied byte-for-byte from the preceding scout's pinned evidence and independently rechecked against their recorded Git blob and SHA256 identities. `pinned/` is archival evidence; execution uses `native/` and the new entrypoint. `PINNED_SOURCE_MANIFEST.json` records those identities and the reused evidence location. `diffs/` contains exact unified diffs for every native module and for the replaced entrypoint.

| Native recipe retained | Setting |
|---|---|
| DDI command overrides | Embedding/encoder/predictor widths 512, three negatives, dropout 0.3, loss label `WeightedHingeAUC` |
| Encoder | HLGNN; learned linear input map; 15 propagation steps; KI coefficients initialized with alpha 0.5 |
| Predictor | Two-layer Hadamard MLP; raw pair score |
| Input | Fresh learned node-ID embeddings; no node features or pretrained embeddings |
| Optimizer | Fresh Adam at learning rate 0.001; no learning-rate decay |
| Training | **500 epochs, batch 65,536**, full epoch traversal, native gradient clipping at 2.0 for encoder and predictor |
| Native sampling | Global PyG sparse nonedge sampling, self-loops excluded, three negatives per TRAIN positive; fresh once per epoch |
| Graph recipe | TRAIN adjacency includes current targets; no target masking, VALID-edge insertion, or random-walk augmentation |
| Evaluation | Every five epochs, same graph encoder and pair MLP, fixed native VALID candidates |

The author HL-GNN batch is independent of NCNC's separately planned batch 24,576. No shortened budget or hidden early stopping is provided. The initial disabled configuration is rejected before numerical imports or artifact access. To release later, the owning qualification must set the artifact contract and gates described in `ARTIFACT_CONTRACT.md`; training is then invoked with `python3 train_ddi.py --config config.json`. That invocation was **not** run during this preparation.

## Explicit adaptations

1. **Fresh model and optimizer per seed.** RNG seeds are set before constructing `BaseModel` inside `run_seed`. Its constructor creates fresh embeddings, HLGNN linear map, predictor and Adam state. The author's `param_init()` still runs. This preserves the native single-run initialization behavior while avoiding cross-run reuse: the native HLGNN reset omits `lin1`, and native main constructs the optimizer before its run loop. No checkpoint or pretrained state is loaded into a seed.
2. **TRAIN/VALID input interface.** Direct OGB dataset loading is replaced with the later strict tensor artifact, so the runner cannot request the held TEST split. The native sparse conversion and adjacency-to-edge-index reconstruction remain. Split weights remain untouched; any graph weights receive the same float conversion as native main. Graph and candidate provenance need the owner's later qualification.
3. **VALID-only selection.** The native TEST-capable evaluator is replaced with `BaseModel.validate`; the unused TEST-capable utility evaluators are removed. VALID Hits@20 chooses one checkpoint per seed; strict improvement retains the earliest exact tie. Hits@50 and Hits@100 are recorded as secondary VALID diagnostics. The source reports per-metric validation-selected results rather than writing checkpoints; this adapter adds a native-ranker checkpoint export. All 500 epochs continue after selection improvements.
4. **Local imports and final-singleton prediction repair.** Author wildcard imports become package-relative. `batch_predict` changes `squeeze()` to `reshape(-1)`, keeping a one-element final batch concatenable without altering the pair computation. No numerical equivalence test was performed.

## Actual objective and serving

The loss name does not determine the actual branch by itself. The preserved `BaseModel.calculate_loss` and `train` use `WeightedHingeAUC` only when TRAIN includes `weight`. With positive score `p`, grouped negative scores `n`, and weight `w`, the native branch sums `w * max(w - (p - n), 0)^2`. With no TRAIN weights it falls back to AUC, summing `(1 - (p - n))^2`. Neither branch is replaced by BCE, normalized, or assigned synthetic margins. Each run records the actual branch.

The native sampler excludes the TRAIN graph and self-loops; it does not exclude VALID or held positives from its nonedge pool. That source behavior is retained. The sampler's shortfall-copy behavior is also unchanged and needs later runtime qualification on the native graph. The same full TRAIN adjacency remains in the encoder during fitting and validation.

Serving remains the HLGNN encoder followed by the Hadamard MLP with dropout disabled during validation. Its propagation normalization, coefficient dtype and weighted power sum are unchanged. No auxiliary counts, responsibilities, member bank, or candidate-batch normalization is added. `valid_best.pt` exports only the selected native encoder/predictor/embedding and identifying metadata, without optimizer state; it is a serving checkpoint, not a resumable training artifact. No served TEST predictions are produced by this adapter.

## Checks and unresolved dependencies

`python3 check_source.py` passed. It uses only stdlib AST and text/hash operations: all 14 Python sources parse, seven pinned files match their source identities, native initialization/training/loss/sampler/encoder/predictor bodies are preserved, the configuration matches the pinned README command plus main defaults, fresh construction is inside the seed call, checkpoint selection follows VALID evaluation, and the exact diffs match. `SOURCE_CHECKS.json` contains the result. These checks do not certify tensor behavior, reproducibility, runtime feasibility or numerical equivalence.

Before execution the owner still needs:

- The declared TRAIN/VALID artifact and its native graph/candidate/weight provenance. The separate root TRAIN census is not assumed complete and was not inspected here.
- A qualified Python environment (the entrypoint uses Python 3.11+ `hashlib.file_digest`) with compatible PyTorch, NumPy, PyG, torch_sparse and OGB plus required compiled PyG backends. No environment was installed or imported. The pinned requirements file's tree identity was available in the scout, but its contents were not retained; exact author dependency versions are not claimed.
- GPU memory/cost feasibility for full-graph HLGNN propagation on every batch with the author batch 65,536 and all three complete 500-epoch seeds. The configured `cuda:0` is a later runtime choice; there is no automatic CPU fallback.
- Authorization and qualification for the eventual execution and output custody. The adapter creates a new output directory, refuses overwrite, records incomplete runs as `in_progress`, and marks completion only after the full epoch budget. No execution release is granted by this packet.

Potential later extension points are the native embeddings/encoder features within training and the loss calculation. Any auxiliary adaptation must retain the native ranker, native loss and sampler, fixed candidates, full budget, and fresh initialization. No such extension is coded here.

All writes and checks stayed within this new directory. No server, numerical import, data/model/checkpoint read, training/scoring launch, PDF compilation, GENLINK, Desktop access, external message, or additional agent was used.
