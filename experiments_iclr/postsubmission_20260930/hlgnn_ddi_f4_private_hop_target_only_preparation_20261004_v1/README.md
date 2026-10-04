# Minimal F4 private-hop target-only DDI source

This source preparation adds one fixed four-member target-only control alongside the saved native HL-GNN DDI recipe. It performs no launch, fitting, tuning, additional numerical check or DDI payload read. Release remains disabled. The separately owned DDI TRAIN+VALID artifact work is not modified or claimed complete here. The frozen Collab family is unchanged.

## Fixed architecture and objective

Use one learned node-ID embedding matrix and the exact verified `SharedPowerHLGNN` implementation with four private coefficient rows. One common input dropout context and one fixed normalized TRAIN operator supply the shared augmented powers `[X, 1], P[X, 1], …, P^15[X, 1]`. Each route has private input/output feature factors and a bias offset around one shared dense map/bias. The constant channel retains exact affine bias propagation when P does not preserve constants. Coefficients are signed, unconstrained, unnormalized trainable parameters initialized with native KI alpha=0.5. There is no cross-call cache.

Each route has a private native two-layer Hadamard pair MLP, with the native width 512 and dropout 0.3. These heads receive ordinary fresh independent initialization and retain their native per-head dropout. The encoder's input dropout is shared. Identity affine factors, zero private bias offsets and identical KI coefficient rows initialize identical encoders; different heads can break output symmetry. This does not establish useful learned diversity.

TRAIN supervision is the arithmetic mean of four **native AUC sums** on the same positives and three native sampled negatives. The native nominal `loss_func="WeightedHingeAUC"` dispatch is retained. With no TRAIN weights, `margin=None` falls through to unchanged native `auc_loss`: `sum((1 - (positive - negative))^2)`. No hinge, sigmoid, fabricated unit weights, degree margins, per-example mean or conditional objective replaces that branch. The F4 entry point explicitly requires the missing-weight contract.

Serving uses the uniform mean of four raw pair scores. Counts, responsibilities and auxiliary outputs are absent. There is no conditional loss, entropy/diversity regularizer, peer-gradient suppression or detached shared embedding.

## Native DDI recipe retained

The recipe stays at learned width 512, K=15, KI alpha=0.5, input/predictor dropout 0.3, Adam learning rate 0.001, gradient clip 2, batch 65,536, three global negatives, full 500 epochs, no learning-rate decay and fresh seeds 0/1/2. The native global sampler is called once per epoch and its candidates are shared across routes. The native shuffled loader and final partial batch are retained. Native clipping is applied to the aggregate encoder group and aggregate predictor group; embeddings are included in Adam and remain outside those clipping groups. Group clipping across four routes is an explicit multi-member adaptation.

The TRAIN graph includes native targets and receives no per-batch target removal, VALID insertion, graph augmentation or new edge weights. The unchanged baseline artifact loader accepts only TRAIN plus fixed VALID and preserves the owning graph/candidate contract. The separate artifact preparation declares no graph or TRAIN weights and the native AUC branch; no artifact is opened by this packet.

Every fifth epoch evaluates the complete fixed native VALID positive/global-negative pools through OGB Hits@20/50/100. **VALID Hits@20** selects the state on strict improvement; the earliest exact tie wins. TEST is unavailable. The unchanged baseline `run_seed` creates a fresh model, embedding, heads and Adam per seed, runs the full recipe, and writes its selected encoder/head/embedding state. It never resumes or loads a fitted/resource state. The selected weights are not an exact optimizer/RNG continuation checkpoint.

`baseline_train_ddi.py`, `BASELINE_CONFIG.json`, and all `native/` files are exact copies of the saved native DDI preparation. `train_f4_ddi.py` is a thin factory entry point using the unchanged baseline loader and fit driver. `f4_model.py` replaces only the multi-member model/training/serving pieces. The original baseline entry point remains usable with its separate disabled baseline configuration and separate output directory.

Seed labels alone do not align random draws across native and F4 fits: extra head initialization and dropout advance RNG differently. Native and F4 share candidates within their respective epoch, not an externally frozen common candidate bank across fits. The parameter and compute budgets are different and must remain explicit.

## Evidence and limits

The single completed CPU float64 equivalence run is copied as compact evidence and hash-bound in `INPUT_BINDINGS.json`. It passed 109 assertions for synthetic outputs and input/all-parameter gradients, including private hop coefficients, irregular weighted P with nonconstant P1, bias and native M=1 reduction. That evidence applies to the unchanged encoder factorization with tied parameters/contexts; it does not validate this four-head trainer, float32 GPU execution, a DDI graph, speed, memory scaling or prediction quality. No extra equivalence run was performed for this packet.

This is an implementation of existing factorization/member ideas. It establishes no methodological novelty, specialization, quality benefit or architectural necessity. A later conditional objective can be compared on the same private-hop parameterization and target/candidate recipe, with counts excluded from served scores; none is implemented here.

The disabled `config.json` must be bound by the owning root to the actual artifact and existing execution controls before any fit. This packet adds no resource-only qualification ladder, transport client or executable launch. `PROPOSED_COMMAND.json` is a disabled argv specification only. Local stdlib AST/hash checks and the sealed manifest preserve source provenance; they do not certify training readiness or completion.
