# Prospective DDI 100-epoch development pilot

This protocol is fixed before any DDI VALID predictive score. Its purpose is to test whether the exact cross-side coupling law transfers to served rank quality on DDI. It is a development comparison on one graph. The original sealed nine-cell 500-epoch proposal is preserved and remains disabled.

## Budget amendment from actual TRAIN cost

Pinned root-adopted runtime receipts establish one complete fresh seed-0 TRAIN epoch for each F4 arm: target-only 12.3811 s, joint 226.0748 s and separate 212.4089 s. Each has 17 updates,1,067,911 records and the 19,335-record tail; all common receipts and joint/separate auxiliary receipts align. No VALID score, TEST data or checkpoint was produced; those runtime weights were discarded. The adoption and individual receipts are hashed in `INPUT_BINDINGS.json`.

Linear scaling of these co-resident measurements gives 187.8603 GPU-hours for the original nine-F4 500-epoch proposal and 37.5721 GPU-hours for the nine-F4 100-epoch portion of this pilot. Native M1, VALID and replay costs are unmeasured. Approximately 38–40 serial GPU-hours is preliminary planning, with no guaranteed forecast: later epochs, contention and standalone throughput remain uncertain. The prospective 100-epoch amendment is based exclusively on this authenticated TRAIN cost. It is shorter than the author's 500 epochs and cannot be described as author-budget parity or as a substitute full-budget result.500 epochs remain conditional competitive confirmation if the fixed development rule passes.

## Fixed family and inherited scientific computation

The family has 12 fresh fits: seeds 0,1,2 × native M1, F4 target-only, F4 joint and F4 separate. Every fit executes 100 full epochs, including all 17 native updates per epoch and every tail record. There is no early stopping, epoch selection by runtime, pilot subset, grid, donor, pretrained state or state reuse from the runtime checks. Every scientific fit constructs fresh learned node embeddings, encoder, predictor and Adam after resetting its seed.

Native M1 calls the unchanged pinned author `native.model.BaseModel`. The three F4 arms use the exact sealed v2 `F4 ConditionalPatternModel`, its pinned original F4 owner sources, complete support/query module and byte-identical exact conditional loss. No model/core is copied or edited. The F4 architecture stays shared 512-dimensional learned node-ID embeddings, shared affine map/private input-output factors and bias offsets, four private signed KI rows over 15 shared augmented powers and four private native two-layer 512-wide Hadamard MLP heads. All serve complete TRAIN through the uniform mean of raw pair scores. Native M1 serves its one inherited raw head. No counts, teacher bits, responsibilities or diagnostic output changes served ranks.

The only author recipe amendment is epochs 500→100. Adam lr 0.001, dropout 0.3, clipping 2, native batch 65,536, three global negatives per positive, KI initialization, no node features, no LR decay, no VALID graph insertion and all widths/layers remain exact. Missing TRAIN weights retain the actual native AUC dispatch: squared-margin sums, uniformly averaged across the four F4 members. Encoder and predictor clipping groups remain native; learned embeddings stay in Adam outside those groups. Config retains the nominal `WeightedHingeAUC` name while recording actual AUC.

The F4 auxiliary stays exactly 64 stratified query positions per update (32 positive and 32 from the already drawn native negatives), lambda 1 in mean native-margin units, complete residual supports, both endpoints excluded, whole-positive-minibatch plus query-identity mask, complete-TRAIN detached teacher, same dropped input and owned scorer RNG. For side losses a_m,b_m and d=max(n_L+n_R,1), joint is `-log(mean_m exp(-a_m-b_m))/d` and separate is `[-log(mean_m exp(-a_m))-log(mean_m exp(-b_m))]/d`. Native target loss T remains its original sum reduction; auxiliary arms use `T+(3 B)*A` with the existing half-positive/half-negative means. All deterministic, zero-count, full-count and empty supports remain. The source uses the existing 8192-slot checkpoint chunks and exact ESP arithmetic. There is no automatic adjustment after outcomes or cap failures.

## Per-cell execution and admission

`train_cell.py` launches exactly one declared arm/seed cell. Required output is a new absolute writable project directory ending `FAMILY_ROOT/ARM/seed_SEED`, outside sealed source. Numerical/model imports and artifact reads follow the disabled release guard and explicit root admission. Root may change only `release_enabled` and `root_admission` in an external release config; scientific fields must serialize exactly as canonical config.

The driver verifies the sealed v2/original-native source pins and qualified TRAIN/VALID artifact/report hashes. It uses the unchanged native loader and `baseline.run_seed` with the prospectively fixed 100-epoch recipe. It deliberately does not call the old full 500 release gate or set an unmeasured full 500 qualification true. Root admission declares scheduling, CUDA allocator fraction, host/wall caps, supervisor and capacity evidence. It authorizes execution under those conditions; it does not establish empirical 100-epoch or 500-epoch feasibility. The driver sets the admitted CUDA allocator fraction; root owns host/wall/capacity supervision.

Default admission request is a 0.30 CUDA allocator fraction,32 GiB host limit and 8 h per-cell wall limit. Source approval remains false. A technical failure or cap stop leaves an incomplete cell and retained receipts. There is no automatic retry, shortened fit, replacement seed, favorable subset or success-only mean. Further technical recovery must preserve the original record and be specified by root prospectively.

## VALID selection and complete comparison

Every fifth epoch traverses all official fixed VALID positives and the inherited shared negative pool with OGB Hits@20/50/100. Select Hits@20 by strict improvement; retain the first exact tie. After all 100 epochs, reload only the current cell's own selected state and perform one complete VALID replay. No passive diagnostic selects a state. Each fit has 20 training VALID traversals plus one own selected replay,21 total; the family has 252. TEST has no loader or score path.

Primary is joint minus separate. Secondary contrasts are joint minus F4 target-only, separate minus F4 target-only, and each F4 arm minus native M1. Nine F4 streams require full prospective receipt alignment over 5,100 three-arm row comparisons (100×17×3 seed blocks), including initial-state/negative/RNG/query/mask/support/teacher invariants. Native M1 uses the same seed labels and native recipe but its different constructor consumes different RNG draws; secondary M1 differences are seed-blocked and cannot claim F4 stream/RNG pairing.

`compare_pilot.py` requires all 12 complete fits and own selected replays before producing any family mean or continuation decision. It verifies all nine complete F4 streams through the unchanged v2 comparator with epochs 100. Report every selected epoch, every seed difference, mean, sample SD, range, descriptive t95(df 2) and all eight sign flips. Metrics are fractions; differences ×100 are percentage points. Three seeds describe optimization variation on one development graph, not graph replication or graph-level confidence. These development scores support neither significance nor generalization/novelty/acceptance claims. A receipt mismatch makes paired interpretation and continuation ineligible.

## Fixed go/no-go for conditional 500-epoch confirmation

Recommend separately prereleased and root-admitted fresh 500-epoch competitive confirmation only when all 12 cells complete, nine F4 streams are eligible, and all three numerical conditions hold:

1. The arithmetic mean of the three joint-minus-separate seed differences is strictly positive.
2. At least two of those three differences are strictly positive; exact zeros do not count as positives.
3. The mean joint-minus-F4-target-only difference across all three seeds is at least zero. This explicitly defines “no pooled regression”: the proposed joint model does not regress in the declared equal-seed average against the same F4 target control.

This is a prospective resource-allocation rule, not a significance test. Native M1 comparisons provide competitive context and are reported regardless of their sign; they add no unannounced threshold. No diagnostic, best seed or secondary gain rescues a failed primary rule. If joint fails versus separate, simplify/close the cross-side coupling candidate; reconstruction benefit alone cannot justify its coupling claim. If both auxiliary arms fail versus F4 target-only, retain the failure and close this candidate without a tuning grid.

A go is a recommendation, never automatic launch or acceptance. The preserved original 500-epoch F4 family and a prospectively fixed native M1 competitive companion would require separate confirmation admission, fresh state and full native budgets before a stronger performance claim.100-epoch selected states cannot warm-start that confirmation. Even a later positive development confirmation would require separately fixed confirmation data and adequate competitive baselines for a generalization claim.
