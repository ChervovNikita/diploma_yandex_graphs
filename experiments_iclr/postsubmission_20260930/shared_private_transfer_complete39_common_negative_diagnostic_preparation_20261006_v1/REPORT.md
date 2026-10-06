# Common negative retention and diagnostic protocol

Completed39 retains full selected member logit banks. Local metadata and the successful D2 reader establish that these banks contain aligned positive scores and all 500 negative scores for each of 227 VALID queries. A common negative diagnostic can use those existing banks without a fit or model forward. The raw banks have not been copied to the current local workspace, and this preparation has not opened prediction payloads or calculated new scores.

## Retained evidence

All 39 collection records name and hash `selected_VALID_logits.pt`. The completed D2 reader required exactly `member_pos`, `member_neg`, `mean_pos`, `mean_neg`, `selected_cycle`, `inputs`, and `checkpoint_sha256`. It checked finite FP32 shapes `[227,M]`, `[227,500,M]`, `[227]`, and `[227,500]`, exact agreement with the saved raw mean pool, selected state custody, and common input hashes. This is stronger retention than per query reciprocal ranks. Neither the saved bank schema nor the D2 result schema includes an explicit winning negative node pair or index list.

The twelve requested banks are E_joint, E_live, S_joint, and J4_joint in b0, b1, and b2. All twelve are referenced in D2 and absent at their recorded paths under this local phase. All 39 recorded bank paths are locally absent. The reported retention is the prior authenticated source phase observation; current source host existence was not checked in this assignment. `BINDINGS.json` preserves exact paths, hashes, selected cycles, member counts, and checkpoint identity strings for the twelve banks.

All twelve executed training records pin the same `run.py` SHA `6d7e75f9bae93ef88b2873f55f4f449ae52b9a0b6768fa808fae391da50ed524` and source manifest SHA `db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f`. The local source has those exact bytes. Its evaluator iterates VALID positives in preserved nonself file order, flattens `pool[ids]` in query and negative order, and reshapes the resulting negative logits to `[query,500,member]`. Every bank binds the same positive file and negative pool hashes. Thus `(query_slot, negative_slot)` identifies the same stored candidate across members, cells, and blocks.

Node pair identities are recoverable by associating these slots with the original hashed `heart_valid_samples.npy`, but that additional held input is unnecessary for the core overlap analysis. The proposed code reads only bank payloads. A future explicitly authorized graph annotation could separately read the fixed positive and candidate files; their values were not read here. The loader source requires the negative pool shape `[227,500,2]`, first 250 candidates anchored at the positive's first endpoint, and last 250 anchored at its second endpoint. Candidate slots may repeat a node pair; slot counts reproduce the evaluator and must not be described as unique edge counts without a separate identity audit.

## Signed margin analysis

For query q, negative slot j, and member m, define `d[m,q,j] = positive[m,q] - negative[m,q,j]`. A negative strict margin means that candidate outranks the positive. Use strict comparisons of the saved FP32 scores, and perform the subtraction in float64 to preserve those signs. Do not impose a threshold, calibrate scores, change pooling, or tune on outcomes.

Let `C[q]` contain slots whose margins are negative in every member, and `U[q]` contain slots with a negative margin in at least one member. Report common and union counts; their ratio is defined only for a nonempty union. Also report each member's strict error count. High common counts can merely reflect weak members, so counts and normalized overlap must be read together. Raw margin magnitudes depend on logit scale and are descriptive across architectures.

Read the saved mean pool to identify `Wpool[q]`, its strictly outranking negative slots. Record `L[q] = |C[q] intersect Wpool[q]|`, the common slots that survive the actual serving arithmetic. Export the common slot indices, all member margins, the float64 mean of those margins, and the margin between saved pooled scores. In exact real arithmetic all strictly common negatives survive an equal raw mean. Separate averaging of FP32 scores can produce ties; therefore use observed `L`, rather than assuming every common slot remains a strict error in the saved pool.

The original evaluator uses a midrank for ties. Since at least L negatives strictly outrank the positive, pooled rank is at least `1+L`, and pooled reciprocal rank is at most `1/(1+L)`. L at least 10 excludes Hits10. This is a lower bound on rank from existing offending slots, not a prediction of complete rank. Common ties and disagreements remain distinct.

Compare E_joint against J4_joint for common counts, common slots surviving pooling, and common fraction of the union. Compare E_live against E_joint as a descriptive characterization of the stopped live arm. Report which E common slots J4's saved pool strictly corrects. S_joint has one member, so its common set is trivially its error set; use it only to ask whether a capable single model strictly corrects those same slots. J4 uses four separately initialized encoders and full native predictors. It changes more than sharing, so this comparison cannot isolate a sharing effect.

## Clear falsifiers

The strict same negative explanation for a particular query requires at least one slot in C. If C is empty on a query where members are weak and pooling fails, unanimous strict misranking cannot explain that query's failure. If common slots do not survive the saved pool, they do not supply a strict serving rank obstruction; the arithmetic exception must be reported.

The stronger comparative claim that E_joint has greater common negative burden than J4 predicts positive E_joint minus J4 differences in mean common count and in common fraction of the union. If both differences are nonpositive in each of all three blocks, that directional comparative claim is falsified on these selected states. Mixed block signs are inconclusive. Neither a positive difference nor a common error proves graph information erasure, identifies the historical training cause, or supports a new method.

## Uncertainty and query dependence

Present b0, b1, and b2 separately, paired differences on identical query slots, and an equal block descriptive summary. The blocks are training repetitions on one graph, split, and negative pool. The same 227 queries recur in all three blocks; queries can share endpoints, neighborhood support, and negative node pairs. Neither 681 query rows nor the negative slots are independent graph replications. The minimal code emits no iid confidence interval or significance claim. Three blocks are too few for a credible population uncertainty estimate.

All checkpoints were chosen by the existing rounded VALID MRR selector, often at different cycles. This retrospective diagnostic concerns selected VALID states. It is not an independent confirmation, a matched training trajectory, or a causal intervention. Complete39's prior live failure disposition remains in force.

## Prepared source and execution status

`analyze_common_negatives.py` is a minimal CPU reader that requires an enabled root release, exact source and binding hashes, all twelve bank hashes before any deserialization, the previous D2 Torch runtime, safe tensor loading, and state/schema checks. It loads no model code, checkpoints, features, positive files, or negative pool file. It prints diagnostics as JSON and never modifies the source banks. The accompanying release is disabled. Only metadata construction and static syntax/source inspection were performed; numerical execution remains unverified.

Existing raw predictions are sufficient for the core question, so no additional TRAIN prediction diagnostic is needed to resolve the retention issue. If the retained banks later become unavailable, the fallback would be a separately authorized TRAIN diagnostic on a fixed small TRAIN query/candidate bank and the existing selected states, with common identities across these twelve cells, endpoint geometry, and query target masking. That would require new forwards, measure TRAIN behavior rather than these VALID failures, and does not recover the missing VALID evidence. It has not been prepared for execution or run.

`LOCAL_RETENTION_EVIDENCE.json` records source excerpts and metadata hashes. The canonical indices, completed39 packet, and earlier TabM assessment are unchanged.
