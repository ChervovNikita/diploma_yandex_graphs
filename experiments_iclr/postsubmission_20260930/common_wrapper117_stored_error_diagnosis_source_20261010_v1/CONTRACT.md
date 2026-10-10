# Stored error diagnosis for the closed 117 fits

## Scope and callable

This is one source-only diagnostic for the already closed SAGE/GCN/GAT common-wrapper families: 63 groups, 117 acquisition fits. It reads only the 27 selected VALID archives for unchanged shared4, ordinary genuine I4 and factorized genuine I4, at every paired seed 7301/7403/7507. No connector, rotation, partial 77, TEST, checkpoint or model execution is included.

[diagnose.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/common_wrapper117_stored_error_diagnosis_source_20261010_v1/diagnose.py>) provides:

    run(sage_report, native_report, new_report_path, archive_roots=None)

The source imports only the standard library until this explicit call; the call uses existing CPU NumPy. The two admitted complete analysis digests are fixed in source. Archive paths/digests and ordered ID/label hashes come from those reports. Optional roots map SAGE/GCN/GAT to exact copied family directories; content hashes must still match. Every bank's frozen count decomposition is checked. An existing output is refused. There is no network, launcher or runtime framework.

The CLI accepts --sage-analysis, --native-analysis and --report, with optional --sage-archives/--gcn-archives/--gat-archives. Root owns any later CPU execution. Source preparation supplies no observed diagnostic result or execution admission.

## Fixed error partitions

Saved member and pooled error flags remain authoritative and must agree with raw-logit and saved-pool argmax decisions. Ties use the lowest class index, as NumPy argmax does.

| Stratum | Definition |
| --- | --- |
| no_alternative_wrong | Every member is wrong and the pool is wrong. |
| aggregation_only_correct | Every member is wrong but the pool is correct. |
| alternative_lost | At least one member is correct but the pool is wrong. |
| alternative_served | At least one member is correct and the pool is correct. |

These four strata partition all 5,274 nodes. Two explicitly overlapping rollups also receive statistics: all_member_wrong combines the first two; served_correct combines the second and fourth. Every summary contains the full cohort and each true class 0–9, including empty classes. Counts are exact. Seeds/classes/nodes are not independent replications.

For each I4 reference separately, Q is the exact set where that I4 pool is correct and shared4's pool is wrong. Partition Q into:

- **Unrecoverable by member selection:** no shared member has a correct top prediction.
- **Recoverable by member selection:** a correct shared member exists but averaging loses it.

“Recoverable” means oracle opportunity among the unchanged members; it does not identify a deployable label-blind selector. A fourth nested panel contains unrecoverable nodes with a common strict false rival. The source reports exact counts and shared/I4 distributions on the same paired nodes, plus the reverse shared-correct/I4-wrong count. No arbitrary cross-bank member-index pairing is used.

## Fixed statistics

Let y be truth, z_m the stored float32 logits and p_m their diagnostic float64 softmax. The actual served pool p is the archived float32 probability mean; it is never replaced by a recomputed pool.

- **Correct-class mass:** p(y) per node, p_m(y) over all member-node events, and p_m(y) over wrong-member events.
- **True-versus-best-false margins:** z_m(y)−max[j≠y]z_m(j), p_m(y)−max[j≠y]p_m(j), and p(y)−max[j≠y]p(j).
- **Correct/wrong confidence:** max[j]p_m(j), separately over saved correct and wrong member events. For mixed nodes, count whether the maximum wrong-member confidence exceeds or equals the maximum correct-member confidence. These are associations, not calibration or causal certificates.
- **False-rival agreement:** the largest number of the four members sharing the same best false class, excluding y even for correct members. Separately report the largest number of wrong members sharing a top predicted class, and unanimous wrong top predictions.
- **Common strict rival:** count false classes j for which every member has z_m(j)>z_m(y). Also summarize d=max[j≠y] min[m](z_m(j)−z_m(y)). Exactly d>0 certifies a common strict competitor for the real-arithmetic softmax of those logits and excludes correction by its unchanged nonnegative class-shared probability mixture. It is not a bitwise certificate for unretained float32 member probabilities. Exact zero is retained as a tie boundary; no fitted tolerance or threshold is introduced.

Member distributions weight each member-node event equally; pool distributions weight nodes equally. Output includes sample counts, means, linearly interpolated quantiles at 0/.05/.25/.5/.75/.95/.99/1, and exact fixed-bin counts. Probability bins are deciles from 0 to 1. Probability-margin edges are −1,−.5,−.2,−.1,−.05,−.01,0,.01,.05,.1,.2,.5,1. Logit-margin edges are −∞,−10,−5,−2,−1,−.5,0,.5,1,2,5,10,+∞. Bins are left closed/right open except the final bin, which includes its right endpoint. Empty groups report zero samples/bins and null mean/quantiles. No raw logit vectors or per-node prediction rows are exported.

Float64 member probabilities are diagnostic operands, as in the existing stable-NLL analysis; archived serving decisions and selected scores remain unchanged. The output records these distributions without refitting, calibration, reselecting checkpoints or ranking a new aggregator.

## One prospective failure signature and interpretation

The single nominated signature is **consistent missing shared rankings**: for both I4 references, at every one of the nine backbone/seed cells, a strict majority of Q consists of shared all-member-wrong nodes with a common strict false rival. The source reports all eighteen component counts and the fixed predicate 2×certified_missing > Q; an empty Q does not pass. This majority is a descriptive triage rule, not a predictive utility gate or significance test.

If observed, the signature gives a specific reason to investigate a training or readout change that changes shared predictions: neither choosing an unchanged member nor its ordinary nonnegative probability mixture can repair those certified nodes. Mass/margin/confidence distributions show whether those failures are near ties or confident errors, without choosing a favorable threshold. It does not diagnose gradient conflict, lost input information, hidden collapse or an architecture cause, and it does not establish that a proposed factor start would help.

If the signature is mixed or absent, preserve that result. A large recoverable Q instead locates a serving opportunity, whose actual cause remains unresolved. All-member-wrong nodes lacking a common rival can sometimes be aggregation-only recoverable; the explicit partition prevents treating every such node as a convex-pooling impossibility. Report the full populations and classes rather than selecting one attractive cohort.

A diagnosis can justify a narrower remedy question. Advancement still requires full-population repairs versus harms, competent references and proper risk under a prospectively frozen study. This source adds no fitted remedy, new accuracy result, confirmation claim or general mechanism list.
