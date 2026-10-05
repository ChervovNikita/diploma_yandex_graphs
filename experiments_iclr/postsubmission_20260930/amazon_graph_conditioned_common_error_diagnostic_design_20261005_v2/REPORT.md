# VALID-to-VALID graph-error recurrence: frozen descriptive V2

**Status:** source preparation only; no execution, tensor/label reads, fitting, hidden extraction or compute admission. V1 is preserved. Root reviews and executes separately. The running private-transfer pilot is unchanged.

## FIT feasibility correction

The saved root-adopted RESULTS_SUMMARY.json gives mean TRAIN-FIT accuracy 99.9591649% shared, 99.9829856% independent and 99.9727766% native single. Existing consumer role metadata gives FIT sizes 9795, 9796 and 9795. Multiplying each mean error fraction by `3 * max(FIT size)` gives an upper bound below 13, 6 and 9 pooled errors respectively across the three splits: hence at most **12 shared, 5 independent and 8 single** pooled-error node/split observations. A unanimously wrong member bank is a subset of its native pool's errors.

V1 is not algebraically forced to zero: a rare erroneous FIT anchor could touch many VALID targets. But its distinct error-anchor support is severely bounded. Repeated edges cannot add distinct error evidence, and its total-target/anchor floor could pass with almost all recurrence indicators zero. That makes it an unsuitable main discriminator. No FIT score/label projection should be admitted merely to discover this in-sample error floor. These bounds use saved aggregate values and role-count metadata only; no original payload or outcome shard was opened.

## One population and matching recipe

Use the existing authenticated selected-bank VALID-only views/labels and the public graph, with all 6123 VALID nodes in each of splits 0, 1 and 2. The native single is independent member 0. Member/single prediction is raw FP32-logit argmax; four-member pool prediction is argmax of mean FP32 softmax probabilities. Ties choose the smallest class. Confidence is each bank's mean member maximum FP32 softmax probability.

For every target v and each competitor c different from its true class, anchors are the other VALID nodes of the **same true class**. Separate direct canonical public graph neighbors from all nonneighbors; exclude self edges and duplicate directed edges. Public degree counts all unique undirected nonself public neighbors before the VALID/class restriction. Nonneighbors may share two-hop context.

Match each eligible same-class neighbor u to at most one nonneighbor a of v, using exactly the same pair for both banks and the single. Require identical true class, degree band `floor(log2(1+public degree))`, both banks' correct-member counts (0–4), and native-single correctness. Require confidence distance at most .05 separately in both banks. Process neighbors in SHA256 node-ID order with salt `common-error-valid-valid-20261005-v2`; choose the unused admissible control with smallest summed confidence distance and break ties by that same order. No control reuse within a target; reuse across targets is recorded. Matching never uses the target competitor, anchor/control wrong-class identity, unanimity, or the contrast. No widening, bin merging or population substitution after outcomes.

This holds coarse member competence/confidence fixed for anchor pairs. It does not remove all feature/community difficulty or supply causal exchangeability. Same-class conditioning removes the anchor label-composition difference within this estimand and excludes heterophilous edges.

## Complete outputs and one primary comparison

For every bank and competitor, retain three anchor events: pool predicts c; all four raw member predictions equal c; fraction of members predicting c. Retain the native single predicts-c event. All are wrong-class events because target, anchor and control labels agree and c differs from that label. For all targets and every fixed pooled-error stratum, true class and wrong competitor, report:

- raw adjacency and **all** nonneighbor numerators, denominators and rates;
- matched adjacency/control numerators, denominators and rates;
- both pair-weighted and target-weighted rates, with empty supports explicitly undefined;
- target-weighted matched adjacency-minus-control recurrence for each event.

The disjoint pooled strata are both correct; shared loss with/without shared erroneous unanimity; shared recovery with/without independent erroneous unanimity; both wrong at the same competitor with all four combinations of erroneous-unanimity flags; and both wrong at different competitors with those four combinations. All 13 strata are retained, including empty cells. Also retain full native-pool and individual-member class confusion matrices, the joint correct-member-count/native-single-correctness histogram in each stratum/true class, and each bank's full 0–4 member-count histogram for every wrong competitor. No node/prediction export or selected favorable stratum replaces these complete counts.

The primary population remains targets where both banks are unanimously wrong at the **same** competitor c_v. For each matched target, E_f(v) is mean `[all-members-predict-c_v at neighbor − at control]`. The splitwise primary contrast is mean `[E_shared(v) − E_independent(v)]`, giving each target equal weight. Retain the single reference, all primary raw/matched rates and full population fraction. Report all three splitwise values; a three-block mean is secondary and is undefined unless all three are defined. This narrow stratum cannot explain the complete accuracy gap or shared-only losses.

## Support and limits

Report complete VALID, stratum and primary denominators; targets with eligible neighbors and matches; raw/matched pair counts; unmatched counts by class/degree/correct-member/single-correctness signature and fixed reason (no nonneighbor in signature, no confidence-compatible control, or controls exhausted); unique neighboring anchors/controls; maximum control reuse; actual matched confidence distances. In the primary population additionally report **distinct wrong-event neighbor/control anchors** for each bank and the single. Empty denominators yield null, never zero. V2 has no numeric “adequate mechanism support” threshold: nonempty means descriptively defined, and effective-event counts remain visible for root's assessment. V1's total-anchor floor is not reused as a power claim.

The same selected VALID labels appear at targets, anchors, controls and in matching; this is deliberate retrospective label-conditioned description. It fits no predictor and cannot be used as a serving rule, fresh development result or confirmation endpoint. Checkpoints were selected using VALID. Directed pairs from one undirected graph are dependent, anchors/controls recur, and the three masks overlap on the same graph. No iid node/edge bootstrap, p-value, independent-split inference or causal graph/sharing conclusion is supplied. Residual overconfidence, communities, selection and unsupported cells limit interpretation. Saved logits do not identify parameter-gradient interference. No positive/negative recurrence result provides novelty clearance or GPU authority.

DIAGNOSTIC.json fixes the settings. SOURCE_BINDINGS.json binds the safe feasibility evidence and unchanged source/custody references. The separate source packet implements only this descriptive contract; root independently reviews it before any execution.
