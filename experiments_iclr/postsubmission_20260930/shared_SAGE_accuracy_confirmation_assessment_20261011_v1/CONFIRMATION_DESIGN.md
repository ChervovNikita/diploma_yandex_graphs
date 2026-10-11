# Prospective Penn94 confirmation design

## Fixed scope and available source

One complete homogeneous Facebook graph, native SAGE, published fixed split 0. Keep all 41,554 public graph nodes for message passing, including y = −1 nodes. Exclude those missing targets from all losses/metrics. Remove gender/target column from node features. Bind the saved PyG one-hot encoding of the non-target metadata, exact dimension/node mapping and undirected/coalesced support before fits; do not mix it with the author's different `label_binarize` encoding or assume a stored edge count from the paper. Payloads and masks have not been acquired or opened in this packet.

CUAI source revision `82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6` contains `data/facebook100/Penn94.mat` and `data/splits/fb100-Penn94-splits.npy`; their git-blob/size metadata is saved. PyG revision `76ff9c2ce18c8cebf52122b57e2aeadce9793d10` supplies Penn94 masks. Choose the first supplied 50/25/25 split, with no selection from task outcomes. Known-project records inspected here contain no Penn94 scoring entry. Root must bind that known-scoring-history eligibility decision and current official-data/source admission. Public target availability and custodial processing do not themselves invalidate unused scientific evaluation; this does not certify every historical public-label access.

## Whole-pipeline label roles

Preserve full official TRAIN and TEST memberships. Sort official VALID ids, apply a label-free permutation with fixed seed 11709, assign the first floor(N_VALID/2) to CAL and the remainder to SELECT. The custodian records counts/hashes without exposing TEST targets to trainers. CAL has no gradient or checkpoint-selection use. Use full public features/topology transductively; the benchmark's raw bundle contains all labels and cannot be passed directly to a training loader. Raw target/split payloads, canonical graph/feature hashes and proper-role safe exports still require admission.

All model/hyperparameter/selector choices precede these fits. All selected states precede CAL temperature fitting. All temperatures, serving modes and failed/terminal records precede one separately admitted complete TEST scorer release. Old `run_tabm` and LINKX evaluators read TEST automatically; the future wrapper retains their model/loss/readout definitions while providing TRAIN/SELECT/CAL-only execution and a separate heldout scorer. It does not execute those stock mains.

## Seven fitted banks, one declared configuration each

| Bank | Purpose and selection | Fits per seed |
| --- | --- | ---: |
| Internal native shared4, probability mean | Candidate; fixed F coherent bank selection | 1 |
| Original boundary shared4, logit mean | Direct original architectural/serving comparator; original Xavier W/private B and logit checkpoint law retained | 1 |
| Matched native-boundary shared4, probability mean | Current native weight/bias/initialization/RNG/loss/selector law, factorize input/output only | 1 |
| Ordinary native SAGE M1 | Capable unfactorized single context | 1 |
| Internal-factor native SAGE M1 | Capable factorized single | 1 |
| Genuine internal-factor native SAGE I4 | Four independent complete bodies, own unscaled CE, optimizers/streams/selectors; selected probability mean | 4 |
| Native LINKX context | One fixed published-source topology/feature reference; separate source-qualified selected state | 1 |

This is seven banks / ten optimizer-body fits per seed, ten paired blocks 0–9: 70 banks / 100 fits. No new context objective, teacher, head, training dictionary or tuning grid is included. Fresh controls are fitted for this confirmation; old selected models are evidence context only.

For the candidate, matched-boundary bank and capable SAGE controls, bind native Model/SAGEConv semantics (depth 2, hidden 128, hidden multiplier 1, LayerNorm, dropout 0.2), first-input Rademacher factor initialization and persistent route dropout streams. AdamW lr 0.001, decay 0; maximum 1,000 updates, stale patience 300, first strict SELECT accuracy maximum, earliest ties. Banks optimize mean member CE; individual bodies use full unscaled own CE. Pool CE supplies no backward. Adapt only admitted graph/feature/class/role dimensions and the binary two-output scorer; the existing hardcoded WikiCS loader cannot run Penn94 as is.

The original comparator preserves the saved BEBlock Xavier boundary initialization, zero-initial private B rows, Rademacher first R, later factors = 1, original sequential mean-member CE and logit-mean serving/selection cadence. Preserve its saved 5,000-update default ceiling and 300-stale-update allowance; record those unequal opportunities and the exact original checkpoint cadence. Fresh seed blocks replace the old runner's use of split id as an initialization seed; this is a prospectively declared confirmation protocol, not reproduction of its historical grid or paper scores. Its recipe contrast with current F remains composite. The matched-boundary bank supplies the clean factor-location comparison at current F's common probability mean and selector.

LINKX uses the saved author implementation, one published script configuration at width 128/layers 2, parser defaults dropout 0.5, lr 0.01, decay 0.001, 500 full epochs, init adjacency/feature layers = 1, inner activation/dropout false, native optimizer/loss and first strict SELECT maximum. This is one source-backed configuration, not a claim of reproduced optimized paper performance or prior local competence. Source/finite-gradient/role/selected-state qualification and complete allowed-development competence diagnostics precede source freeze. No weak-reference shortcut or TEST-chosen recipe repair is allowed. Feature-only MLP, LINK and other task baselines remain explicit breadth gaps rather than an extra active grid.

## Separating placement from pooling

Primary factor-placement comparison is internal/probability versus matched-boundary/probability under the common native constructor, bias, seed/RNG, CE and selector law. It measures the complete trained factor-placement procedure, including its extra factor parameters and optimization trajectories.

For the same selected states of those two banks, predeclare both probability and logit means from the cached member logits. These four placement/readout cells require only the two already planned banks. Their within-state probability-minus-logit contrasts isolate serving at fixed acquired predictions. They do not pretend to be independently selected whole-pipeline logit variants. The original boundary/logit primary comparator retains its own initialization/private-bias/selector law, so its direct recipe gain must not be renamed a factor-placement-only effect.

## Binary scoring and calibration

Penn94 has two reported-gender targets. Use two-class logits with ordinary own CE (equivalent to the benchmark's log-softmax/NLL path) and argmax accuracy. Probability mean is mean softmax(z_m); logit mean is softmax(mean z_m). NLL is −log p_truth and Brier sums the squared two-class probability residuals. The stock original one-output BCE/sigmoid path and ROC-AUC conventions are not this scorer. No heldout threshold selection is introduced.

After every native selected state is frozen, fit one bank-global positive T on CAL only for each fixed fusion: mean softmax(z_m/T) or softmax(mean z_m/T). Log T starts at zero; fixed 500 CPU float64 Adam updates, lr 0.01, betas (0.9,0.999), epsilon 1e−8, no decay, fixed final endpoint, no retries/grid/SELECT or TEST temperature refit. Equal policy means equal scalar capacity and label exposure while preserving each declared fusion. The two alternate pooling views add 20 scalar fits; 70 primary plus 20 alternate = 90 terminal CAL fits. Raw accuracy remains the primary endpoint. Lock every T before TEST release.

## Prospective inference and disposition

Four primary paired accuracy contrasts: candidate versus genuine I4, ordinary M1, factorized M1 and original-projector/logit comparator. Practical margin is +0.2 percentage points for each. Proposed accuracy-confirmation rule: mean gain at least +0.2 pp and positive-effect paired seed inference surviving Holm familywise 0.05 across those four contrasts. Freeze exact paired-test implementation before fits: paired t inference conditional on independent optimizer-seed differences, with paired intervals, all ten signs/values and exact sign-flip symmetry sensitivity. Ten seeds provide no power guarantee or ten-graph replication; the threshold is a new prospective confirmation rule and does not change either closed development verdict.

Report calibrated NLL/Brier and risk uncertainty against the same four controls. Keep the old practical mean +0.02 / each-seed +0.05 NLL harm screens as separately labeled confidence protection. An accuracy confirmation with failed confidence protection is reported as an accuracy/confidence tradeoff; it does not earn a protected-risk claim. Calibrated accuracy, member weakness, costs, class metrics and covariance diagnostics create no additional post hoc accuracy veto.

The matched-boundary and fixed-state pooling contrasts are secondary attribution tests, with their family/multiplicity labeled before release. LINKX is task-native context; if it is stronger, limit the headline to the confirmed SAGE-family comparison. Report every selected-state/member/cohort, calibration failure, cost, memory maximum, acquisition/restoration/serving scope and concurrent wall-time scope. No minimum member competence invented from TEST, no node-iid confidence intervals, no TEST-selected seed top-up, heldout replacement or alternative learning mechanism after failure.

## Readiness and exactly three actions

1. Root binds Penn94 known-scoring eligibility/current input admission, official split-0 authority and exact safe TRAIN/SELECT/CAL/TEST custody.
2. Source-qualify and freeze the seven-bank family, dimension/binary port, original bias/initializer/checkpoint law, fixed LINKX point, exact seed/test/statistical contract and finite resource admission using permitted roles only.
3. On separate execution authorization, close the fixed ten-seed whole pipeline and all CAL endpoints, then one full heldout release and complete interpretation.

This packet is an assessment/prospective design. No source port, role generation, data fetch, fit, heldout outcome read, score recalculation, canonical edit or launch was performed or authorized by it. The fixed-key 580-TRAIN/300-feature WikiCS prototype and a quadratic Penn94 dictionary are outside this design.
