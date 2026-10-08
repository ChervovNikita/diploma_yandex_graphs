# Conditional label posterior and fair independent-four reference

8 October 2026. Thin source-only assessment using saved literature and source
contracts. Root supplied the now-open family/C&S summaries; no current numerical
files, models or data were opened. A historical reference report exposed its
previously adopted benchmark table incidentally; that table did not select this
extension. No primary literature was reread or newly fetched. Current jobs,
sealed sources and canonical memory remain unchanged.

## What the reported gap supports

Root reports that C4 loses U4 in all three blocks. At C4-selected states, native
errors have large true-class logit deficits and high top-one confidence, while
reported correction uplifts are much smaller and few routes repair native errors.
Reported nonzero-residual reach also covers only part of the native-error set.
These summaries are consistent with confidence domination and limited evidence
support. They do not prove that TRAIN-error mismatch caused the failure; aggregate
quantiles are not paired per-node counterfactuals.

The algebra explains one risk. Residual CE has gradient
`softmax(z_B + delta) − onehot(y)`. If the native predictor already fits a TRAIN
query, that gradient can be tiny despite literal query-label masking. Conversely,
repairing a heldout native winner `r` requires
`delta_y − delta_r > z_B,r − z_B,y` for that member. Input-label exclusion prevents
literal echo; it does not remove the query's earlier native supervised exposure.

Root's completed C&S result supplies positive label utility, while its declared
row-normalized scores give true-zero probabilities and infinite NLL. This is not
an ensemble or calibration advantage for the new candidate.

## One concrete extension: a separately supervised label posterior

Use one competent source-native feature predictor, selected by its **own**
declared selector and then frozen. Every comparison uses the same actual native
state and graph/label roles. This is an explicit staged adaptation of the current
concurrent recipe, enabling existing native acquisitions to be reused.

The private attention operator keeps detached native H, feature-only Q/K scores,
label-only values, common masked TRAIN Q and the bias-free zero-context path.
For each route, define `q_m = softmax(ell_m)` and train **CE(ell_m,y)** on Q.
Native logits are absent from this loss. The label branch receives direct masked
classification supervision rather than relying on unsolved native TRAIN errors.
This does not produce out-of-sample native errors or erase supervision already
encoded in H.

Let `q_bar` be the mean route posterior and `a_i` indicate any permitted label
anchor under the declared one-hop operator. Fix one aggregation rule:

`p_i = p_B,i` when `a_i=0`; otherwise `p_i = (1/5)p_B,i + (4/5)q_bar_i`.

The weight is a prospective one-native/four-label-family vote rule, not a claim
of independent Bayesian evidence or estimated reliability. There is no weight
grid. The native floor guarantees `p_y >= p_B,y/5`, hence pointwise
`NLL(p) <= NLL(p_B)+log(5)` in exact arithmetic. Compute mixture log probabilities
with stable log-softmax/log-sum-exp. This prevents the declared true-zero issue
when the native log probability is finite; it does not guarantee accuracy or
calibration. A native wrong-class probability contrast can be reversed when the
label posterior's true-versus-rival contrast exceeds one quarter of it.

Keep one hop for this test. Adding hops simultaneously would confound learning
the posterior with expanding its label support. The unreachable-label limitation
remains, and confident incorrect label predictions can introduce new errors.

### One incisive falsifier

Give a **single** four independent 64-wide label-attention heads and a joint
nonlinear class readout, trained by the same masked label CE. Give it the same
frozen H, contexts, native floor, zero-context fallback and selection opportunity.
It receives the same 4/5 label-family weight despite serving one posterior.

Compare that single with both a four-route bank and four fully untied label
predictors on the same native state. If the capable single matches/exceeds the
bank's complete-population accuracy/NLL, the separately supervised-route utility
claim fails. If untied correctors outperform tied BE routes, compression cannot
rescue a quality claim. Current C4's failed gate remains failed; improvement over
a weak residual or one-path reference cannot rescue it.

Use the same bounded native-floor map for an explicitly named C&S variant before
claiming a calibration advantage. Preserve the original CS06 endpoints; changing
its score map is a separate declared reference, not a repair of its old result.

## Honest native-error exposure

Crossfitting, out-of-bag prediction, stacking and Super Learner are established
ancestry. Honest native out-of-sample errors require that the scored query label
never trained **any label-dependent part of the native predictor**.

A single-backbone construction is possible: permanently reserve TRAIN subset R
from native CE, fit B on A\R, and use R for native-error targets. Keep that exclusion
at serving. This is honest label-heldout prediction on the allowed transductive
graph, but it reduces native supervision and needs competence checks; graph and
class-distribution dependence prevents a universal representativeness claim.
Refitting B on all A changes the predictor those errors describe.

Rotating Q, omitting Q only in the current update, dropout/old-view captures, or
crossfitting only classifier heads on a representation trained with all A do not
make the complete native predictor out of sample. The proposed label-posterior
extension bypasses fitting native error targets; it does not claim to solve this
crossfitting problem. A permanent native holdout is not additionally elected here.

## Can independent4 reuse existing trajectories?

**Yes, when its four native members were already independently initialized and
trained from scratch. They need not be freshly reacquired.** Four label heads on
one B, cloned native states, or different checkpoints of one trajectory do not
satisfy that definition. Preserve original plain-native reference endpoints.

For the **staged extension above**, existing source-native native banks can support
a fair ordinary-four reference. Freeze each member's original own-selected state,
fit one new private label posterior per member, and mean their final bounded
mixtures. Use the actual bank member0 state for shared-B arms if the comparison is
intended to isolate sharing. Check exact graph/node/label roles, source/recipe,
initialization lineage, selected epoch/mode, selector and state custody before
admission. Old residual-CE heads are not posterior-CE heads. Charge historical
native acquisition plus new head fitting/serving; do not claim free native work
or fresh same-hardware parity. No teacher distillation is involved.

For a **concurrent** adaptation, final checkpoints alone are insufficient. New
heads need the changing per-update H and evaluation states, correct old view,
native/mask RNG and coherent own-local restoration. The sealed four-bank source
retains selected states and scalar traces, not a complete H trajectory cache.
Without suitable archived history, native replay or missing native trajectories
must be executed and charged. Three optimizer blocks cannot be relabeled as four
members. Root can perform the custody audit without reopening or retraining
published-paper scores.

## Attribution and disposition

Masked label prediction/label-aware branches: UniMP, GAMLP, GMNN, Echoless-LP and
GOODIE. Probability smoothing/label utility: C&S. Shared/private maps and mean
member losses: BE/TabM/LoMETab. Learned prediction combinations: stacking/Super
Learner. These are ingredients, not novelty clearance. The narrower question is
whether the protected label-posterior recipe earns useful quality under the
capable single, untied and established label-aware references.

One inactive conditional extension; zero implementations, fits, remote operations
or new primary reads. No universal-reach, novelty or superiority claim.
