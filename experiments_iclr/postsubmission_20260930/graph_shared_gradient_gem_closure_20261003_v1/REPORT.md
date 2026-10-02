# GEM closure for the shared-gradient proposal

**Disposition: update nearest-prior attribution; preserve shared-gradient v1 unchanged; add one prospective comparator.** This is a literature and mathematical addendum. It qualifies no source, adopts no driver and launches no experiment. The predecessor manifest remains `53b430ead234682bb8dcfe1703d213f60bccc4422a42f99930c329f01f6574c8`.

## Exact projection attribution

At the saved reference `B=(theta_old, phi_plus)`, hold the current competence support and graph weights fixed. Let `S_m` be the weighted member-relative correct-class margin, `h_m=grad_theta S_m`, `H=[h_1,...,h_r]` (`r<=4`), `a=grad_theta L_pool`, `d0` the native proposed shared displacement and `d=d0+c` the corrected displacement.

The inequality-only operator is

```text
min_d  0.5 ||d-d0||^2       subject to H^T d >= 0.
```

Define virtual losses `ell_m=-S_m`, gradients `q_m=-h_m`, proposed descent vector `g=-d0` and corrected descent vector `v=-d`. The objective and constraints become

```text
min_v  0.5 ||v-g||^2        subject to q_m^T v >= 0 for every m.
```

This is exactly GEM's simultaneous Euclidean gradient projection. A common positive step-size division gives the same mapping in gradient units. Constants added to the virtual losses change nothing. The simultaneous inequalities, minimum correction and Gram/active-set solver are established GEM/general constrained-projection machinery. **GEM is the closest consulted QP predecessor; OGD is no longer the closest QP attribution.** PCGrad remains a different member-task comparator. Virtual-loss equivalence holds locally with the frozen support, weights and private reference; it does not require these virtual losses to be ordinary CE or permanent objectives.

The saved proposal also imposes `a^T c=0`, hence `a^T d=a^T d0`, and rejects the minimum correction when `||c||>||d0||`. Its QP is a GEM-type projection intersected with an affine hyperplane, followed by a cap and finite acceptance rule. The complete saved operation is therefore not literally vanilla GEM. GEM's inequality cone contains zero; a nonzero pooled-progress equality can make the intersection infeasible. Projecting an arbitrary native AdamW displacement retains the exact QP mapping for that fixed vector, although its optimizer/state transition differs from GEM's gradient-step algorithm.

For nonnull `a`, define `P_a=I-aa^T/(a^T a)`, `b=-H^T d0` and `G=H^T P_a H`. Eliminating the equality yields the correctly signed dual

```text
max_lambda>=0  lambda^T b - 0.5 lambda^T G lambda
c = P_a H lambda.
```

This derivation assumes feasible constraints and uses the original Euclidean parameter norm. The saved null/rank/KKT/cap/fallback qualification remains required. No new projection principle is claimed.

## What A-GEM does and does not cover

A-GEM replaces simultaneous previous-task inequalities by one sampled memory-average reference gradient. With fixed nonnegative normalized weights, `h_ref=sum_m alpha_m h_m`, the inequality `h_ref^T d>=0` is exactly its one-halfspace projection after the same virtual-loss/sign substitution when the pooled equality is absent.

Under pooled equality, if `b_ref=-h_ref^T d0>0` and `r=P_a h_ref` is nonnull, the minimum correction is `c=b_ref*r/||r||^2`. If `r=0`, that violated inequality is infeasible; if `b_ref<=0`, `c=0` already meets it. This is an equality-constrained extension of the A-GEM geometry. It preserves only an averaged specialist score. For example, `h_1=e_2`, `h_2=-e_2` and equal averaging give `h_ref=0`: the average imposes no constraint while displacement along `-e_2` erodes the first score. A-GEM does not establish each member's protection, and this addendum proposes no averaged-score arm.

## Differences in the complete operation

The saved graph proposal protects current, jointly coupled member-relative logit margins at a current private-proposal reference. Its eligibility requires TRAIN correctness, target probability above the mean-logit pool and a positive correct-class margin contrast. Its weights use stopped, self-looped row-normalized `P^2` transport of current positive TRAIN advantages and are refreshed at each step. Unlabeled nodes provide topology transit only. It uses no old-task episodic replay. GEM already recomputes memory-loss gradients at current weights; refreshing gradients alone is not a distinction from GEM.

The state operation projects the actual native shared displacement after proposing the native private update, carries the original own-gradient moments/step/buffer transition, imposes pooled-progress equality and a displacement cap, operates for 32 steps, and accepts only after dropout-off finite score/pool/own checks. These are differences from the consulted complete GEM/A-GEM algorithms. Support and weights change between steps, so the operation establishes neither a permanent loss-monotonicity property nor a continual-learning guarantee.

Established ingredients do not decide whether this precise graph/member/state composition is useful. No complete equivalent predecessor was established in this bounded consultation, and no global absence or novelty certificate follows. The retained logit-contrast/identical-shared-Jacobian distinction remains conditional; no paper result supplies a measured gain for this proposal.

## Prospective seventh arm: member-CE GEM-type projection

The original six-arm packet is immutable. `PROSPECTIVE_MEMBER_CE_CONTROL.json` records an addition for root consideration before any outcomes. At the same reference `B`, use all four uniform native TRAIN member losses `L_m` with their exact source masks/normalization and `k_m=grad_theta L_m(B)`. Use the same native `d0`, all shared parameters and pooled gradient `a`, then solve

```text
min_c  0.5 ||c||^2
subject to a^T c=0 and k_m^T(d0+c)<=0 for every member m.
```

Equivalently, use `H_CE=[-k_1,...,-k_4]` in the same simultaneous projection. Keep the cap `||c||<=||d0||`, 32-step window, original native moments/step/buffers/RNG transition, solver qualification and native fallback. Keep the same finite pooled guards (corrected pooled CE no worse than both `B` and the native joint proposal) and finite mean-own guard (corrected mean own CE no worse than `B`).

**The functional guards differ:** the graph arm requires each frozen weighted margin `S_m(corrected)>=S_m(B)`; the CE control instead requires every member CE `L_m(corrected)<=L_m(B)`. Its mean-own guard is redundant under exact per-member checks, but remains explicit for matching and numerical qualification. Ordinary CE protects the full TRAIN loss of every member, including poor members; the graph arm protects eligible subset-weighted margin advantages. CE nats and logit-margin units require prospectively bound appropriate finite tolerances. Equal scalar tolerances or an orientation-only comparison are not justified. Supports, ranks, eligibility, finite rejection rates and correction magnitudes can differ. Graph-specific no-support fallback has no identical analogue in the all-member CE arm; pooled-gradient and native-progress eligibility are matched.

This is a GEM-type member-task control, not a standard continual-memory GEM reproduction. It addresses generic simultaneous member-loss preservation. PCGrad alone does not cover that nearest prior. Identity and permuted `P` arms remain necessary to attribute topology, and the shared-pooled/native arms remain needed to interpret the objective and native displacement. If the added arm is adopted, evaluate the saved practical screen against all six other arms: at least 0.01 nats mean later pooled VALIDATION NLL gain, no more than 0.5 percentage-point mean accuracy decline, and the same gain sign across all three seeds. This is a practical screen, not a significance claim. Keep all attempted/fallback steps and paired seeds.

If member-CE matches the graph arm, graph-conditioned utility is unestablished. If identity/permuted graph matches the original graph, topology value is unestablished. A gain only over PCGrad is insufficient. A gain over every matched control would support this composition on the frozen contract, subject to one unused complete-graph confirmation; it would not prove broad novelty or generalization. Broader competent single/native BE/packed independent comparisons remain required for broader GNNM claims.

## Reading scope and source ambiguity

Two new exact-v1 primary method scopes were completed: GEM `arXiv:1706.08840v1`, *Gradient Episodic Memory for Continuum Learning* (2017-06-26), and A-GEM `arXiv:1812.00420v1`, *Efficient Lifelong Learning with A-GEM* (2018-12-02). Both IDs are absent from consulted v23. Counts are **52 paragraph/heading blocks (46 paragraphs, 6 headings), 2 complete printed algorithm figures, 14 display-math nodes and 64 distinct inspected math nodes/fragments**. The 64 nodes are not 64 separate equations. Scoped optimization proof/argument text was read. There were zero full-paper reads, author-code reads or retained-method rereads. Latest-version dates were metadata only: GEM v6 (2022-09-13), A-GEM v2 (2019-01-09). No source numerical benefit is transferred.

A-GEM v1's presentation of GEM's dual defines negative gradient rows while printing a plus linear term and plus restoration. Taken literally together these signs conflict with its own primal. With scalar `g=-1`, previous-task gradient `g_1=1` and row `G=-1`, the printed minimization is `0.5 v^2+v`, minimized at `v=0`; its restoration gives `-1`, violating the primal. Positive gradient rows would make the printed plus signs consistent. This is a saved-v1 sign/rendering ambiguity, not an audit of later versions or author code. The clear primal and independently derived dual above support attribution; A-GEM's one-constraint formula is unaffected. No extra primary read was made to resolve it.

## Resource arithmetic, separate from merit

The permitted outcome-free timing receipt gives Squirrel 78.64112058182558 s and Photo 5325.0353772901 s mean native continuation. Seven arms × two complete graphs × three seeds gives **42 fits and 31.52144623758623 baseline device-hours** (31.52 h), an increment of 4.503063748226605 h over six arms. This is native continuation only; correction, qualification, acquisition, failures, snapshots, graph transport, contention and confirmation are extra and unmeasured.

The added CE arm contributes at most 192 correction attempts and conservative counts of 2304 member-forward and 3840 member-backward route equivalents. Four projected arms give 768 attempts, 9216 extra forward and 15360 extra backward route equivalents. Adding retained PCGrad/shared-pooled gradient work gives an upper bookkeeping count of 18432 extra backward route equivalents. These are not measured packed cost or a wall-time conversion. The projection workspace remains at most seven shared float32 vectors (`28*P_shared` bytes), plus unmeasured snapshots, activations, buffers and other state. No inference capacity is added. No resource is reserved or requested, and current availability does not decide scientific merit.

The read-only stdlib verifier checks bindings, manifest coverage, declared counts, rational sign/cancellation/equality examples and forecast arithmetic. Its PASS is an integrity/accounting check, not source compatibility, utility, author-code reproduction or a novelty certificate. Root owns source/use-history qualification, tolerances, protocol adoption, execution and resources.
