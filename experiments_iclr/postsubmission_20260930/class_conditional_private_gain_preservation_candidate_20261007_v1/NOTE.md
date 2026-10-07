# Test whether shared updates erase class-specific private progress

**One unadopted hypothesis:** in a jointly trained shared backbone, an ordinary shared Adam step can undo a member's useful own-label private update on one class even while that member's total CE improves. Whole-member averaging hides this cancellation. Retaining that class-specific private progress could strengthen weak members and improve the fixed served pool. Wiki24 motivates the competence question; it does **not** demonstrate gradient interference or explain why Rademacher members were weak.

This is materially different from changing own/pool risk placement in molecular18: it changes the shared update map while retaining ordinary own supervision. It is an attributed class-conditioned constrained-optimization experiment, **not a new optimizer principle**. Saved GEM/PCGrad/OGD and private-learning notes already establish the relevant ingredients. No independent teacher acquisition, new architecture, new paper reading, or execution is proposed here.

## Exact candidate

Use the existing four-member, all-internal BE WikiCS architecture, first-input Rademacher factors, plain two-view mean own CE, fresh joint training and original 1100-epoch horizon. Every member sees every TRAIN label. Shared weights/norms remain trainable; private factors retain their ordinary Adam update. The stress setting is the observed weak-member init arm; this is not a replacement for the ongoing molecular18 study.

At an update, preserve the old shared/private parameters, Adam states, and both ordinary dropout realizations. Compute the same ordinary source gradients and native proposed displacements `u0` for shared parameters and `d_m` for each private block. These include the source's original loss scaling and native moment histories; do not substitute raw negative gradients. Construct the post-private reference `B=(theta, phi+d)` without advancing an additional optimizer step.

Use deterministic serving-mode forwards on the complete TRAIN panel to form each member/class CE `L_m,c`. A pair is active when its class is present and the ordinary private displacement strictly lowers its class CE: `L_m,c(B)<L_m,c(old)`. This includes wrong predictions; eligibility does not require an already correct specialist. Freeze this active roster for the current correction.

At B collect `q_m,c=gradient_theta L_m,c(B)` for active pairs. Choose the Euclidean-nearest shared displacement to the actual Adam proposal:

```text
minimize_u  0.5 ||u-u0||²
subject to q_m,c · u <= 0 for every active member/class pair.
```

Zero is feasible for this linearized problem. Test at most four scales of the projected displacement: `1, 1/2, 1/4, 1/8`. Accept the first whose finite deterministic TRAIN forward leaves every active class CE and every member's total CE no larger than at B. If none passes, apply `u=0` for the shared block and still commit the already proposed private update. If no class is active, use ordinary `u0`. Native Adam moments advance once from the ordinary gradients in every arm; the applied shared displacement is separately recorded. This moment/displacement convention is known and supplies no Adam descent guarantee.

The candidate protects **measured class-specific progress from this private step**. It does not preserve an entire private tangent space or future learnability, forbid all node mistakes, protect development behavior, or guarantee useful diversity. A constraint on the private gain's magnitude alone would be inappropriate: a shared update can shrink that gain by repairing the class itself. The reference here is the actual class risk after the private update, which allows further shared improvement.

## Why it could affect common wrong competitors

Own CE has the corrective cotangent `p_m-y` even when every member prefers the same wrong class. A private step can increase truth probability on such nodes without immediately changing argmax. Keeping its class-specific improvement prevents a conflicting shared step from undoing that early correction. Different private Jacobians can preserve different useful trajectories without deliberately weakening non-specialists or adding hidden repulsion.

This is only a plausible route. Class means can improve on easy nodes while leaving common errors intact; the shared update may already help every class; private updates may have no transferable signal; or the constraints may reduce the shared step to zero. A finite TRAIN guard is not an unused-population competence guarantee. Consequently common-error **repairs minus introduced errors**, served accuracy/NLL, and worst-member competence are decisive. Oracle coverage and gradient cosines are insufficient.

## A discriminating representative plan

Conditional on the running objective-placement work leaving a meaningful member deficit, propose a separately frozen WikiCS screen: four conditions × seeds 6101/6203/6307, complete split0 TRAIN graph, original model/horizon/pooling/selector, all cells completed before comparative opening. No fits or implementation are admitted by this note.

| Condition | Difference tested |
|---|---|
| Ordinary BE-init Adam | Original learning trajectory and competence reference. |
| Faithful member-task PCGrad on shared gradients | Established pairwise conflict surgery; keep private own updates and document its native optimizer integration. |
| Whole-member post-private projection | Same displacement projection, finite trials, moment convention and private updates, but protect positive private progress in total member CE rather than separate classes. |
| Class-conditioned post-private projection | The candidate above. |

The whole-member projection is the primary mechanism control: it separates label-conditional cancellation from generic shared-step restraint. PCGrad is a capable established alternative, not algebraically identical to projecting an Adam displacement at B. All conditions collect the same deterministic panel diagnostics without altering RNG/selection opportunity; actual gradient collections, finite forwards, QP work, memory and applied movement are charged. Forty possible member/class gradient rows are not cheap simply because no teachers are trained. Equal epochs do not imply equal compute.

Record actual finite shared erasure `L_m,c(B+u0)-L_m,c(B)`, active private gains, total-member progress, corrected/zero shared steps and gradient norms. Separately record a cancellation event when an active class is harmed by `u0` although its member's total CE improves. This diagnostic directly tests the premise; pairwise negative cosines do not.

**Stop criteria:** if active positive gains or finite cancellation events are negligible, reject this explanation; if core movement is mostly suppressed, do not call freezing a successful conflict repair. If the whole-member projection or PCGrad matches the useful gain, class-specific preservation is unsupported. If candidate mean or worst-member competence declines, shared common errors merely move, repairs are canceled by harms, or the served accuracy/NLL gain is absent, stop promotion. A favorable three-seed development screen requires positive paired served differences against the native and whole-member controls, improved mean/worst-member competence, and no worse mean served NLL; it remains unconfirmed. No class, coefficient, guard or horizon is chosen after outcomes. Existing competent unit-plus-contrast, single and independent4 results remain quality references with their selection/cost differences disclosed; method superiority would require fresh matched capable controls, including applying the same class restraint to a single model to rule out generic class balancing.

## Prior overlap and graph boundary

| Saved primary conclusion | Overlap and limit |
|---|---|
| GNCL Eq.5; TreeNets own/pool training | Own/member versus pool-risk objectives are prior. This screen holds plain own CE fixed and changes the shared displacement; it does not replace the molecular18 test or show a new loss principle. |
| GEM / A-GEM | Minimum-change simultaneous halfspace projection is directly GEM-type optimization; class losses are virtual tasks. Finite checks are generic constrained optimization. |
| PCGrad / CAGrad / FAMO | Resolving or balancing task gradients is prior. Whole-member constraints can average away class effects; the proposed data grouping is an empirical hypothesis, not novel optimization. |
| OGD; saved private-step transfer / MLDG / MAML ancestry | Protecting functional directions and credit through private learning are prior. Here the actual finite post-private class risk is protected without a differentiated private-update meta-objective; the frozen private proposal is not a new transfer estimator. |
| BatchEnsemble / TabM; FiLM / GNN-FiLM | Learned private factors and affine modulation are prior and unchanged. The candidate creates no extra reachable feature or message direction. |
| Hydra / Deep Sub-Ensembles | Shared-body private predictors and member identity are established. Hydra's inspected method distills separate teachers; this experiment starts jointly from fresh own-label training. Avoiding teachers does not itself establish novelty. |

There is **no defensible graph-specific methodological gap yet**. Graph propagation couples class gradients across nodes and neighborhood contexts, but this projection would also apply to a tabular shared model. It adds no graph-conditioned constraint or new evidence path. A later graph claim would need separately declared structural attribution, not the fact that the backbone is a GNN. GEENI remains method-unresolved; this generic update test does not clear its or any other prior's exact overlap.

Saved conclusions only were reused: `shared_member_gradient_conflict_primary_20261007_v1/NOTE.md`, `graph_shared_gradient_{gem_closure,routing}_20261003_v1/PAPER_CONCLUSIONS.json`, `ensemble_consistent_shared_core_optimizer_prior_20261006_v1/CONCLUSIONS.json`, `adaptation_free_private_learning_regularization_assessment_20261005_v1/REPORT.md`, `shared_backbone_accuracy_mechanism_saved_prior_synthesis_20261006_v1/REPORT.md`, `shared_backbone_native_decision_residual_control_20261007_v1/CONCLUSIONS.json`, and `staged_private_graph_residual_closest_prior_P_note_20261007_v1/P.md`. New primary retrievals/rereads, full-paper credits, raw science payloads, numerical runs, source mutations and root-ledger/status edits: zero.
