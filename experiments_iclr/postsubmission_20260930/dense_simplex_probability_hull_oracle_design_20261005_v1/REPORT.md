# Dense probability-hull correctability: pre-outcome oracle design

5 October 2026. Source-only mathematical design. No predictions, labels, logits, checkpoints or data payloads were opened or hashed; no LP, fit, export, score or scientific computation was executed. No new literature, development arm, compute grid, promotion or TEST admission is proposed. The design uses the saved conditional Brier geometry and its independent mathematical review.

## Purpose and exact limit

Before considering a costly hidden-state export, one small label-conditioned diagnostic can establish which **existing native mistakes are geometrically correctable by a dense scalar simplex mixture of the fixed member probabilities**. It asks what a clairvoyant per-node combiner could do when it is explicitly given that node's true VALID label. Its weights are oracle witnesses, never a fitted rule, deployable gate, OOF score or evidence of heldout performance.

Fix P in R^(4 by C), each row a member's class-probability vector, and a known diagnostic label y. Use the existing floor

    ell = 0.0125,   delta = 4 ell = 0.05,   alpha = 1-delta = 0.95.

The admitted mixtures satisfy w_m >= ell and sum_m w_m = 1. Write w = ell*1 + alpha*v, where v is any four-member simplex vector. The exact output set is

    S_ell = {P^T w} = 0.05 * mean_m P_m + 0.95 * conv{P_1,...,P_4}.

This is the same restricted probability hull in the saved Brier assessment. The Brier identity concerns projection of an unknown conditional distribution q into S_ell. The present diagnostic instead uses the observed class y and a top-1 margin objective; it is neither conditional-risk estimation nor Brier minimization.

## One primary LP

For each competitor c != y, define d_(m,c) = P_(m,y) - P_(m,c). Solve the closed-polytope LP

    maximize over w,t:  t
    subject to:        sum_m w_m = 1,
                       w_m >= 0.0125 for m=1,...,4,
                       sum_m w_m d_(m,c) >= t for every c != y.

Its value is

    t* = max_w min_(c != y) [(P^T w)_y - (P^T w)_c].

The feasible weight set is nonempty and compact, and margins are continuous and lie in [-1,1]. Thus the optimum exists even if weights or maximizing predictions are nonunique. With Amazon's five classes, the primary LP has only five variables (four weights and t), one sum equality, four lower-bound constraints and four class-margin inequalities. This is a fixed-size diagnostic, not training. Forming the gaps and a certificate takes O(4C) arithmetic per node; solver/I/O overhead and actual latency remain unmeasured.

Uniform weights 1/4 are feasible, so t* is at least the minimum margin of the arithmetic probability mean computed from this same P. The LP permits arbitrary label-dependent weights at each node, without graph smoothness, a learnable gate, ridge or a moment estimator. It is therefore an upper capability bound for all uncorrected operators that stay in this exact restricted hull, including globally or locally estimated moment pools and posterior projections. Their achievable performance can be much lower.

### Exact sign interpretation

- **t* > 0:** some allowed mixture makes y uniquely highest. Strict correction is possible in this output family.
- **t* < 0:** every allowed mixture has some class strictly above y. Even a clairvoyant dense mixture cannot correct this node.
- **t* = 0:** y can be a maximizer, but no allowed mixture makes it uniquely highest. A decoder's tie rule determines whether a tied correct prediction is attainable.

A strictly positive value remains attainable with every w_m strictly above the floor: move an optimal weight vector slightly toward uniform weights and use continuity. Thus the strict feasibility conclusion also applies to the existing finite-softmax gate w=0.0125+0.95*softmax(a). A closed-polytope tie witness that needs a weight exactly at its floor might be only a limit of that finite-softmax parameterization; do not report it as an exactly attainable finite gate.

## Ties and degeneracy

The saved decoder chooses the smallest class index at exact probability ties. For that rule, y is correct exactly when its margin is **strictly positive against every c<y** and **nonnegative against every c>y**. If t*=0, a deterministic tie-resolution subproblem can decide closed-hull attainability:

    require all d_c^T w >= 0;
    maximize eta >= 0 subject to d_c^T w >= eta for all c<y,
    with the same weight constraints.

For y>0, tied-decoder correctness is attainable iff eta*>0. For y=0, any weakly feasible optimum is decoded correctly, so no subproblem is needed. This subproblem only resolves the zero-margin case of the same diagnostic; it is not an additional fitted arm or tuning axis. If exact finite-softmax attainability is required, also require w_m >= ell+eta in that feasibility test and maximize eta; the common positive slack enforces both an interior weight vector and all required strict lower-class margins. For y=0 this interior test still checks whether a boundary-only weak witness can actually be attained.

Identical members, repeated rows, zero probability coordinates, redundant competitor constraints and an optimal face are valid mathematical inputs. Identical members reduce the LP to their one fixed prediction. A wrong identical bank has no mixture rescue; this still does not show that a nonlinear decoder or extra context is uninformative. With C=2, there is one competitor and the optimum assigns the residual mass 0.95 to a member with maximum d; multiclass tradeoffs require the joint LP. C>=2 and 4 ell<=1 are required. With a floor of 1/4 the only mixture is uniform; with a larger floor the problem is infeasible. No floor sweep is proposed.

## Why member-correctness and pairwise tests are insufficient

These exact five-class constructions are analytic examples, not task observations or executed experiments. Coordinates are (y,a,b,c,d).

**No member correct, yet a native mistake is correctable.** Three members predict (0.40,0.53,0.05,0.01,0.01); the fourth predicts (0.40,0.05,0.53,0.01,0.01). Every member is wrong. Their mean is (0.40,0.41,0.17,0.01,0.01), also wrong. Put total weight 1/2 on the three first members and 1/2 on the fourth, for example weights (1/6,1/6,1/6,1/2). All exceed the floor, and the mixture is (0.40,0.29,0.29,0.01,0.01), with t*=0.11. The value is optimal because coordinates a+b always sum to 0.58. Thus an 'any member correct' union misses genuine mixture capability.

**A correct member need not suffice under the floor.** One member predicts (0.5001,0.4996,0.0001,0.0001,0.0001), and the other three predict (0.0001,0.9996,0.0001,0.0001,0.0001). The first member is strictly correct. Its weight cannot exceed 0.9625. The maximal true-versus-a margin is 0.9625*0.0005 + 0.0375*(-0.9995) = -0.037, so no allowed dense mixture is correct. The pure correct member is excluded by the floor.

**Separate opponent feasibility is not joint feasibility.** Two members predict (0.30,0.59,0.09,0.01,0.01), and two predict (0.30,0.09,0.59,0.01,0.01). Each of a and b can individually be beaten by choosing the opposite group, but a+b=0.68 for every mixture. At least one is >=0.34, so t*=-0.04. This also rules out treating 'no one common competitor beats y in every member' as an exact correctability test.

## Cheap certificates and honest numerical reporting

A useful dual identity follows from the finite bilinear max-min problem (equivalently LP duality). For lambda on the simplex over competitors,

    t* = min_lambda [ell * sum_m sum_c lambda_c d_(m,c)
                     + alpha * max_m sum_c lambda_c d_(m,c)].

Every feasible w supplies a lower bound L=min_c d_c^T w. Every feasible lambda supplies the displayed upper bound U. A strictly positive certified L proves strict correction; a strictly negative certified U proves impossibility. A dual optimizer also gives an interpretable mixture of competing classes that obstructs the true class. A one-class lambda gives the inexpensive necessary bound

    t* <= min_c [ell*sum_m d_(m,c) + alpha*max_m d_(m,c)].

All these separate upper bounds can be positive when the joint optimum is negative, as the third example shows. Use them as certificates or explanations, not substitutes for the exact multiclass LP.

A future admitted implementation must validate class/member/node ordering, finite nonnegative simplex rows, floor and label bounds. It must not silently clip or renormalize malformed inputs, which would change the hull. It should retain one deterministic feasible primal witness, dual witness, objective interval, feasibility residual and gap for each queried node. No weight-norm tie-breaking fit is needed: classify capability by the value and explicit decoder feasibility, rather than by a particular optimizer.

For reporting, fix a margin guard tau=1e-10, matching the existing source solver tolerance scale. Numerical results count as strict-correctable only with a valid lower certificate >tau, and blocked only with a valid upper certificate <-tau. Bounds must include residual/roundoff correction; tau by itself is not a proof of a bound. Values near zero, a wide certificate gap, invalid input or solver failure remain **unresolved**, not exact ties or blocked nodes. Exact zero-margin declarations require an exact certificate or a separately justified arithmetic bound; the tie subproblem needs the same discipline. Do not change tau after inspecting outcomes.

The primary P convention would be stable FP64 probabilities derived from the frozen existing logits, while the native-mistake set follows the frozen native FP32 accuracy contract. Record any FP32/FP64 mean-argmax discrepancy separately. If a node is already correct under the FP64 uniform mixture, its apparent FP32-native rescue is numerical, not learned reweighting capability. No such values are accessed here.

## What the diagnostic can and cannot decide

A predeclared descriptive output, after separate authority, could report the number of original native mistakes and counts/fractions with certified strict mixture correction, certified impossibility, exact tie outcomes and numerical unresolved cases. Keep each existing designated bank/split separate; preserve the complete native-error denominator and report failures. Report continuous margin bounds as diagnostics. Add no thresholds for promotion, subgroup hunt, extra seeds, fitted route or pool-floor tuning. Oracle weight vectors must not enter served predictions, graph masks, fusion features, hyperparameter selection or fit supervision.

Many strict-correctable mistakes would show that uniform averaging leaves pointwise geometric headroom. They would not show that the available scores/context identify the right weights without y, that an error-moment estimator can recover them, or that accuracy/Brier/NLL will improve. Other previously correct nodes can also be harmed by a learned combiner.

Many blocked mistakes would show a ceiling for **this fixed restricted probability-mixture output family**. They would not demonstrate missing information in the score bank, require hidden features, or justify hidden-export admission. Mean-logit pooling, temperature-adjusted probabilities, class-specific scales, unrestricted score/hidden residual readouts and C&S correction/smoothing can leave this original hull. The margin bound applies before those operations and to the fixed raw P only. A nonlinear score/context readout can exploit information already present in P even when the probability hull cannot place y highest.

Thus the diagnostic distinguishes **correctable versus uncorrectable mixture geometry**, and identifies possible averaging headroom. It cannot distinguish bad learned weight selection from missing label-relevant information. The saved same-information nonlinear controls remain necessary for that stronger utility question. Since VALID selected base checkpoints and y explicitly enters this oracle, any eventual results are retrospective label-conditioned descriptions, with no honest predictive or serving interpretation.

## Scope and bindings

This packet derives elementary convex-hull/LP facts and analytic counterexamples. New literature identities/scopes: zero. Model/data/logit/checkpoint payload access, scientific execution, LP evaluations, hidden exports, fits, source implementation, manuscript/index/canonical edits and TEST admission: zero. The existing development protocol, its compute budget and promotion gates remain unchanged. This pre-outcome design authorizes none of those actions.

Sources: late_pooling_brier_geometry_root_assessment_20261005_v1/REPORT.md; late_pooling_brier_geometry_independent_math_review_20261005_v1/REPORT.md; amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v1/REPORT.md (fixed dense floor, native arithmetic contract and decoder); V2 REPORT.md is additionally bound to preserve the existing amendment context. Hashes in SOURCE_BINDINGS.json cover safe saved documents only. MANIFEST.json and SEAL.json authenticate this design, not an outcome.
