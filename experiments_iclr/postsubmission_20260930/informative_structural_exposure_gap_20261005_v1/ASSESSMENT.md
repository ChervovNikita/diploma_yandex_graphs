# Structural alternatives exposed to native link scores

5 October 2026. Source-grounded design assessment; no fitting, numerical experiment, queue change or quality result.

I would retain **one untested implementation adaptation**: train the served link scorers to distinguish a hidden three-edge matching from the other degree-preserving matchings of the same six endpoints. A finite mixture over complete assignments supplies member responsibilities. This is a distinct operation from the current residual-neighbour reconstruction auxiliary, because its energies use the actual native target logits and its observation creates competing assignments deliberately. Degree-preserving switches, structured likelihoods, finite mixtures and maximum-entropy marginal controls are established ingredients. This packet supports neither a new mathematical principle nor an advantage of sharing over independent networks.

The supplied native-batch observations (1.3599% informative two-sided queries and a joint-minus-separate gradient norm equal to 0.056% of the target gradient) motivate examining observation design. They are implementation/initialization diagnostics, not predictive evidence. Increasing the auxiliary coefficient or merely resampling existing informative masks would not supply a distinct mechanism.

## Exact observation and operation

Let the TRAIN graph be a loopless undirected **unique-pair** graph for this operation. Select three node-disjoint observed TRAIN pairs and orient them as `(a_i,b_i)`, `i=1,2,3`. Orientation and row/column order are randomized without using node IDs as predictive features. The six endpoints must be distinct. The three diagonal pairs are the observed matching. Require the six off-diagonal pairs to be absent from observed TRAIN. Their latent/future link status is unknown; they are sampled alternatives to TRAIN observations, not certified nonlinks.

Remove **every source record and both adjacency directions** for the three diagonal pairs before constructing visible graph `H`. Equivalently exclude all nine candidate pairs from `H`; the off-diagonals are already absent. Recompute the graph quantities consumed by the native encoder from `H`, including coalescing, edge attributes/weights and normalization where applicable. A surviving duplicate must never leave a supposedly hidden edge visible. This is a prospective unique-pair corruption, deliberately different from native record masking; no equivalence to the current fixed cohorts is asserted.

To make the alternatives structurally supported, admit a bundle only when every one of its nine candidate pairs has a visible length-two path in `H`. This is an explicit common-neighbour witness rule, not a claim that all accepted candidates are equally hard. It can leave few or no eligible bundles; no coverage or runtime claim is available. Selection reads TRAIN and `H` only. It does not inspect VALID/TEST pairs to decide eligibility or filter alternatives.

Run each of the four complete native member trajectories on the **same `H`**. Score all nine `(a_i,b_j)` pairs with the native target scorer. No assignment is inserted into the graph; no positive pair is revealed while scoring another candidate. Recursive completion, pair features and encoder state must all retain that same context. Denote the resulting served-type logits by `s_mij`. For each of the six permutations `pi`, define

```
E_m(pi) = sum_i s_m,i,pi(i)
q_m(pi | H) = softmax_over_the_six_permutations(E_m(pi))
q_J(pi | H) = mean_m q_m(pi | H)
L_J = -log q_J(observed_diagonal_matching | H).
```

Every candidate assignment restores one incident edge per endpoint. Thus the hidden degrees and the visible context are identical across hypotheses, while counterpart identities differ. Add this loss to the existing competence loss; the loss weight and schedule would need a prospective choice and are intentionally unspecified here. Retain the declared native serving pool and graph recipe. There is no inference mask, router or likelihood-based serving weight.

This exposes the entire alternative assignment bundle to each member. Its likelihood gives a larger training responsibility to a member already better at explaining that complete matching. It does **not** guarantee that different members specialize, and it does not prescribe different arbitrary masks to them. The proposed learning hypothesis is that members develop different useful interaction judgments after degree/popularity shortcuts cancel within these assignment comparisons, with those judgments affecting the native link scorer directly. Competence and specialization can still conflict.

For this objective, a member's target-score gradient is its complete-assignment responsibility times the difference between the observed matching incidence and its expected matching incidence. The link scorer receives that gradient directly; there is no separate auxiliary decoder whose parameters can absorb all of it. This eliminates the particular decoder-only transfer route, not the need to demonstrate improved served ranking.

## Why two hidden edges are insufficient; a precise marginal control

A two-by-two switch has only two feasible matchings. Fixing its row marginals fixes their complete distribution. It therefore cannot isolate a higher-order matching preference from its marginals; a mixture-sigmoid versus pooled-score difference there could be ordinary ranking/pooling behavior.

For three-by-three matchings, every cell belongs to exactly one even and one odd permutation. Let `epsilon_pi` be `+1` for even permutations and `-1` for odd ones. Starting from the positive six-probability `q_J`, define

```
q_ME(pi) = q_J(pi) + epsilon_pi * c
```

where `c` lies between `-min_even q_J` and `min_odd q_J` and solves

```
sum_even log q_ME(pi) = sum_odd log q_ME(pi).
```

This is the maximum-entropy law having **exactly the same nine row marginals and the same six-matching support** as `q_J`. It retains all marginal link preferences and removes the one remaining assignment dependence available on this support. The scalar root is monotone and must be differentiated implicitly; detaching the fitted root would implement a different training control. The source sketch provides that derivative without selecting numerical tolerances or solver budgets.

Each individual additive-logit matching law `q_m` is itself maximum entropy for its own row marginals. Mixing such laws can leave that family. This is established exponential-family/maximum-entropy mathematics; the operation introduces no new normalizer or compatibility principle and supplies no native-architecture representation guarantee.

Compare `L_J` with `L_ME=-log q_ME(observed_matching)` using identical scorers, observations and competence losses. At a fixed scorer state the marginals match. Independently fitted arms can learn different marginals, so a trained quality difference measures this intervention; it does not establish an isolated causal effect of dependence.

Ordinarily mixing row predictions independently and then conditioning on a perfect matching gives a simpler control, but generally changes the postconditioning marginals. It must not be described as a clean marginal-matched contrast. An explicit conceptual witness is a mixture favoring the three even matchings: all row marginals can remain uniform, while its matching law differs from the uniform maximum-entropy law. This is an algebraic illustration, not a fitted graph result or a representation theorem for the native scorer.

## Controls that can defeat the claim

The strongest ordinary comparator is four separately initialized and trained competent native models, each receiving **the identical saved TRAIN bundles, unique-pair masks and alternative support** and its own `-log q_m(observed_matching)` loss, with ordinary native prediction pooling. This distinguishes the proposed training from a good independent ensemble given the same structural hard alternatives. An untied four-model bank trained with `L_J` is also necessary before attributing any observed advantage to weight sharing; its parameters are independent but its training loss couples models, so call it an untied jointly trained control, not an independently trained ensemble.

A capable single uses one native target trajectory plus a permutation-equivariant matching decoder over the same nine pair features and visible graph context. A shared set/attention decoder can score the six complete matchings with nonadditive energies, including the native score sum, and backpropagate to the native scorer. It can represent assignment dependence beyond a unary score sum. Give it the same masks, alternatives, supervision and competent fitting opportunity; omit the matching decoder at native single serving. Its predictive performance can defeat the need for four target trajectories. Root's current capable-single implementation concerns the existing auxiliary; this packet specifies a future adaptation and does not duplicate or edit that work.

The same-architecture `L_ME` arm tests whether the remaining assignment dependence helps; the ordinary and capable-single controls test whether the complete operation is useful beyond those alternatives. Neither better assignment fit nor greater member disagreement answers the predictive question.

## Closest operations and attribution

| Source at its retained scope | Established operation | Boundary for this design |
| --- | --- | --- |
| Maslov and Sneppen, *Science* 296, 910–913 (2002), [cond-mat/0205380v1](https://arxiv.org/html/cond-mat/0205380v1), paragraph `p8.1` | Swap the endpoints of two observed edges, preserve in/out degrees, reject existing replacement edges to avoid multiple edges | Supplies the switch/null-model ancestry. It randomizes networks for structural analysis; the read paragraph does not train native link scores on hidden matching likelihoods. Three-edge assignment alternatives are an attributed extension of degree conservation, not a new rewiring primitive. |
| Li et al., [HeaRT, 2306.10453v3](https://arxiv.org/html/2306.10453v3), §4.2 and selected G/I passages | Endpoint-corrupted heuristic hard alternatives; RA/PPR/feature ranking; filtering rules and a Collab temporal caveat | Hard alternatives tied to target endpoints are prior. HeaRT explicitly changes evaluation sampling, retaining its training recipe; it is not the present matched TRAIN corruption or complete-assignment member likelihood. Its filtering caveat warns against turning observed-pair status into a future-link label. |
| MaskGAE, 2205.10053, previously scoped paper/code conclusions in index_v57 | Edge/path masking, masked-edge prediction and degree auxiliary updating a served encoder | Rules out novelty for structured corruption or inference-omitted graph supervision. Current proposal's matching constraint and actual target-score energies specify a narrower adaptation. |
| GRAN, 1910.00760, previously scoped paper/code conclusions | One latent mixture component explains a graph block; responsibility gradients | Whole-assignment finite-mixture learning is prior. This design cannot claim a new latent compatibility/mixture principle. |
| CAM, 2405.19375v4; PIFM, 2601.22107v2, previously scoped conclusions | Coherent constrained linksets; joint graph refinement with an NCNC-informed prior | Global coherence and graph reconstruction are prior. The remaining implementable increment is a tiny, deliberately ambiguous degree-conserving TRAIN observation with the native target logits and unchanged serving; its utility is unestablished. |

The loss is structured hard-alternative ranking with a mixture over complete assignments. It is not merely independent edge BCE, because it normalizes over feasible joint assignments and can learn a joint preference beyond all fixed row marginals. However, replacing `q_J` by one member's `q_m` yields an ordinary structured ranking loss; the graph observation and higher-order ensemble law must not be conflated. The exact composition is not established as a published duplicate by these scopes, and that fact supplies no novelty clearance.

## Real failure and selection risk

Collab can contain repeated collaboration records and temporal links. Excluding all observed TRAIN cross pairs while removing all duplicates of each diagonal can teach “restore historically observed unique pairs.” Future collaborations include different histories and repeated pairs. A false alternative may even become a future positive. Good matching reconstruction can therefore hurt future ranking, especially if eligibility favors dense common-neighbour communities and underrepresents sparse useful links. HeaRT's temporal filtering discussion is directly relevant to this failure.

There is also a loss-level failure: a model can improve the sum of the three true-edge scores by improving a companion edge while worsening the anchored queried link. Native per-query ranking is not equivalent to correct complete matching selection. The common competence objective mitigates this possibility but supplies no guarantee. Fully symmetric context/features can prevent useful identity discrimination; all members can collapse to one matching law; a capable single or independent ensemble can match or exceed the bank.

A TRAIN negative-selection leakage risk arises if the selector removes candidates because they appear in VALID/TEST, or uses a full-graph common-neighbour cache containing heldout links. Freeze only TRAIN-based bundle eligibility and compute witnesses from `H`. Do not call unobserved off-diagonals verified negatives. Candidate order/diagonal labels must not enter graph features. No eligibility measurements or downstream quality observations have been made here.

## Implementation and limits

`matching_objective_sketch.py` supplies the six-permutation objective and the differentiable maximum-entropy counterpart. It is unexecuted source, not an admitted trainer or qualified adapter. Graph selection/masking integration is specified above because it depends on the native record/attribute contracts. It requires nine native candidate scores per member and shared context custody; no negligible-cost claim is made.

This is a small attributed learning increment worth retaining as a design, with a clear way for ordinary/single controls to defeat it. There is presently no evidence that it improves predictions, that its eligible observations are common, or that parameter tying helps. Existing frozen Collab/DDI cohorts remain unchanged. No launch is proposed or authorized by this packet.
