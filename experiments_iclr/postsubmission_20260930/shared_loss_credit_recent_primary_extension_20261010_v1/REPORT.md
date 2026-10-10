# Shared-loss credit: focused primary extension

**Decision:** keep the FAMO composition inactive. Shared-only counterfactual loss credit and prioritizing struggling objectives are explicit prior. The remaining question is whether one particular finite-Adam/private-reference composition improves competence and served quality beyond its closest controls. No running family, proposal source or original score changes.

## What this reading changes

| Source and actual scope | Verified operation | Consequence |
|---|---|---|
| **Saved TAG**, [2109.04617v1](https://arxiv.org/html/2109.04617v1), retained §4.1 exact passage | A task's hypothetical shared SGD update is evaluated on other tasks while their private heads and the input batch stay fixed. Relative loss change `Z=1−L_after/L_before` measures positive or negative transfer. | The broad distinction “measure only the shared update's effect” is already established. This closest explicit antecedent was omitted from the prior FAMO memo and is added here. This is saved passage reuse, not a new paper read. |
| **New FairGrad**, Hao Ban/Kaiyi Ji, [2402.15638v2](https://arxiv.org/html/2402.15638v2), §4/Algorithm1; selected §5.5 implementation prose | Uses `g_iᵀd` as first-order loss decrease. Maximizes α-fair utility subject to `g_iᵀd≥0` and a bounded direction. Sets `d=Gw`, solves `GᵀGw=w^(−1/α)` through positive constrained nonlinear least squares, then takes a gradient-descent step. Larger fairness emphasis can favor struggling progress. | Harm avoidance and progress-based shared resource allocation are prior. This is a gradient-matrix solver, not loss-only FAMO feedback. Its linear-independence assumption and raw-step formulation do not automatically cover near-identical ensemble routes or native Adam. The inspected algorithm does not specify our fixed-private proposal convention. |
| **New Task Weighting through Gradient Projection**, Christian Bohn et al., [2409.01793v1](https://arxiv.org/html/2409.01793v1), §2.2 paragraph and complete §3/Algorithm2 | Prioritizes weaker tasks by changing which conflicting gradients are projected. Dynamic probabilities are proportional to the previous epoch's loss raised to γ. When there is no conflict, projections do nothing. | Targeting weak objectives specifically under interference is prior. High loss is not proof of actual shared-step harm. Prose says one sampled task stays intact, while the printed algorithm loops through a sampled random order and can revisit gradients; protected-task permanence/order semantics remain unqualified without code. No reproduction is claimed. |

The new papers are MTL sources, not verified same-label ensemble recipes. Treating members as tasks is an adaptation. The unresolved ensemble-gradient-conflict DOI `10.1145/3638248` remains metadata/abstract only; this extension adds no method claim for it.

## Exact relationship to the proposed score

Our reference states were `B=(θ_old,φ_new)` and `D=(θ_new,φ_new)`, with dropout-off TRAIN monitor losses. Define `Z_ε=1−(ℓ_D+ε)/(ℓ_B+ε)`. Then

```text
r_core = log(ℓ_B+ε)−log(ℓ_D+ε) = −log(1−Z_ε).
```

It is a monotone transform of TAG's fixed-head relative-loss score. Evaluating after private progress, using the realized joint shared Adam displacement instead of each source-task virtual SGD step, and feeding the score into FAMO's four scalar weights with unchanged private cotangents are specific state/update choices. They do not create a new counterfactual-credit or harmed-member weighting principle. No exact complete published recipe equality is established by these scoped reads; no global absence claim follows.

## Conditional usefulness and next decision

Retain the existing **identical block rule with total-progress feedback** as the decisive control, plus native uniform shared4 and capable factorized M1/genuine I4 references. A core-feedback gain must improve actual member and served predictions, not just reduce TRAIN monitor harm. Total-feedback equality removes the need to separate private progress. Native equality removes the need for adaptive shared credit. Reduced harm accompanied by worse pooled quality rejects usefulness.

Do not add FairGrad or weighted projection to the running roster from this literature check. If a later optimizer comparison is justified, its solver/gradient storage, state law and private treatment need their own fixed source contract. FairGrad's practical section reports a fairness-parameter search and an approximate RL solver; those are not a tuning license or portable competence recipe.

The FAMO proposal still pays8 monitoring route forwards/update and one old-core snapshot; these sources supply no cheaper qualified implementation or predictive guarantee. No new useful hypothesis or compute request is added.

## Reading boundary

Checked saved conclusions and exact TAG excerpts first. Six bounded arXiv metadata queries selected **two genuinely new primary method scopes relative to the consulted records**; other hits are discovery only. Both version-specific HTML bodies and exact selected passages are saved. Zero full-paper, proof, author-code, accepted-version or empirical-result audits. An initial HTML extraction mishandled void tags and merged figure containers; it was corrected before sealing, and method/algorithm passages were reread from the corrected blocks. Adjacent published table/configuration wording was incidentally exposed and unused. No numerical job, model/data/label/outcome payload, partial-family read, scientific-host access, owner infrastructure or canonical edit occurred.
