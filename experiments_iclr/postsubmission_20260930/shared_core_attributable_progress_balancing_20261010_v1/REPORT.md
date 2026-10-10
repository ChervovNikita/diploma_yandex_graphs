# Balance shared learning using its own effect on members

**One conditional hypothesis:** use FAMO-style scalar loss-progress feedback to allocate shared-core learning, while retaining each route's original private own gradient. Measure the shared update's contribution separately from private progress. This changes acquisition credit; it adds no attention, auxiliary diversity loss, adapters or teachers. It is unimplemented and changes no current family.

## Why this could preserve competence

A joint update can reduce a member's total loss while its shared component damages that member: a helpful private update can hide the damage. Pairwise negative gradient cosine does not diagnose the realized effect of Adam, moments, finite steps and nonlinear graph computation. Conversely, weak members can reflect capacity or generalization rather than interference. The hypothesis requires repeated actual shared-step harm or lagging progress; otherwise this review supplies no reason to prioritize it.

Let θ be all common parameters and φ the four private rows. For a completed proposal θ⁺,φ⁺, define deterministic, dropout-off **TRAIN** own CE at three states:

```text
ℓ⁰_m = ℓ_m(θ,  φ)          current state
ℓᴮ_m = ℓ_m(θ,  φ⁺)         private-updated counterfactual
ℓᴰ_m = ℓ_m(θ⁺, φ⁺)         actual new state
r_core,m = log(ℓᴮ_m+ε) − log(ℓᴰ_m+ε)
```

Negative r_core means the realized shared change increased this monitor loss after private progress. Total log-loss progress decomposes exactly into private progress plus r_core. This isolates a finite function change on the monitor; it is not a causal explanation of heldout mistakes.

## One proposed update

Maintain four scalar logits w. Use `z=softmax(w)` and normalized shared weights `v_m ∝ z_m/(ℓ⁰_m+ε)`. Supply `∑m v_m ∂θ L_train,m` to native AdamW, but retain `¼ ∂φ_m L_train,m` for every private row, including its normal moments/clock/decay. Weights are stopped when collecting gradients. The same native graph, architecture, own CE, dropout law, horizon, selector and reducer remain.

After committing the update, use

```text
w_next = w − β [ J_softmax(w)ᵀ r_core + γw ].
```

Proposed fixed scalar settings are w₀=0, ε=1e−8 and γ=.001 from the retained FAMO algorithm, with **β=.01 a prospective choice**, not a selected or author-default claim. No weight-rate grid is proposed. The next ℓ⁰ can reuse the previous ℓᴰ. Inverse-loss weighting is part of FAMO; this is not simply assigning the highest weight to the largest loss. The denominator still reflects private learning; only the temporal bidding signal separates its contribution.

## Closest primary and collision limits

[Fast Adaptive Multitask Optimization, arXiv2306.03792v1](https://arxiv.org/html/2306.03792v1), §§3.1–3.3/Eqs10–13, already supplies normalized inverse-loss weights, softmax-logit feedback from loss changes, and weight decay. Its small-step derivation relates to MGDA on log risks. Our counterfactual shared contribution and unchanged private gradients are a block-specific adaptation. Native FAMO, arithmetic-mean stationarity, common-descent, finite-Adam and member-competence guarantees do not transfer. Adam may still harm members; feedback only responds afterward and can overreact to difficult or irreducible cases.

Saved GEM/private-progress projection already separates shared and private effects and covers closest-step loss constraints. This proposal instead changes future shared credit through four scalar states, with no QP, retained four-gradient bank, rejection or hard nonincrease claim. PCGrad, Nash-MTL and MGDA already cover member/task gradient bargaining; the broad idea is attributed optimization, not a novel learning principle. Sparse partial sharing and teacher-free staged correction were also checked and are not reopened.

## Decisive control and falsifier

In a separate fixed three-arm comparison, retain native uniform shared4 and compare two otherwise identical block rules: **total-progress FAMO** uses `log(ℓ⁰+ε)−log(ℓᴰ+ε)`; the candidate uses r_core. Both keep identical private gradient law, scalar settings, initialization, stochastic streams and monitoring cost. This directly tests whether separating private progress earns anything beyond generic adaptive weighting. Existing capable factorized M1 and genuine I4 remain necessary references.

Prediction: core feedback reduces persistent shared-induced monitor harm and improves worst-member prediction quality while preserving mean-member and served quality across complete paired seeds. Reduced TRAIN harm alone is insufficient. If total feedback matches it, separation is unnecessary. If native uniform matches it, adaptive shared credit is unnecessary. If worst-member gains accompany worse pooled accuracy/NLL or large new harms, reject usefulness. Root must lock practical criteria before any fit; no current or partial outcomes select this proposal.

## Estimated work and status

Each update adds two no-grad full-bank TRAIN monitors: **8 route forwards**, plus an initial4; no new gradient vectors beyond one route's ordinary accumulation, and one old-core snapshot. For one three-seed arm capped at1,000 updates, this is at most24,012 extra route forwards. Three monitored arms would add72,036 to their ordinary training/VALID work. Snapshot cost is one shared-parameter copy; monitoring activations can be released route by route. Exact memory/backward implementation remains unqualified. Serving parameters, trajectories and reducer do not change; no wall-time saving is assumed.

This is one actionable, conditional next question, not a launch or broadly useful wrapper claim. One retained primary method scope was revisited; zero new identities, downloads, full papers, author-code audits, numerical jobs, outcome/label payloads, remote hosts or owner infrastructure. Exact passages and source hashes accompany the memo.
