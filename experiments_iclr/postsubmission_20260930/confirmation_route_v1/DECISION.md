# Independent confirmation route

**Register Peptides-func for the training-only sharing-policy branch.** It supplies 15,535 peptide molecular graphs with ten biological function labels. It is a different corpus from PPI, MolHIV and MNIST. The existing molecular adapter preserves categorical atom and bond information and exposes whole GINE blocks for the selector. The official random stratified graph split and macro average precision are widely used LRGB protocols. This is a useful transfer test; it is not a scaffold-disjoint or long-range superiority claim.

No dataset was acquired or loaded, no server was accessed, no predictive score was read, and no experiment was launched. `CONFIRMATION_DESIGN.json` registers the source-level plan. Actual dataset identities, split counts, candidate source, controls, GPU and runner remain unfrozen. **This packet is design-only, not launch-ready.** Historical paper results remain unchanged.

## Ranked decision

| Rank | Route | Decision |
|---|---|---|
| 1 | Peptides-func, bond-aware multilabel graph prediction | Primary confirmation of a sharing policy that first passes complete PPI/MolHIV development gates. Current GINE source needs a peptide loader, masked BCE/AP runner and grouped probe adapter; its core model interface is compatible. |
| 2 | CIFAR10 superpixels, ten-class image-graph prediction | Better independent route for centered narrow SAGE if existing PPI/MNIST development gates pass. Use RGB plus coordinates, CE and accuracy. MNIST's current runner hardcodes input3/name/counts, so it cannot run unchanged. Preserve packed and size-matched alternatives. |
| 3 | Molecular regression or recommendation/link prediction | Defer as confirmation. No completed suitable runner was identified. New loss, negative sampling and leakage choices would first be development. Peptides-struct reuses the peptide corpus and is not another independent task. |

Do not switch routes on confirmation outcomes. A failed Peptides study remains failed confirmation; CIFAR cannot replace it as a favorable result. CIFAR is a prospectively separate confirmation of a different, development-qualified architecture.

## Small primary family, stronger uncertainty

Use the complete official Peptides training partition, full labels, width128, three residual GINE blocks, four members, equal-graph mean-member BCE, AdamW0.001 and batch128. The primary small-backbone budget is100 epochs. This recipe is a controlled transfer of the current method, not a tuned chemical state-of-the-art recipe.

Ten paired seeds0–9 compare the training-only policy with TIED, a development-fixed private block, and one size-matched efficient control selected and locked on development only. All four receive identical data and checkpoint opportunities. A policy choosing no private block legitimately reuses TIED. The fixed-block contrast is essential: improvement from extra capacity alone does not show that the selection rule is useful. Development must already contain all fixed placements, full untying, single, multihead, independent ensemble and closest efficient alternatives before this reduced confirmation family is justified.

The three primary contrasts are policy-minus-TIED mean-member BCE, policy-minus-fixed-block mean-member BCE, and policy-minus-frozen-efficient-control macroAP. Report every seed, mean/SD/range, paired95% seed intervals and paired graph/duplicate-sequence-cluster bootstrap sensitivity. Ten seeds describe optimization variation on one corpus and split. Ten biological targets are correlated and are not ten independent tasks. Even the official graph split can contain related peptides; an input-only identity audit and grouping are required.

If a sign-flip test is used, enumerate the1024 paired assignments and disclose its symmetry/exchangeability assumption; Holm correction applies to the three contrasts. The minimum two-sided p is0.25 with three nonzero pairs,0.0625 with five, and0.001953125 with ten. Statistical reporting cannot rescue a tiny or unrepresentative gain. The practical macroAP margin is0.005. Exact effect/gate rules and unstable-bootstrap handling are in the JSON.

## Training-only policy mapping

Group exact canonical molecules and identical normalized released peptide sequences together. Split only official training groups into P/Q with seed700 and Q into three A/B partitions with seeds701–703; validation/test graphs never enter probe gradients. Fit TIED on P for25 epochs, archive0/10/25, and average graphs equally within each scoring fold. Freeze IDs, unequal group sizes and objective weights.

Apply the existing recurrence-budget-v2 rule unchanged: S is the nonnegative median recurrence, N the median noise statistic; a block must have S>N and positive median recurrence in every partition. Rank eligible blocks by(S−N) per added parameter; earlier block breaks exact ties. Copy at most one full GINE block and permit TIED. Final fits restart the canonical paired initialization. The atom lookup and boundary factors stay outside probe block groups. No threshold/window selection from peptide validation is allowed.

Parameter matching must be recounted with output10, rather than borrowing MolHIV output1 widths. Match the fixed one-block storage cap using actual CPU counts over integer widths8–512, ties smaller, mismatch<=1%. A policy that selects TIED underspends this cap; report that fact. All-layer GINE BatchEnsemble is a declared Kim-inspired adaptation: the recovered thesis tested GCN/GIN/GAT and probability averaging. It is neither an exact reproduction nor new methodological novelty.

## Convergence and compute

Repeated late checkpoint selection is an explicit gate. If an arm selects epoch91–100 and validation AP still has a positive last20-epoch slope in at least3/10 seeds, continue **every primary arm and seed** to the already registered250-epoch cap. Preserve100-epoch results. No extension of only favorable models. The source must support this continuation before first outcomes. The analogous late/improving condition at250 leaves evidence budget-limited; an exhausted resource cap leaves confirmation incomplete.

The expected70/15/15 split gives10873/2331/2331 graphs, but these counts have **not been verified from the archive**. They are planning assumptions. At128 graphs per batch that is85 training steps per epoch:40×100 fits require340,000 steps, or850,000 at the250-epoch ceiling, excluding probe, validation, replay and profiling. Ten half-training25-epoch probes add roughly10,750 minibatches plus fold-gradient passes. The first three seeds form a full-data resource pilot; they do not suffice for confirmation. No mini-subgraph or tiny-graph dataset is substituted.

Published mean size150.94 nodes/307.30 directed edges gives about19,320 nodes and39,334 edges per128-graph batch. A single four-member width128 node-state tensor is roughly37.7MiB, before internal MLPs, messages, autograd and optimizer state. Mean sizes do not bound worst-case memory. Real batch/cost measurement is required; hours cannot be inferred from these counts or CPU preflights. Use a48GPU-hour cap. If a measured pilot projects beyond it, request resources and retain incomplete evidence rather than declaring a three-seed confirmation.

## Stronger baseline boundary

The primary LRGB GINE recipe uses five layers, width208 and500 epochs. More importantly, the **reassessment authors' actual configuration** uses eight layers, width160,20-step random-walk structural features, a three-layer readout and250 epochs with cosine warmup. It is pinned at `Toenshoff/LRGB` commit `c3500cb702dcc5bdbe02783b94f24eebb7fb52c9`, `configs/LRGB-tuned/peptides-func-GINE.yaml`.

A three-layer100-epoch matched pilot cannot establish superiority over these methods. The narrow confirmation can establish whether the sharing rule transfers at its fixed operating point. A broader quality claim additionally requires a faithful, fixed stronger-reference run with its actual model/loss/provider/evaluator code and separately measured feature costs. Do not copy a YAML alone and call it reproduced. This packet adds no large HPO suite and does not launch stronger references before the development gate.

## Spectral mechanism outside PPI

Ten graph-level binary labels are compatible with centered member-gradient statistics;121 node-level targets are not mathematically required. Whole peptide groups, objective weighting and missing labels need their own frozen definition. However, the current correction targets SAGE's square neighbor map. GINE has no identical map. A specific square bond-aware transform must be fixed and checked for zero-coefficient forward identity, gradients and native update scale before any molecular spectral fit.

The current mixed PPI F2 scores are local gradient proxies, not a predictive gate: selected bases beat raw bases in nonzero low-label cells but lose to raw bases in every full-label seed. Keep that adverse evidence. The spectral branch needs predictive selected/random/raw/TIED development with matched ranks and common warm start first. It is **not launch-ready independent confirmation** here. MulticlassCE, regression and link losses also require new target-map and data-unit gates; an objective-general algebraic statistic does not validate those ports.

## Access and resource request

Before any confirmation metric, lock the method, development decisions, source, data, folds, controls, evaluator, checkpoint rule, continuation rule, all ten seeds and analysis. The future provider separates train/validation targets from test targets; it must not reuse the old test-label-fingerprinting freezer. Public test labels are not cryptographically sealed. Input-only split/overlap audits are permitted; no test target digest, class summary or predictive scoring precedes the lock. After all checkpoint identities are final, score test once and keep every outcome. The corpus cannot remain fresh confirmation after that unsealing.

**Resource specification:** one exclusively allocated idle CUDA GPU with explicit UUID/index and authorized repository path; >=24GB VRAM, A10040/80GB preferred, subject to actual preflight memory. Reserve up to48GPU hours after the development gate, then establish ETA from the full-data resource pilot. All data, environments, caches, temporary files and results stay inside the authorized project. No sudo. Only the design and primary-source packet are ready now; no runnable peptide confirmation or reserved GPU is claimed.

## Primary source provenance

The official LRGB provider/config/evaluator sources are pinned at `2410e9ee8d9309dd2120809d119a46092788eb89`; PyG2.7.0 provider at `76ff9c2ce18c8cebf52122b57e2aeadce9793d10`. The official peptide raw/split MD5s are recorded in the design. No raw graph or split archive was requested. Retrieval manifests retain the initial metric-path404 and the corrected source. Retrieved code was inspected as text and never executed.

Sources: [LRGB](https://arxiv.org/abs/2206.08164), [reassessment](https://arxiv.org/abs/2309.00367), [pinned reassessed GINE](https://github.com/Toenshoff/LRGB/blob/c3500cb702dcc5bdbe02783b94f24eebb7fb52c9/configs/LRGB-tuned/peptides-func-GINE.yaml), [official peptide provider](https://github.com/vijaydwivedi75/lrgb/blob/2410e9ee8d9309dd2120809d119a46092788eb89/graphgps/loader/dataset/peptides_functional.py).
