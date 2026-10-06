# A concrete CMCL challenge to finite-response specialization

**Design only.** Interpret the current response experiment first; no new fit, grid, source framework or held scoring is admitted. Original scores and manuscript judgments stay closed.

## Missing primary settings are now bounded

The saved CMCL method/Algorithm1 was reused. Only missing configuration/training/aggregation paragraphs, supplement appendixA and six small author-code scopes were added. No new full paper or distinct method is counted. The official paper footnote authenticates the code repository; the inspected snapshot is commit57f41f4b166b8544c56580f9d32ee98b997e3f59 (2018), not certified as the exact2017 results revision.

| Detail | Primary evidence |
|---|---|
|Confidence objective|KL(U||P); owner CE, nonowner beta*KL; current owner score CE-beta*KL up to a common constant|
|Published overlap|K owners per item; large-CNN search beta{.5,.75,1,1.25,1.5},K{2,3,4}|
|Author default/example|M5,K4,beta.75; README/run uses exact-gradient version0, flags default version1|
|Aggregation|Unweighted mean probabilities; author top1 argmax uses their equivalent probability sum|
|Implementation detail|Code clips probabilities at1e-10 for KL; feature-sharing is differentiable masked sum, not weight tying|
|Training discrepancy|Supplement describes Nesterov; inspected ResNet code sets use_nesterov=False|
|Unknown|Exact table-winning beta/K and exact2017 implementation revision; no graph optimum supplied|

The supplement provides full image optimizer/schedules in PRIMARY_SETTINGS.json. They do not supply graph defaults. The inspected ResNet feature-mask path remains stochastic at evaluation when enabled; the close comparator below uses the published core version0 with optional feature sharing disabled.

## One close comparator, with adaptations explicit

Use the published exact-KL overlap loss on the same shared4 common400 native continuation: M4,K3,beta.75. K3 is a published overlap value and gives3/4 owner supervision, close to the author4/5 example. K4 onM4 would give every member the true label and eliminate confidence specialization. Beta.75 is an authenticated default, not a reported optimum for Amazon. No search is proposed.

For each training node choose the three lowest current scores a_m=CE_y(p_m)-.75*KL(U_5||p_m), with fixed member-index ties and discrete ownership stopped. Differentiate the batch sum of owner CE plus nonowner.75*KL. Give S andR equal aggregate role mass, use the existing core/private SGD rates from a single incoming state, commit simultaneously, and retain the fixedH16 endpoint. Serve the mean of all four native softmax probabilities. These shared architecture, state, role weighting and optimizer choices are an explicit objective transplant; they do not reproduce the independent image experiment. Adding all-member CE to this literal loss would define a different control.

## What confidence specialization can already explain

Saved selected VALID descriptions show shared wrong-pool confidence92.60%, all members wrong on93.60% of shared pooled errors, and only+.16pp averaging benefit. These are dependent descriptive motivation, not new evidence. CMCL may lower nonowner confidence/NLL/Brier and allow owner CE to repair mistakes through ordinary training. It can therefore improve a pool without measuring a finite private response. If equally confident members all choose the same wrong class, current scores can tie; assignment contains no measured evidence of which member will learn the corrective distinction. Confidence can fall while the wrong class remains unchanged.

Finite response must add complete-pool quality and net repairs beyond that ordinary confidence mechanism while preserving individual competence. A future separately admitted evaluation should retain all-panel repairs-minus-new-errors, frozen common400 all-four-wrong strata, all15 existing confidence bins, every member's accuracy/loss and pooling gain. Increased disagreement or reduced confidence alone does not establish corrective specialization. The matched first-order control remains necessary to attribute a gain specifically to the finite remainder; CMCL differs in objective and label timing and cannot make that attribution by itself. Actual work and resources must accompany comparisons.

## One small graph-specific extension

After the current experiment is interpreted, consider one synchronous neighbor vote in the CMCL owner score: from ordinary owners v0, form row-normalized training-neighbor owner frequencies q_i,m and chooseK3 owners from a_i,m-q_i,m once, stopping these votes. Isolates useK/M, a common score shift. All losses and serving remain the same. The unit vote coefficient is an explicitly untested design constant, not a CMCL setting; no grid or router is proposed.

Keep this extension only if actual-edge votes improve net repairs/pool quality over ordinary CMCL without weakening members and a pre-frozen graph-permutation control does not explain the gain. Otherwise reject it. This one-step rule is not exact minimization of a graph objective. Neighbor owner coherence may reinforce shared mistakes; the saved graph diagnostic finds stronger unanimity in a narrow matched population, with no larger general shared pooled wrong-class recurrence. It supplies no graph-causal guarantee, generalization theorem or novelty verdict.

Primary sources: [published main](https://proceedings.mlr.press/v70/lee17b/lee17b.pdf), [official supplement](https://proceedings.mlr.press/v70/lee17b/lee17b-supp.pdf), [author code snapshot](https://github.com/chhwang/cmcl/tree/57f41f4b166b8544c56580f9d32ee98b997e3f59). Exact settings, excerpts, receipts and read accounting are sealed with this design. All temporary complete PDFs, renders and source bodies were removed; no prepared native source, dataset, labels, checkpoint or prediction payload was read or hashed.
