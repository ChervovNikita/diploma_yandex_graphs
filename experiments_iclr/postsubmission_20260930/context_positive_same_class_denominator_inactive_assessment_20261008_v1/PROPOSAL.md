# Unselected same-class denominator rows: inactive assessment

8 October 2026. Mechanism and retained-prior assessment. No source/gate changes, fits, current outcomes or scientific input arrays.

## Judgment

**Yes: the current objective can repel unselected same-class TRAIN objects and can thereby undermine competence. Whether it actually does so is unestablished.** Label-compatible positive selection does not make the complete contrastive objective label-compatible in every gradient direction. The frozen method explicitly retains unselected same-class rows as denominator distractors; this is deliberate within-class discrimination, not an implementation defect.

**Excluding exactly those rows is a distinct, scientifically interpretable conditional ablation worth retaining as an inactive proposal.** It removes an identifiable loss component without new capacity, learned weights or extra labels. It is not a new contrastive principle or a justified rescue of the active family. Preserve the current complete context9 comparison and gates; no execution or source preparation is authorized by this note.

## Exact change and its limits

For an anchor i in either original cross-view direction, let Q_m(i,:) be the fixed row-normalized target after the original scored-panel restriction. Let P_m(i)={j:Q_m(i,j)>0}, C(i)={j:y_j=y_i}, and U_m(i)=C(i)\P_m(i). The proposal retains

```
E_m(i) = P_m(i) union {j:y_j != y_i}.
```

Use the same original targets, scores s_j=cos(h_i,h_j)/tau, views and reductions. All original positive entries, including the cross-view self positive, remain in E, so the denominator is nonempty. No Q mass is moved to unselected same-class objects. Apply the rule independently to each original direction's anchor/target convention.

With Z=sum_all exp(s_j) and Z_E=sum_E exp(s_j), one row's losses are

```
ell_full = log Z - sum_j Q_j s_j,
ell_mask = log Z_E - sum_j Q_j s_j,
R = ell_full-ell_mask = log(Z/Z_E) = -log(1-rho_U) >= 0,
rho_U = sum_U exp(s_j)/Z.
```

Writing p_j=exp(s_j)/Z, the full score cotangent is p_j-Q_j. For j in U, Q_j=0 and this is positive: a Euclidean score step lowers that similarity. Masking sets that direct cotangent to zero. For j in E, the new cotangent is p_j/(1-rho_U)-Q_j, which exceeds the old cotangent by p_j rho_U/(1-rho_U). **Every retained cotangent changes**, including stronger different-class repulsion and less negative, or possibly positive, selected-positive cotangents. This is denominator renormalization, not merely deleting selected parameter-gradient terms. A lower masked auxiliary loss is automatic at fixed scores and is not evidence of improved learning.

At parameter level, the full objective is F+alpha(A_mask+R_mean), where F is the unchanged mean-own CE and R_mean has the same symmetric-view/member/object reductions. Under a small Euclidean step, the masked-minus-full first-order change in CE is

```
Delta F_mask - Delta F_full = eta alpha <grad F, grad R_mean>.
```

Removal helps CE locally when that inner product is negative and removes useful CE-aligned credit when it is positive. It can also be zero through parameter Jacobians. This is a same-state raw-gradient calculation; Adam history, finite steps and shared/private coupling prevent a native descent guarantee.

## Why the mechanism is plausible, and how it can fail

Unselected objects may share class evidence that the coarse prediction should preserve. A static top-neighbor mask can instead make them compete because of nuisance degree, feature or neighborhood variation. In that case the added within-class discrimination can spend shared/private capacity on distinctions that do not repair class margins. The proposed exclusion permits neutrality toward those rows while preserving the chosen context attraction.

But classification does not require every same-class hidden pair to be close. The same repulsion can preserve useful subclasses, resist oversmoothing or supply beneficial regularization. Its removal can weaken specialization or let nuisance positives dominate. TRAIN label agreement is not a proof that the removed gradient conflicts with CE, nor a diagnosis of active-family errors.

Further limits are concrete:

- Retained positives still share a softmax denominator: a selected positive with p_E>Q can still be repelled. Masking does not eliminate all same-class repulsion.
- If a panel contains no different-class candidates and a row has only its self positive, the masked row loss is identically zero. With several positives it can merely match their relative target proportions. The auxiliary can weaken substantially on class-skewed panels.
- At parallel normalized same-class embeddings, same-class cosine derivatives vanish. Exclusion does not guarantee a useful differentiated positive signal or prevention of class collapse. Renormalized different-class gradients may still change.
- Masking changes class- and route-dependent denominator sizes and effective gradient scale at the fixed alpha. It may strengthen harmful cross-class directions or remove useful hard negatives. A favorable result would evaluate this complete intervention, not isolate all of those effects.
- Hidden neutrality or separation does not ensure better competitor margins. Own CE, mean/worst member competence, actual pooled repairs **and** harms, full-population NLL and the existing competent references retain their roles.

## Nearest saved priors

The retained SupCon/BotSCL conclusions already establish label-aware multi-positive cross-view contrast. In the all-same-class positive version, same-class candidates have positive target mass rather than being unselected zero-target distractors. The proposed restricted-positive, restricted-denominator rule differs from assigning positive mass to every same-class example, but belongs to established supervised/label-aware contrastive design. No original SupCon body or implementation was reread or newly qualified here.

Saved Chen et al.2204.07596v1 distinguishes meaningful subclass structure from arbitrary within-class spread and supplies weighted class-conditional contrastive ancestry. Its exact denominator cannot be assigned to this proposal from the saved conclusions. Xue et al.2305.16536v1 supplies assumption-specific collapse/feature-suppression analysis, not a diagnosis of this model. PMGCL, DOI10.1007/s13042-025-02924-2, remains publisher-abstract-only: true-positive probability estimation and multiple positives are prior, while its exact denominator treatment is unresolved. Saved AMCL and CGCL reinforce shared/multiple-view contrastive ancestry, without certifying this exact TRAIN-class mask operation.

Thus retain the attribution and bounded unresolved complete-rule comparison. Do not claim novelty for removing label-compatible distractors, positive mining, graph contexts or persistent route positives; no new paper search or body reread is needed to state the mechanism.

## One representative conditional contrast

If the denominator question remains worth answering after complete family closure, the narrow contrast is:

| Fixed policy | Targets | Auxiliary denominator |
| --- | --- | --- |
| Original shared ROUTE | Original fixed Q_m | Original complete panel |
| Masked shared ROUTE | The identical fixed Q_m | Selected positives plus all different-class panel rows |

Use every original fixed seed, the same unit initialization, two-own-view CE, panel, temperature0.2, coefficient0.05, native trajectory/selector, serving and complete paired-population readout. Reuse original terminals only if their complete identity/selection custody is compatible; never choose favorable seeds or checkpoints. No coefficient, temperature, mask-size or learning-rate grid, learned mask, new teacher or altered gate. This specifies an inactive method contrast, not permission to fit.

A credible favorable finding would preserve or improve member competence and full-population served accuracy/NLL while improving actual competitor rankings and net repairs beyond the full-denominator ROUTE reference. Lower auxiliary loss, fewer within-class pair pushes, more hidden agreement, or a good baseline-error cohort alone cannot establish the intended benefit. Record excluded counts and denominator composition so an effect due to weakened auxiliary exposure is disclosed. If specialization weakens or harms offset repairs, retain that failure; do not rescue the contrast with retuned alpha.

### The existing COMMON matching does not transfer automatically

If the mask is derived from each target's support, ROUTE retains P_m while COMMON retains support(Qbar)=union_m P_m. Their denominators now differ. Even at identical scores, the average target terms match but the log-normalizers need not; the original expected shared-gradient cancellation and equal-denominator control no longer apply. A masked ROUTE-versus-masked COMMON result would combine target allocation and denominator support changes.

A common union denominator for every route could preserve equal denominators, but it would continue treating other routes' selected same-class rows as distractors in a route where they are unselected. It answers a different question. The representative two-policy contrast above avoids silently redefining the frozen COMMON control. It tests denominator utility narrowly; it cannot establish factual context semantics, superiority to ordinary all-same-class supervised contrast, or a benefit unique to shared BE factors.

## Scope and inactive status

Read retained literature conclusions/scopes and the persistent-positive follow-up first, plus targeted current method/frozen objective declarations and prior mathematical assessments. No primary body, paper outcome, target tensor, prediction archive, model, server or current/partial comparative result was opened. A keyword scan of an older saved competence report incidentally exposed historical scalar summary text; it is not used here and its underlying files were not opened. All derivations are hand algebra; no numerical fixture was needed.

This folder is an inactive proposal only. Running source, frozen targets, active conditions, protocol permissions, gates, indices and ledgers remain unchanged. Whether the excluded component actually limits competence is still an empirical question.
