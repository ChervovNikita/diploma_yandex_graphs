# What the completed PubMed study changes

The most useful result is a distinction between **member competence** and **correct alternatives**. The plain shared bank has stronger mean members and a better selected validation pool than the original common-selected independent bank. It also has fewer nodes with any correct member. Its good aggregate accuracy therefore does not demonstrate useful member specialization.

| Seed | Shared pool errors | Shared errors with no correct member | Shared pool errors with an available correct member | Independent errors with no correct member | Independent pool errors with an available correct member |
|---|---:|---:|---:|---:|---:|
| 9101 | 347 | 343 | 4 | 229 | 164 |
| 9203 | 364 | 356 | 8 | 244 | 144 |
| 9307 | 365 | 363 | 2 | 241 | 136 |

Each row describes the same 3,942 validation nodes on one previously encountered graph and one frozen exploratory split. Seeds change optimization, not the split. Counts come from the original selected predictions; all original signatures and integer scores matched. No new paper scores or heldout outcomes were computed.

## A precise limit on prediction selection

In every shared no-correct-member case, at least one false class has a strictly larger logit than the true class in **every** member. Softmax preserves this class order. Any convex weighted average of those fixed probabilities preserves the same ordering: each weighted difference is positive, and at least one weight is nonzero. Selecting one existing member is also covered by this statement. The actual FP32 pooling diagnostic found no exception in this cohort.

This is elementary convexity, not a new theorem. It applies to the fixed probabilities and their class order. Class-specific transformations, new hidden-state interactions, new information, or retraining can change the premise. Different members being wrong about different false classes would not establish this obstruction.

## Decisions

1. **Stop the failed contrastive-reconstruction continuation.** CORE changes real predictions, but its repairs and introduced mistakes do not consistently improve the complete roster. Keep the negative and its costs. A different implementation needs a distinct mechanism and evidence-based reason, not a new name.
2. **Resolve the competent-reference question.** Individually selected independent members and a native single averaging four dropout losses distinguish checkpoint selection and gradient averaging from ensemble benefit. These controls take priority over another diversity grid. Neither the current pool advantage nor the common-error counts establish novelty or ordinary-ensemble superiority.
3. **Target useful alternatives without sacrificing ordinary learning.** The private specialist-credit candidate keeps complete ordinary supervision on shared weights while giving private factors additional sample-specific credit. Known CMCL ancestry and gradient-scaling controls are mandatory. The typed-context candidate supplies conditional label-availability evidence and needs same-information additive/single/independent controls. A cross-member hidden-state interaction is a separate prospective idea, subject to a closest-prior check and matched extra-head controls.

## Workflow after each complete study

The history researcher updates one compact synthesis, including contradictions and repeated ideas. The combination researcher identifies whether two mechanisms repair distinct errors and prepares baseline/A/B/A+B only when that reason exists. The new-method researcher uses the updated failure map to check a structurally different candidate against primary literature. Root launches the smallest full-task study that can decide the hypothesis, preserves all fixed outcomes, and advances supported candidates to unused confirmation.

Source setup checks should resolve concrete execution risks and then stop. Repeated administrative receipts, renamed losses and duplicated identical trajectories do not replace scientific progress. A fresh manuscript review follows supported paper claims; an engineering review is not an acceptance verdict.
