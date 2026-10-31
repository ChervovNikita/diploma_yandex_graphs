# Validation decision analysis: prospective diagnostic after launch

This protocol describes how the frozen shared-message study changes predictions. It separates recovered correct decisions, lost correct decisions, and disagreement that leaves both models wrong. It then checks how member decisions contribute to pooling and whether these patterns vary with node degree. These analyses describe the completed validation predictions; they do not change a model or select a new method.

The study was already running when this protocol was prepared. **This is a prospective diagnostic planned after study launch, not a preregistered primary endpoint.** This analyst inspected source and engineering metadata only. No labels, predictions, checkpoints, scientific scores, graph arrays, SSH sessions or GPUs were accessed to prepare the plan. This statement concerns the analyst's access and does not certify blinding of the entire team.

This folder is a disabled proposal. A separate implementation, a completed-family integrity audit and a separate approved CPU/disk request are required before analysis. None is supplied here. It is not an authorization to inspect partial runs, extend the training allocation, load test labels or promote a finding into the paper. Changes require a new version; keep this sealed version intact.

## Models and retained states

Use both tasks, Reddit2 and ogbn-products, all five arms and all three seeds (17, 29, 43). The four-member reference, `exact_member`, keeps a private neighborhood message for each member. It is a GNNM reference, not an independent deep ensemble. `shared_mean_message` averages normalized member states before a single shared neighborhood propagation while retaining private root, nonlinear and head states. `detached_mean_message` has the same forward shared message but stops the entire neighbor gradient path, including self-member and cross-member paths. A comparison with this arm cannot isolate cross-member feedback. The two single-model controls use the frozen base and parameter-matched widths.

Analyze the final epoch for all 30 cells first. Produce the same diagnostic atlas separately for all 30 validation-NLL-selected states. Keep the selected epochs visible; the selected-state analysis is sensitive to selection on these same validation nodes. Do not choose whichever role looks better, merge the roles, select a new epoch or pick favorable seeds or members. No further forward pass, training, tuning, inference, checkpoint load or dataset-provider access is needed.

## Aligned decision changes

Use the retained raw-logit pool and its argmax. A tie selects the smallest class index, as in the frozen scoring code. Keep validation node IDs in their stored order and require exact equality with the frozen target bundle. Do not repair an ID mismatch by sorting, taking an intersection or dropping nodes.

For each task, seed and checkpoint role, compare all ten unordered arm pairs. The ordered arm list in `PROTOCOL.json` fixes which arm is the reference and which is the candidate. Explain `shared_mean_message` versus `exact_member`, and `detached_mean_message` versus `shared_mean_message`, first; retain the complete ten-pair atlas regardless of direction or size of effects.

For every pair, report counts and fractions of validation nodes for:

- both predictions correct;
- reference correct and candidate wrong (lost correct decisions);
- reference wrong and candidate correct (recovered correct decisions);
- both predictions wrong, split into the same wrong class and different wrong classes;
- predictions that differ.

The net accuracy change, in percentage points, is `100 × (recovered − lost) / number of nodes`. Overall this must agree with the frozen accuracy contrast. Changed decisions are not all improvements: two different wrong predictions are disagreement without recovery. A node changing state does not establish the mechanism that caused the change.

## Members and pooled decisions

For each arm and state, count how many members predict the correct class at each node. Report every possible count, from zero to the number of members, its node denominator and the number for which the retained pool is correct. This shows whether pooling preserves correct member decisions and whether a pool can recover a class that none of the member argmaxes selected.

Also report pooled-correct/all-members-wrong and pooled-wrong/at-least-one-member-correct counts, each both as a fraction of all nodes and conditional on its stated denominator. Retain all members. Do not select a new best member. The existing validation-selected member index, if used for context, is fixed from the selected state and reused at the final state.

For every member pair in each four-member arm, report both-correct, exactly-one-correct, both-wrong/same-class and both-wrong/different-class counts. Report total decision disagreement and error Jaccard with their denominators. Separate disagreement with one correct member from disagreement with both members wrong. These are descriptive categories; a correct member in a pair does not by itself mean pooling uses that decision successfully. Average pair statistics within a seed before summarizing optimizer seeds. Six dependent member pairs do not supply six independent repetitions. For single models, pooled and member decisions must match; member-pair metrics are not applicable.

## Structural degree groups

Use only the existing `degree.npy` arrays listed in `INPUT_CONTRACT.json`. Degree here counts **incoming raw edge entries** in the exact stored graph, including duplicate entries and any existing self-edge entries. It is computed before duplicate coalescing and is the degree used by the frozen incoming-mean normalization. It is not a count of distinct neighbors. Do not reconstruct a graph, change edges, add loops or open features or a public dataset-provider pickle.

Use the fixed groups: 0; 1–4; 5–16; 17–64; 65–256; and at least 257 incoming entries. Produce all arm and pair diagnostics for each group and for the complete validation split. Always show node counts. Keep empty groups as zero counts with undefined fractions; retain groups below 30 nodes and mark them small. Do not change boundaries after looking at results. These groups overlap only with the separately reported overall split, not with one another.

Node degree is an observed structural property, not an intervention. Associations cannot establish a causal reason for performance changes, distinguish feature effects from graph effects, or support claims about a population of unseen graphs. Small-group differences are descriptive.

## Summaries and uncertainty

Show the three seed values for every reported contrast. For the declared arm-pair quantities—net accuracy change, lost-decision fraction, recovered-decision fraction and decision-disagreement fraction—report an unweighted mean across the three paired seeds and a descriptive two-sided 95% Student-t interval (`df = 2`, critical value `4.302652729911275`). Use the sample standard deviation of those three seed values divided by the square root of three. Keep intervals untruncated and label their units. Produce these summaries separately by task, role and degree group. An empty denominator gives no interval. Do not average tasks or checkpoint roles.

These intervals concern optimizer variation on one fixed graph and split; three seeds give weak information about that variation and cannot diagnose the approximate normality assumption. They are not simultaneous intervals across the many diagnostic comparisons. This protocol makes no significance decisions, multiplicity-adjusted claims, acceptance judgments or generalization claims. Validation nodes and member pairs are not independent experimental repetitions. Do not use a node bootstrap or node-level hypothesis test. A zero sample variance yields a zero-width descriptive interval, not proof of certainty.

## Outputs and integrity

The prospective output tables are specified in `PROTOCOL.json`; no result tables or illustrative numerical data are created now. Each future table must include task, role, seed where applicable, arm or pair, stratum, raw counts, denominators and undefined-value reasons. Retain losses, ties and gains. Keep the full atlas even if a later main-text figure focuses on the two designated contrasts.

Three fixed figures are proposed: lost/recovered decision counts with all seed points; member disagreement separated by correctness; and a signed degree-group map for all ten arm pairs with denominators and a separate panel for each task and role. Use a fixed divergent color scale across panels, set from the complete atlas rather than selected favorable rows. Empty values are visibly marked. Do not render figures until a separate implementation and execution request are approved. No node IDs, logits, node-level predictions or label payloads are copied into outputs.

First require all 30 fits, all 30 replays and the qualified family lock. The later request must bind the exact family audit, admission, target freeze, approved analysis source and output location. Use stored fit logits only after replay qualification; do not load an additional replay-logit copy. Verify input hashes before and after use. This analysis needs the two existing validation-target bundles and two degree arrays; it requires no training targets or test labels. Missing cells, malformed data, mismatched identities, an integrity failure or a resource limit stop the whole analysis without a partial scientific summary or automatic retry.

The plan includes explicit count identities for checking an implementation. They are consistency checks, not new training runs or evidence that the scientific hypothesis holds.

## CPU and disk proposal

The retained float32 member and pooled logits contribute 1,285,847,040 payload bytes for 60 inputs. Node-ID payload in those inputs contributes 15,125,280 bytes. The two degree arrays contribute 21,455,952 payload bytes, and target-plus-ID bundles contribute 1,008,352 payload bytes. These are metadata-based payload calculations; serialization overhead is separate and actual files are not inspected here.

Stream one logit input at a time and retain compact decisions/correctness only for the five arms in the current task, seed and role. Proposed limits are four CPU threads, 2 GiB process RSS, 600 seconds for the whole supervised process, and 128 MiB new output disk. GPU use is zero. Do not duplicate numerical inputs.

The engineering load/hash and artificial metric timings yield a **5.20-second component proxy**, not a total runtime bound. The new reductions, degree extraction, integrity checks, imports and plotting have not been measured. A 30–120-second total expectation is an unmeasured planning estimate; the proposed 600-second limit is the stopping rule. Root must reserve the complete allowance and account for it separately against the remaining wall and disk budget. These diagnostics cannot enlarge or alter the frozen training allocation, forecast or estimand. `RESOURCE_PLAN.json` gives all formulas, engineering refs and exclusions.
