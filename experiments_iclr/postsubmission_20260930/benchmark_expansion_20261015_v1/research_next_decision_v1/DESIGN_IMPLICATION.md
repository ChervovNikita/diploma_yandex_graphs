# Normalization policy before large benchmark compute

15 October 2026. The final study review should explicitly decide which normalization setting evaluates the extensions. Preserve original GNNM with its original shared running state as a named comparison. A consistent private-BN extension setting is a credible alternative where native BN exists, conditional on the complete nine-fit review. Applying every extension to the current shared-BN setting risks spending substantial compute on a limitation already observed in MolBACE. Applying private BN everywhere would extrapolate one setting and confound extension gains unless its matched controls use the same policy. The frozen 6,460-fit roster and counts are unchanged by this note.

## What the completed evidence supports

In MolBACE GIN with dropout 0.5, private BN raises mean member AUC from 62.01 to 72.63 and pooled AUC from 67.78 to 73.61; independent four remains ahead at 74.85. Every seed improves over original GNNM. Fixed-weight bank assignment further establishes sensitivity to route–bank correspondence: own banks average 73.61 AUC versus 59.06 tied and 57.98 across cyclic mismatches. The tied construction is not the original shared EMA, and one mismatch improves seed 101. These findings establish a consequential normalization issue in this setting, not a universal molecular-architecture result.

The probability-loss tradeoff matters: private-BN pooled BCE is 1.086 versus 0.719 for original GNNM. Better ranking therefore does not establish better probabilities. Different selected epochs, EMA clocks and stopping histories also prevent attributing the entire gain to correspondence alone. Completed evidence and exact source pins are in [MEMO.md](MEMO.md) and [SOURCES.json](SOURCES.json).

## Three design choices

| Choice | Scientific value | Cost or interpretation limit |
| --- | --- | --- |
| Keep shared BN for every extension | One source-consistent normalization convention; preserves existing method definitions. | Estimates extension effects conditional on a locally weak running-state policy. A null result need not reject the extension under competent route normalization. |
| Use private BN consistently for BN-containing extensions | Tests the new mechanisms against a stronger observed shared-backbone reference; avoids spending a full matrix estimating the same normalization contrast repeatedly. | Requires matching each extension's ingredient, objective and information controls under private BN. Gains over original GNNM alone combine normalization and the extension. Transfer beyond the completed cell remains unestablished. |
| Stage a limited normalization diagnostic | Resolves a decision-changing ambiguity on one representative setting before broad allocation. | Supports only that mechanism and setting. It should not become an automatic normalization cross-product across all arms. |

A consistent extension policy must retain **both original GNNM and a private-BN GNNM reference** wherever the latter is used. The extension's primary comparison is its exact mechanism control under the same normalization. For C6 that means mean/global/conditional under the same all-block J, not original own-F GNNM; for C2 it means structured/permuted responsibility with uniform reference. Native M1 and genuine I4 retain their original native normalization. No equipped M1/I4 fits are implied. BN-free backbones retain their native setting; do not add BN to make the policy universal. Neither arm-by-arm VALID winner selection nor silently combining private BN with every method is a defensible policy.

## What the full nine fits must resolve

Review BN effects in every geometry/correction cell, all signed per-seed contrasts and AUC/BCE tradeoffs. Determine whether the combined private-BN model improves on its individual ingredients and its private-BN reference. A less-negative interaction alone is insufficient. Retain member quality, pooling behavior, selection/stopping epochs, failed or capped outcomes and actual costs. No missing cells should be imputed or favorable subsets promoted.

Coherent benefit across these cells would support a common private-BN setting for a subsequent bounded molecular extension screen. Heterogeneous effects or substantial proper-loss harm favor a limited diagnostic and a narrower claim. Neither outcome establishes that C2 responsibility or C6 aggregation has the same normalization interaction: their credit and served objectives differ. These are decision branches, not recommendations based on partial nine-fit scores.

## Smallest honest comparison before committing large compute

First finish and review the existing nine-fit study. Then, if an extension merits a bounded screen, compare that extension with its already specified matched controls **under one normalization policy fixed before training**, on one retained representative BN-containing molecular setting. Preserve original GNNM and I4 as references, reusing completed controls only when source, policy and selection bindings match exactly. This is sufficient for a conditional mechanism claim; it does not require crossing all extensions with normalization.

If normalization dependence would change the large-study decision, use one sentinel mechanism contrast under both policies. Both treatment and its matched control must be represented under each policy; adding only a private-BN treatment to a shared-BN control cannot distinguish the effects. Escalate no further without a specific unresolved decision.

Where ordinary TRAIN BN permits it, prospectively retaining shared and route-specific TRAIN histories during otherwise required fits can supply identical-weight, identical-epoch readouts without a second optimization trajectory. Such a readout separates state sensitivity at those weights; it does not replace comparison of separately selected stopping policies. Existing shared histories cannot reconstruct missing private banks, and VALID must not fit or recalibrate them.

The appropriate default for the final review is therefore to **retain original GNNM, treat private BN as a substantive common-reference choice for extensions, and resolve that choice in a bounded stage before committing the large budget**. The complete nine-fit evidence must determine whether that stage retains shared BN, adopts private BN or requires a sentinel contrast. No run, roster edit, new count or broad grid is authorized here.
