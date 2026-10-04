# Current method decision

5 October 2026. This is a research decision, not a new manuscript result.

The objective is a useful shared-backbone ensemble that improves predictive quality over a competent single and an ordinary independent ensemble. The literature is informing specific training operations and controls. The objective has not been achieved.

## What is being learned

The running Collab and DDI extension trains one member to explain compatible missing connections around both endpoints of a link. The separate control mixes explanations at the two endpoints independently. The same visible context, target supervision and native prediction pool are retained. This tests whether learning whole patterns improves the predictions actually used for link ranking.

A proposed next operation applies pattern supervision directly to native link logits. Hide three disjoint observed TRAIN edges. Score the six possible matchings of their six endpoints in one common masked graph. Each matching gives every endpoint exactly one restored edge. A member assigns a probability to each whole matching; the loss rewards the mixture probability of the observed matching. This supplies direct gradients to the target scorers, with no separate decoder to absorb the auxiliary signal.

The degree constraint provides a precise invariant: adding any member-specific row or column offset to the nine scores changes every complete matching energy by the same constant and leaves its conditional law unchanged. The corresponding logit gradient sums to zero within each row and column. This removes additive endpoint-score shortcuts from this particular comparison. It does not eliminate graph selection bias, false alternatives or nonadditive degree effects. It is an established property of assignment likelihoods, not a new theorem.

## How the literature changes the decision

- Embedding contrast and negative correlation have close prior. Greater embedding distance alone cannot establish better predictions.
- DIVE already learns diverse graph masks; adding learned views alone is insufficient novelty.
- MaskGAE establishes masked structural supervision; GRAN establishes complete-block mixture likelihoods and member responsibilities.
- Degree-preserving switches are attributed to Maslov–Sneppen. HeaRT's temporal and alternative-filtering caveats prevent treating TRAIN-absent pairs as verified future negatives.

The proposed composition is therefore a bounded adaptation of established ingredients. Its exact prior overlap remains to be resolved. Neither the literature search nor the working loss implementation establishes novelty or predictive superiority.

## Execution evidence and next decision

One bounded CPU call verified the implemented matching probabilities, preservation of matching-law edge marginals, single/identical-member nulls and first-order gradients. Maximum gradient discrepancy was 6.77e-10 on constructed inputs. The check took 0.538 seconds; inclusive child time was 2.480 seconds. There were zero fits or updates and no graph, checkpoint or prediction reads. See `matching_loss_cpu_execution_root_20261005_v1/RECEIPT.json`.

The new loss also has a serious serving-transfer risk. A critical assessment identified a four-member path that improves matching likelihood while keeping mean target logits unchanged. Direct access to target logits therefore does not guarantee improved served predictions. No new matching fits are released from the CPU check. Any future predictive screen must score the declared native served predictions, preserve its complete fixed comparison, and include the same structural observations for the ordinary ensemble and capable single.

The exact current queues were verified at 23:48–23:49 UTC on 4 October: Amazon 10/15 fits complete (active update 601/2700); Collab 5/9 (separate seed1, epoch6); DDI 3/12 (separate seed0). No recorded failures, new launches or comparative outcome reads occurred. These cohorts continue unchanged. Their complete results are the next predictive decision point.

The completed older five-seed Collab family remains an incomplete success: private completion gains 0.8483 percentage points over single64 but trails independent4 by 0.3389 points. These consumed TEST results are exploratory evidence for the current search, not confirmation of the new intervention. Original paper scores remain unchanged.
