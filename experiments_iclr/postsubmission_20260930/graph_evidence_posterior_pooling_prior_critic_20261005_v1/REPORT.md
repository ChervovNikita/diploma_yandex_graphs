# Graph evidence posterior pooling: closest-prior criticism

## Decision

**Reject this as a new methodological direction for the present search.** The proposed normalized graph-likelihood gate over shared-backbone members is already derived in the project's saved `conditional_pattern_serving_alignment_note_20261004_v1/NOTE.md`. Its published ingredients also have direct ancestry. Replacing uniform pooling with posterior responsibilities is a legitimate model change, but neither that change nor graph coupling/shared factors establishes novelty.

An empirical adaptation could still be useful. No new method, advantage, theory result, protocol or launch is proposed by this packet. Exact duplication of every possible neural implementation by a published method is not established, and this is not a global novelty search.

## Can inference normalize an unknown edge label?

Yes. Let C be the declared context, B the observed pair of endpoint patterns, m a member and y a binary label. A coherent local model can set

    J(m,y,B|C) = pi_m(C) p_m(y|C) q_m(B|y,C).

If the component laws are normalized and the observed-pattern probability is positive, then

    P(m,y|B,C) = J(m,y,B|C) / sum_{j,t in {0,1}} J(j,t,B|C).

The prediction sums this posterior over m at y=1. Both candidate label values are evaluated; the true inference label is not supplied to a gate. Labeled joint-likelihood training instead uses the posterior over m conditional on the observed training label. Conditional label NLL and joint label/pattern NLL are different objectives and must not be equated.

These are ordinary latent-mixture/Bayes operations, not a new graph inference principle. If q is independent of y within a member, the expression reduces exactly to the saved note's structure-likelihood weighted probability pool. If q depends on y, it is a class-conditional generative mixture. A label-conditioned implementation that selects q using the true validation/test label would be invalid; marginalization avoids that problem.

Normalized probability pooling is also a different predictor from averaging raw NCNC logits. A sampled-negative ranking score is not automatically a calibrated population link probability. Calling its sigmoid a posterior does not resolve that distinction.

## Published overlap already retained in index v56

| Prior and retained scope | Established operation | Limit of the comparison |
|---|---|---|
| Newman–Leicht, *Mixture models and exploratory analysis in networks*, PNAS 2007, DOI 10.1073/pnas.0610537104; PDF pp. 2–4, Eqs. 1–13 | Observed neighbour likelihoods yield latent graph-class responsibilities proportional to a prior times an attachment-likelihood product. | Global node-class preference tables, rather than the exact proposed neural query/member composition. |
| Amini et al., *Pseudo-likelihood methods for community detection in large sparse networks*, 2013, DOI 10.1214/13-AOS1138; §§2.1–2.2, Eqs. 2–5 | One latent class explains a graph row's block-count vector; degree conditioning produces a multinomial mixture and likelihood responsibilities. | Aggregated counts and graph pseudolikelihood, not distinct fixed-cardinality endpoint identity subsets. |
| Liao et al., *Efficient Graph Generation with Graph Recurrent Attention Networks*, arXiv:1910.00760v1; §§2.1–2.3 and retained pinned loss-source scope | A mixture component jointly explains several block edges; graph-conditioned mixture weights, component likelihoods and responsibility gradients model dependence. | A graph generator; the retained scope does not certify an identical discriminative ensemble gate. |
| Chen–Liu, *Statistical Applications of the Poisson-Binomial and Conditional Bernoulli Distributions*, 1997; primary pp. 875–878 | Exact fixed-cardinality binary subset laws and their summed-product normalizer. | A graph neural parameterization remains an adaptation; the distribution and normalization are prior. |
| Retained Link-MoE method/source conclusions | Pair-specific structure/feature gates combine complete expert scores; a learned combiner is an established link-prediction alternative. | A discriminative gate is not automatically a normalized generative responsibility. Supervision and inference work must be matched. |
| Retained NCNC/IECNC and GraphLP/GAD-NR conclusions | Learned neighbour completion and structural/neighbourhood reconstruction already support graph predictors. | These scopes do not prove equivalence to every proposed local likelihood gate. They remove broad novelty claims about graph evidence or reconstruction. |

Conditional Neural Processes were suggested as further conditional-likelihood ancestry. Their primary method was not read in this task, so no exact CNP overlap assertion is made. The retained direct graph priors already settle the generic rule; an additional primary read is unnecessary for this negative conclusion.

## Why the current source cannot serve this gate directly

The saved source audit shows that on complete TRAIN the present residual teacher subsets have kL=kR=0. Every component's normalized conditional-subset likelihood is then one; posterior weights are uniform. Nonzero native completion scores do not change that teacher-law fact. Removing only the queried edge does not resolve it under the audited endpoint exclusion.

A useful observed-pattern likelihood would require a separately fixed split between encoded context and observable evidence, consistent at training and serving. The queried target cannot enter the evidence. This requirement and the inference-mask proposal are already recorded in the saved serving amendment; they are not a newly found method here. If C already determines B, conditioning on B adds no information beyond C. A restricted model may still benefit from a different inductive bias, but the conditional-information identity supplies no fitted-model improvement guarantee.

Exact local normalization does not make overlapping endpoint laws a normalized whole-graph density. Degree conditioning does not remove candidate popularity. Shared factors do not guarantee component specialization, and concentrated responsibilities do not guarantee that a member predicts the queried edge correctly. These limitations are already established in the retained conclusions.

## One conditional empirical opportunity

The remaining question is **whether identity information in a fixed, label-blind observed endpoint pattern improves served decisions beyond the same information given to a competent discriminative gate**. This is an untested adaptation question, not a genuinely new mechanism established by this review.

The decisive comparison would hold allowed graph information, member bank, training supervision and charged work constant. A normalized posterior would face uniform probability pooling and a capable discriminative structure/feature gate. A count-preserving identity control would distinguish arrangement-specific evidence from degree/popularity or generic confidence. A competent structured single predictor and independent ensemble remain necessary for a broader quality claim. A gain only over uniform pooling would establish a pooling adaptation, not necessity of the proposed likelihood mechanism.

This is a conceptual discriminator, not another protocol: no masks, seeds, datasets, tuning budgets, gates or execution are frozen here. It must not be fitted or selected against already consumed TEST predictions. Without a distinct graph operation or representative independent evidence, do not prioritize the duplicated posterior candidate as methodological progress.

## Read accounting

Index v56 and the bound saved conclusions/notes were read first. This packet performs zero new primary retrievals, searches, primary rereads or paper-reading count additions. No datasets, models, scientific servers, numerical packages, GPU tasks, private running scores or canonical files were accessed or changed. Input hashes and the exact reused scopes are preserved in `INPUT_BINDINGS.json` and `CONCLUSIONS.json`.
