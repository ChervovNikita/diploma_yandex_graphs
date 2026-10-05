# Train-only transfer prior and implications for the shared-backbone method

## Result

The proposed learning rule has a concrete implementation difference but no established methodological novelty or predictive advantage. Two additional primary method scopes remove a potential misleading novelty claim: transfer-oriented optimization with no adaptation at inference is already established, and shared features with persistent separate source-specific heads are also established. This does not by itself disprove utility of the graph recipe.

## MetaReg

**Balaji, Sankaranarayanan and Chellappa, MetaReg: Towards Domain Generalization using Meta-Regularization, NeurIPS 2018.** Primary publication: https://papers.nips.cc/paper_files/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html . The exact published PDF and extracted text are retained here. Inspected scope: Figure 1 caption, Sections 3.1–3.4, Equations 1–3, and Algorithm 1; related-work discussion only for the MLDG identification. This is a scoped method reading, not a full-paper reading or an independent verification of its results.

The source explicitly uses one shared feature network and separate task networks trained on each source domain. It differentiates through virtual head updates from a current head on one source domain to learn a regularizer that improves loss on another source domain. The summary freezes the feature network after initial supervised training. Final training uses the learned regularizer and a single fresh feature/task pair on all source domains; no new target-domain labels are needed at inference.

Consequently, shared features, separate persistent heads, train-only cross-source episodes, and differentiation through a private update are prior ingredients. Our proposed rule instead updates the shared core through private updates, evaluates the served mean-logit ensemble with an own-loss anchor, and recomputes committed private updates at the new core. Those are recipe differences. MetaReg does not show that arbitrary endpoint-separated subsets on one graph are genuine source domains, or that this recipe beats a capable single or untied ensemble.

## MLDG

**Li, Yang, Song and Hospedales, Learning to Generalize: Meta-Learning for Domain Generalization.** Exact inspected manuscript: https://arxiv.org/abs/1710.03463v1 . Retained PDF SHA256: `e0e2de498e7eed8569f9737bc169135ae886805f721cbf864cc3914eba49a872`. Inspected scope: Methodology / Meta-Learning Domain Generalization, supervised-learning subsections through Final-Test, Algorithm 1, and Analysis of MLDG through Equation 7 on PDF pages 2–4. The experiments, reinforcement-learning method, and alternative variants were not evaluated in this scope.

MLDG divides existing source domains into meta-training and virtual testing domains. Its objective scores training loss and the virtual-testing loss after a training-gradient step. The final model is deployed without adaptation. Its first-order expansion already relates the objective to agreement between training and virtual-testing gradients. Therefore neither train-only learning-to-generalize, the extra derivative through an update, nor a gradient-transfer interpretation can be claimed as a new principle here.

Our graph proposal partitions supervised queries by endpoint incidence, rather than sampling independently specified source domains. It adapts private routes while updating a tied core for the ensemble prediction and member competence, then serves recomputed committed weights. Node reuse and shared neighborhoods make those graph episodes correlated. We must call them a transfer regularizer on one graph, not an unbiased cross-fitting estimator or certified domain-generalization task distribution.

## Concrete consequence for method development

Keep the existing frozen frame comparison unchanged. The new learning rule remains source development, not an adopted successful method. If exact native derivatives are feasible, a representative pilot needs both the capable adapted single and the untied ensemble using the same adaptation schedule. A detached-update control isolates adaptation credit; exposure/masking-matched random separation and ordinary training isolate the endpoint-specific argument. Equal loss weights remain the existing prospective choice, not a new tunable novelty claim.

For a one-step private SGD update, a first-order expansion of outer loss gives the familiar inner/outer private-gradient agreement term. MLDG already supplies that interpretation. The ensemble outer loss changes the outer gradient through the served mean and competence anchor; the graph query construction changes which learning transfer is measured. A contribution must be supported by those differences improving prediction over the required controls, not by renaming the meta-learning operator or adding embedding disagreement.

The native sparse derivative feasibility check can fail because of an implementation limitation. Such failure must be recorded as native-runtime incompatibility; it is not evidence against the scientific hypothesis. A declared exact constant-adjacency differentiation wrapper could be assessed separately with forward, first-derivative and mixed-derivative parity. It must not silently replace the native gradient in a fit.

## Reading custody

Two new scoped primary method readings are recorded in this packet. The failed Semantic Scholar discovery request is retained. NeurIPS index and abstract retrievals are discovery, not reading credit. No experiments, numerical model executions, validation/test outcomes, original-score changes, novelty clearance or manuscript review occurred during this reading.
