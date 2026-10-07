# Method ancestry and bounded claims

This implementation reuses the published portable internal-BE v2 primitives,
the reviewed context integration v2 training/selection/replay, and the exact
context-mask/alignment helpers. SOURCE_ORIGIN and exact predecessor diffs identify
these sources. The actual public CLI has not yet completed native runtime
qualification or an external full training reproduction.

- Supervised cross-view same-class contrast is established by SupCon and graph
  adaptations such as BotSCL (arXiv2306.07478v1). Shared query/key maps, channel
  scaling and heterogeneous signed graph attention are already prior ingredients.
- X/X-PX/PX/P²X signatures use established graph-filter/root/high-pass/low-pass/
  multi-hop contexts. HLCL (arXiv2303.06344v1) already contrasts shared-weight
  high/low-pass graph encoders. The predictor’s graph is not modified here.
- Chen et al., arXiv2204.07596v1, distinguishes subclass clustering from arbitrary
  within-class spread. Xue et al., arXiv2305.16536v1, studies feature suppression/
  class collapse under specified encoders/distributions. Neither proves that
  this model collapsed or that these fixed signatures improve coarse-label accuracy.
- Positive mining in graph contrastive learning, Zheng and Cheng,
  DOI10.1007/s13042-025-02924-2, published16February2026, describes multiple
  positive targets and a Beta Mixture Model in its publisher abstract. The
  accessible scope was abstract-only; complete-rule duplication remains unverified.
- BatchEnsemble, TabM and retained CDLG/DICE/HGEN/DIVE/SuGAr/DIBS/FoRDE work are
  direct sharing, factor, pairing and diversity ancestors. Different per-member
  contrastive targets are not asserted to be a new primitive.
- The native backbone is the pinned Polynormer implementation supplied separately
  by the caller from its upstream repository. No new backbone or auxiliary head
  is introduced.

The controlled question is whether persistent label-compatible context targets
improve useful shared-route predictions beyond their common-target and fixed
permuted controls, and beyond ordinary plus objective-matched single/untied
references. The common target matches aggregate normalized target mass. At equal
deterministic score matrices the average loss/score cotangent agrees; native
independent dropout does not imply equal scores, Jacobians or trajectories.

Unselected same-class rows remain denominator distractors. This can harm
competence; TRAIN label compatibility is not a competence guarantee. Positive
coverage is an unavailable oracle and can fail to be served by the probability
pool. Hidden separation, a lower auxiliary loss or a larger aggregation gap is
not enough to establish practical utility, causal graph evidence, novelty or
generality. The original staged design treats three seeds on one selected
WikiCS development population as exploratory. General superiority requires
prospectively chosen unused split/task evidence and capable controls.

This source does not alter any running study, select a favorable arm after
outcomes, change original-paper scores or claim model performance. All runtime
outputs from the public CLI identify caller executions separately from the
registered author family.
