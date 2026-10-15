# Structural private credit: close prior art and the remaining test

15 October 2026. **ACE already specifies different shared-backbone and private-expert training risks. Shared/private gradient separation must not be claimed as novel.** Two scoped primary method reads are added here; neither is a full-paper or code audit.

## Primary evidence

| Source | Exact core read scope | Finding relevant to this mechanism |
| --- | --- | --- |
| Cai et al., [ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot](https://arxiv.org/html/2108.02385v1#S3), ICCV 2021 | §§3.1–3.4, Eqs. 1–7; saved `ace_blocks.json` blocks 19–38, especially `S3.SS4.p3` | A generalist expert sees all categories. Other experts receive overlapping category subsets. The paper explicitly says only the generalist loss updates the shared backbone; specialist losses update only their own branches. |
| Kang et al., [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/html/1910.09217v2#S3), ICLR 2020 | §§3–4, Eqs. 1–4; saved `decoupling_blocks.json` blocks 18–35, especially `S3.p5`, `S3.p6`, `S4.p3` | Instance-balanced sampling gives each original example equal probability. cRT then freezes the representation and retrains a reinitialized classifier with class-balanced sampling. Different representation/classifier training distributions are established practice. |

**ACE is the closer antecedent.** Its §3.4 specifies: “only ℰ₁ updates the backbone.” This occurs in parallel training, not merely in a frozen-backbone second stage. Its private branches duplicate the final ResNet block and use LWS classifiers; specialist class-subset cross-entropy is supplemented by an interfering-class logit penalty. It scales learning rates by subset volume and uses fixed classwise averaging of rescaled expert logits. An input-conditioned inference router is not required by this construction. These are paper specifications; the author implementation was not audited.

Kang et al. supplies earlier representation/classifier decoupling ancestry, but cRT is a second-stage classifier procedure, not a simultaneous residual ensemble. Neither paper's image results establish an effect on a native GNN.

## The narrower graph hypothesis

For the proposed mechanism, native Adam receives shared gradients from the unweighted mean of the routes' native own-risks **F**, and private residual gradients from **W**, using the same pre-update state. Unlike ACE's single generalist loss for the backbone, F includes every route's ordinary supervised risk. Unlike ACE's class exclusion, every observed example keeps positive weight in every route.

Here “class-balanced responsibilities” means balanced **assignment mass within each observed TRAIN class**, not replacement of the native class prior or BCE reduction. In the declared structural law, graph-size quartile memberships give weights `0.5 + 2*q`; the degree analogue is task-specific. The whole four-route weight vector is permuted within class for the control. This preserves its vector multiset, route/class totals and each example's total supervision while disrupting the structural association. Finite permutation need not make empirical correlation exactly zero. Inference retains the ordinary mean-logit pool.

This can test whether persistent graph-size emphasis gives weak members useful skills on particular structural subpopulations. It is plausible only if those subpopulations contain relevant unmet prediction tasks and the private residual capacity can learn them. Positive weights, symmetry and specialization alone provide no member-quality guarantee; emphasizing a difficult subset can also weaken a route elsewhere. A class-frequency or static-logit benefit should not be mistaken for structural skill.

**Unweighted shared credit is not a guarantee against shared-representation degradation.** W changes private states, which change later route Jacobians and subsequent F gradients. The shared trajectory can therefore differ even though no W gradient is directly assigned to shared parameters. ACE's motivation for protecting representation learning is prior rationale, not a theorem validating this graph implementation.

## What an improvement could support

- If structured assignment beats both the exact permutation control and the same-capacity uniform-responsibility control under the frozen evaluation, the defensible claim is that **the association with graph size helped this shared native GNN under the tested conditions**. This supports the specific assignment, not a new general separation-of-risks principle.
- If structured and permuted assignments improve equally, generic weighting or symmetry breaking remains an explanation; structural semantics have not earned credit. If structured beats permutation but not uniform, it is not an overall quality improvement.
- Better pooled quality alone does not show that weak members became competent or that the shared representation was preserved. The existing member/subpopulation quality and pool repair/harm diagnostics should carry that interpretation, with BCE/calibration tradeoffs retained. No claim of universal graph transfer, guaranteed diversity, uncertainty estimation or superiority to independent ensembles follows.

The useful contribution, if supported, would be a carefully controlled adaptation of known risk separation to **positive, persistent, within-class graph-structural private credit**, with native shared training and fixed serving. The exact combination was not found in these scoped reads; this is not an exhaustive novelty clearance or proof of absence.

## Read accounting and retained evidence

Read first: `structured_resampling_prior_scan_v1/CONCLUSIONS.md` (Owen 2007; Owen–Eckles 2012, abstract-only ancestry), plus existing RIDE/SADE notes. They receive no new reading credit. Exact title/arXiv checks found no entries for the two sources above in v72's 260 `paper_records` / 655 `existing_packets`, or the checked literature-note scopes. The v72 index SHA256 is `cfffe743dd0f73e5f3f2fa065ec284c3ce11db769103432384008c8242f4cc1e`; this is a targeted deduplication check.

ACE's CVF title/abstract and keyword-discovery excerpts were also visible: saved blocks 5, 7, 42–44, 46, 48, 50, 52–54 (at most the first 2,000 characters of each). Kang's author README was returned in full for identity and the representation/classifier workflow; numerical tables were visible but not audited or reproduced. No numerical image gain, theorem proof, author training code, dataset, checkpoint, graph experiment or TEST result was adopted. Discovery-only queries are not additional method reads. No literature-index totals were changed.

Retrieved primary HTML and deterministic paragraph extracts are retained in `sources/`. Exact byte receipts:

- `ace_html.txt`: 279438 bytes; SHA256 `50bec0dff29b70d474c318ad69ebe0572a35abbbfd30469c113d5e1e2ac5e593`.
- `decoupling_html.txt`: 235905 bytes; SHA256 `9764758eb2608626ac060d8ad296c384a61880916d49c585276102e40eb263b2`.
- `ace_blocks.json`: 35409 bytes; SHA256 `352931e12143dc620ee0f85dadae691cd6b901d17bb0ec801560403be043cf09`.
- `decoupling_blocks.json`: 45748 bytes; SHA256 `6db2aa19981e7191d8c1d8f82e9ba766e8ee7a2f30eed4b6a81e1018b0cd8a96`.
- `ace_landing.txt`: 6293 bytes; SHA256 `c88746bec2863ddfff0176c22c404a976303dc8638116c41f78bc2b1b1b34240`.
- `decoupling_readme.txt`: 9552 bytes; SHA256 `51cf787082a16b1ba4ad2de90db4925696f147ebb609d11674c39d13b40970d4`.

Source identity URLs for the additional locators: [ACE CVF page](https://openaccess.thecvf.com/content/ICCV2021/html/Cai_ACE_Ally_Complementary_Experts_for_Solving_Long-Tailed_Recognition_in_One-Shot_ICCV_2021_paper.html); [Kang author README](https://raw.githubusercontent.com/facebookresearch/classifier-balancing/main/README.md).
