# Pair-specialized shared rankers: narrower prior-art conclusion

15 October 2026. Source/literature follow-up only. **The proposal has a defensible empirical question, but no established methodological novelty.** Two new scoped primary reads strengthen the ancestry beyond ordinary single-ranker AUC optimization. The existing 21-fit proposal, coefficient, quartiles, controls and serving rule remain unchanged.

## Closest verified operations

| Primary source and scope | Established operation | Relationship to the proposal |
| --- | --- | --- |
| Yang, Xu, Cao and Huang, *Learning Personalized Attribute Preference via Multi-task AUC Optimization*, [arXiv:1906.07341v1](https://arxiv.org/html/1906.07341v1), complete **Model Formulation** section, including final objective | For a given attribute, each annotator is a persistent task with its own positive–negative pair set and score `f_i(x)=W_i^T x`. The model jointly optimizes the sum of normalized pairwise squared losses with **`W_i = theta + G_i + P_i`**: common, grouped and user-specific parameters. Regularizers couple the tasks. | Direct precedent for persistent pairwise-risk specialization with shared and private parameters. Its tasks reflect distinct annotators' labels; they are not artificial similarity regimes for one label target. It serves the relevant user's ranker, not the fixed mean of same-task members. Linear scores, squared surrogate and grouping regularization differ from our native nonlinear routes and logistic risk. |
| Hu, Zhong and Yang, *Multi-block Min-max Bilevel Optimization with Applications in Multi-task Deep AUC Maximization*, [arXiv:2206.00260v3](https://arxiv.org/html/2206.00260v3), **§3.1–3.2** | §3.1 explicitly uses a **shared encoder `w_l` and task-owned heads `w_h^i`**, with a sum of task AUC objectives evaluated after task-specific CE-gradient updates. §3.2 defines task partial-AUC risks using positive–negative score differences and selected high-scoring negatives. | Direct deep shared-parameter ranking precedent, including CE and AUC in one training procedure and different pair-risk support by task. It uses bilevel/min-max optimization and model-dependent partial-AUC selection, not our one scalar BCE-plus-logistic objective with fixed input-only positive weights. The inspected application formulation predicts task-specific scores; it does not establish uniform same-task ensemble serving. |

Neither source is merely a post-hoc mixture of independently trained rankers. Conversely, **neither inspected formulation is the proposed ensemble**. Different task labels or operating ranges need not yield useful complementary errors when all routes ultimately solve one molecular label task. No published numerical gains or convergence/generalization theorem are transferred to our GIN. These are scoped method reads, not full-paper, proof or author-code audits.

## What is already established

Pairwise AUC surrogates and logistic ranking consistency were already attributed in the preceding Gao–Zhou note. The two reads above additionally establish joint training of shared/private rankers on persistent task-specific pair risks, shared deep encoders with private AUC heads, and CE-assisted AUC training. Retained ACE/sMCL, shared-branch ensembles, and the project's structural-weighting recipes establish expert supervision, persistent assignment and parameter sharing. A positive-weight floor, the symmetric-pair target calculation, quartile construction and changing the surrogate from squared to logistic are not, by themselves, new learning principles.

The proposed distinction is the **complete adaptation**: one native GIN with its existing private input residuals and BN state; one observed binary target; positive symmetric pair weights based on frozen TRAIN bond-composition similarity; native BCE plus pairwise logistic risk through all parameters; and the unchanged mean-logit prediction. The question is whether these imposed roles produce competent, useful alternatives under that fixed serving rule. Its answer remains empirical. The available scopes neither establish the benefit nor certify that no earlier paper implements this combination.

## What the current comparison can establish

The existing common-pairwise control separates ordinary ranking-objective benefit from assigning persistent roles. The exact whole-pair-vector permutation separates the chosen structural association from generic persistent weighting. Native-BCE and common-pairwise M1/I4 retain the user's reference bar and expose an auxiliary-objective explanation. If the structured bank fails those declared comparisons, the fixed recipe fails; an argument about specialization or a lower surrogate does not rescue it.

Two stronger claims would require stronger evidence **in a separate future decision**, not additions to these 21 fits:

1. **A benefit caused by sharing:** genuinely independent members trained with the *same structured pair roles*, BCE, reductions, selection opportunities and mean serving. Common-pairwise I4 alone does not isolate sharing from the role assignment. Until then, a favorable result is a comparison of complete procedures.
2. **An advantage over conventional shared/private multitask rankers:** a declared shared-encoder/private-score model using these same pair tasks and loss, with capacity and training opportunity disclosed. Existing unrelated literature results are not that comparison. A simple affine-head mean also collapses to one affine readout, so the comparator's function class and claims must be explicit rather than treating head count as independent predictors.

No new fit is recommended here. The useful present claim, if the experiment succeeds, is that this **fixed pair-role assignment improved this shared native model** beyond its matched controls and reference procedures. Architectural novelty, superior learning-to-rank generally, or a causal sharing advantage would exceed the design.

## Search boundary and unresolved leads

Targeted v72 title/identifier checks and retained scope/note searches found no previous method-read records for these two papers. Existing shared-ensemble, router, staged-learning and Gao–Zhou findings were reused. This is deduplication, not an exhaustive literature census.

Two relevant discovery leads remain **unread**: Chapelle et al., *Multi-task learning for boosting with application to web search ranking* ([DOI:10.1145/1835804.1835953](https://doi.org/10.1145/1835804.1835953)), and *RankExpert: A Mixture of Textual-and-Behavioral Experts for Multi-Objective Learning-to-Rank in Web Search* ([DOI:10.1145/3711896.3737258](https://doi.org/10.1145/3711896.3737258)). Author-locator attempts did not yield the former body; the latter's ACM PDF returned 403 and exact arXiv title search returned no entry. Metadata cannot determine their exact risk-sharing or aggregation operations. Their unresolved status further rules out novelty clearance from this narrow search.

`READ_SCOPE.json` records the two precise primary scopes, metadata-only discovery and access failures. `SOURCES.json` binds retained inputs. No experiment, implementation, scientific remote access, data/tensor/checkpoint inspection, TEST read or Git operation occurred.
