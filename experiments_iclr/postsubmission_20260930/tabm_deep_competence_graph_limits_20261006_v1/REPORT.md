# TabM deep rereview: competence, diversity and a graph diagnostic boundary

6 October 2026. **One existing primary paper was examined in depth; no new paper identity or novel mechanism is claimed.** TabM does not guarantee competent standalone members. Its reported strength is the collective prediction of individually weak but useful submodels. Completed39 does not establish the corresponding quality mechanism for endpoint-live transfer.

## Exact reading scope

Reused the retained **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**, Yury Gorishniy, Akim Kotelnikov and Artem Babenko, ICLR2025, **arXiv:2410.24210v3** PDF. SHA `a6988aa10d726e99c706dc92856c5a07f4e3f9fa854d08c8f854f06dc83ea34f`. No external retrieval, new author-code read or source implementation.

Assessed comprehensively for this question: PDF pp3–10 (§3 method, §4 experimental results, §5 all analysis/ablations), pp14–22 (A.1–A.4 limitations, B expanded results, C dataset choices, D.1–D.10 experiment/ablation setup and TabM/MLP tuning tables). On pp26–37, read the AppendixE definition and the MLP/TabM-family result rows for all46 datasets; other methods' rows were not audited. Figures1–7,9–11 and Tables1–10 were inspected as relevant to their sections; new full-page visual checks cover pp5,9,10. References, most introduction, detailed tuning recipes for unrelated baselines and full remaining table contents were not read. This is **one deliberate extended rereview, not a full-paper read**. Previously scoped method/A/D passages are expressly rereads, not new paper credit. New depth is the connected experimental/ablation interpretation and setup, not another keyword search.

Saved index/recipe notes were checked first. Classification probability-pooling and paper-code independent-head details already in those notes were reused; their source payloads were not reopened. Mechanical extraction of37 pages is not37 semantic page reads or paper credits. READ_SCOPES.json specifies the limits.

## What TabM actually establishes

| Mechanism | Primary evidence and relevant limit |
|---|---|
| Joint ensemble training and selection | §3.3 compares independently selected MLP×32 to **TabMpacked**, which has no weight sharing but joint ensemble-aware stopping and HPO. Packed improves before sharing is introduced. Ensemble-aware learning is therefore a competing explanation for gains. |
| Sharing and first adapter | BatchEnsemble gives TabMnaive; removing its very first input adapter gives poorer TabMbad. Keeping just that adapter gives TabMmini. Full TabM randomizes that first adapter and initializes other multiplicative adapters at1. The first adapter acts **before the first shared feature mixing**, not merely at the final head. |
| Member training | Each branch's own loss is optimized; loss of the mean prediction is observed at serving/validation. The paper's primary rule does not train a pooled-loss substitute. A.4 advises branches not exchange states and keep separate training losses. Shared batches are an allowed starting point, with only minor average degradation (§3.4). |
| Individual versus collective performance | §5.1/Fig5: individual submodels look overfitted, while their collective predictions generalize better. Best-head TabM[B] is no better than MLP on average in Fig6; TabM gains about2.15% relative to MLP while TabM[B] averages about−0.06%. These are paper tabular metrics, not graph MRR units or guarantees. |
| Member count and activation coverage | §5.3/Fig7: useful k depends on width/depth and excessive k can hurt. D.7 uses17 datasets, five seeds and separately tuned learning rates per k. §5.4 reports fewer dead ReLU units at k32, measured on2048 training objects at selected checkpoints. Different input adapters give a shared neuron multiple activation opportunities; this is not a proved mechanism for a graph encoder computed before private branching. |
| Collective pruning | TabM[G] chooses additions by collective validation improvement, not individual-head ordering; its average retained count is8.8±6.6 out of32. This is established validation-dependent selection, not authority to select Citeseer heads retrospectively. |

The main benchmark covers46 datasets (37 random and9 domain-aware splits), 28 regression and18 classification tasks. Models are tuned per dataset, typically50–100 Optuna iterations and15 evaluation seeds, using AdamW, clipping1.0, no learning-rate schedule, early stopping patience16. TabM's width/depth/dropout/learning-rate spaces differ from MLP's; k32 is not tuned. Per-dataset tables contain exceptions, e.g. ordinary TabM is below MLP on Ecom Offers. It is not a universal dominance result.

Crucially, **Fig5 is a different diagnostic regime**: depth3/width512, no early stopping, no dropout/weight decay, learning rates separately selected from a25-value grid. Its four displayed datasets are not a general experiment proving that weak members always help. A.4 explicitly leaves normalization backbones unstudied. None of this supports declaring the frozen graph recipe invalid, importing TabM's HPO, or attributing a graph improvement to parameter savings.

## Compare the actual completed39 evidence

These are frozen selected VALID summaries from227 positive queries/500 negatives each; no new score calculation or TEST read:

| Cell | Mean member MRR | Pool MRR | Pool−mean member RR | Mean pair RR-error correlation |
|---|---:|---:|---:|---:|
| E_end_joint | .292948 | .296971 | .004024 | .931153 |
| E_end_live | .219032 | .243122 | .024089 | .738203 |
| J4_end_joint | .271605 | .291541 | .019936 | .813800 |
| U_end_live | .224114 | .258518 | .034403 | .635695 |

E_joint has stronger members, higher error correlation and a smaller pooling gain, yet the highest mean MRR here. E_live has more query-level diversity but much worse served quality. Thus **a lower correlation or larger pooling gain is not itself a quality intervention**. Weak heads are not automatically disqualifying under TabM's interpretation, but the collective must repay that weakness; E_live does not outperform its competent controls. E_joint's modest mean difference from J4 (+.005431, two ofthree blocks, lower Hits10) is development evidence, not novelty or confirmation.

The methods are not equivalent. TabM performs simultaneous own-loss learning with early branch diversification/private heads and classification probability pooling. E_joint uses the graph's shared encoder/predictor factorization, a half own/half mean-logit pool outer loss and three ordinary inner/outer/repeated-inner passes. E_live additionally differentiates a persistent private support response and then recomputes commitment. TabM gives no endpoint-disjoint transfer assumption, native-Adam-history theorem or graph-pseudotask justification. The frozen endpoint-live versus matched-random comparison already loses allthree blocks, so that benefit claim stays closed.

## ONE graph-specific failure hypothesis and its falsifier

**Hypothesis:** the common graph encoder/neighborhood representation causes members to misrank the **same hard negative edge identities**, leaving no useful cancellation for the raw-mean pool. TabM's tabular experiments do not examine common graph aggregation before private prediction branches, shared endpoint neighborhoods or query-specific500-negative ranking. Its first-adapter ablation cannot certify recovery of information lost upstream in a shared graph representation.

For an existing query q and retained negative edge j, define d_mqj=s_m(q)−s_m(j). If all members have d_mqj<0, their raw-mean pool also misranks j. With K_q common strict outrankers,

\[
\operatorname{rank}_{pool}(q)\ge1+K_q,\qquad
RR_{pool}(q)\le(1+K_q)^{-1}.
\]

Ten such negatives suffice to exclude Hits10. This is elementary fixed-mean ranking algebra, **not a new graph theorem or algorithm**. The graph-specific cause—representation/neighbor aliasing—still requires evidence beyond the signs.

A strong global sign-lock explanation predicts no useful ranking differences under exact arithmetic. Existing nonzero pooling gains and substantial unequal query relations contradict that universal explanation. A partial common-negative explanation remains possible, but **current D2 summaries cannot establish it**: they contain query RR/error correlations and pair Hit10 overlap, not offending negative IDs or topology-conditional errors.

The saved counterexample makes the limit concrete. Give each of four members a distinct group of11 bad negatives among500: its group scores+1, other members score−1, positive score0. Each member ranks the positive12th, but the pool scores every bad negative−.5 and ranks it first. Add a second query everyone gets right: all member RR-error correlations and pair Hit10 IoUs equal1 despite perfect pooled recovery. If the same11 negatives instead score+1 in every member, all member query-rank/error summaries are unchanged, but pooling cannot recover. **Common query errors do not imply common offending negatives or graph information erasure.**

A falsifying use of the already retained predictions is therefore specific: on failed pool queries, inspect common strict wrong-negative counts and compare matched ordinary shared/native-four predictions; widespread failed queries without common offenders would refute the proposed explanation as the dominant failure mechanism. Demonstrating shared offending edges still would not prove a graph cause without topology/representation evidence. This assessment did not open raw prediction/negative-edge payloads or perform that new scoring; it only specifies the falsifier. No new fit is needed to ask it, but it is outside this turn's no-held-scoring scope.

## Actionable conclusion and competitor

**No defensible novel gap survives this reading.** The remaining question is an unestablished diagnosis about same-negative ranking failures, not permission to rename generic sharing, early adapters, private heads, pruning or pooling as a graph method.

The closest existing strong competitor is **J4_end_joint**, with **S_end_joint** and **E_end_joint** retained as competence references. The closest primary conceptual competitor is **TabMpacked**: joint untied learning and collective selection already improve quality without sharing. A graph adaptation of TabM's first-adapter/private-head/own-loss recipe is an attributed baseline, not a novel component, and current graph differences cannot be repaired by importing several settings together and attributing any gain to one mechanism.

Preserve the stopped live-credit disposition. Interpret future quality work through actual served performance and member competence, with the equally changed single/untied controls already required for the known ranking-loss exploration. The graph diagnostic above can rule out an explanation, but this packet establishes no new learning rule, predictive gain, tuning grid, fit, source implementation or manuscript claim.
