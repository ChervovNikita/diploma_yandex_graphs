# RankExpert: bounded primary-source access follow-up

15 October 2026. **RankExpert remains unread at method/experiment level.** No primary manuscript was obtained. This follow-up resolves the requested access check, not the scientific content: it adds no verified method, experiment, or novelty finding and changes neither prepared study.

## Identity, deduplication and access result

The lead is Yuchen Li et al., *RankExpert: A Mixture of Textual-and-Behavioral Experts for Multi-Objective Learning-to-Rank in Web Search*, KDD 2025, [DOI 10.1145/3711896.3737258](https://doi.org/10.1145/3711896.3737258). Crossref and two coauthors' publication pages confirm this identity. The retained exact-title/identifier checks found the earlier metadata and failed-retrieval records, but no retained primary method-read record. This is a scoped deduplication result, not an exhaustive literature census.

The ACM PDF again returned **403 HTML**, not a PDF. Public author routes did not supply an accepted manuscript:

- [Haoyi Xiong's homepage](https://sites.google.com/site/haoyixiongshomepage/) and [full publication list](https://sites.google.com/site/haoyixiongshomepage/full-publication) name RankExpert but contain no manuscript link in its entry.
- [Yongqi Zhang's publication page](https://yzhangee.github.io/publications/) names the exact paper and links only to its ACM DOI page.
- [Linghe Kong's homepage](https://www.cs.sjtu.edu.cn/~linghe.kong/) had no exact RankExpert/title-identifier match in the retrieved HTML. A candidate Mingxin Cai GitHub Pages URL returned 404; it was never verified as an author site.

Crossref supplies only the ACM PDF location. ORCID furnished the Xiong/Kong homepage locators. Semantic Scholar returned 403; DBLP and DuckDuckGo returned challenge pages; Google returned a redirect/script page without usable results; Bing's extracted result set was unrelated. These responses are access/locator evidence only. The earlier zero-result arXiv title query was reused, not repeated. The search stopped after these bounded routes; no predecessor or substitute paper was opened.

All **19 top-level HTTP retrieval attempts** in this follow-up went through `remote_control.call` on the guarded allocation. Raw response payloads remain in the repository's remote `.cache/rankexpert_followup_v1`, outside the publication package. `RETRIEVAL_INDEX.json` records their paths, hashes and statuses. Local JSON files contain receipts, bibliographic metadata and bounded locator extracts. No raw manuscript was copied locally, and no access challenge was bypassed.

## What can and cannot be concluded

The retained abstract advertises textual/behavioral inputs, multiple ranking objectives, distillation and weighted fusion. That description explains why the lead was relevant; **it does not establish its computational design**. This package does not classify RankExpert as either a same-task shared ensemble or a conventional multitask ranker. Without primary text, the following remain unverified:

| Required distinction | Unresolved evidence |
| --- | --- |
| Shared same-task members versus task-specific heads | Exact shared/private blocks, expert targets, supervised units and parameter/update ownership. |
| Ensemble prediction versus one stronger fused ranker | Serving equation, whether every member solves the same target, and whether a fixed uniform mean is actually evaluated. |
| Small overhead versus added modeling/training resources | Private parameter counts, teacher/distillation costs, extra modalities and inference computation. |
| Useful complementarity versus ordinary multitask/model improvement | Relevant baselines, ablations, independent-member controls, pooling gains, error repairs and harms. |
| Reproducible reported gain | Complete datasets/splits, metrics, tuning/selection protocol, uncertainty and offline/online experimental results. |

No numerical claim, proof, empirical benefit or exact sharing/aggregation operation is transferred from the abstract. The access failure also supplies no evidence of absence of close prior art and no novelty clearance.

## Effect on the prepared decisions

**MolBACE pairwise: unchanged.** Preserve the 21-fit proposal: three seeds of common/structured/permuted shared pair loss plus twelve genuine native bodies with the common pair auxiliary; native BCE plus `0.5 R_pair`; the existing fixed bond-composition quartiles/positive weights; all-parameter updates; and mean-logit serving. Retain native-BCE references and every declared rejection comparison. The prior verified shared/private multitask AUC precedents still apply. The common-pairwise I4 control still cannot isolate a causal sharing benefit from structured pair roles. RankExpert supplies no newly verified reason to add a teacher, gate, objective, arm or architecture.

**PubMed transfer: unchanged.** The saved default remains the existing 24-trajectory native PolyFormer-Cheb comparison, conditional on root's complete ongoing-study report and measured costs. Its priority rests on testing transport to another task/backbone, with explicit prior PubMed exposure. Inaccessible RankExpert content neither strengthens nor weakens that priority. Pairwise remains the prepared alternative; neither launches automatically.

This was source-access work only. No fit, data/logit/checkpoint/TEST inspection, partial running-fit quality read, GPU computation, implementation change or Git operation occurred. The helper's mandatory hostname/GPU-UUID route guard is not a scientific workload. Root retains publication and launch decisions. A future method assessment requires an actually accessible primary manuscript; no further search or new branch is scheduled here.
