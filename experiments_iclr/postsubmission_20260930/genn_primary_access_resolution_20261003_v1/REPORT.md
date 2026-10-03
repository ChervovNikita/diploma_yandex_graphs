# GENN primary-access resolution

3 October 2026. Bounded follow-up for *Graph ensemble neural network*, Information Fusion 110 (2024), article 102461, DOI **10.1016/j.inffus.2024.102461**. Registered authors: Rui Duan, Chungang Yan, Junli Wang and Changjun Jiang.

## Decision

**Primary method and author source remain unavailable in this six-route search. Stop this access route.** A useful new locator was found: OpenAIRE groups the published DOI with SSRN manuscript DOI **10.2139/ssrn.4535927**. The targeted SSRN delivery request returned 403. Its manuscript bytes, version, complete authors and relationship to the published method were not verified.

No assertion about GENN's exact parameter sharing, losses, update order, inference pool or gradient routing follows from these metadata records. Exact overlap with either the current objective policy or the member-filter family remains unresolved. Neither inaccessible access nor an empty search response is novelty evidence.

## Saved evidence consulted first

The prior saved Crossref record identifies the exact journal DOI, title, authors, article number and publisher text-mining locators. The saved OpenAlex item supplies no OA PDF or abstract. The prior ScienceDirect abstract-landing request returned 403; **that URL was not retried**.

The already read HGEN introduction describes GEN as incorporating ensemble operations through GNN training rather than only at prediction. This remains a secondary characterization, not GENN primary evidence for its operators, losses or sharing. The earlier source scope and its evidence hash are retained in `INPUT_BINDINGS.json`; no new HGEN primary reading occurred.

## Six distinct targeted routes

This packet conservatively counts each new HTTP request as one route. There were six requests, six distinct URLs, no retries and no additional linked-source requests.

| Route | Target | Result and limit |
|---|---|---|
| 1 | Semantic Scholar exact DOI lookup for a manuscript/preprint locator | HTTP 200; exact title, DOI and four registered authors match. `isOpenAccess=false`, PDF status `CLOSED`, empty PDF URL; no arXiv external ID. Metadata only. |
| 2 | arXiv exact-title preprint query | HTTP 200; zero results. This exact query's failure does not prove no differently titled preprint exists. |
| 3 | GitHub exact-title repository query | HTTP 200; zero repositories, `incomplete_results=false`. This does not exclude differently named, unindexed or private author code. |
| 4 | OpenAIRE exact DOI repository/institutional lookup | HTTP 200; one grouped result. Journal instance marked `CLOSED`; SSRN DOI 10.2139/ssrn.4535927 listed with `UNKNOWN` access. Tongji and Guangzhou University associations appear in the metadata, but no institutional manuscript download is supplied. |
| 5 | Saved first-author ORCID researcher-URLs endpoint | HTTP 200; empty researcher-URL list. It supplies no author-homepage manuscript or code locator. |
| 6 | OpenAIRE-linked SSRN manuscript delivery endpoint, abstract ID 4535927 | HTTP 403. Error bytes saved; no manuscript was obtained or read. |

Exact URLs, request times, status codes, final response URLs, raw response hashes and bodies are in `ROUTE_RECEIPTS.json` and `routes/`. This was a targeted access check, not an exhaustive literature or code search. The Crossref publisher API locators were retained but not requested after the six-route limit. No blocked-page workaround or author contact was attempted.

## What the inaccessible source would need to resolve

The current proposed policy has a trainable common block \(W\), private blocks \(\phi_m\), and member logits \(z_m\). From one unchanged pre-update state it uses

\[
g_W=\nabla_W\operatorname{CE}\!\left(\frac1M\sum_mz_m,y\right),\qquad
g_{\phi_m}=\frac1M\nabla_{\phi_m}\operatorname{CE}(z_m,y).
\]

The original private \(1/M\) factor is retained. Shared and private gradients are combined for one optimizer step; private gradients are not recomputed after a shared update. Deployment uses mean raw logits followed by softmax. This description is bound to the saved objective-policy assessment; it is not a new implementation or experiment.

| Required operation-level fact | GENN finding |
|---|---|
| Which feature/graph maps are shared, private, frozen or independently initialized? | Unresolved. |
| Which adjacency, hop, spectral or learned-neighborhood operators act on each member, and where relative to nonlinearity? | Unresolved. |
| Does supervision use individual CE, pooled CE, both, distillation or diversity terms; with which reduction factors? | Unresolved. |
| Are updates simultaneous from one state, alternating, staged or sequentially recomputed? | Unresolved. |
| Does serving sum/average raw logits, average probabilities, fuse embeddings, learn a gate or deploy one learner? | Unresolved. |
| Are pooled-loss paths to private blocks or individual-loss paths to common blocks selectively stopped/replaced? | Unresolved. |
| Is a trainable common backbone combined with private learned graph filters and persistent predictive members? | Unresolved. |

The graph-member-filter question concerns learned member-specific graph/neighborhood evidence on common feature maps, particularly private filtering before nonlinear processing. The existing sealed review found a head-information limitation but close learned-filter-bank and neighborhood-ensemble ancestry, and proposed zero pilots. GENN cannot currently be used to claim either complete equivalence or an exclusion from that family. HGEN's secondary training description makes GENN a relevant unresolved prior; it does not resolve the table.

## Read accounting and future custody

- New primary paper method scopes: **0**.
- Full-paper reads/certifications: **0**.
- Author-source files read or audited: **0**.
- Retained primary method rereads: **0**.
- Metadata/request routes: **6**, with one unresolved SSRN manuscript locator.
- Experiments, current outcomes, grids and pilots: **0**.

`PAPER_CONCLUSIONS.json` records an access-limit conclusion, not a primary-method conclusion. `METHOD_COMPARISON.json` retains each unresolved operation instead of inferring it from a title or citation. The SSRN record is a candidate locator, not a certified paper-version alias or a newly read unique paper. The saved filter-gap packet and index_v31 are hash-bound and unchanged; no index append or canonical ledger/status edit occurred.

**Resume only for a specifically supplied accessible manuscript/code, changed access, or a newly authorized named route. Do not repeat these six requests or the earlier blocked ScienceDirect landing by default.** This preserves the unresolved prior without treating repeated access failure as scientific evidence.
