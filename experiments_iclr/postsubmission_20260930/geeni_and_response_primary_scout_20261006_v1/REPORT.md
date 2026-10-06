# GEENI access limit and a close specialization antecedent

6 October 2026. Base index 72 and active supplement V6 were consulted first. Closed BGNN/AdaGCN/AKD/E2GNN/GETS/A3GCN/SMCL/FAGEL and the recent functional-update/geometry scopes were excluded. This packet adds **one new bounded published primary method scope, CMCL**, not two papers or a full-paper/code audit. GEENI remains unresolved. Canonical index, active supplements, pilot and scores are unchanged.

## GEENI: exact identity, no recovered algorithm

**Efficient ensembles of graph neural networks**, Amrit Nagarajan/Jacob R.Stevens/Anand Raghunathan, DAC 2022, [DOI 10.1145/3489517.3530416](https://doi.org/10.1145/3489517.3530416). Crossref/OpenAlex identify the CC-BY published version; SemanticScholar confirms the same publisher PDF. The [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3489517.3530416) returned 403. Two guessed Purdue faculty paths returned 404. A legitimate institutional research/publication locator was accessible, but bounded exact-title/navigation extraction produced no usable GEENI paper link. A first-page GitHub acronym search yielded no repository authenticated to this work. These limits do not establish that no public copy exists.

No GEENI method body or author code was read. Its loss, training/aggregation, shared/private parameters and error-correction mechanism stay **unresolved**; title/efficiency/OA metadata cannot settle overlap. Existing bibliography identity is reused, not counted as a new paper reading.

## CMCL: specialization designed for probability pooling

**Confident Multiple Choice Learning**, Kimin Lee/Changho Hwang/KyoungSoo Park/Jinwoo Shin, [official ICML 2017/PMLR 70 PDF](https://proceedings.mlr.press/v70/lee17b/lee17b.pdf), exact 937,011-byte body SHA256 5dcc65f0cc0a5a421b40d9ef2ff4822aaae86775b6d2c8af05fe4976bea85ac6. Published source ID pmlr:v70/lee17b; linked arxiv 1706.03475 is metadata only, not a separately read/equivalent-certified version. Checked index/active/saved-scope metadata had no matching prior CMCL method scope; that screen is not a novelty census.

Read §§2–4.2 and complete Algorithm1 on physical pages 2–5 plus §4.2's leading page 6 continuation. Formula/algorithm/feature pages 3–5 were visually checked. Contribution/§3.3 discussion, Fig 1 accuracies, Fig 2 entropy plots and one later page 6 empirical sentence were exposed but no performance claim is adopted. Only concise excerpts and the algorithm are retained; temporary full PDF/renders/read-page text were removed.

The confident oracle loss assigns exactly one owner per training item: its true-label loss plus beta*KL(U||P) for each nonowner. Algorithm1 chooses the owner minimizing CE_m+beta*sum_(n!=m)KL(U||P_n), so it uses current prediction/loss/confidence, not observed SGD learning response. A single batch update trains the winner on its label and the others toward uniform output. Version1 replaces the exact KL gradient by unbiased uniform-sampled-label CE. The objective directly addresses a known failure: confident wrong non-specialists can spoil average-probability top 1 even when an oracle member is good. Simple averaging/voting is the declared deployment motivation; no unique implementation of every test-time mask detail is certified.

Optional feature sharing is concrete: each model retains its own W_m, while its layer input SUMS its own activation with elementwise Bernoulli-masked activations from other members. It is not concatenation or a common weight matrix/private-factor architecture. Cross-member feature dependence already exists. Its precise gradient ownership/test-mask implementation is not audited without author code.

Preserve source caveats: prose reverses the KL description relative to Eq3/Algorithm1/§4.2's explicit KL(U||P). Also, the paper calls min noncontinuous; continuous individual losses give a continuous minimum, potentially nondifferentiable at ties. Neither wording is adopted as a theorem. Fixed-assignment independent-model prose should not silently erase the optional feature-sharing dependence.

## Specific overlap and one falsifiable gap

Current-error/confidence-dependent specialization, correction of overconfident non-specialists for ordinary probability aggregation, and member feature exchange are established operations. The native proposal instead measures a finite own-CE private-step margin response, retains all-member competence CE, balanced graph allocation, live adaptation credit and original-phi recommit. Those are scope-bound operational differences; they do not clear novelty, and GEENI remains an open dependency.

The remaining conditional question is whether that measured response causes transferable complete-pool repairs beyond the already matched live first-order utility and ordinary own/pool loss, rather than merely changing confidence or current-loss allocation. Reject finite-specific corrective attribution at the declared setting if the complete pool/member-competence criterion fails, repairs are canceled by new errors, or improvement is confined to S, oracle members or calibration of unchanged wrong labels. This is an interpretation obligation, not a new scientific arm, source implementation or execution admission.

**Accounting:** 12 HTTP requests: 10 metadata/locator and 2 primary-access requests; 9 returned 200, one 403 and two 404. One new bounded primary method, zero full papers, author-code scopes, performance adoptions, scientific payloads, numerical runs or server scientific actions. No failed access or search absence adds reading/novelty credit.
