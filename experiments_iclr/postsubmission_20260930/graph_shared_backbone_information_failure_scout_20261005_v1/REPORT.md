# Graph return-message contamination and common representation loss

**No shared-ensemble successor survives this scout.** The distinct failure examined is graph return-message contamination: repeated propagation combines newly reached node information with messages that immediately return over an edge. If a common backbone collapses a label-relevant distinction before private predictors receive it, more private heads cannot recover that distinction. The two new exact method scopes repair propagation inside one learner. Neither makes an ensemble necessary, and accurately fitting the current shared predictor does not establish that this failure occurs in it.

**New reading:** two primary identities, five complete subsections and 21 HTML paragraph/theorem/list-item containers; zero full-paper reads, retained primary rereads, author-code reads or result-section reads. Three public arXiv metadata searches produced nine distinct identities. HOPPER and NBA-GNN were absent from all 241 v64 records and from the targeted local saved-scope search before retrieval. The 2017 line/hierarchical GNN identity `1705.08415` was retained despite title changes and was not reread.

## Exact new methods

| Source and inspected scope | Actual operation | What it covers and what remains unproved |
|---|---|---|
| Herath, Gopakumar and Sahu, [HOPPER, arXiv:2608.09031v1](https://arxiv.org/html/2608.09031v1), §2.3, §§3.1–3.2; Eqs.3–16 | A single model extracts a sequence of hop states, then processes that sequence independently per node. A graph-summary/hop-conditioned hypernetwork emits coefficients for a finite-memory recurrence using `A`, `D` and `I` (Eq.15). Feature attention and LayerNorm modify the extracted output rather than the recurrent structural state (Eq.16). Fixed adjacency and nonbacktracking extraction are displayed in Eqs.6–7. | Already supplies a capable single with graph/hop-adaptive extraction and separation of propagation distance from nonlinear processing depth. Coefficients themselves depend on the initial features, so the structural recurrence is linear in its state **conditional on those coefficients**, not a globally fixed linear map across arbitrary feature inputs. No stability recipe, theoretical guarantee or published quality result is adopted; Appendix B, proofs and experiments were outside scope. |
| Park et al., [Non-Backtracking Graph Neural Network, arXiv:2310.07430v1](https://arxiv.org/html/2310.07430v1), §§3.1–3.2; Eqs.1–5 and Proposition1 statement | Uses a hidden state for each directed edge. The update to `j→i` aggregates `k→j` for `k` in `N(j)\{i}` (Eq.3), excluding the reverse-edge echo. Initial states encode both endpoints and edge features; final node prediction pools incoming and outgoing states separately (Eq.4). A degree-one exception permits a return update to avoid dead ends. Eq.5 gives a normalized residual NBA-GCN example. | Direct single-model response to redundant message walks and upstream information obstruction. The tree access-time proposition is a motivating statement, not a certified general-graph theorem in this packet. Edge-state storage and message updates are part of the method; no free shared-cache or runtime claim follows. |

These are graph propagation operations, not meta-learning or generic diversity penalties. The v64 scope for Chen et al., `1705.08415v4` (record222), already credits nonbacktracking line-graph states, learned graph operators and node/edge incidence maps. HOPPER additionally bounds the recent learnable extraction route. A private nonbacktracking branch would borrow those operations rather than introduce a new ensemble principle.

## A falsifiable failure statement

Let `R_q = T_theta(G,X,q)` denote **all** graph-dependent inputs available to every member for query `q`, and let `z_m = h_phi_m(R_q)`. For any fixed model state, a deployed pool of these member predictions is still a function of `R_q`. If two graph/query inputs have the same `R_q` but require different targets, changing only private heads, their diversity penalty or credit assignment cannot distinguish them.

This is a conditional information-loss statement, not a claim about every shared model. Its assumptions fail if a member has additional graph context, if the shared encoder changes and separates the inputs, or if a supposedly row-local prediction also consumes other graph states. No current NCN query collision, over-squashing measurement or fitted quality failure is asserted here.

A concrete graph witness uses the five-node tree with edges `(r,a),(r,b),(a,u),(b,v)`, scalar features, and a deliberately restricted common representation `R_r=(A^2 X)_r` without a raw-feature or hop-state skip. At the root,

`R_r = 2*x_r + x_u + x_v`.

Two input configurations have the same `R_r=2`:

- Configuration A: `x_r=1`, `x_u=x_v=0`.
- Configuration B: `x_r=0`, `x_u=x_v=1`.

Set the root target according to whether the newly reached distance-two sum is positive. The targets differ, while every head receiving only `R_r` has identical input. A nonbacktracking second-hop state `(A^2-D)X` gives respectively `0` and `2`. A single receiving both `X_r` and `(A^2X)_r` can make the same subtraction. This is exact symbolic reasoning, not a model run, native-architecture qualification or measured dataset result. It also shows why an information-preserving skip can defeat the proposed need for a specialized graph operator, let alone an ensemble.

The empirical hypothesis that return-message redundancy impairs a *particular trained* backbone remains falsifiable: a competent single with the same task information and explicit hop/cavity states may eliminate the error. The restricted witness cannot substitute for evidence that the trained common representation actually loses the relevant information.

## Why sharing matters—and why it does not rescue this candidate

Sharing one compressed graph representation makes an obstruction common to every downstream member. Private head disagreement after that compression cannot create missing path information. An independent ensemble fed the same fixed lossy representation also fails this witness. Independently trained upstream graph encoders are a stronger alternative because they may retain different graph distinctions; their quality cannot be inferred here.

Sharing before propagation, for example only a node-feature map, can amortize some computation while private edge/hop states remain distinct. That is a different sharing boundary with real private graph work. Sharing a richer bank of graph states is also possible. The saved BUDDY scope already notes that independently trained predictors can share deterministic caches too; cache reuse alone does not identify learned sharing as the source of quality.

The strongest capable-single comparator is one accurately trained predictor given the **same complete graph-state information** as the proposed bank: an NBA-GNN where directed-edge memory is essential, or a HOPPER-style extractor plus a sufficiently capable hop-sequence processor where depth/return cancellation is essential. Raw-feature and hop-state residual information must be allowed. A small single restricted to the original compressed node embedding is an inadequate comparator.

The strongest ordinary ensemble comparator is separately initialized and trained copies of that competent graph model, with identical task information, supervision and evaluation. An untied bank receiving the same objective is additionally required if a later proposal couples member training. Actual graph operations, edge-state memory, feature work and tuning opportunity must be disclosed; no comparison or compute release is proposed by this packet.

## Saved directions excluded rather than rediscovered

The adopted v64 and saved rejection notes already cover:

- Walk-length/view-mask diversification, graph-conditioned routing, filter banks and capable structural singles (including GPM).
- Structural-response disagreement that may change member behavior while leaving the served mean-logit predictor exactly unchanged.
- Degree-preserving assignment observations and complete-matching likelihoods, with a capable nonadditive single and independent ensemble able to defeat the ensemble claim.
- Persistent private-learning meta-gradients and their existing single/untied attribution limits.

This scout does not rename one of those operations. The newly inspected failure is upstream return-message redundancy and lost graph information. The proposal to repair it with downstream shared-ensemble private learners is rejected because it acts after the obstruction, or, when moved upstream, adopts an already available single-model propagation remedy without evidence that sharing improves quality.

## Disposition

**No new operator, representative fitting experiment or source preparation is recommended.** The failure is concrete, but the ensemble-specific hypothesis does not survive the information boundary and capable-single comparator. A new-data fit would currently test an attributed propagation architecture or familiar multi-branch capacity tradeoff, not a justified new sharing mechanism.

This is a scoped rejection, not a global literature absence proof, quality verdict on an existing fitted model or claim that every shared graph ensemble is useless. No experimental result is adopted. The one new folder retains exact scoped-source conclusions, selected method text and hash receipts. No canonical status/index, frozen experiment, model/data/checkpoint/outcome payload, server/18.77/MacLink, compute allocation or queue is accessed or changed. The saved structural-exposure note contained quoted initialization diagnostics; those numbers were excluded from this reasoning and no raw diagnostic artifact was reopened.
