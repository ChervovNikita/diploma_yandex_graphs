# Direct efficient graph ensemble baselines and the remaining comparison

8 October 2026. Local saved-source assessment. Accuracy is the objective. No new
primary method scope was needed; no full-paper credit is added. This packet
does not implement a method, revise a protocol or admit execution.

## Conclusion

**Retain one actionable baseline: a Packed-Ensembles graph port with four untied
members and gamma = 1.** Give each member its complete nonlinear graph computation,
normalization, readout and optimizer state. A source-derived whole-model parameter
budget can determine its private width before outcomes. Keep a competent native
single and ordinary full-width independent4 as accuracy references. The packed
arm tests a published capacity/independence tradeoff; packing full-width members
is an execution of ordinary independent4 and should not be counted as another
statistical method.

The strongest confound is **member capacity and optimization competence**.
Parameter matching shrinks packed members while a shared model can retain wide
member states. A shared-family win over underpowered or poorly tuned packed
members would not establish useful sharing or a better diversity mechanism.
Parameter, edge-channel, activation and time budgets are different constraints.

The sources expose a concrete comparison gap: can the current compact graph
ensemble retain competent members and useful prediction alternatives when judged
against an optimized untied construction on the same backbone? They establish
no new graph-specific operation or general accuracy advantage. Grouped kernels,
low-rank private adapters, rank-one factors, private normalization, branches and
graph transplantation are established ingredients.

## MIMO fidelity depends on the example

Havasi MIMO samples independently allocated TRAIN examples into input slots,
concatenates those inputs before its ordinary hidden body, predicts the matching
slot labels with a summed member NLL, and repeats the target input at inference
before probability averaging. Input correlation and batch repetition are part of
its capacity/exposure contract. Independent tuple allocation does not make graph
observations statistically independent.

| Regime | Defensible MIMO adaptation | Fidelity boundary |
| --- | --- | --- |
| Transductive node graph | Complete feature-only SIGN/polynomial token rows can be paired with their own TRAIN IDs/labels after graph preprocessing; the learned body then operates on independent rows. | Naively concatenating aligned node slots from one continuing message-passing graph repeats the same example throughout training. Shuffling raw features against unchanged adjacency corrupts their graph context. Coherent permuted graph copies need separate operators, and global attention needs component separation. |
| Inductive molecular graphs | An invariant graph encoder followed by MIMO on independently paired graph embeddings is a declared late-fusion port; charge the encoder on every intact graph. | Padding atoms from differently sized unordered molecules provides no semantic correspondence. A disjoint union is graph batching, not an unchanged image-style MIMO trunk. A learned/cached encoder changes the training boundary and its upstream cost must be included. |
| Link prediction | MIMO can act on complete support-compatible query features, pairing each endpoint/subgraph example with its own edge label and allowed negatives, then repeating the target query at serving. | A post-encoder MIMO decoder is a decoder-only port. It cannot supply distinctions missing from the common encoder/query input. Shared endpoints, target-edge masking and query-dependent supports must survive pairing. It does not establish end-to-end MIMO of a continuing graph encoder. |

Equal numbers of views or heads certify none of those contracts. Repeated-input
training at rho = 1 is a legitimate limiting setting with a correlation limitation,
not evidence that arbitrary multiview graph training reproduces native MIMO.
MIMO's negligible image-model boundary overhead is not a GNN cost guarantee:
the saved PolyFormer example has a source-derived 36.4% parameter increase at M = 4.
That is algebra for the saved architecture, not timing or a current experiment.

## Why packed members are the cleanest port

For fixed linear aggregation, `P[H1,...,HM]=[PH1,...,PHM]` permits channel packing
without changing the functions. Every edge still processes the packed member
channels. Learned GAT edge scores, softmax denominators and Polynormer global
K/V reductions remain member-specific; sharing them would alter the baseline.
Normalize within each member and preserve private affine parameters. Independent
initialization and correct per-map fan-in matter. Gamma = 1 avoids adding a second
subgroup sparsity handicap before member competence is established.

The same construction applies to molecular graph mini-batches without atom
alignment: each member processes every intact graph with its own state and
readout. For link prediction, keep each member's encoder, query/completion
features and decoder private, on the same admitted supports/negative schedule.
Packing a common encoder with several private decoders is a shallow comparator,
not the complete untied packed baseline.

The paper's `alpha=sqrt(M)` matches interior quadratic parameters, not necessarily
the full model. The saved four-half-width PolyFormer example has 12.5% more total
parameters than its native single because boundaries, norms and biases matter.
Choose any strict budget width from the complete source count and legal head/FFN
divisibility. Source qualification, measured cost and a finite equally allocated
VALID configuration budget would remain separate future work.

## Direct shared alternatives are substantial prior

| Saved alternative | What is already established | Limit for this accuracy question |
| --- | --- | --- |
| BatchEnsemble / TabM / Kim graph-BE | Shared matrices with private factors and complete member trajectories; Kim places factors inside GCN/GIN/GAT layers. | Vectorization does not remove member-dependent nonlinear graph work. Factor placement/initialization/objective/pooling require attribution; Kim's exact author initialization/norm recipe remains unresolved. |
| N-Ens / Tiny-DE | Shared non-normalization maps and private normalization; N-Ens adds a continuing temperature-controlled log-softmax scale regularizer. | N-Ens executes separate forwards; its regularizer is not Shannon entropy. Tiny-DE's inspected sequential recipe freezes the shared model before private norm fitting; its parallel loss is unresolved. No graph accuracy transfer follows. |
| LoRA-Ensemble | A shared frozen transformer base with private additive low-rank attention factors and classifiers; inspected training uses mean member CE and probability-mean evaluation. | The native frozen/pretrained base is a different acquisition/training contract. Charge one-base acquisition. A trainable cold graph backbone is a declared adaptation, and low-rank placement is not novelty. |
| Shallow graph ensembles / DPOSE | One shared graph encoder and a final committee; atomistic work trains mean/variance with regression NLL and studies head covariance. | Cheap and relevant architectural ancestry. Molecular energy/force UQ does not establish node classification or molecular property accuracy; task loss/readout must be adapted. Private heads cannot recreate distinctions absent from their complete common input. |
| CAMERO | Shared bottom/private top networks, independent representation perturbations, own task loss and consistency regularization. | A graph classifier is a port. Matching perturbation/view counts does not establish MIMO, and consistency is not a guarantee of useful error complementarity. |
| ONE / PCL | Joint shared/private branches with concurrent ensemble/peer teachers and competence/distillation losses. | These do not first acquire an independent teacher ensemble, so they remain relevant ancestry under the stated exclusion. Their extra loss, gate/mean-teacher state and native deployment differ from an ordinary four-member mean. No graph recipe or utility is certified here. |

Methods that first train independent teachers and then distill or merge them are
outside this assessment's retained baseline. No additional branch, normalization,
LoRA or head arm is proposed by this packet.

## One representative comparison and its interpretation

Use the same native Polynormer node-graph boundary as the closed WikiCS motivation
for a separate prospective whole-family comparison: current shared construction,
the single PE4 gamma = 1 operating point above, competent native single and ordinary
full-width independent4. Pair complete seed blocks; preserve TRAIN-only fitting,
the admitted graph visibility, full local/global stages and the checkpoint
selection population. Use a declared common probability reducer for a new native
PE comparison, and disclose any separate raw-logit reducer. Preserve all original
scores. All variants need the same finite tuning allocation rather than a weak
zero-search port for the new comparator. This is a comparison blueprint, not a
new grid, protocol or execution authorization.

Judge served accuracy first, with NLL, mean member accuracy, coverage, repairs,
introduced errors and pool harms to explain the outcome. More disagreement or
embedding spread alone is insufficient. Charge preparation, selection and failed
work, optimizer storage, every nonlinear trajectory, graph/edge-channel work,
pooling and cold/warm serving under the same hardware. A benefit against narrowed
PE alone cannot support a claim against full-width independent4. A complete
development gain would still need a separately frozen unused confirmation.

The closed Wiki15 report supplies only motivation: it reports near-redundant
selected decisions, no pooled-only rescues and no useful primary combined-objective
gain on its validation-selected population. It does not localize the cause to
sharing, attention or initialization, and does not predict this PE result.
No partial graph12 or Mol18 outcome was read or used. Broader invention and the
single8 sequential implementation remain outside this packet.

## Reading boundary

Started with saved literature memory and scoped control reports. Relied on the
saved MIMO v2 Section 2/3.5/Appendix B passages, PE v4 Section 3 and selected
Appendix B/F/H/K passages, source-readiness conclusions, and saved direct shared
method conclusions/scopes. `SAVED_METHOD_SCOPES.json` retains exact locators and
inherited source hashes; original primary HTML/PDF and author code were not
reopened. Existing historical result snippets in saved reports were excluded from
the assessment's outcome evidence. There were zero new primary identities/scopes,
full-paper reads, retrievals, source audits, proofs or numerical executions.
`READ_BOUNDARIES.json` and `INPUT_BINDINGS.json` record actual local reuse and
its limits. Canonical documents, ledger, sources, owners and protocols are unchanged.
