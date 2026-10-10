# Graph-conditioned private members: literature boundary and one held question

The generic mechanism is already published. TopLoRA (NeurIPS 2025, DOI 10.52202/085713-4552) generates a token-dependent positive diagonal matrix and places it between low-rank factors: `B diag(exp(RMSNorm(Theta X))) A`. Official PDF pages 3–5 and the pinned author forward agree on this operation. Source was read, never imported or executed. Its published setup adapts pretrained models; inherited PEFT trainability was not independently audited here.

TabLoRA, which we had already read, trains a live shared backbone and private low-rank/input adapters from scratch using complete own losses. This pass adds its input-adapter scope and does not count it as a new paper. Combining that training arrangement with TopLoRA and graph context has clear ancestry. Retained GNN-FiLM, GRANOLA, GaRA and MoSE scopes also cover featurewise/graph-conditioned transformations and structural routing. We have not established methodological novelty.

ATLAS is a useful exclusion. Its frozen retained-domain atlas supplies local PCA directions, routed contractions and a norm gate for one shared residual. The base and atlas stay frozen. Its retention motivation does not provide a guarantee for our live shared core, and its reported results are not imported.

## One empirical question, held before implementation

Test whether member-private structure-conditioned adapters improve both individual member competence and the served probability average during joint training. This is a competence hypothesis for a known-component composition. It is not currently a new method claim.

A concrete illustrative form is a live shared affine map plus a private low-rank residual whose diagonal gate receives a fixed label-free structural context. Every member keeps its complete own loss. The native checkpoint rule and arithmetic probability pooling stay in the comparison. No pretrained I4 teacher or external head is introduced. The structural context remains undecided; no cache, adapter source, or experiment was authored.

The necessary controls are a capable same-information conditional single/four-loss model, a common conditioner across members, constant and feature-only conditioning at controlled capacity, the existing factorized-M1 controls, and proper individually selected same-information independent members. Runtime and storage must include the conditioner and shared reusable cache. A gain explained by the conditional single or common conditioner would not establish the proposed ensemble-specific value.

Root's closed PubMed result supports further competence analysis but does not identify a missing conditioning signal. Factorized single-member optimization remains an unresolved explanation. Root's closed IMDB error counts also include many substantial-margin common mistakes, so a low-confidence-only repair gate is unsupported. No partial CMCL, 18.77 quality, TEST outcomes, raw prediction archives, or original-score recomputation were accessed during this continuation.

**Disposition:** hold until factorized-M1 and the current complete CMCL evidence close. Root must fix the operator, context, controls, full training budget and representative seed/split scope before an experiment. Positive exploratory results would still need unused confirmation and a stronger novelty argument. Null results are recorded without tuning through them.

## Reading and provenance

TopLoRA and ATLAS have bounded primary method scopes in this packet; no full paper or full repository is claimed. TabLoRA is a repeat. Official primary documents, exact retrieval receipts, scoped passages, pinned author sources and extraction receipts are retained. PDF text extraction used local hashed pure-Python tools, with no PDF compilation, model imports, training, or existing source edits. The empty unsealed TopLoRA bibliography locator was corrected using ATLAS's exact `bib.bib11` entry. Predecessor seals were untouched.
