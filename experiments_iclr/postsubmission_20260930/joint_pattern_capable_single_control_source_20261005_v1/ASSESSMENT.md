# A native single with the same conditional endpoint-pattern auxiliary

5 October 2026. Source only; unqualified and unexecuted.

`single_control.py` implements the missing operation: one fresh native width64 NCNC encoder, recursive scorer and scalar target decoder, with a four-output auxiliary emission head on the recursive scorer's shared penultimate features. It consumes the same native masked graph, endpoint pairs and complete residual supports. Its training labels and exact conditional-Bernoulli core are the existing queue's `ObservationTeacher` and `training_pattern_losses`. This is an auxiliary-supervision control with one target trajectory. It contains no four-member factor maps or independently decoded target members.

## Exact source operation

The execution root's `remote_control.py` binds `exact_cb_support_bucket_paired_predictive_preparation_20261004_v1` (manifest `0d0899d94f0a20d317f7e30491f3ed5b35c293d462f28a0baf3f4ea350e9d229`). That packet's `pattern_model.py` uses `PatternTwin`/`CompletionDecoder`, whereas its `pilot_model.make_native` constructs the already qualified ordinary N64 scalar NCNC source. The latter is this adapter's target baseline.

For each positive or negative query group, preserve the native call order: outer full-node `xlin`, left depth-zero recursion, right depth-zero recursion, outer scalar decode. The left counterpart pairs are `(v,w)` for residual neighbors of `u`; the right pairs are `(u,w)` for residual neighbors of `v`. Each depth-zero call still performs the second full-node `xlin`, all candidate common-neighbor sums, and native `xijlin`/`xcnlin`/`lin` maps. Empty candidate calls execute as usual.

The pinned native last decoder map is `lin[8]: Linear(64,1)`. Define its penultimate feature exactly as

`phi = lin[0:8](xcnlin(common) * beta + xijlin(endpoint_product))`.

The original scalar scorer is still `lin[8](phi)`. A new native `Linear(64,4)` supplies four raw auxiliary emissions from this same `phi`; use the existing `eta = scale * (emission - offset) + log(pt)` transformation. Four output rows share the same context and nonlinear scorer trunk. Each row can prefer different counterpart identities on both sides. The complete-side conditional laws are mixed through existing `J_K` or independently through existing `J_K_sep`. No new normalizer, sampler, auxiliary weighting, count head, target gate or optimizer setting is introduced.

The emission uses `nn.Linear.reset_parameters` through its ordinary constructor, creating distinct random rows. CPU `fork_rng` restores the native stream after this added initializer; the constructor reuses the caller's existing fit seed through `make_native` and introduces no seed search. No identical-head copy or donor state is used. Component permutation symmetry, component collapse and weak structural gradients remain possible. Additive emission biases, offset and fixed-pt shifts cancel from a side's fixed-count subset law; the four bias parameters therefore do not identify a structural preference. The head has 260 parameter entries; the native 38,147 total becomes 38,407. No resource or capacity match is claimed.

## Native target and gradients

The target uses the original scalar candidate score, original native `clampprob` soft weights and one original scalar outer decode. It serves that one raw logit. Auxiliary components are neither scored as target members nor pooled into the target. The auxiliary head is omitted at serving. Direct conditional teacher-law gating would still be uniform on complete TRAIN and is not part of this control.

During auxiliary training both complete candidate-scorer trunks have autograd. The four emissions train those shared native nonlinear maps and the encoder through the existing exact conditional NLL. Main target loss sees detached scalar completion scores, as required by native NCNC; outer features and the scalar target decoder remain differentiable. The native scalar final projection has its ordinary main-loss gradients; it is not an auxiliary emission row. The auxiliary can improve the target through shared representations, but no transfer guarantee follows.

The provided `objective` uses unchanged positive and negative query-mean native log-sigmoid losses, plus coefficient1 times the sum of positive and negative auxiliary query means. Labels are looked up only after model forward on the exact supplied support. Core empty-support zeros remain in all-query means. The graph provider must continue deleting original positive record IDs before symmetric coalescing, allowing surviving duplicates; encoder dropout does not define the residual support. The source accepts no teacher counts, teacher adjacency, true target label or arm flag in a predictive forward.

## What this comparison would distinguish

At one visible context, take penultimate candidate features that distinguish each side's two identity alternatives. Different auxiliary rows can prefer the paired alternatives `(a,b)` and `(c,d)` at `k_L=k_R=1`. The existing shared-component law then expresses endpoint identity association. The source has this finite-mixture capacity without four private full-node transformations or four target decoders. This is a conditional capability statement; an equivariant scorer with indistinguishable candidate features can still forbid it, and no dataset-specific competence has been established.

A future comparison of this single's `joint` versus its own `target_only` asks whether endpoint-pattern supervision helps an ordinary native target. Its `joint` versus `separate` asks whether cross-side association helps within this shared trunk. Comparing the resulting native scalar single with the current F4 joint bank asks whether a competitive target can obtain the benefit without four full member trajectories. These are controlled questions, not a launch plan. Different architecture/RNG schedules and capacity mean a single-versus-bank result cannot isolate one universal cause. A favorable F4 cohort screen would still require this capable single comparator; it would not prove that four trajectories are needed.

The single still pays the full encoder, three full-node scorer transforms per query group, all complete candidate feature/decode work, four scalar auxiliary projections per slot, teacher lookup, exact four-component ESP work and auxiliary backward. The F4 source instead has private per-member transforms/scorers and target decodes. This operation removes the repeated member neural trajectories; it does not remove complete-support DP or guarantee speed/memory savings. It currently retains candidate autograd directly and lacks the bank's activation-checkpoint wrapper. Runtime and full native-batch resource qualification are pending.

## Reuse and readiness of the saved alternatives

| Existing packet | Actual readiness and relation to this control |
| --- | --- |
| `ncnc_cardinality_single_comparator_implementation_preparation_20261003_v1` | Concrete C64-D4 source; all runtime gates and release remain pending. One total-count law and unary-weighted subsets; four sampled native decodes change the prediction route. Condition on `(k_L,k_R)` and its count-only terms cancel, leaving separate unary side laws. It therefore cannot supply this fixed-count endpoint identity-association control. Reuse it for its declared count question; do not rename it as a qualified current comparator. |
| `ncnc_structured_single_control_source_proposal_20261004_v1` | C and S are proposals, without source qualification or fits. C's two-sided count potential also cancels at fixed side counts. S has four internal density heads and full covariance attention but changes the target decoder; its stated moments concern the unconditioned Bernoulli law and are not the current conditional-law moments. Reuse its conceptual scope; do not duplicate or silently adopt it. |
| Ordinary native N64 in bound `pilot_model.make_native` | Nearest already qualified single operation: one native scalar target and its own soft completion. It has no current conditional endpoint-pattern auxiliary. This adapter reuses its source construction; the new head/gradient adapter itself is not runtime-qualified. |

No established saved packet implements this shared penultimate four-emission auxiliary with the unchanged native scalar target. That is the bounded missing source operation supplied here, not a new claim about mixture distributions, model-class superiority or novelty. Existing native100 pilot and500 confirmation controls retain their current definitions; this packet changes no existing plan or queue.

## Remaining adaptation boundaries

The source is reusable by a future caller supplying the bound native constructor, existing graph helpers, the native adjacency factory (already present in `cardinality_model.native_adjacency`), existing observation teacher and conditional core. It has no data loader, fit loop or launch entry point. Before execution, native scalar parity/dropout-RNG order, complete conditional-loss/gradient interfaces, distinct-row initialization, no teacher leakage, optimizer membership, complete-support memory and native validation replay must be qualified. No new constants or budget are selected here. Existing global-RNG native mask/negative hashes are not replay arrays; same seed across the different neural schedules does not certify identical realized masks.

Only this packet was written. Source text, metadata, AST and hashes were used; no project numerical module was imported, model instantiated, outcome score viewed, graph/checkpoint payload read, training or remote work performed, or agent spawned. Static checks establish syntax, source custody and structural call boundaries only.
