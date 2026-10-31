# Shared-transition/private-head attribution control v1

Status: **unexecuted source only**. This separate packet preserves both sealed earlier source packets. It implements an established recurrent-mixture control with private ordinary or factored output heads; no new methodology, novelty, performance or resource-admission claim is made.

## Exact change and inherited contracts

`contracts.py` reuses the sealed AR packet's complete binary `ScheduleDomain`, immutable fixed positive pi, generated-record and independent-baseline contracts, source encoder and full-mixture reverse-KL objective. `SharedTransitionLedger` reuses its audited encoder/matrix/reduction accounting. The manifest binds the original module hashes. No reused source is edited, copied into a changed fork, imported or executed during authoring. Future package-relative imports require the parent source directory to be available as a package root; that setup remains unqualified.

One ordinary authorized source `Model` with dropout zero and node-wise LayerNorm/Identity still encodes the entire graph once per objective batch. All heads and sampled/replayed histories reuse the same static H. The source fingerprint remains `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`. Graph binding, deterministic source provenance and fixed-parameter batch contracts retain the earlier limitations.

The transition uses the earlier exact input and GRU convention: `x=[H_v,previous_label,blocked_flag]`, with zero start token; reset/update gates and candidate from split input/recurrent affine maps; a post-update hidden-state head. Here `W_x:[3h,p]`, `W_h:[3h,h]`, `b_x:[3h]`, `b_h:[3h]` and `h0:[h]` are **fully member-independent**, with `p=d+2`. `_transition` has no branch argument. There are no private recurrent/input factors, private initial states, branch embeddings, head-logit/probability feedback or mixture-weight inputs. Actual emitted labels remain the ordinary autoregressive prefix input. State updates use no in-place mutation.

`ordinary_heads` supplies private `W_o:[M,2,h]` and `b_o:[M,2]`. `factored_heads` supplies one shared `W_o:[2,h]` and private `r_o:[M,h]`, `s_o:[M,2]`, `b_o:[M,2]`, with logits `(W_o(r_o_m*h))*s_o_m+b_o_m`. All initial parameter tensors are caller-supplied after admission; no dimensions, initialization, sample counts, pi or scientific settings are frozen here. Binary laws depend on logit differences, so larger raw head parameter counts alone do not prove a larger distribution class.

## Same law, exact same-prefix reuse

The inherited simple full conflict graph and fixed node permutation determine the same hard-core mask. An active emitted neighbor forces zero; every free step has binary log-softmax. Hidden state advances at all nodes. Each complete branch law is normalized and has the common theta-independent independent-set support in exact arithmetic. All feasible complete assignments have positive probability; infeasible ones have zero probability.

For one supplied complete y, shared h0 and transition imply identical hidden states across heads by induction on the recorded prefix. `full_log_densities` therefore executes **one** N-step recurrence and one mask computation per step, evaluates all M heads, and returns every full `log q_m(y)`. This is exact computation reuse, not averaging or substituting a representative private state. Repeated identical assignments in distinct sample records are not deduplicated.

When a generating branch is supplied, the one shared recurrence retains that score's autograd graph. Its head consumes the live state. Every nonself head consumes `state.detach()` under `no_grad`, producing an exact density value without a cross-head autograd graph. The imported objective detaches the full all-member mixture reward and multiplies it by the generating complete log score. Thus shared recurrent/encoder parameters receive the required generating-score gradient, with no second recurrence for a separate gradient replay. The explicit normalized-density derivative still integrates to zero under the inherited fixed-pi/common-support assumptions.

With `generating_branch=None`, the entire full-density vector is computed under no-grad. `cross_member_log_densities` uses one full-vector call per generated record. Compatibility `log_density` computes/charges the whole vector before selecting a scalar; looping that scalar API M times would forfeit reuse and must not be reported as the efficient path.

`generate` remains unconditional, using a caller-selected head and N independent supplied uniforms. Each sampled history runs a distinct charged recurrence, because head choices can produce different labels/prefixes. There is no generation-time shared-prefix optimization, RNG, replay of sampling values, numerical-failure retry or future-lock clamping. Clamping arbitrary future labels would not implement conditioning.

The reused objective includes all M density contributions, generating branch included, and offers the same full-matrix mean-branch-KL objective ablation. Fixed pi, positive known fugacities, valid iid/stratified draw allocation, an independently obtained baseline and frozen parameters/H throughout the objective batch remain required external contracts. No own-sample-including baseline or leave-one-out mixture density is introduced.

## Costs and controls

Write `C_R=3h p+3h^2`, `C_H=2h`, and `R=C_R+7h` for shared transition parameters (including both gate biases and h0). The ordinary-head control has `R+M(2h+2)` decoder parameters. The factored-head control has `R+2h+M(h+4)`. Both exclude the common source encoder. Compare with the earlier private-factored decoder's `C_R+2h+M(p+15h+4)`. These are the documented tensor-shape counts, not parameter/capacity matching.

For K schedules, one objective batch charges one source encoding, K generation recurrences with one head each, and K density recurrences with M heads each. Selected forward multiply terms are

`N[2K C_R + K(1+M) C_H]`,

plus factored-head diagonal work `N K(1+M)(h+2)` when applicable, recurrent/head biases, gates, masks, softmax, scores, rewards and all external task/RNG/reference work. The private-transition packet instead replays KM recurrent trajectories for full cross densities. With K=MS, this control still has `M^2 S` head density contributions, but only MS density recurrences. No cross-head contribution is omitted. Forced steps still compute and charge heads in this source.

Only K generating-score graphs are retained, sharing the one common encoder graph; the counter is named `retained_generating_score_graphs`. It is **not** a completed-backward counter. Actual backward work, wall time, transfers, Python/synchronization overhead, optimizer state, activation/container memory and transient allocations are unmeasured. Generation and density recurrences are separate; sampling self-density reuse is not implemented. The cheaper intrinsic mean-KL path would need only one head per density recurrence, and must remain disclosed alongside the full-matrix objective ablation.

`PAIRED_COMPARISON_REQUIREMENTS.md` specifies source/replay equality and attribution checks. Parameter/cost matching, initialization, task/outcome/control freeze, strong baselines and numerical/resource qualification remain root responsibilities. Existing phase/diagnostic caps, failed work and the one-A100 limit are unchanged. This packet grants no execution or fit permission.
