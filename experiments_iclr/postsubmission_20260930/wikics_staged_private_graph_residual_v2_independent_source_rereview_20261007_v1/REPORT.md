# Independent v2 source re-review

This is an independent source **re-review with the prior v1 source-review context**, performed by the same reviewer. It is not a fresh-agent review or a paper review.

Reviewed source: `wikics_staged_private_graph_residual_method_preparation_20261007_v2`.

Manifest SHA-256: `84779fdd8fd697c568850862e0deeeabd11116c41c4cde800f590924d4eea8db`.

**Decision: the exact v2 discarded full-TRAIN numerical attempt is source-approved within the fixed external bounds. The complete cost attempt is conditionally source-approved only after an exact v2 numerical pass and the exact native-only qualification pass. No fit release, novelty, empirical quality, or superiority is certified.** No source blocker to these bounded qualification attempts was found. This does not predict that the strict numerical reference or its ordinary-forward memory demand will pass.

## Source custody and scope

The v2 manifest and all 17 named file hashes/sizes match. All seven Python files parse with the standard-library AST parser. V1's manifest remains `a64b708f17d478dc1128c2a6a9032fa82bbff9edc8f380e266e4e41e64702d2a`. Native vendor bytes are identical between versions. AST comparison of `method.py` shows only `bank_epoch` and `single_epoch` changed; the model classes, native constructor/step, dropout/RNG operators and correction loss are unchanged. Their additions are optional qualification observers, plus diagnostic work metadata. The ordinary fit calls leave observers at None.

The seven scientific arms and seeds17/29/43 retain all 21 unique cells. The only changed PLAN fields are qualification/custody/status/schema declarations. Disabled jobs, command activation and root adoption remain disabled in this source packet. No Python source module, torch runtime, model, dataset, checkpoint or held result was imported/opened/executed. No GPU, SSH, launch, source edit or subagent was used. This report and its hash bindings are the only new local artifacts.

## V1 findings and repair assessment

| Finding | Source evidence in v2 | Assessment |
| --- | --- | --- |
| F1: duplicate deletion | `qualify_numerical.py:46–74` isolates reference lifetimes; mode cleanup at107 no longer deletes a variable already removed in a nested replay | Fixed. No analogous definite cleanup failure found. |
| F2: graph gradients only referenced while asleep | `94–96` runs first and awakened references for each of four paths in both modes; `58` applies a manually derived cotangent to the same actual graph; `67–72` compares actual gradients, Adam-updated parameters and moments | Covered prospectively. All graph groups must be zero first and nonzero after wake. |
| F3: single only global/connectivity | `109–155` compares two full checkpointed/ordinary updates in local and global modes, with logits/loss, all active gradients, all parameters, both Adam states, five streams, caller RNG and path/decoder/native wake | Covered prospectively, without omitting the ordinary full-model reference. |
| F4: arbitrary passing native receipt | `common.py:62–86` requires native_qualification, exact c0d6 native manifest, runtime, original tolerances, both stages, two native references, whole epoch costs, TRAIN roles and caller RNG restoration | Fixed for the required gate. |
| Identical cache producer | `run.py:160–196` binds exact E_stage producer job/output authority/source/program/arm/seed/plan/data/donors and detached finite tensors | Exact producer custody is now enforced. |
| Final cache function closure | `run.py:221–225,239–241` links the model freeze to cache authority/payload and closes that model freeze from the final FREEZE | Fixed. Model state alone is not presented as the bank's complete served function. |
| Diagnostic cost omitted | `qualify_cost.py:40` enables fit diagnostic VJPs for each bank case | Fixed for this cost omission; full-fit selector/I/O forecasting remains outside these probes. |

## Numerical reference logic

The private observer snapshots pre-update bank parameter state, checks that state before every requested route's gradient, deep-copies that route's parameters and reconstructs its Adam state with cloned CPU tensors, and computes a same-forward VJP with

`(p-y + p*(w-sum(p*w)))/580`, where `w = (sum(detached peers)+p)/4 - y`.

This is the declared gradient of own CE plus twice the mean pool Brier. The own-only branch removes the second term. Autograd produces the private parameter VJP on the exact same live dropout/message graph; the observer does not perform a second model forward, change parameters, advance a route stream, or fill actual .grad. `total.backward()` then creates actual .grad, and actual Adam updates occur only after all requested gradients have been collected. The shadow Adam receives detached clones of the independent cotangent VJP; its state loader receives cloned pre-update state, so the replay does not share actual optimizer moment tensors. Deep-copy parameter order matches the same unchanged module architecture; actual optimizer ownership is already enforced by ResidualBank construction.

Every path is checked first with exactly zero incoming gradients and a learnable output head, then with nonzero raw/h/GAT/root/LayerNorm/beta groups and referenced populated Adam behavior. Asymmetric learned peers are explicitly required before the four-route reference (`102–104`). The state snapshot check there rejects an early parameter update. Own-only is independently referenced after wake. Frozen donor values and absent donor gradients are checked in both shared modes; the untied four-fresh-donor bank receives a separate ownership/storage/reference exercise. Its ownership exercise is global, while the shared private gradient/reference and hidden interfaces cover both modes; mixed selected untied modes are not separately simulated. No full1100 training is claimed.

The capable-single reference is `copy.deepcopy(single)`, including its registered donor/paths/readout, optimizers and route states. `optimizer_ownership` verifies that both copied optimizers own every copied active parameter exactly once. It runs checkpointed actual and ordinary shadow updates sequentially from corresponding parameter/Adam/stream states, retains CPU copies of actual forward logits/loss, and compares all active gradients, all parameters, both Adam moments and all five stream states. Local native-inactive parameters are explicitly required to have no gradient, rather than being omitted from parameter comparison. The second pair starts from populated state and a nonzero joint output. Each graph group and the hidden decoder must wake; the selected native mode must learn in both steps.

`use_reentrant=False, preserve_rng_state=True` remains appropriate for a grad-free raw input and trainable native parameters. Each original forward runs inside its isolated route stream; backward recomputation uses checkpoint RNG preservation. All forwards and backward finish before any optimizer step. Transient native classifier hooks are removed in finally. The new reference compares these semantics at the unchanged tolerances rather than treating source plausibility as a numerical pass.

One minor receipt wording caveat remains: `path_reference` always records `populated_Adam_reference: true` (`73`), including the first update whose pre-update Adam state is empty. The later reference genuinely exercises populated moments. This is not a missing check or an attempt blocker; read the flag as overall replay coverage, not proof that the first update began populated. The phase labels and complete first/after-wake records retain the distinction.

## Scientific controls and cache custody

E_stage and E_own retain ordered 4×100 private updates; E_joint retains100 four-route bank epochs with common pre-update deterministic, detached peers and delayed Adam steps. E_joint is interleaved coupled coordinate fitting, not end-to-end joint ensemble training. These controls preserve the same cached selected donor, private draws, path width/support and100 updates per path. E_joint/E_own must load the exact E_stage authority at the bound producer output and its hash-bound payload; a new same-donor forward cannot satisfy that contract.

S_continue retains live end1100 native weights/Adam/RNG plus400 native CE epochs and its old eligible selector. I_native retains four fresh independently acquired1100 prefixes and four ordinary100-epoch continuations, separate selectors/optimizers/streams and post-fit pooling. U_stage freezes four own selected native donors and runs the same staged correction, rather than being mislabeled ordinary independent4. S_paths keeps the selected native model/Adam with live end1100 RNG, all four full graph paths, nonlinear2048→512→10 readout and a live selected native function. It has no hidden cache. A local-selected single trains the active local function and the four paths; it does not silently switch to global mode. These documented start policies remain distinct from the native end-state continuations.

Final changed predictor states and bank cache metadata are frozen before their first endpoint VALID score in the fit source. This re-review does not approve that fit program's activation or comparative endpoint analysis. The prescribed source-native VALID selectors, fixed residual endpoints, complete population/competence/error-flow criteria and no-rescue rules remain prospective and unchanged.

## Complete cost attempt and bounded approval conditions

The cost source covers native1, native4, E_stage, E_joint, E_own, S_paths and U_stage in both local/global modes: exactly14 cases. Fresh correction banks pay cache construction and one real TRAIN warm step before measurement; measured routes have nonzero residual outputs and initialized optimizer moments. All bank cases enable the diagnostic pair where the loss includes Brier, so their elapsed/memory measurement conservatively includes the work omitted in v1. E_own correctly has zero pool diagnostic work. S_paths pays its real checkpointed native and four-path rematerialization inside the measured epoch. Native4 holds four models, a disclosed storage upper bound compared with serial native fits.

The native cost cases include first-update Adam allocation; the separate pinned native qualifier also measures after populated steps. These are fresh-state epoch probes, not complete donor convergence, selector, checkpoint, cache serialization, CPU transfer or deployment measurements. Full original horizons, source selectors and all artifacts still require conservative root forecasting before any later fit decision. No measured cost or speed claim exists yet.

The exact numerical program may proceed only with a fresh root-owned job and output, this review/source manifest and program hashes, the exact TRAIN-only projection manifest `978b382d1f95af23512606ef6ac9530f31ee510efcf36efb35d059bd0ad7615e`, the pinned runtime/allocation/GPU, and the existing owner. Keep the proposed bounds: at most1200s soft/1800s external hard per qualification job, owned GPU24GiB, RSS32GiB, logs8MiB and own output2GiB, plus the source's one-GPU and resource-inventory checks. Root must confirm/enforce the external hard/resource bounds; this review does not itself launch or establish them.

The cost program may proceed only after a complete numerical pass of this exact v2 source and an exact native-only pass. `common.py:90–102` enforces changed-source numeric-before-cost, in addition to exact native receipt semantics. Both jobs must remain discarded full11701-node/442907-edge/580-TRAIN-label qualification, with fits_authorized false and VALID/TEST value access false. All14 cost cases must finish. Preserve failed/partial receipts; no retry, cap expansion, shortened reference, relaxed tolerance or family fallback is approved.

Strict separate-forward checkpoint/ordinary FP32 comparison can expose the known native/GAT variation even with mathematically corresponding execution. The ordinary reference may also exceed its fixed memory budget. Either result is a failed gate for the present packet, not permission to omit it or infer success from connectivity. Only actual full receipts can resolve these uncertainties. Approval here is for the bounded attempts, never a numerical result, novel method, empirical superiority, or future fit release.
