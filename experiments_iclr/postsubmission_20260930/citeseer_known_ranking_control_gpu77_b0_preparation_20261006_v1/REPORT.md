# Known ranking control: minimal 77 b0 preparation

## Scope

Three new b0 controls are prepared: S, J4 and E, in the original ranking packet's order. Wave1 assigns S to physical GPU a998 and J4 to8ced; E follows on a998 after both wave1 fits have successful observed terminal closure and complete60-cycle freezes. Other six prospective ranking cells remain gated. The nine original BCE anchor metadata bindings are preserved and no anchor is rerun. This packet launches nothing.

## Scientific source and explicit schedule change

`transfer_step.py`, `models.py` and `private_adam.py` are byte-identical to the ranking packet. The fixed loss remains `2*softplus((negative-positive)/2).mean()` for inner, outer and repeated-inner committed ordinary Adam passes. No sampler, RNG-consuming code, architecture, optimizer, dropout stream, positive/negative list pairing, union mask, serving rule or selector was changed.

`custody.py` is copied exactly from the reviewed77 `custody77.py`. It binds peptide, the authorized repository, ordered a998/8ced physical inventory and exactly one prospectively selected visible physical UUID. Runtime/input/source loaders remain the existing contract.

The original runner allowed max60/eval5/eleven VALID misses. The new runner requires exactly60cycles/eval5 and disables result-based early stop. This is an explicit prospective schedule change under the user's all60 request, not byte-identical schedule behavior. All three archived b0 BCE anchors themselves recorded60 cycles. The first maximum rounded4 complete VALID MRR remains the selector and serving remains the arithmetic mean of raw member logits without adaptation. Successful fit freezing now also requires60 completed cycles. `SCHEDULE.diff` records this small change.

## Actual new-loss gate before any fit

Two disabled qualifier jobs cover only their assigned architectures: E/S on a998, J4 on8ced. Each architecture pays one first-episode actual ordinary step and one separate direct-native-module reference step. The reference uses the analytical logit VJP of the fixed loss and ordinary `torch.optim.Adam`; it calls neither `own_loss`, `outer_loss`, nor `ordinary_episode_step`. It preserves E/S means and J4 inner sums/outer times4. Complete committed parameter/moment parity, direct serving, exact fresh actual/reference initialization, three Adam updates, dropout replay/one advance, native modes and global CPU/CUDA RNG are checked. The copied comparator helper bodies and original immutable parameter/moment tolerances are retained.

Each architecture then starts a fresh model, Adam and dropout streams and pays exactly one complete TRAIN cycle:3870 outer positive examples, all61 episodes including the tail,183 ordinary updates. It writes a raw architecture `COST_RESULT.json`, episode metadata and assigned-GPU raw qualification `RESULT.json`. It loads only authenticated TRAIN/features and discards all models/states. VALID/TEST values, metrics and checkpoints are absent. There are three actual cost cycles total, with no repeated architecture qualification on the other identical GPU.

Root must independently review this exact source and qualifier, release and run the two jobs under the existing owned-process helper, verify both terminal receipts and costs, and combine their E/S/J4 entries into a new all-three manifest-bound gate. `ALL_THREE_GATE_TEMPLATE.json` describes that metadata merge. The runner's existing all-three architecture/exact-source admission guard is unchanged. Root binds the three raw measured cost receipts into an adopted plan before any fit. Old BCE FP32 results and cost precedents cannot supply the new gate or measurements.

## Runtime, input and fixed bounds

The exact existing repo-local CPython3.11/Torch2.1.2+cu118 runtime is bound; no installation is needed. The original official acquisition manifest and four existing public roles are used at their authorized77 paths. Root's new runtime/resource and cached-input-path metadata receipts are pinned, while each child still authenticates the actual accessed TRAIN/features or TRAIN/VALID bytes through existing custody.

Qualifier E/S soft900s/hard1200s, J4 soft600s/hard900s. Fits preserve original per-cell caps: E soft15000s/hard18000s; S/J4 soft18000s/hard21000s. Owned CUDA10GiB, RSS64GiB, child logs8MiB, own fit output1GiB, fresh free CUDA12GiB and finite pre-child wait3600s are fixed. No automatic cap expansion, method skipping, retry or horizon shortening is permitted. Existing unrelated jobs may coexist; their processes and host configuration are untouched. Root reported42,583MiB free on each GPU, but actual new costs and fresh preflight remain prerequisites. Qualification/cost and fit caps remain distinct. The conservative fixed two-wave fit phase cap is47100s, including fixed wait/cleanup allowance; serial allowance is71700s.

`COMMANDS_DISABLED.json` gives exact child argv/env/cwd and binds the existing reviewed77 ownership helper. These are child descriptions requiring root-owned supervision and activation. Existing30-cell/block queue mains cannot be executed unchanged for this three-cell control. No new process or guard framework is included. Root may stage this disabled source while qualification is pending. A qualification-only start/completion is not the detached full training launch promised before shutting down the other Mac.

## Verification and interpretation limits

Only stdlib AST/hash/metadata checks ran locally. All six Python files parse; source manifest closure, copied source identity, unchanged runner `metric`/`admissible`, copied comparator helper AST identity, exact three disabled fit jobs, two assigned disabled qualification jobs, all60 schedule and nine BCE anchor metadata bindings pass. No prepared source was imported, no numerical framework or payload was opened, no server command was run and no child/fit launched. Independent review and numerical/resource qualification remain pending root.

This is a known ranking-objective CONTROL/calibration ablation. Historical bitwise initialization and archived logical draw replay remain unverified; that limitation is disclosed and is not a fabricated stop condition. Host/runtime/seed/source matching does not establish identical historical trajectories, novelty, method-specific superiority, simultaneous member improvement, causal historical-gradient attribution or heldout performance.
