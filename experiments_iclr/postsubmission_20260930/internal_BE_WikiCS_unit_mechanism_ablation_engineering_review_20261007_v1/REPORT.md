# Independent static engineering review of WikiCS VJP recomputation v2

Reviewed 7 October 2026 against pinned portable v2 `Session.train_step`, `Session.forward`, the model/factor/objective code, native Polynormer, original driver/transition and the separate two-lane owner. Source seal: `c6ef37631a0669df45b690071f3eed70c98f53a0e97fed1d902f82325b11dd1c`. Sealed source/public payload hashes match. Only workspace source files and metadata were read; no numerical imports, graph payloads, training, server access, parity fixture or sealed-source edits occurred.

**Disposition: no remaining material source defect found in the reviewed replay path or patched owner.** This is static engineering review, not scientific admission or a numerical-equivalence certificate. Scientific12 remains subject to root's separate admission.

## Gradient and objective correctness

[recompute.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/internal_BE_WikiCS_unit_mechanism_ablation_source_20261007_v2/recompute.py:38>) reconstructs the exact shared-bank loss: two CE views averaged by .5, mean across four members, plus the original .05 alignment and .05 residual terms where enabled. The same evenly spaced maximum512 TRAIN targets and .2 temperature remain. Plain skips both auxiliary functions; the per-session facade zeros only the declared inactive component. It adds no member/view rescaling.

The loss is evaluated jointly on all detached output leaves before its cotangents are collected. Thus cross-view, cross-member and class-centering derivatives remain in those cotangents; this is not independent per-member auxiliary minimization. The original objectives depend on parameters through logits/representations, so no direct parameter-loss term is omitted.

Each replay calls backward with **both** logits and captured representations as roots. Autograd adds the classifier path and the auxiliary representation path; it does not double the CE path. Shared weights accumulate contributions from all eight replays, and private factor slices retain their member-specific pullbacks. One zero-grad precedes the bank, and one native Adam step follows it. No averaging is applied to the already scaled cotangents. The parameter-version and finite-gradient guards cover the completed accumulation.

## RNG, modes and native trajectory structure

The shadow calls use the original forward twice in train mode under no-grad. Replay restores pre-shadow streams and follows the same view0/member0–3 then view1/member0–3 order, using the original CPU/CUDA `fork_rng` protocol. It checks equality to the shadow endpoints after replay, so the extra eight forwards do not introduce extra stochastic views or advance the persistent streams twice. Known native backward paths use stored forward values/masks. Member context restoration is compatible with factor indices captured in the graph.

The pinned native model uses LayerNorm and has no mutable normalization buffers; installation rejects buffers/BatchNorm. Native hooks and factor contexts are restored after each forward. Train mode is set before each shadow/replay update; evaluation uses the unchanged public forward/pool under eval/no-grad. The full driver still restores the selected joint local model/optimizer and switches to global mode at epoch101 while retaining the live end-local RNG. Checkpoint selection and serving remain original. Floating-point accumulation order changes, so no bitwise trajectory claim follows.

## Qualifier and resource scope

The qualifier exercises actual fullgraph local/global updates for P/A/R/C, checks16 forwards, eight VJPs, one cotangent collection and one Adam bank transition per update, then CPU-mapped save/reload, CUDA parameters/moments, CPU byte RNG, restored global flags and complete5274-object serving. Its nonblocking prediction-difference diagnostics are appropriate to the stated mathematical-objective claim; no toy bitwise gate is required.

Root reported completion of all eight updates/128 forwards in85.66 seconds and maximum reserved memory11.3203125GiB, below the32GiB cap and the allowed24GiB minimum. These are **root-supplied runtime facts**, not independently opened receipts or measurements in this review. Root must bind the complete qualifier receipt and external owned-process resource evidence to the staged release. Torch reserved memory and external process GPU accounting are distinct; the owner enforces the latter. Source estimates alone do not qualify capacity or long-horizon time.

## Separate owner correction verified

The initial owner passed the32400-second total envelope as the helper's active deadline; the helper could then wait another five seconds for termination. I reported that mismatch. Root changed [owner.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/wikics_unit_mechanism_gpu77_scientific_activation_root_20261007_v2/owner.py:73>) to pass `cfg['external_active_seconds']` (32390), retaining the declared10-second cleanup allowance inside32400 total. The corrected bytes were reread; SHA-256: `84671ddcca0e6edc4fae84f4d4865109a849eaf5877192bf86303029623e3941`.

The two lanes load separate ownership-helper modules with separate GPU bindings, launch children in fresh sessions, apply owned-tree caps, require observed exits and absent CUDA/PID rows, and stop a lane on failure without retry. No remaining material issue was found in that scoped owner check. The reviewed owner hash must match regenerated staging/BUNDLE bytes.
