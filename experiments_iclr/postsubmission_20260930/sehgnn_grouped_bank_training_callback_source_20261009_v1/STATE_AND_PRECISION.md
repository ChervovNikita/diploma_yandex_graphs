# State, scoring and prospective IMDB amendments

## Own objective and optimizer

The new IMDB study uses the mean of four native mean-entry BCE losses. It adds no factor prior. Both deduplicated native slow and disjoint private factor groups use native Adam `.001`, zero weight decay, default Adam moments. This is the root's prospective IMDB amendment, fixed before candidate fitting or quality/outcome access. The older Tolokers `.0005` centered-prior geometry scope is untouched. The sealed source-helper `Config` and source are unchanged: ordinary own update remains caller-owned.

Four streamed native AMP forward/backward tapes share the same old slow/private values; no optimizer step occurs between members. `loss/4` implements mean-own gradients, then one caller-master GradScaler step/update is attempted. Distinct private gradients/states are physical objects. Slow parameters have one Adam state. An overflow skip is not misreported as an actual Adam step. Each native own call persists that member's BatchNorm running mean, variance and counters.

## Precision

Ordinary own updates preserve native CUDA AMP plus GradScaler. All source TRAIN reference/replay/trial calls explicitly disable autocast and run native FP32, with FP32 native parameters and caches and an explicit differentiable `.float()` score boundary. This is a prospective IMDB numerical-validity amendment before qualification/outcomes: the helper uses unscaled `autograd.grad`, and casting only final AMP logits cannot prevent cotangent underflow upstream in half layers. A scaled replay alternative would require separately reviewed code and qualification; none is supplied here. Every source-correction control must use this same FP32 path and charge its cost.

Native eval is FP32, including the complete-TRAIN eval guards and complete known-role selection/serving. Native task output stays five logits. The helper receives explicit `bernoulli_marginal_logits`. All targets are complete binary TRAIN query-by-label entries on the same device; each label probability is pooled separately. Pool selection uses `-mean(logsumexp(observed_member_logps)-log(4))`, the stable BCE of the full-input per-label probability mean. There is no across-label normalization or product/joint-label mixture.

The unchanged helper retains its frozen replay tolerance and strict finite guard constants. This packet does not loosen them after outcomes. Exact replay is a helper precondition to be checked on actual native calls, not a new tiny floating-output parity campaign. Selected reconstruction reports practical output/probability/prediction drift while checking model/buffer/Adam/scaler/RNG restoration exactly.

## Distinct RNG and state transactions

The root must freeze four distinct member RNG seeds explicitly. After the exact native prototype has been initialized/placed, native `set_random_seed` creates each owned Python/NumPy/Torch CPU/CUDA state under caller-stream capture/restore. No native slow model reset occurs. The original native train loader remains generator=None and draws from the master stream. Each own member call temporarily installs/advances only its own stream and restores the master stream. No member's stream is aliased with another state.

Every reference/replay/trial token binds member, real source/full view, mode, complete row order and the post-own native buffer identity/version. A member's factual and family tokens reuse its same frozen stream, giving matched dropout masks for the native shape-preserving views. Tokens do not advance own streams. Every callback captures caller RNG and modes and installs **scratch clones** of all registered buffers. It then restores the original buffer object references and mode flags, followed by exact caller RNG restoration. The returned replay tape retains its scratch BatchNorm tensors. Copying saved values back into the tensors held by that tape would increment their versions and can break native backward; this implementation avoids that operation.

The transaction never calls a parameter-containing model state_dict restore. Current slow/private values remain live, including the helper's private trial candidate. Rejection and accepted private value handling are owned by the unchanged helper. The integration verifies slow identity/version, all gradient fields, the complete optimizer state, scaler and member/master streams before/after correction.

## Checkpoint metadata

Caller fit/snapshot/restore identity is normalized to exact built-in Python strings, integers, finite floats, booleans, None, dictionaries, lists and tuples; arbitrary metadata objects and tensors in identity are rejected. This removes a native qualifier's reported `TorchVersion` serialization hazard: that str subclass survives JSON but introduces a global rejected by `torch.load(weights_only=True)`. The final bank snapshot traverses the entire checkpoint tree and accepts only exact detached CPU Torch tensors plus plain primitives/containers (string/int dictionary keys accommodate native Adam state). Native tensor state is already copied by the exact native `cpu_tree`; this boundary does not rewrite values or allowlist foreign pickle globals. Source preparation executed only a stdlib metadata str-subclass witness. Actual weights_only save/load qualification remains root-owned.

## Caches

The exact pinned IMDB forward has no persistent unregistered tensor cache and modifies no feature/label/data_size metadata; its derived feature/label dictionaries are local variables. Every callback stages detached advanced-index slices of the complete canonical CPU caches and never passes those canonical tensors by reference. It captures/restores the integration's active staging pointer and verifies source/full mapping identities, key/order/shape/dtype/device and tensor versions. Native original buffers are restored by object reference. This cache transaction is specific to the pinned source: extensions with an unregistered mutable native cache need an explicit successor transaction and review.

Full and removed-family caches must be supplied by the actual separate literal raw-support constructor; declarations alone are insufficient. All25 feature and12 TRAIN-label keys survive; same path names across namespaces remain separate. Complete TRAIN row IDs are explicitly token-bound, not a shuffled current minibatch, TRAIN+VALID role mix or unassigned subset.
