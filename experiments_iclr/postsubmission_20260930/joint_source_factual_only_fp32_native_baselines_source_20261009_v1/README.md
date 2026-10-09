# Factual-only native FP32 baseline extension — disabled

9 October 2026. This separate source adds `factual_native_single_P` and `factual_independent_native4_P`. It preserves sealed joint V1/V2 and all ten existing conditions. Together they supply the twelve declared conditions for the next prospective pilot freeze; no combined freeze, runtime release, qualification or fit is created here.

The baseline gap is ordinary factual-only native training. The existing native single and independent additive P controls train on four supervised views. They cannot replace these two full-input native controls. Historical AMP results remain context, not matched FP32 baselines. This extension asserts source availability only, with no quality or resource result.

## Callable integration

`factual_contracts.Config` is disabled by default and requires this exact source seal, adopted V2 source review, native qualification and original role/full-context/study bindings. Scientific fits additionally require actual factual qualification and a new combined quality/resource/custody freeze. Body and stochastic seeds must be frozen with the paired joint family; independent4 uses four distinct native bodies and streams. Adapter seeds are empty.

`prepare_factual_context(rt, modules, input_root, role_file, costs, config)` uses the original role loader and native full feature/TRAIN-label propagation. It returns only the full context. It does not construct source-absent views, resplit roles, acquire data or create teachers. A root-owned qualified existing full context can also be supplied; retain its actual setup cost log and bindings.

`make_condition(rt, modules, ctx, costs, config)` and the two named constructors create a thin subclass of the injected immutable V2 Session. The module mapping adds `joint` to the seven original dependencies. Hash/namespace checks require the actual V2 module and its own contracts. Native storage ownership is checked directly because the existing private-gradient ownership verifier requires nonempty private blocks; these bodies deliberately have none. They retain all native83,659,532 parameters and three private BN modules per body.

The integration reuses V2's native BN schema/storage checks, exact public role guards, factual evaluation, own optimizer verification, owned model/buffer/Adam/RNG snapshot/restore, serving and cleanup. The fit loop is the literal fixed V2 loop with the factual constructor and policy, own selectors, and no source diagnostics; it is not another scheduler or runtime framework. All public forward tokens must have `source=None`. The source-ablated diagnostic method is explicitly unavailable.

## Actual factual training policy

Each active native body optimizes `B(sigmoid(z_full), y)` averaged across every TRAIN row and five labels, coefficient1. There is no division by4, committee-loss gradient, adapter, private penalty, source credit or source-absent supervision. Independent losses and native storage/Adam are disjoint. Every full native forward/backward/Adam is paid.

There is one live FP32 full TRAIN forward and backward per active body per epoch. Native dropout uses the declared owned stream. The existing scratch helper clones the canonical running buffers; the forward captures native post-BN state and RNG. After scratch exit, post values are copied into the restored canonical buffers, leaving the live tape's scratch tensors untouched. Each native BN counter advances persistently once. No extra no-grad reference or gradient replay is imposed on these controls. All active bodies' gradients finish before their separate Adam steps. Adam is lr0.001, betas(0.9,0.999), epsilon1e-8, decay0, with autocast disabled and no GradScaler.

Stopped native bodies make no TRAIN/peer calls and receive no gradients, Adam updates or BN/dropout/RNG advancement. They retain terminal weights and buffers while other bodies train. Full factual evaluation remains charged for all serving members. Each body selects its own full VALID BCE, strict earliest ties, zero-based maximum200 and literal `epoch-best_epoch>50`. No best-state restoration occurs during remaining training. Final serving freshly constructs all bodies and restores each own best separately, with FP32 probability mean and strict `>0.5` decisions.

Snapshots add the factual policy, one-view coefficient1 normalization, actual one-live-pass buffer policy and V2 review binding. The common V2 `buffer_policy`/schema tag is retained solely for the reused native state reconstruction contract; `factual_buffer_policy` and `training_policy` explicitly describe this control's training. Actual BN tensors are retained in the native model state dict and verified exactly on restore.

## Cost and qualification boundary

Full native setup and constructors are charged to the supplied entry cost log. No source-view setup cost is paid or invented. With A active bodies an epoch pays A TRAIN forwards, A P backwards and A Adam steps, with factual evaluation paid separately. Setup performed elsewhere needs retained root log custody. Fit scalar timing/CUDA peaks exclude prior setup and the first constructor; the result binds the entry log prefix/final hashes and records process-highwater RSS, output storage and authoritative cleanup status. Nested costs overlap.

See `QUALIFICATION_PLAN.md` for the bounded later full-input check and `PROTOCOL.json`/`RELEASE.disabled.json` for inactive declarations. No numerical providers/models/data/outcomes/checkpoints were loaded and no numerical tests, servers, qualification execution or fits were run in this source task.
