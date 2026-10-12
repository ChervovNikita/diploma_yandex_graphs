# Inactive meanSAGE attribute feedback model core

This separate core implements the ten ownership/operator variants of the sealed attribute-feedback proposal. It changes no trainer, optimizer, data provider, selector, current sign-dropout source or scientific roster. It has no Torch/NumPy/scientific imports or numerical work at module import. Static AST compilation is complete; no model was constructed, no tensor was loaded, and no numerical run or runtime qualification occurred.

## Exact model behavior

`build_arm(torch,F,arm,block)` accepts injected runtime modules after root admission. It returns one joint model for P1/F1/S4/PA1/SA4/SM4/SX4/UX4. Genuine I4 and IA4 return **four separate M1 model objects**. Fit them with four disjoint optimizers, full own CE/L2 and own selectors; collect their separately restored models only for serving. UX4 is one jointly trained bank with four disjoint bodies and feedback maps. It has one mean-own objective and one joint selector. It is not genuinely independent I4.

Each source layer uses the same native dropped input for aggregate and skip, mean aggregation on the existing complete support, additive bias and clipping by `max(L2 norm,1)`. First-layer ReLU and clipped final logits are retained. Raw sparse input dropout changes stored values only and retains indices including dropped zeros. Hidden dropout is ordinary elementwise p=.4. Input X is not normalized. Factors act after that shared route-local input mask; aggregate and skip have separate input/output factors.

For factor arms, the effective matrix is `diag(r) W diag(s)`. Factors start from independent per-route Rademacher generators; they learn freely. The first-layer penalty is `.05` times squared effective aggregate and skip matrices, averaged over joint routes. The ordinary source arms have the exact source matrix penalty. Extra factors change optimization and gauge behavior, so F1 and genuine factor I4 remain required competence controls. Feedback maps are unpenalized in every equipped arm, as frozen.

The only new path is first-layer node state to column-normalized attribute hubs, a shared or private rank 16 nonlinear map, row-normalized attribute readback at fixed scale .25, and clipping before the source second-layer dropout. SA4 and equipped singles/I4 use self context. SM4 maps a peer mean. SX4 averages mapped donor contexts excluding the recipient. UX4 uses its own donor map for each body. No donor is detached or frozen. Own CE sends live gradients across recipient/donor paths.

`Vmap=0` preserves the corresponding native forward at initialization. `Bmap` uses Glorot uniform and a dedicated CPU seed. Source slow maps consume the source Glorot law independently of factor/feedback draws. Constructor CPU RNG is forked/restored, and CPU-only default-generator seeding avoids modifying CUDA streams. Route 0 factor/feedback starts are paired with F1/first genuine I4 body; corresponding shared arms match. This is distribution/equation custody, not bitwise TensorFlow equivalence or numeric parity evidence.

## Caller contract

A joint model returns logits of shape `[members,34493,5]`. TRAIN requires `route_rng_context(m)`, a context manager supplied by root that activates and then saves a persistent independently owned native dropout stream for that route. The first and second calls continue the same route stream. It must preserve caller CPU/CUDA/global state, never reseed per update, and be included in selected-state restore. This core implements no RNG manager, checkpoint, data role, optimizer or selector. Each body exposes its initial `post_constructor_cpu_rng` for custody; that constant is not the mutable training RNG state.

Use full qualified public raw X and native adjacency/degrees. The adjacency has 530,417 already-prepared entries including 34,493 self entries; add no loops or symmetry. Degrees must be positive `[34493,1]`. Build `C,D` once with `incidence_from_raw_binary_features(torch,raw_X)` before any dropout, from the same full UNDROPPED X, then reuse them in all equipped arms. Zero rows/columns naturally produce zero outputs. No label argument, native feature normalization, nearest-neighbor search or N×N feature graph exists.

For a joint bank use mean own TRAIN CE plus `model.regularizer()`. For each genuine I4 model use its full own CE plus its own regularizer. Root supplies the sealed source TF1 Adam formula, latest-tied-minimum total SELECT loss, 50-stale/100000-update ceiling, full synchronized joint restoration or genuine independent own restoration, and the frozen calibration/release order. Never apply one optimizer or selector to the four models returned for I4/IA4. Do not use `no_grad` or detach on TRAIN donor paths.

`inspect_parameter_ownership(core)` is a deferred runtime guard for exact trainable counts and distinct parameter-object/storage ownership. It checks separate genuine I4 models, UX4 bodies and UX4 feedback maps. It does not inspect data, validate RNG independence or qualify runtime. Root must call it only after admission and bind the returned receipt.

## Binding and cost

`SOURCE_BINDINGS.json` binds the sealed research manifest and source-derived meanSAGE ancestry. `DATA_COST_BINDING.json` states the dimensions, role protocols, unavailable payload hashes, parameter counts and paid work. It does not substitute public dimensions for actual source/data/resource qualification.

Across three blocks the panel remains 30 banks, 48 optimizer trajectories and 57 physical bodies. Candidate storage is 607,924 parameters; genuine equipped I4 and coupled UX4 each have 2,227,636. The candidate has one shared rank 16 map pair, UX4 four private pairs. Every incidence propagation, nonlinear map, live backward, effective regularizer, selector, export and CAL pass is paid. No timing/peak estimate was measured.

This factory is incomplete orchestration by design. It is not a fit launcher, provider adaptation or runtime admission. Root first closes the running sign-dropout study, then decides qualification and execution of this distinct representative mechanism.
