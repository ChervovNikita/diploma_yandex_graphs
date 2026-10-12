# Fixed context keys: separate SAGE port

This packet implements the ten named source arms from `native_SAGE_fixed_context_key_source_20261011_v1`. It reuses pinned native Model/TABMModel, source FactorLinear/common routes, and the fixed-key target and loss equations. It does not claim the archived trainer/roles/seeds/horizon are reproduced. All real-data work is disabled metadata preparation. Synthetic engineering QA is separate from predictive evidence.

## Source custody and adaptation

| Source | SHA256 |
| --- | --- |
| Original native `models.py` | `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303` |
| `common_routes.py` | `84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773` |
| `factors.py` | `9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3` |
| `fixed_context_key_objective.py` | `4e21ed2968e2f9c3fb97aa17bdf8370c6a52f9f9338cd6d760492ca235446944` |
| Fixed-key `run_family.py` | `b24256e07af818f88d358b55f6cbd58cc61a8ab4b3c4042002fbeb5d16dc3c6e` |

The archived driver is authenticated but never imported/executed. Its source `prepare_targets` and `tensor_bank` functions are called unchanged. The source `context_auxiliary` AST replaces only the literal shape constants580/128/300 with explicit TRAIN-row/hidden/key dimensions; every loss operation and count equation stays unchanged. The source query-map generator receives an isolated globals copy containing those dimensions. The generalized bare map is hidden-width to feature-width, initialized using the same private CPU float64 QR/sign algorithm, then float32. Feature width must be at least hidden width. Provider class count is passed to the native classifier; every TRAIN class must have two examples. Missing classmates fail without filtering or repair.

The present finite matrix remains width128/depth2 and SAGE only. Native constructors, LayerNorm/GELU/residual ordering, shared bias/normalization ownership and source FactorLinear placements are retained. Only the first input fast R is separately seeded Rademacher; other R/S rows are initialized to one. Parameters are installed before AdamW, with native/query parameters owned exactly once. The query map has its own group at the same LR and WD0, and is shared across the four routes of one shared body. Independent bodies own separate query maps and learned states. Their paired-seed query-map initial bytes are deliberately identical; native body/dropout initialization differs by paid body index.

## Ten source arms and private gradients

| Arm | Fits per comparative seed/split | Routes per fit | Auxiliary |
| --- | ---: | ---: | --- |
| ordinaryM1_key_mixture | 1 | 1 | Mean of four complete fixed-key objectives |
| factorizedM1_key_mixture | 1 | 1 | Same mixture on source factorized M1 |
| genuinefactorizedI4_key_mixture | 4 | 1 | Four separate own-selected mixture learners |
| genuinefactorizedI4_key_assigned | 4 | 1 | Paid body m uses exactly Q_m/K_m |
| shared4_key_private | 1 | 4 | Route m uses fixed dictionary m |
| shared4_key_mixture | 1 | 4 | Each route averages four complete objectives |
| shared4_key_uniform | 1 | 4 | Same fixed keys; uniform nonself classmates |
| shared4_projected_peer_private | 1 | 4 | Live projected same-route peers; both gradient paths |
| shared4_key_moment_private | 1 | 4 | Literal `(1 - q·QK)/temperature` |
| shared4_key_shuffled_private | 1 | 4 | Class-preserving permutation of both Q axes only |

Keys are normalized fixed public-feature contexts X, X−PX, PX and P²X. Teacher affinities use ordered TRAIN labels only; every nonself classmate remains a positive and every nonself TRAIN row remains in the dictionary denominator. All keys, weights, support masks, IDs, labels and moments are detached. Moments are the actual float32 Q@K reductions and are never normalized. Unit logZ coefficient is literal1, independently of small cast row-mass errors. Mixture means four complete objectives with separate denominators; it does not average keys first.

The total loss is mean factual own CE plus0.05 auxiliary. Auxiliary gradients reach the live pre-head representation, native trunk/fast factors and trainable query map. They do not reach detached targets or the native classifier head through the auxiliary path. The live-peer arm keeps both query and peer sides live. The shuffled arm moves only the two Q axes; factual key rows remain unchanged. Same-state generic SupCon is not a substitute for these operations.

Shared4 preserves input TRAIN-bank row order. Every M1 and separately fitted I4 body uses stable sorted TRAIN IDs, matching the source boolean-mask order. Both Q axes, fixed keys, IDs/labels, shuffle permutation and moments are transferred to that actual hidden-row order. Checkpoints preserve actual target tensors/hashes, private member dropout streams, query-map model/optimizer state, selected epoch, complete modes/flags and selector state. Strict restore rejects changed target/key/moment bytes.

The query map is used only while fitting. Source arm serving averages probabilities; original-GNNM reference serving averages raw logits. Four source boundary routes are one shared body/optimizer, while genuine I4 means four separately paid bodies, optimizers, private streams and own selected checkpoints.

## Prospective finite standardized matrix

Use prepared WikiCS20 custody, nine LR×dropout points (`.0005/.001/.002`×`.1/.2/.4`), width128/depth2/WD0, search seeds71101/71103 on split0 once, and comparative seeds71201/71203/71207 on all20 splits. Every model fit completes5000 optimizer updates. The archived source uses fixed1000 maximum/patience300 and different roles/seeds; no unchanged recipe claim is made. SELECT is the prepared role, CAL is separate, and evaluation targets remain sealed until the root global closures.

Own strict earliest SELECT accuracy selects each checkpoint. Whole-cell HPO ranks mean own-selected accuracy, then proper loss and stable config ID. Assigned-I4 search pays four body/context trajectories per point/seed and ranks their macro accuracy. That is an explicitly standardized HPO rule, not pooled-I4 HPO accuracy. Mixture-I4 uses member-equivalent factor-M1 HPO choice, and still retains all four separate comparative acquisitions. No source arm is merged into a historical control based on outcomes.

Competent references are ordinary CE M1, source factorized CE M1, original boundary GNNM, ordinary CE I4 and factorized CE I4. The latter two explicitly reuse their already declared CE body0 and add three own fits. Original GNNM is allowed only on new compatible data; the original five datasets remain excluded.

| Quantity | Declaration |
| --- | ---: |
| Search fits | 270 |
| Comparative fits | 1500 =60 cohorts×(16 source+9 reference) |
| Total paid model fits | 1770 |
| Registered model optimizer updates | 8,850,000 |
| Native route forwards while fitting | 17,040,000 |
| Separately owned query maps | 1176 |
| Search selectors | 12 |
| Served banks / exports / CAL fits / evaluations | 900 each |
| Scalar CAL updates | 450,000 |
| Total run records | 4482 |

Each fit has one optimizer object and one physical slow body. Shared4 and original-GNNM fits execute four native route forwards per update; single/I4 fits execute one. Query/dictionary/moment operation counts come from the source helper. Public signature/target preparation is charged once per context factory/fit, with no unimplemented amortization claim. Export reconstruction can repeat preparation; complete acquisition/export cost qualification remains a root resource task.

Resources must be explicit, finite and labeled provisional until target-bank/query/native CUDA memory and rate are measured. Planning capacities for three workers do not authorize access to18.77. The plan keeps dependencies on the allocation and two slots per GPU; launch is false and human GO is mandatory.

## Engineering scope

`tests/test_context_keys.py` covers source hashes, complete finite counts/reference cohorts, all ten arm constructions/updates, optimizer/query ownership, source-versus-adapted loss and gradient equality at the original dimensions, both live peer paths, classifier-head gradient exclusion, shuffled-Q-only axes, literal unnormalized float32 moments, private RNG, disjoint I4 storage, and strict checkpoint/target custody. Numerical tests use synthetic arrays only. Real-data compilation imports no numerical library, deserializes no role tensor, scores no model, and creates no GO. CUDA qualification and whole-root release remain required.
