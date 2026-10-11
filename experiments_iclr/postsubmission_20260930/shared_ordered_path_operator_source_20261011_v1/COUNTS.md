# Parameter derivation and full graph cost estimate

## Count derivation without runtime imports

Saved `models.Model` uses an input map D→h, L complete residual SAGE modules, final affine LayerNorm and h→C output. With multiplier one, each SAGE module has:

| Component | Shared W/b or norm parameters | Private factor vectors per member |
| --- | ---: | ---: |
| SAGE lin_l, h→h with bias | h²+h | 2h |
| SAGE lin_r, h→h without bias | h² | 2h |
| FFN linear_1, 2h→h with bias | 2h²+h | 3h |
| FFN linear_2, h→h with bias | h²+h | 2h |
| Block affine LayerNorm | 2h | 0 |
| **One module** | **5h²+5h** | **9h** |

The saved PyG constructor has mean aggregation, project=False, normalize=False and root_weight=True, with precisely that left/root bias convention. Existing FactorLinear gives M private input/output vectors around each matrix, without a private bias. Thus

`native_shared = D*h + 5L*h² + (3+5L)*h + h*C + C`

`native_private = M*(D+(2+9L)*h+C)`.

For D=300/C=10/h=128/M=4, shared2 has 205,194 shared and 11,480 private parameters, total **216,674**, matching the complete saved F report. An added native block costs 82,560 shared + 4,608 private = **87,168**. Shared3 has **303,842** parameters.

The path maps add `5h²+2h=82,176` shared and `7Mh=3,584` private parameters, total **85,760**. Shared2+path therefore has **302,434**. Shared3 is only **1,408** parameters (about 0.47%) larger and has radius three. This is a decisive capacity/reach control, not a proof of equivalent function classes or compute.

A full same-path factorized M1 has **291,136** parameters; four disjoint such bodies have **1,164,544**. Their complete inputs and own optimization matter more than matching the shared bank's smaller storage. The formula and AST checks instantiate no numerical model.

## Full TRAIN call computes the full graph

At N=11,701/h=128/P=8, this source constructs 93,608 path encodings per route and 374,432 per shared4 call. It computes the whole factual graph even when only the 580 permitted TRAIN labels supervise the result. Selecting TRAIN IDs afterward does not reduce the current dense forward or saved path tape. It adds no encoder pass: the native nonlinear stem is reused.

| Added path work per shared4 full-N call | Integer scope |
| --- | ---: |
| Forward dense MACs | 25,305,612,288 |
| Approximate TRAIN forward plus two dense backward products | 75,916,836,864 |
| TRAIN plus one full SELECT forward | 101,222,449,152 |
| Common int64 [N,8,4] indices | 2,995,456 bytes |
| Common Boolean [N,8] mask | 93,608 bytes |
| Materialized FP32 tuple features per route | 191,709,184 bytes |
| First-map output per route | 47,927,296 bytes |
| One native-width node field per route | 5,990,912 bytes |

Backward MACs are a planning approximation, not counted execution. Gathers, factor multiplication, GELU, mean, native propagation, mask checks, optimizer, selections and transfers are additional. Marginal mode pays the same gather but one first-map row per node instead of eight; it cannot be described as matched compute.

For autograd planning, the first FactorLinear may retain both original gathered input and privately scaled input, plus its linear output and GELU input. The output map adds node-width fields. This is roughly **0.5 GB of saved path tensors per route**, or about **2 GB for four simultaneously retained route tapes**, with temporary activations, allocator/GEMM workspaces and native tapes extra. Sequential route backward can have a different peak; this packet's existing wrapper forward builds all route graphs and owns no backward scheduling. The recorded old native F allocated peak was 1,164,760,064 bytes. Do not add that historical maximum to the shape estimate and call the sum a measured current peak. A several-GB full-TRAIN qualifier is plausible; actual peak remains unmeasured.

## Conditional time estimate for qualification and acquisition

One shared4 full TRAIN forward/backward plus one SELECT pass adds about **202.445 GFLOPs** in path dense maps using two FLOPs/MAC. At an assumed effective path-map throughput of 1/5/10 TFLOP/s, this is **0.202/0.0405/0.0202 seconds** of incremental dense work per update-plus-selection pair. These are conditional arithmetic estimates, not GPU throughput observations; gather/scatter, native work, launch overhead and I/O are absent.

The complete closed F acquisition scopes sum 71.676 seconds for 1,558 optimizer steps, about **0.0460 seconds per recorded update scope**, including that procedure's acquisition/selection work. It is a useful historical anchor, not an isolated step measurement or a transferable forecast. Combining that anchor with the conditional dense terms gives an indicative **0.066–0.248 seconds per update-plus-selection**, before additional unmodeled work. For 1,000 such pairs, this is **66–248 seconds per shared4 bank**; no 1,000-update horizon or number of banks is admitted by this illustration.

The M1 path maps have one quarter of the shared4 work; four separately optimized I4 bodies sum to the same map work only at equal completed update/selection counts. Actual stopping histories differ. Their parameter/optimizer inventory and own selectors remain paid. Shared3 has an extra full native sparse SAGE/FFN module rather than the path maps; its time must be measured separately, not inferred from the near-equal count.

Root's actual full-TRAIN qualifier should bind the provider, complete graph, dtype, zero-start equality and RNG/gradient/ownership, then measure forward/backward/SELECT scope and allocated/reserved peak for each primary procedure. No such measurement was performed here. Forward counters provide the eventual actual work ledger; conditional arithmetic cannot substitute for qualification or observed utility.
