# UniMPv2 current Torch reference source — packet v2

10 October 2026. `unimp_v2.py` is a provider-injected model/topology adapter;
`run_reference.py` is a disabled complete WikiCS single-reference runner. Source
preparation used saved pinned code and stdlib only. No Torch/NumPy import, graph
payload, fit, selected metric, partial retrieval outcome, TEST or remote operation
was performed. Numerical and actual full-TRAIN qualification remain root-only.
V2 narrowly repairs the runtime UUID guard/path admission and RNG documentation;
the model file and scientific recipe are byte-identical to sealed v1. See
`CHANGE_NOTE.md`; v1 remains preserved.

## Resolved operation

| Source operation | Explicit implementation |
|---|---|
| Three virtual features | Append three copies of the **original raw factual feature mean**, before nonaffine feature LayerNorm. No labels enter their initial values. |
| Virtual edges | The author's `add_vnode` includes all N+3 sources to each virtual destination, including virtual-to-virtual edges. Then symmetrize, lexicographically deduplicate, add/deduplicate every self-loop. The virtual nodes form a clique. |
| Edge identity | Store canonical source/destination edges plus an explicit destination/source permutation. Attention and every weighted message use the same receiver order. No detached/cached learned attention survives a forward. |
| Label input | Only visible TRAIN labels are embedded Normal(0,1), nonaffine LayerNorm/ReLU transformed and added to normalized feature rows. Label width equals raw-feature width. No query/VALID/TEST labels are inputs. |
| Attention | Q/K/V biased Xavier-uniform maps; score `q_receiver·k_source/sqrt(head_width)`; softmax over incoming edges separately per head. |
| Hidden mask | One uniform draw per edge shared across heads; add `(keep−1)*10000` **before** softmax. This is finite −10000, so an all-masked incoming group still normalizes. The .2 uniform value mean includes all edges. |
| Hidden aggregate | `.8 attention-weighted values + .2 incoming-uniform values`; concatenate three 100-wide heads. |
| Hidden APPNP | Three steps of `D^(-1/2) A D^(-1/2)` using indegrees on the augmented undirected graph. Each step has a different learned scalar gate on `[propagated,h0,propagated−h0]`; combine `(1−gate)*propagated+gate*h0`. No fixed .2 teleport here. |
| Residual | A separate scalar gate on `[skip,out,out−skip]` combines `gate*skip+(1−gate)*out`. Hidden layers then apply nonaffine LayerNorm, ReLU and .3 dropout. |
| Final layer | Four class-wide heads averaged after the same .8/.2 value blend; separate residual gate; no hidden APPNP, LayerNorm or ReLU. |
| Final attention APPNP | Reuse that final layer's **pure head-mean softmax attention**, not its .8/.2 value blend. Ten steps of `.8 P_attention current + .2 h0`, with live gradients through the attention on every step. |
| Input dropout | .1 after feature/label addition; ordinary inverted Torch dropout, disabled for evaluation. |
| Training | Uniform in-place shuffle of all 580 TRAIN positions each epoch; first 377 visible (.65 floor), remaining 203 CE targets. One mean complementary-target CE; no separate native CE or retrieval mixture. At serving expose all 580 TRAIN anchors. |
| Update | Adam lr .001, betas .9/.999, eps 1e−8, **coupled L2** coefficient .0005; not AdamW. Exactly 1,500 epochs, no early stopping. |

The graph example files expose receiver-grouped softmax and a reused attention
tensor but do not independently audit the legacy PGL internal send/recv
permutation. This adapter resolves the intended edge-aligned mathematical
operation explicitly. Legacy edge-order/RNG/bitwise equivalence is unclaimed.

`L.fc([p,h0,p−h0],1)` is represented by one concatenated 3D→1 affine map. Its
three weight blocks and bias have the author's uniform bound `1/sqrt(3D)`.
Residual/skip maps use `1/sqrt(fan_in)`; input/hidden LayerNorm uses eps 1e−5 with
no trainable scale/shift. Current Torch and Paddle random sequences need not
match despite matching initialization distributions.

## Explicit runtime adaptations

The author declares `attn_drop` **int64** but supplies CLI .1. Saved source does
not establish whether legacy Paddle rejects, casts or otherwise accepts it. The
sole policy here is `float_0p1_pre_softmax_explicit_adaptation`, acknowledged at
model construction and root release. It is not identical-runtime certification.
No alternative dropout search is added. Evaluation omits mask draws; the author
feeds −1 to keep every edge. Their evaluation operators agree mathematically,
while RNG consumption differs. The **label episode shuffle** uses an isolated
seeded NumPy RandomState. Hidden pre-softmax edge masks use `torch.rand` on the
model device and share the seeded Torch model RNG with ordinary feature dropout.
They are not NumPy draws or a separately isolated edge-mask stream. Checkpoints
save NumPy episode RNG/order and Torch CPU/CUDA RNG states separately.

Dimensions change from author arxiv feature/label width 128 and classes 40 to
WikiCS width 300 and classes 10. The same raw WikiCS features/edges receive the
author's declared graph transformation. No candidate hidden representation is
substituted. The runner loads only root-bound TRAIN `x,edge_index,ids,y` and
VALID `ids,y`, with 11,701 factual nodes and 5,274 complete VALID rows.

The stock TEST loop is removed. Each epoch scores only complete VALID, selects
the strict first accuracy maximum, stores model/Adam/Torch and mask RNG/order,
then restores and replays the selected predictions after the full horizon.
Replay tolerance 2e−6 is an engineering guard, not legacy parity or utility.
The runner's final complete file requires all three seeds 7301/7403/7507;
there is no retry/resume or survivor subset. Activation recomputation from the
legacy RecomputeOptimizer is not used: this changes memory/runtime, not the
declared forward operator. Actual feasibility remains unmeasured.

## Entry and remaining concrete qualification

Root can load the source adapter without importing Torch and supply its admitted
provider later:

```python
topology = author_topology(torch, factual_x, factual_edge_index)
model = build_model(torch, 300, 10,
                    attention_mask_policy=ATTENTION_MASK_POLICY)
logits = model(topology, visible_train_ids, visible_train_labels)
```

`FREEZE_TEMPLATE.json` and `RELEASE_TEMPLATE.json` are disabled. The runner checks
root release, model/runner/freeze hashes, positive **complete** retrieval decision,
explicit mask adaptation, literal runtime host/GPU/provider and prior numerical
and full-input resource/replay qualification before scientific imports/fitting:

```text
python run_reference.py --freeze ROOT_FREEZE --release ROOT_RELEASE --output FRESH_OUTPUT
```

The runner first verifies hostname `anogena-2-0` and exactly one `nvidia-smi`
UUID, `GPU-44039938-fd82-41d2-fefd-de71514e2fac`, with stdlib only. The freeze
retains the literal `anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru`
destination and port 2222. CUDA visibility, when set, must contain that UUID;
there is no index/name substitution. After imports, CUDA availability/count and
the exact root-qualified Torch provider string are checked. No device-properties
`.uuid` attribute is needed from Torch 2.1.2.

Freeze, release, their bound decision/qualification receipts, TRAIN/VALID NPZ
paths, source hashes and the fresh output are resolved and confined to the
authorized allocation's `experiments_iclr/postsubmission_20260930` directory
before their bytes are opened. Absolute explicit paths are required; symlink
escapes are rejected. The output must match the freeze and cannot reuse an
existing directory or extend either sealed source packet. These checks apply to
explicit study artifacts; ordinary library/runtime caches remain allowed and
no filesystem namespace is altered. Runtime checks are source-only here and
have not been executed locally or remotely.

The meaningful numerical qualification is specific: edge-order/per-head softmax
and final-attention reuse; three independent propagation gates plus residual
gate; vnode clique/raw means; whole-query label removal before **all** propagation;
parameter gradients through final attention and gates; actual one-update
full-TRAIN source loss with VALID truth absent; selected state/RNG replay and
full-input peak/time. It must include this float-mask convention and coupled-L2
Adam. No previous homogeneous-wrapper pass qualifies these new operations.

Scatter softmax needs a current Torch provider with `scatter_reduce_`; hidden
normalized propagation uses sparse COO `mm` with fixed values. Attention value
aggregation materializes edge×head×width tensors. For E augmented edges, each
hidden edge-value tensor alone has E×300 float32 elements; forward/backward have
additional buffers. Three attention aggregations, six hidden propagations and
ten final propagations make 19 logical source operator steps per forward. The
Torch implementation makes two reductions per attention layer (weighted and
uniform), so physical neighbor reductions are 22, plus the three edge-score
operations and dense maps. The full schedule adds
1,500 TRAIN and 1,500 VALID forwards plus selected replay per seed. Root must
measure actual memory/time; no feasibility, speed or competence claim follows
from these source counts.

Disposition: stronger-reference source is prepared for review; execution and
baseline competence are pending. License, exact pinned ancestry and static
verification are included. No generic baseline queue or scientific release is
created by this packet.
