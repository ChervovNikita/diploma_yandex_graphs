# Disabled WikiCS live-route block source

This packet contains callable scientific source for one fixed live exchange and its parameter-matched route-separable control. It is unadmitted. No model, numerical library, tensor, initializer, data provider, TRAIN update, restore, GPU or job was executed during preparation. There is no CLI, full runner, owner, launcher or placeholder qualifier.

## Fixed model and forward

Use the exact pinned WikiCS native Polynormer: width512, one head, seven local blocks, two global blocks, pre_ln=false. The fresh facade currently targets original plain `be_unit` F training: four complete private-factor routes, two full stochastic own-label views, original Adam and mean-probability serving. It is a working library facade, not a full-fit driver.

`blocks.py` creates one 32,768-weight block, M4/D512/r16. Exchange uses parameter-free LayerNorm inside its branch, Q/K/V projections, off-diagonal route self-attention and a zero U output map. The matched control uses the same normalization and a shared512→32→512 ReLU residual with zero C. CPU tensors are initialized from one isolated generator; no nn.Linear/default reset consumes native RNG. The isolated seed is870000000+100000*optimizer_seed. Both initial complete deterministic functions equal their native baseline. Zero initial Q/K/V (or B) gradients are expected because the final map starts at zero; only the final map initially receives the task cotangent.

`lockstep_forward.py` implements the smallest synchronization required:

1. For each route, resume its existing RNG stream and select its existing factor rows. Execute native input dropout, stem, dropout and local block0 in exact pinned order. Keep both the raw block0 state and its original first `x_local` term live.
2. Stack the four raw states as[N,4,512] and run the single exchange/control. No detach, frozen teacher, new graph bank or final-only fusion is used.
3. For each route, resume its prefix RNG endpoint and existing member context. Feed that route's changed raw x into h_lins[1], local_convs[1] and lins[1]. Continue all six remaining local blocks with native x_local addition. Call the original global_attn/ln/head when global mode is active, or the original local head otherwise. Slice the original selected ids only after the full graph forward.

The local block helper preserves GAT+parallel affine, ReLU, dropout, sigmoid beta and LayerNorm product mixing in native order. The original global module itself is called without replacing its implementation. Mutable factor/scorer selectors are used sequentially, never concurrently. The exchange is route-slot-specific at the same node; later native GAT propagation transports changed states along unchanged graph edges and recalculates its native attention. It is not an edge receiver gate.

The existing per-member stream consumes the same ordered native dropout calls in prefix then tail when the output map is zero. The new exchange/control draws no forward RNG. This is a source argument; full-device numerical parity and stream endpoints remain for root's admitted check.

## Fresh construction and restore binding

`session_adapter.py` privately clones the exact public Session constructor and inserts block registration inside the sole WikiBackbone wrapper before the existing optimizer factory. Original Adam therefore owns all original body/factor parameters plus the new block; there is no optimizer retrofit. The inherited plain train_step uses the new Session.forward and performs full joint mean-own backpropagation. The original source/module/class is unchanged.

The facade also clones only the original save/restore primitives, changing their schema and adding a descriptor that binds block kind, dimensions, initializer seed, native/public source and new file hashes. It preserves original model/Adam/member-stream/RNG and local/global restoration. Cross-kind or original snapshots are rejected by the new identity; this is training-continuation source, not a newly selected serving checkpoint or full-run selection protocol.

Load the new directory as a Python package in a fresh reviewed process; `adapted_public(original_public, kind)` returns the fresh callable Session class. Its constructor requires explicit task wikics/arm be_unit and the pinned native path. No call is authorized by this README or disabled admission file.

## Material integration changes still needing root review

The old member-at-a-time replay is invalid for a coupled exchange. This facade deliberately uses full joint autograd; native past replay counts, peak memory and runtime qualification cannot be inherited. Both own-view graphs and all four live prefixes/route dependencies can increase memory. If root needs segmented/recomputed joint gradients, that is an additional unreviewed implementation, not an available fallback in this packet.

The forward can enter the matching existing private-local-scorer controller context if a reviewed Session already owns it. The facade does not install that controller, does not alter scorer ownership/initialization and does not implement alphaF/relationJ policies. Reusing the graph-relation constructor/credit permissions or replay would require separate review: existing guards explicitly require the original Session.forward/source and would reject this new forward or unknown block parameters. Do not bypass those guards.

Fresh coupled source/control require their own descriptor/admission binding in any full owner. This packet supplies no borrowed historical costs, launch commands or qualified numerical claims. Native/original files remain byte-bound and unmodified.

## Cost and interpretation

One fixed site adds32768 parameters in either source arm. Matrix arithmetic is131072N main dense MACs plus512N attention MACs for exchange; the control has the same main dense count. Normalization, softmax, storage, backward and coupled-prefix memory are additional. All native graph/global work remains. There is no speedup or near-free claim.

The block is ordinary set self-attention; Cross-stitch/sluice establish continuing stream communication. A flattened structured wider predictor represents the coupled forward. A useful same-task result can support a placement/objective comparison, not a new attention primitive or an expressivity separation. Root's later scientific comparison must retain competent WikiCS M1/genuineI4 and own4, individual mean/worst quality, exact acquisition and served retention, old repairs and outside harms. PubMed's cached-token I4 numbers are motivation only and are not WikiCS references.

STATIC_SOURCE_CHECK.json records AST parse/compile and import/source checks only. ADMISSION_DISABLED.json is false throughout. No scientific quality or runtime was opened or measured.
