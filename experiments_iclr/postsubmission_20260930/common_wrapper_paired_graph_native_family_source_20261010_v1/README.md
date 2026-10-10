# Attributed paired graph / LoRA / local pilot

Callable entry: `run_family.py --config <JSON> --output <fresh-directory>`. Required keys are the existing `native_repo`, `common_routes`, `factors`, `train_npz`, `valid_npz`, `device`, three paired `seeds`, and explicit `backbone` (`GAT` or `SAGE`). Root freezes the configuration and criteria. Source preparation used AST/text checks only: no numerical import, execution, payload read or host operation.

The original native trainer, immutable common wrapper, own mean CE, full graph/roles, AdamW law, strict VALID selection, probability reducer and full model/optimizer/stream checkpoint restoration are reused. No owner, launcher, qualifier or automatic retry is added. Other entries and completed outcomes remain unchanged.

## Acquisition roster

Default: nine arms × three seeds = 27 groups / 45 acquisition units per backbone. The five original references are ordinary M1/genuine I4, original all-map factorized M1/genuine I4, and original Rademacher shared4. Four new shared banks are `shared4_coherent`, `shared4_paired_graph`, `shared4_rank1_lora_graph`, and `shared4_paired_local`.

Set `fresh_arms_only: true` for the fixed four-new-arm roster: 12 groups / 12 acquisitions per backbone, 24 across the fixed GAT/SAGE pair. Its completion file declares `full_comparative_family_complete: false`; completion means its acquired roster closed. Acquisition never reads old reference payloads. Root must bind every immutable reference record, source/config/role/seed/owner identity and cost before joining the full nine-arm comparison. No outcome chooses a backbone, subset or seed. Integration supplies no scientific quality evidence.

## Private maps and counts

All correction arms retain learned diagonal r/s factors everywhere, initially one; their input stem has no Rademacher change. The original five references retain their original starts. Corrections extend thin `FactorLinear` subclasses while retaining original W/b/r/s Parameter objects. Additional banks are registered before device movement and AdamW. Existing member context selects their route through the inherited member field; original native forwards are unchanged.

| Backbone | Graph correction sites per block | Local pair site | Extra coordinates across four routes, L2/H128 |
|---|---|---|---:|
| GAT | `conv.lin`, input H | FFN `linear_2`, input H | 2,048 for each graph-pair / graph-LoRA / local-pair |
| SAGE | `conv.lin_l` and `conv.lin_r`, each input H | FFN `linear_1`, input 2H | 4,096 for each graph-pair / graph-LoRA / local-pair |

The helper derives and checks this parameter match from actual map shapes. SAGE neighbor bias and all native biases remain outside output scaling. Rank1 LoRA acts at the graph sites only. Each route/site seed is `family_seed + correction_seed_offset + member_seed_stride*route + 1009*site_ordinal`; default correction offset is 5000081. Graph pair and graph LoRA draw the same incoming unit direction. Directions are normalized independently using local CPU generators, preserving native/dropout RNG streams. Local sites have their own ordered shape-specific draws.

Each pair has separate equal-valued `u1/u2` banks; each LoRA map has unit incoming `lora_a` and zero outgoing `lora_b`. The existing r/s role reporter does not classify these extra banks; `RESULT.json` explicitly records their names, sites, dimensions, seeds and counts. Optimizer ownership checks all registered parameters. State dictionaries restore the full body and additional banks strictly.

## Function, tangent and cost limits

In column notation, pair maps are `D_s W H(u1)H(u2) D_r`; LoRA maps are `D_s(W+b*a^T)D_r`. The native bias is added afterward. At coherent starts, equal normals and zero outgoing LoRA preserve the native function in real arithmetic. These are known orthogonal-adapter/ensemble ingredients, attributed to the scoped OFT/BOFT/Householder/LoRA review in `paired_Householder_native_graph_operator_triage_20261010_v1`; there is no novelty claim or bitwise claim.

At equal unit normal n, the pair tangent is `2[(v1−v2)n^T−n(v1−v2)^T]`: dimension at most d−1 and rank at most two. The pair globally affects one plane, with `rank(Q−I)≤2`; it is not arbitrary SO(d). Cold rank1 LoRA instead starts with an outgoing-vector tangent and zero incoming gradient. For SAGE, local input dimension 2H differs from the two graph H-dimensional sites. Equal raw counts do not equate tangent freedom, effective step size or usefulness. The published plain-SGD first-step formula is not an Adam formula. All arms retain the same nominal AdamW law; no rate grid or displacement matching is added.

Reflection denominators use the raw squared normal norm, differentiated normally. Nonfinite or zero norms fail; there is no clamp, fallback, retraction or hidden normalization optimizer. LoRA vectors, TRAIN loss and evaluated logits must also be finite. Only Q is orthogonal; learned diagonal factors and shared W do not guarantee preserved full-map norms, calibration or competence.

Two reflections use O(Nd) vector work without constructing private dense matrices. LoRA uses its rank1 dot/vector work. Every original full-graph route trajectory still runs; added gradients, Adam moments, checks and storage are paid. Results record corrected-map/reflection/rank1 call counts, parameter/checkpoint bytes, observed acquisition/readout time and memory. The fresh roster has ceilings of 24,000 Adam/backward steps and 96,000 TRAIN plus 96,000 VALID route trajectories across both backbones. Actual early stops/costs are retained. No isolated speed/memory advantage is claimed.

Root performs the necessary representative training/restoration qualification, then decides the finite frozen pilot and post-closure joined readout. Checkpoints, traces and prediction payloads stay on the authorized host; TEST stays closed.
