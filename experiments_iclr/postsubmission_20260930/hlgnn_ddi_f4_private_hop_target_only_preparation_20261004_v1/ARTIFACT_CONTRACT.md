# Later TRAIN/VALID contract

This is a proposed input interface. No artifact was produced, opened, or qualified by this packet.

The owner must supply one tensor/primitives `.pt` artifact with **exactly** these top-level keys. It must have no TEST split, TEST candidates, TEST labels, saved predictions, model parameters, optimizer state, or pretrained embeddings. The adapter uses `torch.load(..., map_location="cpu", weights_only=True)` only for this artifact and rejects extra top-level or split keys.

| Key | Required value |
|---|---|
| `schema` | String `hlgnn-ddi-train-valid-v1` |
| `num_nodes` | Positive Python integer; same node indexing throughout |
| `graph_edge_index` | CPU `torch.int64`, shape `[2, E]`; the native OGB DDI TRAIN graph, including its native reverse orientations |
| `graph_edge_weight` | `None` for an unweighted native graph, or aligned finite CPU tensor `[E]` if the owning native graph actually has weights |
| `train` | Dictionary containing `edge`: CPU `torch.int64` `[N, 2]`; optional `weight`: native aligned finite CPU tensor `[N]` |
| `valid` | Dictionary containing exactly `edge`: CPU `torch.int64` `[V, 2]` and `edge_neg`: CPU `torch.int64` `[M, 2]` |

TRAIN `weight` must be present **if and only if** the owning split actually supplies it. Preserve its values, dtype and alignment. Do not manufacture ones, reinterpret graph weights as loss margins, or use degree counts as margins. Absence executes the pinned AUC fallback; presence executes the pinned WeightedHingeAUC branch. The declared `train_weight_present` is checked against the artifact.

The graph must contain the training targets and be derived only from the native TRAIN graph. Keep the native target inclusion, self-loop treatment and edge weighting. Do not insert VALID edges, remove targets per batch, normalize the graph in the artifact, or augment it. The adapter retains the author's `ToSparseTensor` conversion and reconstructs `edge_index` from the sparse adjacency for the native global sampler. HLGNN's forward performs its own `gcn_norm` and propagation.

VALID candidates must be the fixed native DDI split and use the author's global negative pool for OGB Hits@20, Hits@50 and Hits@100. Preserve candidate order and membership; no candidate resampling, graph insertion, per-positive regrouping, or alternate rank metric. Node IDs in this DDI interface must be in `[0, num_nodes)`; the native mean embedding for an unseen index is preserved in the model but no unseen nodes are admitted by this contract.

Before release, the owner must record node mapping and native split provenance, confirm graph equality to the native TRAIN graph and absence of held split edges in the graph, confirm graph weight semantics and TRAIN weight presence, and confirm the fixed VALID negative protocol. The loader checks structure, dtypes, ranges, finite aligned weights, and declared SHA256; those checks alone do not establish these provenance claims. A qualification report is a required declaration, not a qualification performed here.

`config.json` must point to that artifact and SHA256, declare actual TRAIN weight presence, identify the qualification report, and mark the TRAIN/VALID, runtime and full-budget gates only after the corresponding work is complete. TEST remains unavailable to the adapter after release. Separate held-TEST access is outside this source preparation and would require its own protocol.
