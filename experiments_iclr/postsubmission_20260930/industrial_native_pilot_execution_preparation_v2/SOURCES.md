# Source and runtime attribution

The native baseline is GraphPFN's paper implementation at pinned commit [`3b9b115490249cc777227c846babfb55f35bd8c4`](https://github.com/yandex-research/graphpfn/tree/3b9b115490249cc777227c846babfb55f35bd8c4), using GraphLand's released Tolokers2 RL task. This preparation retrieved no new primary papers and does not reproduce a published score.

| Source | Scope in v2 |
| --- | --- |
| `paper/bin/base/gnn.py` | Exact `Model` class, two-logit CE/native AP training recipe; saved followup source remains unchanged. |
| `paper/lib/graph/deep.py` | Exact native custom GAT, activation, FFN, residual/pre-LayerNorm backbone and module factory segments from sealed v1. |
| `paper/lib/graph/data.py`, `paper/lib/graph/util.py` | Exact transforms/enum; public-only loader oracle with explicitly declared target-removal/public-node-count adaptation. |
| `paper/lib/deep.py` | Exact parameter-group factory; zero-decay predicate restricted only after reachable frozen-model module checks. |
| `paper/lib/metrics.py` | Native scipy-softmax/class-1 validation AP source evidence. Worker metrics use compact validation rows only. |
| `paper/pyproject.toml`, `paper/uv.lock` | Published environment metadata; disclosed as distinct from the chosen cu118 compatibility environment. |
| PyTorch v2.4.0 `torch/csrc/cuda/Module.cpp` | CUDA properties source inspection: UUID property is absent, so device identity is bound using mounted NVIDIA minor and proc driver metadata. |
| Official Torch cu118 and DGL Torch2.4/cu118 indexes | cp312 Linux x86_64 wheel existence metadata; no wheel bytes were retrieved by this source author. |

`SOURCE_RETRIEVALS.json`, `CANDIDATE_WHEEL_INDEX_RECEIPTS.json`, `TORCH_SOURCE_RECEIPT.json` and the pinned tree bind saved source/index bytes. `EXTRACTION_LINEAGE.json` gives exact extracted segment hashes. `SOURCE_BINDINGS.json` binds older packets, originals, the archive receipt only, and root's separate runtime capability metadata.

`boundary_adapter.py` is byte-identical to sealed v1's new native adapter, including the exact channel layout and AdamW transport distinct from R17's coupled-Adam rule. `ad_qualifier.py` is byte-identical to v1's native bridge of the borrowed R17 route source. Only common-model binding and gradient qualification are called here; no route initialization, Bernstein bands, controls or continuation training is invoked. R17's own classes/optimizer/scientific protocol remain unchanged. Prior algorithm attribution is retained in R17 and v1; this packet makes no originality claim.

The source-only checker confirms saved file sizes/SHA256 and pinned Git blob identities, exact extracted text, the public-oracle AST after declared removals, closed direct import/mode scope, frozen matrix/policy bytes and sealed predecessor payloads. It cannot certify import compatibility, numerical behavior, GPU memory, native primitive determinism or actual namespace isolation. Those remain root execution gates.
