# Independent review of completed quality-v3 real qualification

**Finding:** no concrete evidence-chain blocker identified. The completed run supplies bounded train-only engineering paths for every arm on both complete graphs. It establishes neither predictive quality nor the full scientific/resource admission.

The local stdlib review passed 21 grouped checks over 148 source/metadata files. It verified source manifest `607c487f83696a21d5792c3f22b408491f9fb55c8f7e0b1dd1527b8f68653dcc`, request `7fba61788f913fc029b505b98ab3f5b6dfcfe5e6ca1666945ee27797b7d41199`, all 27 real artifact files, their exact variant contents, runtime/GPU/import provenance, source-bound inner/outer supervision, the same-source/runtime synthetic 540-pass chain, and full-input acquisition/prepared/fingerprint metadata links. Whole outer wall time is **156.969560489 seconds**; inner wall time is 156.648605913 seconds. No Torch/scientific-package import, dataset/label/CSR/checkpoint payload read, remote operation or source change occurred in this review. Export/CSR bytes were checked by the sealed remote loader; this auditor checked their mutually bound metadata without reopening those payloads.

## Paths and scope

- **24 paths:** 21 qualified and three infeasible. Reddit2 qualifies all 12 variants. Products excludes exact/shared/detached packed execution without checkpointing because reserved memory exceeds 60 GiB. The excluded rows remain present, ineligible for selection, with no smaller-graph fallback. At least one feasible path exists for each arm.
- Each qualified variant performed six fresh seed-17 engineering optimizer updates: cold update 1, two warmups, then three samples. The complete-input equivalence check compares fresh first-step gradients within the same arm across feasible execution/checkpoint variants, at absolute/relative tolerance `1e-5`. Maximum reported absolute error is `3.583729267e-6`. The separately bound synthetic 540 checks provide additional small-graph numerical evidence. This does not compare different scientific arms for equality, replay 300-update trajectories, or establish seed-wide feasibility.
- The request, observed runtime and result agree: sole authorized A100 80GB UUID, Torch `2.1.2+cu118`, PyG `2.7.0`, NumPy `1.26.4`, four CPU threads, float32, AMP/TF32 false, deterministic algorithms false, CUBLAS `:4096:8`. The exact generated SAGE hash and seven installed generator/template receipts match policy. This verifies completed-runtime provenance; it is not sandboxing.
- No validation targets or quality scores were loaded, and no weights/logits were retained. The validation timing is a pooled-forward plus label-free GPU reduction proxy, not the scientific member/pooled CPU metric pipeline. Peak RSS is a lifetime high-water mark of about 4.55 GiB for this short run; it does not qualify future checkpoint/logit host copies.

## Recomputed engineering choices and speed

Selection is correctly recomputed from each variant's **median of three paired `train + label-free validation / 5` samples**, then variant ID. The fastest feasible exact path is **serial without checkpointing on both graphs**.

| Task | Fastest exact, seconds | Selected shared, seconds | Shared cost versus exact | Selected detached, seconds | Detached cost versus exact |
|---|---:|---:|---:|---:|---:|
| Reddit2 | 0.227952 | 0.230389 | +1.07% | 0.218319 | −4.23% |
| ogbn-products | 1.393275 | 2.056424 | +47.60% | 1.968501 | +41.29% |

Shared/detached use packed execution, without checkpointing on Reddit2 and with checkpointing on products. Both singles select packed execution without checkpointing. Selected shared reserved peaks are **8.119 versus exact 2.813 GiB** on Reddit2 and **54.025 versus exact 21.152 GiB** on products.

Shared is about 1.156×/1.149× faster than the fastest feasible *packed* exact path on the weighted proxy, but that comparison omits the faster serial exact path. Against fastest feasible exact, shared shows no weighted-cost reduction in these observations. Its label-free validation component is 1.707×/1.561× faster than selected serial exact; that supports a narrowly described proxy improvement, not a trained deployment latency or full-study speed claim. Three samples at one seed in one process provide descriptive engineering measurements; the roughly 1% Reddit2 difference is not evidence of a reliable timing separation.

## Remaining admission and forecast interface

1. Prospectively freeze feasible training/validation/deployment paths and checkpoint choices in the scientific/preflight binding. Every referenced choice must match a qualified complete-training variant under the admission source; outcome-independent engineering selection does not authorize scientific fitting.
2. Obtain all ten separately authorized cost-preflight rows, schema `gnnm-shared-message-bounded-cost-preflight-v2`. Input rows require task, root arm, report `{path,sha256}`, request reference, execution policy, GPU UUID, supervisor directory and outer START reference. The source uses one warmup and three measured updates per child, at most 900 seconds, 1 GiB transient file and 60 GiB reserved memory.
3. `resource_forecast.forecast(rows, fixed_costs)` returns schema **`gnnm-shared-message-expanded-resource-estimate-v2`**. Exact fixed-cost keys are `validation_export`, `prepared_input_and_operator_work`, `source_protocol_runtime_audits`, `supervisor_startup_manifest_json_overhead`, and `prior_GPU_wall_seconds`. Use conservatively measured/bounded values. Charge four fit-artifact byte-hash passes; separately charge all **45,000 trace JSON reads/parses, 9,000 writes, 150 FIT_END reads/parses and 30 FIT_END full-trace writes**, along with other source/import/audit/supervisor/scalar-transfer costs and every failed/successful prior job.
4. Independently qualify the CPU proxy, numerical replay/I/O overhead, persistent artifact/host-memory envelope and complete time/disk ledger. Admission requires 30 fits/9,000 updates, exactly the 12 cost-component names in predictive-v2 `contract.binding`, forecast at most 16,200 seconds, reserved memory at most 60 GiB and added disk at most 200 GiB. Forecast naming alone does not satisfy this gate.
5. Complete source/protocol/runtime/input/parameter-control and validation-export source gates, then issue an explicit source-bound root scientific admission. Only then perform separately authorized validation exports, freeze their completed receipts, and execute every fixed-300 fit/replay and complete family lock. The current engineering result grants none of these permissions.
