# Independent scientific audit: complete Squirrel development study

Date: 2026-10-03  
Verdict: **The complete study corresponds to the reviewed source and correctly satisfies the frozen Photo development trigger. Its observed advantage is very small and insufficient to establish practical or general predictive superiority.**

## Completeness, source, and selection

Audited all 17 fetched original terminal/source-bound files and all 12 continuation traces against transfer hashes. `STAGE1.json` SHA-256 is `0bd82d13fd29765658b2c6059a12a881ed7b4d9c7b9b4a6db5ebc968ef8b32aa`. Its admitted manifest is the independently reviewed v2 `3f2d8faa81039fd6e5ceb5fecf7ea4b46bf725b516dbb040827926a92ddde452`; the fetched frozen metadata matches the reviewed local copy. Launch, root release, staged mode, and all six donor identities agree with the immutable root choice `ab9d154ca62f033bfdc99a20af2789a1b802b5a3b1131a5a70ba47dec90a9113`.

All three paired seed/split blocks `(17,0)`, `(29,1)`, `(43,2)` completed, with **12/12 selected fits and zero failed/deferred arms**. Reported native identity, active AD, disposable frozen-factor Adam equivalence, joint construction, and every fresh-warm/installed-arm comparison passed. Observed qualification closures total 186 completed, zero failed. All 48 declared original fingerprints and three admitted validation fingerprints were reported preserved. No successful subset was selected.

The configuration is native PolyFormer-Mono cfg0, K4, mean member CE, mean-raw-logit predictor NLL, cap1950/patience250, and epoch0-eligible earliest strict selection. Every trace has the full ordered finite epoch sequence; each recorded selection equals its independently recomputed earliest minimum. All four arms select epoch **4** in seed17 and epoch **1** in seeds29/43, corresponding to native update54/51. They stop at continuation254/251, exactly 250 updates after selection. No prescribed midpoint is reached. Thus selected predictions arise after only one or four continuation updates, with no later validation minimum in the observed stopping window.

Saved checkpoint/logit arrays were not opened or replayed by this audit. The successful source path requires NLL replay agreement within `1e-6`; this audit verifies source/receipt/trace correspondence, without independently re-evaluating the serialized model. No original labels, TEST payload, remote call, or native/model computation was used. Original server artifacts remain the source of record.

## Recomputed results

Full-node's mean validation NLL is **1.364794214567**. Control means are remasked **1.364821036657**, common **1.364847262700**, and permuted **1.364824056625**. Full minus control is negative for all three means, so the exact frozen strict-mean trigger is **true**.

The table uses **micro-nats** (`1e-6` nats); negative favors full-node.

| Control | Seed17 delta | Seed29 delta | Seed43 delta | Mean delta | Blocks favoring full | Paired sample SD | Nominal 95% paired t interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_remasked | -0.954 | -48.399 | -31.114 | -26.822 | 3/3 | 24.012 | [-86.471, +32.827] |
| common_only | -164.866 | +41.008 | -35.286 | -53.048 | 2/3 | 104.080 | [-311.598, +205.501] |
| full_node_permuted | -66.400 | +24.438 | -47.565 | -29.842 | 2/3 | 47.942 | [-148.936, +89.252] |

All three intervals include zero. These intervals use only three paired effects and an independent-normal-block approximation with two degrees of freedom; **they are not calibrated confidence statements for this design**. Blocks share one graph, pair different seeds with different splits, can overlap in roles, and use source validation for checkpoint selection. They do not identify separate seed/split effects or independent-node significance. Multiple correlated control comparisons and conditional followup further limit inference.

## Practical size and sensitivity

- Mean NLL reductions are only **0.00197%**, **0.00389%**, and **0.00219%** versus remasked, common, and permuted controls. The corresponding geometric true-class probability ratios are approximately **1.0000268**, **1.0000530**, and **1.0000298**. Accuracy, calibration, or heldout benefit cannot be inferred from these small average differences.
- Seed17's primary full-versus-remasked margin is `9.54e-7` nats, comparable to the pinned replay tolerance. The overall mean advantage is larger, but this block's primary contrast is almost tied.
- Omitting seed17 changes full-minus-common mean to **+2.861 micro-nats**, making the descriptive leave-one-block-out trigger false. Omitting seed29 or seed43 leaves it true. The original frozen trigger remains true; this calculation describes its sensitivity.
- The complete study took **2,941.54 s (49.03 min)**. This receipt provides no speed advantage claim. Three small development contrasts and early selected states supply weak evidence for a useful full-support or topology mechanism.

## Photo status and claim boundary

The six-donor/mode freeze and original Photo decisions are unchanged: Polynormer-r cfg0 final-global predictor, three matched seed/split blocks, four paired arms, 950 global updates, fixed midpoint450, mean member CE, earliest strict predictor-NLL selection, and the same pooling/transport/RNG policy. **No sealed executable paired Photo driver exists in the supplied/local evidence at audit time.** The Squirrel driver explicitly does not launch it. Photo preparation and exact-global native qualification remain necessary under the existing frozen plan; the tiny positive trigger does not qualify Photo's model or permit tuning its decisions.

Final labels remain closed. The defined confirmation still requires all six attempts and all24 successful frozen selected fits before the separate once-only six-pack release, scoring every arm. This completed Squirrel development screen establishes its arithmetic trigger and engineering correspondence; practical utility, broad superiority, generalization, and novelty remain unsupported.

`AUDIT_RESULTS.json` preserves all twelve values, paired calculations, trace-selection evidence, and all 29 supplied file hashes. `audit_completed.py` reproduces the receipt arithmetic from those inputs. The findings retain the unfavorable blocks and uncertainty without changing the frozen gate.
