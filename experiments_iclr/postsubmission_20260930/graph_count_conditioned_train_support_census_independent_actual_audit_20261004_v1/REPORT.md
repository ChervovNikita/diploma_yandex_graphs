# Independent audit of the completed TRAIN support census

## Verdict and evidence scope

**PASS for the collected metadata and integer projection, with 963 local consistency and custody checks.** The census completed one authorized attempt, with no fit, optimizer update, features, VALID/TEST access, retry, signal or unresolved cleanup recorded. This verdict does not admit a scientific fit or establish predictive quality.

The audited candidate source manifest is `e6c7f6bc64e9d3454f2a7881acb16f493ec00b2126de0397b62c7b64a8b064e2`. The original supervisor launch records PID 3219898 and start ticks 1725006651; the physical terminal records owned child PID/session/group 3219899 and start ticks 1725006669. Its observed unreaped exit was normal status zero, followed by direct collection and physical session closure. These are existing receipts, not new process observations. Supervisor `LAUNCH`/`STARTED` payloads are pinned by custody but were not supplied locally; a separate root monitor review owns that process-chain assessment.

All fetched receipt bytes match their recorded hashes. Source/release/plan, physical/terminal, output/final custody and projection-to-terminal links agree. The custody inventory contains exactly 52 source/control/runtime rows: 15 rows were rehashed locally, 36 runtime rows were compared with authority JSON, and the interpreter hash was matched to its authority. The only scientific input descriptors are `split/time/train.pt` and `raw/edge.csv.gz`; this auditor read neither data file.

The three original seed receipts, totaling 16,218,256 bytes, remain on the server. This auditor did not fetch or independently rehash those raw receipts. Instead, the root projector's exact recorded command was reconstructed from the statically reviewed source and original descriptors, checked against its command hash, and matched to a successful transport whose returned JSON equals the saved local projection. That source rehashes each original seed file, recomputes every histogram scalar, validates all joint margins, and then omits the histogram rows. The independent audit recomputes projected joint/marginal accounting and every reported aggregate. It does not claim a second independent histogram recomputation or reconstruction of graph supports from tensors.

## Census coverage

Seeds 0, 1 and 2 each contain all 17 complete native batches in order, both populations and exactly 65,536 queries per population per batch. There are 51 masks, 102 population receipts, 3,342,336 query occurrences per population and 13,369,344 side occurrences in total.

Each stream selects 1,114,112 of the 1,179,052 TRAIN records (94.4922%); the native iterator drops 64,940 records. Thus “complete census” means every complete batch in the declared native schedule, not every record in each seed. Native negative sampling and the iterator run after a census seed without the model/dropout RNG prefix. They are not a replay of a past fit. Repeated masks and seeds share one TRAIN graph, and these counts are occurrences rather than unique examples or independent graphs.

## Support that can train the conditional identity law

Let `r=min(k,n-k)`: `r=0` is a forced pattern, `r=1` a categorical/complement categorical law, and `r>1` a genuine subset law. Both sides must have `r>0` for a shared mixture component to express cross-side association in the conditional identity term.

| Stratum | Positive occurrences | Positive fraction | Negative occurrences | Negative fraction |
|---|---:|---:|---:|---:|
| Either side variable | 1,015,984 | 30.3974% | 158 | 0.004727% |
| Both sides variable | 194,664 | 5.8242% | 1 | 0.0000299% |
| Both variable, at least one genuine subset | 115,762 | 3.4635% | 0 | 0% |
| Both genuine subsets | 45,453 | 1.3599% | 0 | 0% |
| Either side genuine subset | 306,577 | 9.1725% | 1 | 0.0000299% |
| Both categorical/complement categorical | 78,902 | 2.3607% | 1 | 0.0000299% |
| Both forced | 2,326,352 | 69.6026% | 3,342,178 | 99.9953% |

The positive coupling opportunities occur in every batch: 3,644–3,946 queries (5.5603–6.0211%) have two variable sides; 785–1,027 have two genuine subsets. Per-seed coupling counts are 64,971, 64,748 and 64,945. This is recurring support, although those seed counts are not independent replications.

The 9.1725% union of positive genuine subsets is insufficient as a claim about shared genuine coupling. Only 3.4635% of positives have two variable sides and at least one genuine side. Within the coupling-eligible positives, 59.4676% involve a genuine side; the remaining 40.5324% are categorical-product mixture cases. Those categorical cases can still distinguish a shared mixture from separated side mixtures.

With matched component and mixture-weight definitions, shared `J_K` and separated `J_K_sep` coincide on the conditional identity term when either side has a forced pattern. A forced side has conditional probability one and zero identity gradient. Pooling equal positive and negative populations reduces the joint-variable fraction to 2.9121% and the coupled-genuine fraction to 1.7318%; only 9.0566% of all side occurrences are variable. An all-query or all-side average can therefore substantially dilute the conditional identity gradient. Any normalization or stratum weighting must be declared as an objective choice, since it changes the target objective. This observation concerns the identity term: an additional count predictor or link classifier can still learn from zero-count negatives.

## Prospective cost and the remaining implementation risk

| Recorded arithmetic proxy, both sides | Positive | Negative |
|---|---:|---:|
| Total slots, all three streams | 137,641,433 | 52,235,722 |
| Mean slots per batch | 2,698,851.6 | 1,024,229.8 |
| Total `sum(n*r)`, all three streams | 81,314,804 | 4,504 |
| Mean `sum(n*r)` per batch | 1,594,407.9 | 88.3 |
| Mean weighted genuine transition work per batch | 1,075,179.5 | 2.8 |
| Genuine `(n,r)` groups per batch | 1,413–1,605 | 0–1 |
| Unweighted genuine group slot loops `sum(n)` per batch | 121,369–139,724 | 0–71 |

Positive left/right mean support sizes are 25.05 and 16.13; negative sides are 7.81 each. Maximum support size is 373. Positive maximum `r` is 18 on the left and 17 on the right. These maxima are componentwise maxima and need not occur on the same query.

An optimized ESP recurrence can multiply the `sum(n*r)` proxy by the bank size `M`; it must also score slots and prepare ragged groups. A single direct group-by-`(n,r)` implementation still faces roughly 130,083 genuine group-slot iterations per positive batch. The grouping assumes correct complement handling; grouping separately by `(n,k)` may require more dispatches. Small forward rolling-state sizes do not solve Python/kernel dispatch or autograd-retention costs. The largest single projected genuine group has 339 rolling count cells per bank component, but that is not peak training memory: many groups, logits, saved recurrence states, graph/encoder tensors and tensor metadata may coexist.

Across an average positive-plus-negative batch there are 3,723,081.5 slots. One dense float32 logit array alone is about 14.20 MiB times `M` (113.62 MiB for `M=8`, 227.24 MiB for `M=16`). These examples are arithmetic illustrations, not recommendations for `M`, measured allocations or full autograd bounds. Negative supports remain a substantial scoring/packing cost even though their conditional identity signal is nearly absent. An autoregressive implementation that evaluates every existing slot also incurs those slot costs; early forced-decision handling must be specified and verified.

The metadata census itself used 16.73 seconds of worker wall time, 18.77 seconds of observed physical wall time, 1,952,186,368 bytes of sampled owned-session RSS, 354,232,832 allocated CUDA bytes and 463,470,592 reserved CUDA bytes. This qualifies only the declared census. Its physical wall scope excludes later parent collection/hash/write, and RSS was sampled every 0.25 seconds. GPU peaks were cooperative, there was no instantaneous OS RSS ceiling or escape sandbox, and actual default dtype/CUDA/autocast details beyond the original guards were not independently measured. These figures cannot qualify a differentiable model or native training schedule.

## Scientific implication and next evidence

The count geometry supports continuing with a bounded, representative feasibility qualification of the conditional subset idea after source review. Positive batches contain thousands of variable pairs and hundreds of two-sided genuine subsets; an outright “no useful support” conclusion would be wrong. A bank with multiple components has an opportunity to express coupling on a recurring minority of positives. The geometry does not measure association strength, conditional predictability, effective sample size, calibration, improvement over `J_K_sep`, or latent graph completion.

Any subsequent learning claim needs the following assumptions made concrete:

1. Exact slot identity, mask timing, count/complement orientation and teacher membership agree with the native graph operation, including duplicates and endpoint order.
2. TRAIN teacher ones denote observed incidences hidden by the current mask. Teacher zeros denote unobserved TRAIN incidences, not verified latent nonedges. Generalization to missing future/latent links needs its own evidence.
3. Counts used for conditioning are available or predicted at inference under the declared graph. Teacher counts must not silently become unavailable inference features or held-out labels.
4. Mixture components, shared/separate weights and objective scaling are defined consistently. Multiple components alone do not guarantee learned association; the two-variable strata only make it possible.
5. Stable float32 forward/backward computation, ragged packing, complement cases, slot permutations and genuine subset gradients are verified under a bounded native schedule. The census contains no gradient or predictive check.
6. The next resource qualification measures actual differentiated dispatch and retained memory with representative group counts, native positive/negative packing and the complete required update schedule. Passing tiny fabricated CPU fixtures does not close that gap.

No new source launch, numerical workload, server read, model/data-array read, retry or manuscript edit was performed by this auditor. Earlier failures and sealed packets remain preserved.
