# Cost and failure custody

The registry declares6cold qualifications,6common warm trajectories,30initializations and30continuations. Shared acquisition/qualification/warm costs are counted once for the study; each arm additionally discloses the full standalone acquisition/qualification/warm cost plus its own initialization/continuation/serving costs. A shared warm model is not treated as free. No matched-compute or cheapness claim is supplied.

Every scientific phase synchronizes the admitted CUDA device around timed operations and records elapsed seconds, peak allocated/reserved bytes, and process maximum RSS with its process-lifetime scope. The terminal additionally records phase wall time including admission/source hashing, runtime imports, graph I/O, copies, checkpoint/logit saves, trace output and final freeze hashing. Its last small terminal-JSON write is excluded; retain the root outer process wall receipt too. Costs from failed measured operations are written in `finally`, including completed suboperations and partial method/epoch locals.

Warm costs include all50PolyFormer updates or all200local+50global Photo updates, every validation pass, best-local model/Adam copies, selected local restore, all native parameters/biases and inactive state. Native attention/message-passing trajectories are unchanged. PolyFormer materializes all13polynomial token tensors. No persistent preprocessing cache bypasses these costs.

Initialization costs include restoring the donor, disposable native/K4 Adam audit copies, K1 and K4 clones, detached functional shared/buffer snapshots, complete source-gradient qualification, normalized S/permutation where used, every VJP/JVP/SpMM, every Armijo trial, installation and actual/reference forwarding. The unchanged method counters are upper-bound operation accounting and are reported alongside actual elapsed/peak receipts, never substituted for timings.

| Operation beyond warm | Graph/random/permuted | Common descent | Unchanged warm copy |
|---|---:|---:|---:|
| Initializer VJPs | Up to5 |1 |0 |
| Initializer JVPs | Up to4 |0 |0 |
| Residual SpMMs |3 |0 |0 |
| Candidate route forwards | Up to24 |0 |0 |
| Common/fallback route forwards | Up to24 | Up to24 |0 |
| Exact-warm AD and Adam checks | Required and costed | Required and costed | Required and costed |
| Installed actual/reference forwards | Required and costed | Required and costed | Required and costed |

Common descent receives an empty sparse metadata S because the sealed function signature requires S; it performs no graph normalization/permutation/band/JVP work. Warm copy bypasses the initializer completely and preserves theta0 exactly. Random tangents require the graph-derived tangent norm, so their norm-estimation graph work is charged. The topology permutation preserves spectrum and edge/degree inventory while features/labels/train IDs remain fixed.

Continuation costs include every actual update/validation pass, all copied best checkpoints/Adam history, epoch0 selection, absolute native-stage midpoint output when reached (PolyFormer continuation950/native1000; Photo continuation450/global500/paid-total700), final selected restore and output. The K4 model performs four full native graph/attention trajectories in member order. Shared parameter bytes do not imply fewer than4graph trajectories. Unique parameter and aliased state-dict bytes are both disclosed.

All output directories are new and prospectively named. An exclusive registry claim prevents retries in a different directory. Dependencies must match the current registry's completed terminal for their exact phase/arm; another registry's same-context warm state cannot substitute. Separate immutable lineage authorization binds the exact source/cohort/anchor and complete predecessor list before registration. Failure records retain exception/traceback, partial bounded method attempts, partial update/stage counters, elapsed/peak measurements and no-retry flags. A failure blocks the complete cohort and final read for that study; later engineering repairs require a new declared source/study version with prior attempts retained. Finite common/unchanged fallback is kept as a result, never removed or replaced.

The comparison requires exactly30source-selected continuation freezes and all72successful registered phase terminals. It cannot silently omit a seed, select one lucky graph, swap a warm checkpoint, broaden configs or substitute a cold baseline for a matched arm. Comparison output is canonically `registry_anchor/comparison` and final output `registry_anchor/report`. One exclusive SOURCE_COMPARISON_CLAIM and one FINAL_REPORT_CLAIM live at the registry anchor. A different comparison path/SHA cannot reserve another final read. Failure after either claim retains it and blocks an unrecorded retry.
