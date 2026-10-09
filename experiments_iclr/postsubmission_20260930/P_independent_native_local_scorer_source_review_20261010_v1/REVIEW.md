# Independent native local scorer source review

**Ready for root adoption as a source-only review; no new blocking source defect found.** Reviewed source seal `c69aa86b6682ddbaf97f88d5bbd5d9d1f48a3aa67ef892b19d93d296779f5037`, manifest `ae044bc9f749eeb9941d482c69fe2f7ff9a2d65961f2b2003933f28e3999be7a`, and entry `ae0b4655164db9a18cbc35c4f6cf279cea3d16aa4077ba497f9922ab55099f16`. This verdict does not establish runtime qualification, scientific launch admission, utility, superiority, novelty, or manuscript acceptance.

## Initialization and RNG

`initializer.py` replaces only the four rows of each of the fourteen existing local `att_src`/`att_dst` banks, each bank shaped `[4,1,1,512]`, before the original fresh Adam is constructed. Registration is retained. The uniform support is the native PyG Glorot law `sqrt(6/(heads+channels))` from the final two dimensions, here `1` and `512`; it does not use Torch Xavier fan interpretation.

Each row uses a separate CPU `torch.Generator` with seed `100000*optimizer_seed + 770000000 + 10000*layer + 100*side + member`, where source is side 0 and destination side 1. Across optimizer seeds 6101, 6203, and 6307, the 168 integer identities are distinct, spanning 1380100000–1400760103. Matching optimizer seeds intentionally reuse matching draws across αF and relationJ. There is no search or redraw: equal member-row digests cause failure. A successful constructor declares 56 draws and 28,672 scalar draws.

Explicit generators and `uniform_(..., generator=...)` avoid default-stream seeding. Source guards compare all non-scorer body-state hashes and default CPU/active-device CUDA RNG bytes before and after the reset. Original member dropout streams are subsequently constructed by the unchanged Session. These guards were inspected, but model construction and runtime RNG preservation were not executed or established by this review.

## Original trainer and integration

`integration.py` changes a private copy of original `make_session` at exactly two places: the constructor facade callback and the copied-row start assertion, replaced by an exact initializer-receipt check. The remaining constructor AST matches the original, and its recorded adapted AST hash is `8c551f4b46367105a802946829e4cf4714111257349aa2027a61282650d3b725`. The temporary `verified_hook` replacement is restored in `finally`; the hook loader creates a fresh module per call. This is adequate for the declared sequential construction workflow, without establishing a general concurrent-construction guarantee.

`train.py` delegates to the immutable original trainer. Source inspection of original `make_session` and `run_complete` confirms the original F/J math, β=0.5, disjoint VJP groups from the same old state followed by one Adam step, joint strict selector, local restore, live end-local streams, 1,100/100 epoch horizons, and mean member class probabilities at serving. Only αF and relationJ are admitted. No auxiliary objective, contrastive addition, or teacher is introduced; shared heads, global QK/value parameters, and unit factors retain the original constructor path.

## Selected reconstruction

`reconstruct_selected` checks arm, policy, seed, source/partition/constructor/initializer/integration/AST/stable-start/base-source bindings, a Boolean local/global flag, and matching model configuration. It constructs a fresh exact model, loads the selected state with `strict=True`, restores selected mode/configuration, and enters evaluation mode. Its `train_step` refuses training. It does not resume an optimizer or RNG stream, or perform checkpoint reselection. The stable initializer receipt removes only elapsed time.

## Qualification and disabled cells

The qualifier declares αF/relationJ × native local/global at fresh seed 6101: four complete TRAIN updates on the 11,701-node, 580-label batch, with no VALID forward/metric, TEST, or fabricated numerical fixture. Its source checks fresh Adam, finite recipient/complement gradients, absent inactive global gradients in local mode, live tied-QK pullback in global mode, original dropout-stream bytes under forked RNG, and one Adam after all old-state VJPs. Per case, the expected work is eight shadow and eight replay forwards, two cotangent collections, sixteen reverses, one Adam, and one RNG check. Actual attempt/completion counters are preserved on failure. A qualification receipt must bind the training GPU and exact source. No qualifier ran in this review.

All six αF/relationJ × 6101/6203/6307 releases remain disabled, with authorization/review/qualification flags false and qualification/source-manifest launch bindings null. They retain full 1,100/100 horizons, β=0.5, auxiliary false, original GPU assignments, 32 GiB owned GPU/RSS limits, and existing finite ownership helpers. TEST, automatic retry, comparative opening, and anchor reuse remain disabled. This review adds no owner, launcher, scheduler, or execution authorization.

## Scientific limits and verification

Copied starts are exploratory paired anchors only under exact source, recipe, selector, route, endpoint, terminal, and cost custody; no anchor state initializes a new fit. Historical single/independent4 results have different recipes/selectors and do not establish fresh paired superiority or a sharing mechanism. Native independent scorer initialization is established practice; no graph novelty claim is made. Existing scores, outcomes, failures, and source packets remain unchanged.

Mechanical verification passed for 24 packet payloads, 24 SOURCE_MANIFEST payloads, 13 bound source files, five dependency manifests and 63 dependency payloads. Five Python files were AST-parsed without import or execution. The exact two constructor substitutions, actual old/new qualifier diff, seed identities, six disabled cells, four-case disabled qualifier roster, and original full-task/frozen-screen equality were checked. `VERIFICATION.json` records these checks and their limits.

Navigation correction: this packet has no `NARROW_SOURCE.diff`. Its concrete delta evidence is `BASE_QUALIFIER_DELTA.diff`, `ADAPTED_MAKE_SESSION_AST.json`, and the complete small source files; absence of the alternate filename is not a defect.

No initializer, model, entry, static-check entry, qualifier, numerical provider, fixture, dataset, checkpoint, array, server, or remote runtime was executed or opened for this review. Preserved report/source payloads were bound mechanically where applicable; no numerical outcome interpretation was added. Root can adopt this immutable review as the independent source-review result while retaining all separate qualification and scientific admission gates.
