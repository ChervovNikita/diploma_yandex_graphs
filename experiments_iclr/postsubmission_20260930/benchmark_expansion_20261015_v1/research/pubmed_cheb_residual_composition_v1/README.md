# PubMed-Cheb residual block-credit composition

New source implementation under the frozen second-setting protocol. **No numerical lifecycle, dataset/model/tensor execution, GPU, fit, SSH or Git operation was performed here. No 24-fit launcher or scientific admission is supplied.** Root reviews and publishes exact bytes before any engineering execution.

## Fixed setting and three interfaces

The exact source case is native PubMed / PolyFormer-Cheb: width128, K6 (seven tokens), one layer,16 heads, FFN64, multi2, q2, dropout .30000000000000004, attention dropout0, ordinary/attention Adam LR .0005/.0005 and WD .001/.0001. Native operators, NormalizeFeatures, polynomial basis, complete graph, LayerNorm, head and per-parameter source Adam groups/defaults remain unchanged. There is **no BN** and no added normalization state.

Literal source split0 / seed1941488137 has11829 TRAIN labels, balanced [3943,3943,3943], and3943 VALID labels [72,1919,1952]. Class0 is only1.83% of VALID. The frozen complete-VALID accuracy selector remains unchanged: first strict maximum, no checkpoint0, matched minimum2000/patience100/cap10000. Per-class accuracy/NLL and macro metrics remain descriptive; no source split amendment or new selector is introduced. This graph has substantial prior development exposure and is not unused confirmation.

The three method IDs are `pubmed_cheb_uniform_residual_v1`, `pubmed_cheb_structural_residual_v1`, and `pubmed_cheb_permuted_residual_v1`. `descriptor(root, seed, variant, manifest_sha256)` creates only an in-memory original-runner descriptor for frozen seeds101/202/303; there is no queue or fit-launch entry point. `BASE_CASE.json` is copied from the unchanged native smoke case; `BINDING.json` retains actual provider/data/basis/role receipts and analytic accounting. Every own file and24 retained source/metadata files are checked before transform.

## Implementation and reuse

`extension.py` composes the existing portable residual-only conversion, preserving all native Parameter objects and removing the untrained boundary BE banks/hooks/ownership. It uses native `lin1` output [19717,7,128], before attention, with one route's rank8 residual applied to every token: `h + U_m GELU(V_m h)`. The retained isolated FP32 QR seed+901003 produces four disjoint rank8 down subspaces; up matrices start at zero. Initial evaluation therefore matches a fresh native constructor. `lin3` is observer-only with no private output factor/residual.

Four native route forwards use the original NodeSession activation-checkpoint/dropout orchestration. All three variants use the original `step_private_responsibility` two-collection updater at one old state, followed by one original Adam transition. Every native coordinate, including LayerNorm/head, receives unweighted F: mean of four original full-TRAIN mean CE losses. Only residual down/up receive W. Uniform returns the **same F object as W**, avoiding an extra FP64 reduction discrepancy; structured/permuted W uses the retained fixed-FP64 weighted native mean helper, with no weight renormalization or new regularizer. Coupled decay stays in Adam. This block field need not be the gradient of one scalar objective.

The structural table uses actual source-preprocessed incoming edge-record counts at TRAIN IDs, with duplicate multiplicity/existing self records retained and no new edges/normalization. Only TRAIN labels assign within-class fractional tied quartiles, `w=.5+2q`. The permuted arm moves whole four-route vectors once within class using isolated seed+7030031. Uniform still records the structural counterfactual/table for custody and identifiability, but applies weights1 and W=F. Each route retains the exact original class mass and each record has mean weight1. A wholly uniform structural table rejects the contrast; there is no replacement metric or strength search.

Portable nested STATE remains authoritative. A new outer STATE binds method, exact descriptor/source receipts, TRAIN degree/tie/permutation record and exact weights. Restore rejects changed method/table/weights, then invokes the original portable/native restore. The immutable v2 optimizer helper accepts replaced group dictionary objects only after genuine restore, while preserving ordered live Parameter IDs, group boundaries, **every** option and defaults. Outside restore dictionary identities remain guarded. No frozen producer/protocol file was edited.

## Exact analytical overhead

Counts use pinned native Linear/LayerNorm/attention declarations and preset; actual construction still must confirm them.

| Quantity | Coordinates / bytes |
| --- | ---: |
| Native learned parameters |592435 |
| Input residual down/up |8192 =4×2×128×8 |
| Total learned parameters |600627 |
| Removed untrained boundary BE coordinates |3560 |
| Extra FP32 residual parameter bytes |32768 |
| Extra dense Adam two-moment bytes |65536 |
| Total FP32 parameter bytes |2402508 |
| Dense two-moment bytes if all parameters receive gradients |4805016 |
| Fixed FP64 CPU TRAIN weight table |378528 |
| New BN parameters/buffers |0 |

Adam scalar steps, serialization, native basis/data, retained autograd graphs, temporary FP64 weighted losses and activation-checkpoint recomputations are separate. A GPU copy of fixed weights is created for weighted W. Parameter bytes do not establish memory or speed. The first residual down **data gradient** is zero at zero up; original coupled Adam decay can still change down parameters. The qualifier checks gradients, not a false claim of unchanged down parameters.

The8192 residual coordinates are analytically about1.383% of the592435 native parameters. Both counts and the fraction remain pending actual constructor confirmation.

## Prepared bounded engineering qualifier

After root review/staging and the normal outer literal-route hostname/GPU guard, run the repository `.venv` interpreter with this interface:

```text
qualify.py --project-root R --manifest-sha256 ACTUAL_MANIFEST_SHA --variant uniform|structured|permuted --device cpu|cuda:0 --output FRESH_PROJECT_DIRECTORY
```

CPU requires empty `CUDA_VISIBLE_DEVICES`; CUDA requires explicit0/the authorized UUID. Each invocation performs exactly two full-graph candidate Adam transitions (first, then fresh-restored second), two native constructions, three complete VALID calls, and eight reverse collections including four diagnostic collections. The selected role has its actual3943 records and [72,1919,1952] support checked, with no TEST scoring. No selector is called. Initial native parity, F/shared-W/private same-old-state gradients, uniform versus weighted credit, first up/down gradient pattern, fresh-restored optimizer semantics/bit-exact VALID and changed-state rejection are checked. An engineering checkpoint is explicitly unsuitable as scientific initialization.

The qualifier counts attempted/completed top-level native calls, including checkpoint recomputations and derivative checks; its conservative bound is256 attempts. Nonreentrant checkpoint early stopping can produce fewer completed top-level hooks than attempts, so do not call every attempt a complete production forward. Work from recomputation is retained rather than hidden behind four logical views. Actual fit-level epoch or throughput claims are outside this two-update check.

Run variants separately, stop on failure, and preserve RESULT/failure/closure/work receipts. The script fixes Torch and OMP/MKL/OpenBLAS to two CPU threads, checks CPU CUDA visibility/context, and installs a900-second invocation alarm plus SIGTERM cleanup reporting. Root's existing owner should additionally enforce a900-second external timeout with a10-second hard-kill cleanup bound, and actual output/memory admission; no new execution framework is added. Wall time includes preflight hashing. VALID and update-with-diagnostic times are separately recorded. CUDA peaks include both retained engineering constructors, diagnostic gradients and restore overhead; they are conservative engineering peaks, not isolated production residency or pair-resource qualification. A separate isolated production timing measurement remains needed for a full-fit ETA. CPU/GPU actual checks are **pending**, with no engineering or scientific launch authorized by this package.

## Source checks and remaining work

`python3 -B check_source.py` runs nine stdlib/JSON/mock checks: source pins, exact preset/descriptor/original-runner schema, analytic parameter counts, exhaustive private ownership, optimizer option/default/dictionary restore semantics, exact fractional degree ties/permutation, exact uniform-F identity, provider support and no eager tensor imports/launcher. These checks construct no model or tensor and do not read data/checkpoint payloads. SOURCE_CHECKS records their actual scope; they are not numerical qualification.

After review, root may separately authorize the prepared engineering checks and assess actual full-graph CPU/CUDA resource costs. A selected-node route collector and complete24-fit scientific launcher/admission remain **unimplemented here**. Future scientific admission must recheck the BINDING data/basis/role pins before constructing models, retain all24 trajectories/18 bank results under the frozen protocol, preserve per-class minority harms and expose all signed CE/accuracy/cost results. The immutable MolBACE composition and running pilot are unaffected.
