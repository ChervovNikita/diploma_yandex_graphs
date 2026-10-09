# IMDB development schema and native SeHGNN preparation

The pinned public source supports a representative IMDB development experiment with full transductive support, explicit TRAIN/VALID roles and an unopened TEST member. `role_loader.py`, `build_roles.py` and `qualify_schema.py` are ready for source review. Their public entries default inactive. They introduce no model, fit, monitoring or orchestration.

## Public archive and actual acquisition status

The canonical HGB README at commit `ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe` links the [complete NC release folder](https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing), announced on 2 March 2023. Its public listing identifies **IMDB.zip**, file ID **18qXmmwKJBrEJxVQaYwKTL3Ny3fPqJeJ2**, with **2,102,389 compressed bytes**. The [public download route](https://drive.google.com/uc?export=download&id=18qXmmwKJBrEJxVQaYwKTL3Ny3fPqJeJ2) and folder metadata are preserved in `ARCHIVE_ROUTE.json`. The legacy Tsinghua URL returned HTML headers and cannot identify the archive.

Root reports successful acquisition with SHA256 `dc98438c28f738ab1e7ba07aaeca4fc2e035d837b7bb3955fdd4f61f14dd81e1`, followed by streaming extraction of five development/schema members into the normal allocation repository. The initial extraction failure at an arbitrary 128 MiB limit is retained. **node.dat is 185,065,301 bytes**. TEST remained unopened; this agent accessed no actual archive member or data outcome. These trusted parent facts are preserved separately from public-source observations.

## Source schema and relation directions

The actual SeHGNN loader uses three development inputs:

| File | Fields and role |
| --- | --- |
| node.dat | Global ID, name, type ID and optional comma-separated dense attributes. Zero-based IDs occupy contiguous blocks in type order. |
| link.dat | Raw head ID, raw tail ID, relation ID and scalar weight. Native IMDB constructs binary support and row-normalizes each relation. |
| label.dat | Development movie ID, name, type ID 0 and comma-separated positive label IDs. TRAIN and VALID are explicit partitions of this pool. |

`label.dat.test` couples official TEST membership with values. The safe reader has no interface for that member and never lists, stats, hashes, opens or parses it. All movie IDs outside development remain **unassigned**. Full transductive feature/support processing and development fitting can proceed; final official TEST membership and scoring wait for a frozen confirmation stage.

The source expectations are M/movie **4,932 × 3,489**, D/director **2,393 × 3,341**, A/actor **6,124 × 3,341**, and K/keyword **7,971** with identity features. Type offsets are 0, 4,932, 7,325 and 13,449. Published output dimension is **five Bernoulli labels**. The validator checks these source expectations against the actual inputs; the packet itself supplies no observed class/feature counts.

The six raw semantic blocks, in native first-appearance order, are **MD, DM, MA, AM, MK, KM**. Numeric raw relation IDs must come from the file; none are invented here. A raw row `(head,tail)` becomes CSR `(row=head,column=tail)`. SeHGNN interprets column as message source and row as destination: MD therefore performs **D → M**, while DM performs **M → D**. The same rule applies to actor and keyword pairs. The validator checks complete exact reverse supports, all six real families and one unique director per movie. It records raw duplicate counts and reproduces native coalesced binary support. Non-unit weights trigger a declared scope failure rather than silent conversion.

## Fixed roles and schema qualification

`build_roles.py` reads development IDs while leaving label values opaque to partition choice. It uses a fresh `NumPy RandomState(seed)`, a sorted int64 ID array, one shuffle, the first `floor(0.2*N)` for VALID and the remainder for TRAIN. This matches the native seeded global shuffle practice. Seeds **1–10** are frozen once; all descriptors bind the node, link and development-label hashes. Changing development values can change provenance hashes, but cannot change role choices.

After publication and review, root fills `SCHEMA_QUALIFICATION_RELEASE_TEMPLATE_DISABLED.json` with actual extraction hashes, this final source seal and a fresh normal output. Run the existing repository interpreter with:

```text
<existing-python> -B <source>/qualify_schema.py --qualify --release <enabled-schema-release> --input-root <development_inputs> --output <fresh-schema-output>
```

This one entry freezes the ten descriptors, streams the full node/link/development data once into reusable buffers, validates every split, observes one exact integer partition against native global NumPy seed/shuffle, and writes `SCHEMA_REPORT.json` plus all role files. It records counts, widths, relation IDs/directions, permitted hashes, TRAIN/VALID positive counts, CPU time and process RSS. It never imports Torch, constructs a model, scores TEST or starts a fit. Failures and partial role files remain retained.

The loader's buffers contain only explicit TRAIN/VALID targets. Its separate label-propagation source places TRAIN values at TRAIN rows; zeros elsewhere mean absent source labels and are never supplied as target truth. Keyword identity remains implicit in the schema loader. A native SeHGNN integration must allocate the original identity features and preserve its complete architecture.

## Memory and competent reference

The streaming parser does not load the 185 MB text into memory or impose a 128 MiB member cap. Source shapes predict **182,652,180 bytes** of dense float32 attributes, approximately 174.2 MiB, plus topology and parser overhead. Hashes use 1 MiB streaming chunks. Ten role freezes share one before/after identity check; the real schema is loaded once across all ten splits.

The native reference has a larger footprint. Keyword identity adds approximately 242.4 MiB. Twenty-five full-movie feature channels occupy approximately **1.91 GiB**; raw and cloned feature caches can occupy approximately **3.82 GiB** before label-metapath products and framework intermediates. The literal 512-wide model with 25 feature plus 12 label channels is source-predicted at **83,659,532 parameters**. These are symbolic counts, with unmeasured allocator and wall-time costs. A full-data qualifier should allow realistic CPU memory, such as 32 GiB if available, and measure the native 24 GiB-or-larger GPU requirement before the scientific fit.

The inspected author recipe is 200 maximum epochs, width/embed 512, four feature and label hops, two feature projection layers, four task layers, dropout 0.5, input dropout 0, Adam LR 0.001/decay 0, batch 10,000 and CUDA TRAIN AMP when qualified. Selection is strict minimum full VALID BCE, earliest tied minimum; the literal patience test allows 51 nonimproving epochs. `--residual` is false, while **internal task-MLP residuals remain active** and the final five-logit LayerNorm remains nonaffine. Preserve both. `NATIVE_DRIVER_REQUIRED_CHANGES.md` gives the precise TRAIN/VALID-only driver seams.

## Meaningful graph evidence sources

Actor, director and keyword families are actual typed relations in the native source, with explicit reverse blocks checked by the validator. Movie attribute input M and mediated contexts **MAM, MDM and MKM** supply distinct conditional information paths. The four-hop reference has 25 feature paths and 12 returning movie-label paths, enumerated in `SYMBOLIC_BACKBONE_BUDGET.json`.

These paths support a concrete specialist question: do members depend differently on cast, director or keyword context while maintaining strong individual and pooled genre predictions? Shared dense attributes, TRAIN-label propagation and a capable semantic-fusion model can explain gains without useful private geometry. Any later source-response assay should use a common, prospectively fixed intervention and complete classifier responses alongside original-input quality. Changing a relation or a nominal keyword feature is a sensitivity test; it does not establish a label-preserving causal intervention. This packet implements no such assay and grants no specialization, novelty or universal-adapter claim.

## Verification and evidence limits

Synthetic fixtures exercise role isolation, TRAIN-only propagation labels, label-independent partition choices, exact reverse-pair checks and alias rejection. A protected poison TEST file is never opened or hashed. They supply no real-dataset, native NumPy-output, model or backbone qualification. Source hashes, exact read ranges, public metadata and unsuccessful source-document lookups are retained. Previously used and reserved datasets remain recorded in the earlier inventory; current development acquisition preserves a separately frozen TEST opportunity and does not certify an unused corpus.
