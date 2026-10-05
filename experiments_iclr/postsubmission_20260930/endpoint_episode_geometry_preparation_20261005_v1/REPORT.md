# TRAIN endpoint episode geometry: prepared source

Status: source preparation only. The three Python files were parsed with stdlib AST and hashed; none was imported or executed. No TRAIN payload, feature payload, heldout input, fit, model, checkpoint, metric, server, or GPU was accessed for this preparation.

## Prospective measurement

The disabled job proposes one TRAIN-only CPU geometry measurement with seed 20261005, outer positive batch 64, 256 inner positives and 256 inner negatives per route, and four routes. These sizes were fixed before geometry outcomes and are not adopted. The native nonself TRAIN population has 3870 rows: one complete outer permutation therefore produces 60 full episodes and a final episode of 30 positives and 30 index-paired negatives. No positives are dropped and no outer query is redrawn.

The input contract pins TRAIN SHA256 `1e97ad3a73ecfeb69489aa6d92d02dc44c2925b4056a8e4cd4b548f1c0ebaa27`, 3918 raw rows, 48 self loops, and 3870 unique undirected nonself rows. Native row order and orientation are preserved after removing self loops. The authorized node population is 3327. Feature authority is the pinned prior receipt for float32 `entity_embedding` of shape [3327, 3703]; this inspection hashes receipt metadata and never opens the feature payload.

`native_cycle` delays NumPy, torch, and PyG imports until called. It runs native PyG `negative_sampling(to_undirected(TRAIN), nodes)` and a full CPU `torch.randperm`, using the independent prospective data seed `(seed + cycle) % 2**32`. It restores Python, NumPy, and CPU torch global RNG state. It seeds the CPU default generator only; CUDA generators are neither seeded nor accessed. This stream is not claimed to reproduce the frozen fit's global draw sequence. The native bank may contain directed/reverse equivalent negative facts; indices and those repetitions are retained, while the bank is checked against the entire TRAIN positive graph.

The driver requires a concrete root review, measurement authorization, an exact sealed source manifest and three module hashes, exact input counts/hash, receipt metadata hashes, pinned runtime versions, expected host, a fresh reviewed output directory, and a confirmed external 360-second hard bound. CPU threads are 2, interop threads 1, CUDA is hidden, soft budget 300 seconds, one attempt. Runtime bindings are torch 2.1.2+cu118, NumPy 1.26.4, and PyG 2.7.0. No model or optimizer is constructed.

## Shared module interface

`episode_geometry.py` uses only stdlib and performs no data load at import.

| API | Contract |
| --- | --- |
| `neighbors(train, nodes)` | Validate unique nonself full TRAIN and return full-node undirected neighbor sets. |
| `validate_negative_bank(train, negative, nodes)` | Reject self/out-of-range/full-TRAIN-positive collisions; preserve equivalent negative duplicates. |
| `route_streams(seed, members, control=False)` | Independent Python Random positive/negative streams for each route; distinct control offset. |
| `outer_batches(order, edges, outer_size)` | Require a complete permutation and include its final partial batch. |
| `endpoint_episode(train, negative, outer_ids, streams, inner_size)` | Outer negatives use the same IDs as positives; endpoint union contains both query classes. Sample per-route inner IDs without replacement only from queries with neither endpoint in that union. |
| `matched_random_episode(train, negative, reference, streams, graph)` | Same outer queries, route count and class counts; exact per-route full-TRAIN pre-mask degree/CN strata. Random inner queries may overlap outer endpoints, but exclude equivalent outer query facts. |
| `outer_only_fallback(episode)` | Geometry proposal for an infeasible draw: original outer IDs/endpoints, no inner queries, `feasible=False`, `fallback_adopted=False`. |
| `mask_indices(*episodes)` | Union of positive target IDs; deduplicate only the support mask. |
| `support(train, nodes, removed)` | Return kept TRAIN IDs and full-node neighbor sets after the specified mask. |
| `describe_episode(...)` | Query/fact/node coverage, aggregate and per-route degree/CN strata before/after masking, outer context, support entries, equivalent-fact and cross-route repeats. |
| `exposure_summary(...)` | Candidate outer/inner index, fact, and node exposure; unseen positives; virtual-plus-recompute query counts; actual optimizer updates remain zero. |

Positive/negative inner draws are without replacement by index within a route. Different routes may share positive IDs or equivalent negative facts. A single capable head may concatenate the same four streams, retaining every repetition; F4 and untied four consume the corresponding stream per member. The training source owns the architectures, objective reductions, parameter partitions, optimizer state, and update convention.

## Mask and control proposal

The endpoint arm's own support masks its outer positive targets and its inner positive targets. To give the random control the same support while removing every supervised positive target, the paired support proposal additionally masks both arms' inner positive targets. Both arms then use the exact same union support and full node population. The driver measures endpoint own-mask support separately so this extra context removal is visible.

Only selected positive targets are masked; other TRAIN edges incident to outer endpoints remain context. No VALID/TEST positive is consulted when constructing support or negatives. Matching uses `(min_degree, max_degree, CN_count)` in the full TRAIN graph before masking and is asserted separately per route and query class. Post-mask strata are measured for both arms, not claimed to remain equal. Exact per-ID or node exposure is not matched; the receipt reports that remaining bias. The paired expanded mask is an unadopted proposal.

## Infeasibility and outputs

An endpoint episode is infeasible when either eligible class has fewer than 256 IDs. The outer draw is retained, no stream is sampled for that episode, and no redraw, padding, smaller batch, or substitute pool is used. A matched-control stratum failure also rejects the pair and discards partial control selections. The default proposal rejects the entire candidate before a fit if any episode in the measured cycle is infeasible. Ordinary outer-only fallback geometry is reported for endpoint failures but never adopted or executed. A feasible cycle establishes geometry feasibility only; later cycles and a complete training schedule require their own prospective feasibility gate.

The planned receipt contains metadata only: all episode eligibility counts; rejected indices; tail/full-data schedule; retained context and support counts; before/after degree/CN and node coverage for both arms and every route; inner endpoint-overlap counts; positive/negative equivalent-fact repetitions; exposure over every TRAIN ID and native bank ID; paired-mask exposure. A feasible route's planned inner loss evaluations are counted twice for virtual adaptation and recomputation, consistent with a single committed private-update convention. The driver performs zero updates, so candidate exposure is never described as optimizer supervision.

The output is `RESULT.json` or `FAILURE.json` in a fresh approved directory. A successful measurement can still reject the candidate geometry. It exports no raw query list, model state, checkpoint, accuracy, ranking metric, or recommendation selected from heldout outcomes. Endpoint separation is conditional within-graph transfer regularization; this source makes no unbiased cross-fitting or new-node guarantee.

## Static verification and next gate

`STATIC_VERIFICATION.json` records AST checks, delayed numerical imports, CPU-only seeding/restoration declarations, complete-cycle scheduling, explicit route guards, disabled job flags, fixed sizes, and source hashes. `SOURCE_MANIFEST.json` seals all packet files except itself. The manifest hash must be supplied in a separate reviewed job; this disabled template is kept unchanged.

Root review and separate authorization are required for the one bounded TRAIN geometry execution. Candidate sizes, paired common mask, and ordinary fallback remain unadopted after source preparation. Existing derivative and oracle preparation/execution packets are preserved.
