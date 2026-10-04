# Fresh graph-view source review

Source correctness inspection of the immutable `accuracy_first_graph_view_source_preparation_20261004_v1` package. No sources were edited; no package code, numerical imports, servers, real data, scientific outcomes or PDF compilation were used.

No blocking source flaw was identified for the supplied synthetic CPU engineering checks. One defect blocks the CUDA RNG/replay invariant when an admitted caller uses an unindexed CUDA device. Source inspection supplies no numerical or scientific acceptance claim.

## Actionable finding

**F1 — P2: unindexed CUDA devices omit the CUDA RNG stream.** In [bank.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/accuracy_first_graph_view_source_preparation_20261004_v1/bank.py:22>), `rng_state`, `restore_rng` and the explicit CUDA seed branch test `str(device).startswith("cuda:")`. PyTorch also accepts `"cuda"` and `torch.device("cuda")`; model/tensor transfers can use CUDA while snapshots store `cuda=None` and restoration skips that stream. This invalidates next-update dropout replay for that spelling. The CPU path is unaffected.

Create a preserved successor that normalizes/resolves the selected CUDA device consistently before seeding, snapshot and restoration, or rejects the unindexed spelling clearly. Verify the accepted-device resolution/rejection with a source or stub check; CUDA numerical replay requires separate admitted evidence.

## Inspected behavior

- Untied copies preserve each initialized factor row, copy interior/W ownership, and call the unchanged bound v6 member forward without a constructor/reset or explicit RNG draw.
- Eight passes implement four `[CE(native,TRAIN)+CE(probe,TRAIN)]/4` contributions: zero once, backpropagate each immediately, step Adam once.
- One strict pooled VAL selector retains earliest ties across both stages. Actual-update 200 restores selected-local model/Adam while retaining the live schedule/RNG, then enters global mode. Full final restoration permits a local winner.
- Phase-relative dependency descriptors and exact source excerpts matched. Runtime compiles the reverified neural bytes and records resolved paths.
- Semantic masks delete floor10% of unordered equal/different TRAIN pairs, both directions, while retaining all loops and survivor order. Nulls sample a class-agnostic TRAIN pair pool with matching coarse degree-bin counts; overlap is allowed and recorded.
- TRAIN labels feed fitting and view construction; separate VAL labels feed native pooled selection/reporting. Probability pooling averages four class-softmax vectors. TEST labels have no reader.

## Limits

Numerical correctness, full Amazon behavior, resource cost and CUDA determinism require admitted runtime evidence. The final exact raw-logit check relies on deterministic evaluation. Official data/compact-label provenance remains the caller responsibility; coverage/bundle hashes establish custody and consistency. `verify_bundle` does not independently reconstruct semantic masks. Selected images support state/function restoration, while mid-run clock/selector/trace restart orchestration is absent. Real fixed-mask coverage and an external freeze remain pending.

The broader quality controls are explicitly unimplemented: modern single/conventional ensemble references, paired synthesis, worthwhile-gain rule and confirmation/TEST history. This prevents a final quality claim and does not prevent synthetic CPU engineering checks. VAL selection/reporting remains exploratory.

## Integrity

Both required hashes matched before and after inspection:

- Manifest: `135e30466c703b1929eadd680b7152330620e4cde75152ec69317458580efa2a`
- Protocol: `55f4146b450f2e04c518e07df9a118bb21f7dc65e18dc659eb34e6dbe8edfb46`

All 13 manifest payloads, 10 bound descriptors and 8 exact source excerpts matched on both integrity sweeps. Seven Python files parsed as syntax without execution. Detailed source locations and scope are in [REVIEW.json](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/accuracy_first_graph_view_fresh_source_review_20261004_v1/REVIEW.json>).
