# Squirrel installation-witness diagnostic v3

**Source only; unexecuted.** This is one Squirrel17 engineering followup, with no predictive continuation or qualification promotion. V1/v2 and their failures remain preserved. No selector setting, tolerance, scientific source file, numeric policy or native warm schedule is changed.

## Observation and intended distinction

The v2 report rejects `fixed_first_graph_pair` at `head.R`: 25 FP32 values differ, with maximum absolute difference `1.1920928955078125e-07`. The original strict predicate remains unchanged. This discrepancy does not establish whether the returned slices were altered or the independent geometry reconstruction differed.

V3 observes the actual installation path. It temporarily wraps `driver.select_initializations`, which the driver imports directly, and `selector.install_head`. At selector entry it freezes the actual prototype state, optimizer, RNG and modes. Each install wrapper calls the original installer once and immediately copies the intended slices and full installed model state to CPU. It captures trial and returned installations; it does not retain a strong model/optimizer reference.

Each witness has a monotonic serial and a weak reference to its precise model object. Matching uses `weakref() is returned_model`, never a numeric object ID alone. Dead trial objects and recycled IDs therefore cannot match a returned model. Both runtime hooks restore on context exit, including failure. The source file and instrumentation are explicitly bound and disclosed; the runtime helper bindings are **instrumented**, rather than claimed unchanged.

## Separate exact diagnostics

`SOURCE_CUSTODY_DIAGNOSTICS.json` checks every returned arm against its own installation witness and intended slices. It separately checks non-head state against the actual frozen prototype, optimizer/RNG/modes, and storage independence from the donor/checkpoint and other arms.

`INDEPENDENT_RECONSTRUCTION_DIAGNOSTICS.json` retains the original independent common/basis/radius reconstruction and exact comparisons for every arm. It additionally compares the independently rebuilt prototype/optimizer/modes with the actual captured selector inputs. Every mismatch remains a failed check. The collector proceeds to other arms, without converting an exact failure into a tolerance-based success.

The original per-arm mean-logit forward guard and live-factor probe are explicitly **not executed** in this focused diagnostic. The new observation path adds no model forward, Adam trial or RNG draw. Complete original native qualification is not established, even if all attempted witness/reconstruction comparisons happen to pass. Unexpected preparation errors are preserved separately and uncompleted comparisons are marked.

## Actual input and warm custody

Separate saved diagnostics established unequal v1/v2 warm model/optimizer states despite matching RNG/specification/stage/mode, and unequal repeated native Squirrel preprocessing from identical loaded raw inputs. Those are separate repeatability concerns; they do not prove the cause of the returned-head discrepancy or the warm-state differences. Their existing receipts are bound in `SOURCE_BINDINGS.json`.

V3 hashes the actual loaded raw features/edges, authorized compact TRAIN/VALIDATION packs and native preprocessed input. It saves `ACTUAL_PREPROCESSED_INPUT.pt` containing the already-produced `graph.teacher_input` and `teacher_edge_index`, without recomputing preprocessing or changing its arithmetic. `PREPROCESSING_SNAPSHOT_RECEIPT.json` binds its storage bytes/file hash and logical tensor hashes. The snapshot hashes must match the preceding actual-input fingerprints exactly.

`FRESH_WARM_FINGERPRINTS.json` records logical model, optimizer and RNG fingerprints from the existing fresh CPU checkpoint, plus input/preprocessing identities and native specification/metadata. Tensor fingerprints include dtype, shape, byte count, byte order and raw logical-byte hash. The Torch archive file hash is recorded separately; serialization differences are not equated with logical state differences. These observations claim no repeatability or causal result.

CPU copies, hashing, preprocessing snapshot I/O and witness capture incur costs. The existing `Costs` recorder charges fingerprint/snapshot operations and the aggregate instrumented selector operation; snapshot receipts record storage. Captures change runtime allocation/timing and are disclosed. They do not change training, selection, RNG, or the original trial budget.

## Root integration

Use the existing owned one-child transport wrapper with a fresh output directory. Replace its runner source binding with this packet's sealed manifest and include the bound `install_witness.py`, input bindings and provenance. The callable remains:

```python
runner.run_qualification("Squirrel", fresh_output, device="cuda:0")
```

It retains the authorized one-GPU destination/UUID checks, full prescribed Squirrel50-update fresh warm and existing early engineering checks. Photo is rejected. No launch or retry is performed or authorized by this source packet.

Outputs include `INSTALL_WITNESSES.pt` and `INDEPENDENT_RECONSTRUCTION_HEADS.pt`. The custody JSON maps returned arms to witness serials; it supplies the live-object matching evidence for the archived snapshots. Large tensors remain server-side. The report status distinguishes failed source custody, retained independent reconstruction failures and diagnostic completion; **no qualification-pass status exists**. An existing wrapper that treats every non-pass status as child failure will therefore preserve a non-pass outcome even when this diagnostic completes.

`STATIC_CHECK.json` contains syntax/source checks only. It does not qualify the instrumentation at runtime. Interpretation remains conditional: matching intended/installed/returned states alongside a different independently reconstructed head would support geometry-recomputation drift; failed witness custody would identify a returned-state discrepancy needing investigation. Neither observation alone establishes methodological correctness.
