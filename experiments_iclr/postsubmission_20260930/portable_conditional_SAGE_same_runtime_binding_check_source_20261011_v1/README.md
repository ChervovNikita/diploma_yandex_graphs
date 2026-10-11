# One same runtime portable binding check

Source only. Root owns one actual execution with the existing allocation `.venv` and a 600 second external watchdog. The sealed portable bundle remains unchanged.

The check verifies portable manifest `d52b185077d81692472a0bedc002df864c1c7a34f6de55b4e398126c2ce80781` and all files it names. It binds only portable project modules, installed pinned PyG sources, and the original full ten-block Physics TRAIN/SELECT receipt metadata in `INPUT_BINDINGS.json`. A new local config uses actual `LOCAL_RUN_PERMITTED`/`LOCAL_VALIDATED` semantics, keeps `max_updates=1000`, and uses finite 600 second wall, 48 GiB CUDA and 64 GiB RSS caps. Original scientific configs are checked by hash and never written.

Exactly one active block (seed 8101) and fresh E4 are exercised. `engine.construct` returns a Family; `family.make` returns `(model, optimizer, streams)`. The script directly invokes bundled `own_ce.update(..., mode="sequential")` once: four quarter-scaled TRAIN backwards and one AdamW step. It saves model, Adam and route RNG, mutates one value in each category, restores the saved state strictly, and checks exact state equality. One eval forward on label-free SELECT IDs must be finite float32 `[4,150,5]`. No reference bank, additional update, output replay comparison or qualification suite is added.

The existing constructor loads safe TRAIN/SELECT fields, including SELECT truths; those truths are never evaluated. The unchanged own-CE updater internally computes TRAIN loss diagnostics; no accuracy, SELECT NLL, selection metric, predictive output array or loss value is reported by this check. No CAL/TEST target, current scientific checkpoint, fit, `acquire`, exporter or scorer is accessed. Bytecode writes are disabled for the sealed bundle.

From the allocation repository, under root's external watchdog, the single intended entry is:

```
.venv/bin/python experiments_iclr/postsubmission_20260930/portable_conditional_SAGE_same_runtime_binding_check_source_20261011_v1/check.py --package experiments_iclr/postsubmission_20260930/portable_conditional_SAGE_source_20261011_v1 --output <fresh-check-directory> --permit-one-binding-check
```

Outputs are a new check config, constructor/local validation records, `fresh_binding_state.pt`, and either `COMPLETE_BINDING_CHECK.json` or a retained failure. The completion marker certifies only this existing-runtime binding/update/restore path. It does not certify clean installation, another host, external portability, the full CLI or predictive quality. No execution occurred during preparation.
