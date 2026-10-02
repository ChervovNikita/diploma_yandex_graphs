# Tolokers2 native baseline and pilot preparation

**Prepared and unexecuted. This sealed packet does not authorize a run.** It contains a frozen bounded protocol, four unexecuted scientific drafts, pinned source evidence, and a standard-library metadata checker. All dataset, runtime and numerical qualification gates remain open.

## Frozen next study

| Item | Specification |
|---|---|
| Task | Complete Tolokers2, official RL, transductive, binary AP |
| Native model | GraphPFN paper custom residual GAT; width 512, three pre-LayerNorm blocks, four heads, GELU, two-logit CE |
| Native matrix | c0: lr 3e-4/fraction none; c1: lr 1e-3/fraction none; c2: lr 3e-4/fraction quantile-normal; dropout .1; seeds 17/29/43 |
| Native selection | Nine fits maximum; each max 1,000 updates/patience 100; earliest strict validation-AP best; select config by mean over seeds, exact tie c0,c1,c2 |
| Label custody | Public role freeze first; compact train/validation labels in an isolated worker; test labels CLOSED |
| Optional extension | Only after native competence and actual warm qualification: five K4 arms × three seeds; fixed warm50 donor, max 950 continuation |
| Controls | Reuse selected three native fits as single/independent3 references; three paired four-head single-body controls; full costs and member mismatch disclosed |

This is a small subset of the published search, **not its winning optimum**. The saved config says `add_self_loops=true`, but pinned GraphLand loading discards that argument and removes loops without readding them. This protocol preserves the pinned **no-loop source behavior** and records the discrepancy.

## Files

- `PROTOCOL.json`, `CONFIG_MATRIX.json`, `ROLE_POLICY.json`: frozen scope, settings and labels.
- `ACQUISITION_PHASES.json`, `QUALIFICATION_GATES.json`: phased access and required stop/pass conditions.
- `NATIVE_SOURCE_CONTRACT.md`: model, preprocessing, AD and AdamW details.
- `HANDOFF.md`: remaining work and interpretation limits.
- `native_gat_adapter.py`, `route_initializer.py`, `closed_label_preparation.py`, `closed_native_loop.py`: future reviewed drafts; no scientific CLI or complete admitted runner.
- `SOURCE_RETRIEVALS.json`, `DGL_SOURCE_RECEIPTS.json`, `SOURCE_BINDINGS.json`, `INITIALIZER_LINEAGE.json`: evidence custody and unchanged R17 lineage.
- `check_metadata.py`: source/receipt/protocol checks only. `python3 check_metadata.py --verify-seal` reads the sealed packet and emits JSON; it does not import the drafts or compile them.
- `STATIC_METADATA_CHECKS.json`, `MANIFEST.json`, `SEAL.json`: metadata-check receipt and packet integrity.

No archive bytes, dataset or label arrays, model weights, model/provider imports, scientific runs, SSH, GPU operations, installation, or compilation were used to prepare this packet. No new primary paper was read; the two industrial-paper conclusions were reused.
