# Fixed-projector PPI F2 audit v1

This audit tests transfer of the local denoising proxy to two unused training graph units. It applies three already fixed bases to gradients at the same tied P-only epoch25 source: selected, matching-rank random, and matching-rank raw centered-gradient. It performs no optimization, final training, accuracy evaluation, or AdamW-benefit measurement. The earlier code, input locks, masks, checkpoints, D/E gradient records, replay reports, and basis locks remain byte-identical.

## Prospective freeze

`freeze_audit.py` is standard-library only. Before any F2 gradient is obtained, `AUDIT_INPUT_LOCK.json` binds all six seed0/1/2 x label10/100 cells; their original checkpoints, masks, contracts, source states, D/E records, exact replays, and basis packets; every prior integration file; the spectral implementation manifest; and the new audit code/protocol. It checks that the six-cell collection passed and F2 was unopened. No rank, basis, threshold, graph partition, or mask may change after this lock.

F2 is exactly `ppi:train:09` and `ppi:train:18`, in that order, from the original Q10 split. The existing P10/D4/E4/F2 identities, seed700/704 splits, original node-mask rule, n0=1, and selected cap16 are retained. The saved random/raw basis bytes are reused without a new seed or eigensystem. Matching ranks are checked, including zero. No D/E selection is repeated on F2.

Runtime first verifies all frozen bytes and replays each D/E record. It strict-loads the original tied source checkpoint, checks the same AdamW250 fit evidence and masks, and verifies source/wrapper/zero-adapter forward equality. It binds all six F2 graph/mask/source fingerprints in `F2_PREGRADIENT_GATE.json` before autograd starts. The source state must equal that recorded during D/E collection.

Only the official training loader is constructed, with existing-cache-only download/process guards. F2 targets are now authorized training labels. Before a graph is passed to the model or collector, unobserved targets are physically NaN and the archived observed-node mask is applied. Validation/test datasets and metrics remain absent. P/D/E target tensors are not supplied to the F2 collector.

## Unchanged statistic

For each of the three original h128 neighbor matrices, save its own-member, own-graph mean BCE gradient G_im, without member or graph scaling. Use `[2,4,128,128]` raw float64 evidence. Center members separately within each F2 graph, then call the unchanged `v2_abi_fix_01.spectral_core.output_statistics(raw,n0=1)`:

    Y_m = mean_i D_im
    Sigma_F = sum_(i,m) (D_im-Y_m)(D_im-Y_m).T / [4*(2-1)]
    S_F = mean_m Y_m Y_m.T - Sigma_F/2
    N_F = Sigma_F
    T_F = S_F - N_F

Each complete graph has equal weight regardless of node count. For each fixed saved U the primary target score is exactly `tr(U.T @ T_F @ U)`. Rank0 has an empty projector and score0; it remains a valid result. Save per-target traceS/traceN/traceT and direction diagonals for decomposition, and the unnormalized sum of traceT over the three disjoint fixed targets. Selected-minus-random and selected-minus-raw sums are descriptive comparisons fixed before seeing F2. No rank normalization, clipping, alternative n0, new basis, or favorable gate is introduced.

The same common-source F2 raw gradients feed all three projector scores. No private coefficient update is taken. The values concern the fixed-projector gradient denoising proxy at that local source state; they do not measure a loss change, optimizer trajectory, calibration, accuracy, or resource benefit.

## Evidence and limits

Save each cell's sealed raw-gradient record, all reconstructed covariance tensors and basis projections, unchanged basis packets and their hashes, checkpoint/source/input/code/environment identities, and exact replay report. Collection preserves parameter/buffer values, existing gradient objects/values, modes, and RNG, including failure restoration. An independent replay command reconstructs all covariance/projection bytes from the recorded gradients and fixed basis bytes without loading labels or models. It also checks that earlier evidence remains unchanged.

Two graph units provide no useful basis here for a confidence interval or p-value. No CI/p-value is computed. The six cells reuse the same graph identities and nested masks; seeds, members, directions, nodes, and feature coordinates are not independent task replications. Biological independence and the IID assumptions behind the covariance interpretation are unverified. Selected D/E scores remain affected by selection, and no selected-score unbiasedness is asserted. Small or negative F2 values, or controls matching the selected score, must be reported directly; no criterion or rank revision follows. This bounded audit supports or weakens a local mechanism hypothesis only.

## Coordinator commands

The input lock is generated locally with no torch execution before the new F2 run:

```bash
python spectral_privacy/ppi_f2_audit_v1/freeze_audit.py \
  --output spectral_privacy/ppi_f2_audit_v1/AUDIT_INPUT_LOCK.json
```

Transfer the exact audit code and lock. Run in the existing repoenv, from the phase directory, using the coordinator's supervised CPU launcher:

```bash
python spectral_privacy/ppi_f2_audit_v1/audit_f2.py run \
  --lock spectral_privacy/ppi_f2_audit_v1/AUDIT_INPUT_LOCK.json \
  --output spectral_privacy/ppi_f2_audit_v1/audit_run01
```

The script sets torch intra/inter-op threads1, collects all six cells, and immediately replays every saved JSON record. Then perform an independent label-free replay:

```bash
python spectral_privacy/ppi_f2_audit_v1/audit_f2.py replay \
  --lock spectral_privacy/ppi_f2_audit_v1/AUDIT_INPUT_LOCK.json \
  --run spectral_privacy/ppi_f2_audit_v1/audit_run01 \
  --output spectral_privacy/ppi_f2_audit_v1/replay_run01
```

Each output directory must be new. Preserve failure traces/artifacts; any concrete repair needs a new sibling/version and lock. The author performed local standard-library AST and preservation/freeze checks only and launched no server job.
