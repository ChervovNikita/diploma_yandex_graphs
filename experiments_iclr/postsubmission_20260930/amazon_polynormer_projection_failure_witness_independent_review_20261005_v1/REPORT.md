# Independent bounded assessment of the actual ID9 failure witness

5 October 2026. Root explicitly authorized reading only `WITNESS.json` and `DIAGNOSTIC_RESULT.json` in the preserved witness observation. Their bytes are bound here. No original graph/logit/label/checkpoint or result-score payload was accessed; no source import, numerical solve, fit, scoring or SSH occurred. This is interpretation of recorded arithmetic diagnostics, not a numerical reproduction or V3 approval.

The diagnostic reports `WITNESS_EXTRACTED_NO_FITS_OR_SCORES`, zero fits/optimizer updates/metric calls/scored-fold outcomes, wall4.693285524845123 seconds and peakRSS428380160 bytes. The exact V2 and extractor hashes match their prior source reviews. Witness SHA256 is `4a5310a1549d231fc1a2dd65838d39030a986cea1cd1ca8d022034e59efe0700`,19026 bytes.

## What the record establishes

Native node709 is the first actual `chosen` NaN row of the first0–2048 batch, with source internal offset0. The source-trace scaled A,b and scale match the exported witness fields exactly as JSON values. The node is in scored D; its true label is neither used by the projection context nor exported. The posterior is computed only from permitted B anchors under the already reviewed extractor. Thus the diagnostic isolates the numerical rejection without a target outcome.

The recomputed free face[1,2] records weights `[0.0125,0.7617879182059456,0.21321208179598564,0.0125]`, zero floor violation, zero inactive-dual error and scaled free-stationarity error1.920150236728313e−12, below the1e−10 threshold. Its sum error1.9311219290329973e−12 exceeds the old strict1e−12 candidate-sum threshold. In this recomputation it is rejected by equality roundoff, with the other recorded feasibility tests passing. Its augmented KKT singular values are approximately2.7244,0.7341,2.2254e−6; recorded rank is full3. Therefore this particular face's rejection is not a rank cutoff event.

The member rows are concentrated in a common class with very small contrasts. The full member-probability singular values range from1.9952 to7.7788e−12, and the full-face augmented KKT has a6.7928e−17 direction that the old pseudoinverse discards. These values document strong conditioning/rank sensitivity, but that full-face rank loss is separate from the[1,2] equality rejection. The record does not prove a unique causal explanation for every failing face.

## Repair relevance and limits

Centered Helmert coordinates impose the sum equality through the parameterization and solve member contrasts directly, so they address the observed sum instability without relaxing the floor/sum/KKT thresholds or changing the convex objective. Direct SVD also avoids squared Gram conditioning. The proposed machine-rank cutoff remains an explicit numerical approximation with the limits in the prior arithmetic review. No exact-cutoff-equivalence claim follows from this witness.

The prior source-review reporting limit remains: all15 face records are single-row recomputations, which can round differently from the failed batched einsum/pseudoinverse path. The actual trace NaN row and exact scaled fields establish the failed instance; the[1,2] recomputation is strong evidence of the candidate rejection mechanism, not bitwise replay or a certified exact optimizer. Root's affected-only qualifier will test the actual V3 projection against the witness and an independent reference. No unrelated head/sparse/moment requalification is requested.
