# Full TRAIN episode geometry: actual result and limits

One source-reviewed CPU measurement finished in 10.91 seconds with 515,633,152 bytes peak RSS. The complete fixed Citeseer TRAIN population was used; no model, optimizer, feature payload, VALID or TEST was accessed. The server and local 10,489,968-byte result have identical SHA256. All source and TRAIN bytes remained unchanged.

All 61 endpoint-separated episodes and matched random controls were feasible. Minimum eligible pools were 2,913 positives and 6,519 negatives, above the fixed 256 per-class requirement. All 3,870 TRAIN positives appear exactly once in the outer pass; the final episode retains its 30 positives and negatives. No redraw, padding, truncation or fallback occurred. Every endpoint arm has zero inner queries touching its outer endpoints. Pre-mask degree/common-neighbor strata match exactly for each route/class; post-mask strata and per-node exposure are not asserted equal.

Own positive-target masking removes 937–984 of 3,870 facts. The proposed common support additionally masks random-control targets and removes 1,541–1,610 facts, about 40–42% of the graph. This expansion is substantial and must be identical across candidate and matched controls. It is not ordinary native support.

Each route receives 15,616 inner positives and negatives in one complete outer cycle. Mean positive exposure is 4.035 per TRAIN ID; 91–103 positive IDs are unseen by a given private route during that cycle, while every positive appears in the outer pass. Virtual and recomputed passes evaluate the same inner queries twice but commit one private update. A comparison only against the much cheaper unchanged native schedule would be inadequate to attribute benefit to the proposed learning mechanism.

This admits one-cycle geometric feasibility only. Candidate sizes, common mask, optimizer/update rule and full training schedule remain awaiting the exact method-source and cost review. Future seeds/cycles must retain prospectively defined eligibility/failure behavior. No accuracy, superiority, novelty or graph-generality result follows from this measurement.
