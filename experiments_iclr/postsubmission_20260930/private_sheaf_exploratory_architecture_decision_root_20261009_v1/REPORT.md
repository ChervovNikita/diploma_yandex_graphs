# Exploratory backbone decision after complete attempt review

The original15-attempt screen remains `incomplete_no_freeze`:13 reports completed and2 failed their reconstruction log-probability check. No result was overwritten, rerun or relabelled. Both failed fits completed500epochs. A previously declared once-only diagnostic found exact parameter/buffer restoration, zero changed decisions in TRAIN/VALID, and validation AUROC differences below0.000001. No further numerical investigation is scheduled.

## Full panel

The table uses the original selected-epoch scores for every scheduled seed, including the two failed report slots whose stored metrics were exposed by the diagnostic. It contains no survivor averages. These are development results after checkpoint selection; seed standard deviations are descriptive, not confidence intervals or independent test uncertainty.

| Configuration | Mean stored selected VALID AUROC (%) | Seed SD (points) | Original failed reports |
| --- | ---: | ---: | ---: |
| feature_mlp_10_64_64_2 | 72.675 | 0.113 | 0 |
| d2_f32_L2 | 83.039 | 1.006 | 0 |
| d2_f32_L4 | 84.214 | 0.594 | 0 |
| d4_f16_L2 | 83.469 | 0.389 | 1 |
| d4_f16_L4 | 84.586 | 0.041 | 1 |

## New comparison

Use d4_f16_L4 with the unchanged native optimizer,500epoch ceiling and200epoch patience. This choice follows the full panel and is explicitly exploratory. Freeze fresh base seeds7409/8501/9607 and disable member0 reuse. Each independent reference will have a complete native body, separate optimizer and own best checkpoint; the shared model will have only private incidence-map factors, four complete paths and a mean own-member loss. Additional diversity/context regularization is zero in this comparison.

Exact restoration and finite serving remain required. New reports will record predictive materiality and consistently use one reconstructed selected state, without a maximum-log-probability-only gate or evaluation retry. Original protocols and failed reports remain unchanged.

This establishes a useful backbone for testing the ensemble hypothesis, not a GNNM advantage, graph-causality proof, new method, unused-data result or acceptance recommendation. A scientific launch still requires reviewed source and one full-input shared-gradient qualification.
