# Complete MolBACE development pilot

Three paired training seeds, one selected VALID scaffold split. Test data remain closed.

| Method | VALID AUC mean ± sample SD (%) | Δ vs base (points) | Δ vs I4 (points) | Mean BCE |
| --- | ---: | ---: | ---: | ---: |
| Native single | 73.32 ± 1.17 | +5.54 | -1.53 | 0.779 |
| Independent ensemble | 74.85 ± 0.50 | +7.07 | +0.00 | 0.890 |
| Base GNNM | 67.78 ± 4.39 | +0.00 | -7.07 | 0.719 |
| Alignment + variance | 71.17 ± 2.66 | +3.39 | -3.68 | 0.551 |
| Private input correction | 71.76 ± 3.91 | +3.98 | -3.09 | 0.941 |
| Both ingredients | 69.28 ± 1.80 | +1.50 | -5.57 | 0.882 |

Private input correction improves base GNNM in 3/3 seeds, remains behind I4 on average; geometry helps 2/3; their combination helps less. Retain planned ingredient comparisons and inspect private BN follow-up. No method promoted as an ensemble improvement.

The selected-score CSV contains all three seeds for every method. REPORT.json includes paired differences and approximate t intervals with their assumptions. Three seeds and selected validation measurements cannot establish significance or transfer.

All 24 fits completed; 3,680 whole TRAIN epochs are recorded. Pilot elapsed time was 7.81 hours, with another existing training worker sharing the GPU. Worker costs are preserved separately.

Every actual epoch/selector is audited below. Patience shows validation stopping under the frozen rule; it does not establish optimization stationarity or remove seed/split uncertainty.

The last-20-epoch training-loss summaries compare a fit with itself. Auxiliary and native losses are not interchangeable. Selecting early and stopping later can indicate overfitting; a lower last training loss does not by itself justify training longer.

This analysis reads existing JSON/JSONL measurements only. It starts no training, scores no new predictions, and changes no frozen decisions.
