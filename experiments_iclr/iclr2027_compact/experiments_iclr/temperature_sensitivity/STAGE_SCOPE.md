# Validation-only scalar temperature sensitivity

This stage covers every one of the 72 validation-selected HPO cells. It is a
post hoc probability-quality diagnostic. The test set had already been
inspected before the diagnostic was designed, and the same validation nodes
used for checkpoint/hyperparameter selection are reused to fit temperature.
The method is standard scalar temperature scaling; it changes probabilities
and leaves model weights and class decisions unchanged. It is not evidence
of an untouched generalization gain or an additional architecture novelty.

The source and protocol were frozen before validation extraction/fitting.
`TEMPERATURE_DESIGN_FREEZE.json` preserves the original source hashes;
`TEMPERATURE_DESIGN_AMENDMENT_V2.json` records two pre-execution hardenings
and binds the running source. `validation_only/` contains only the original
float32 pooled validation logits and frozen validation indices/labels, for
all 72 cells. `TEMPERATURE_VALIDATION_LOCK.json` contains all 72 fitted
inverse temperatures and validation objectives. The fit command wrote this
lock before the score command opened any test array. Its SHA-256 is
`4fe03b34d5b733ad19d13ee4430c470b7d8adeccfaf1390ed02ed6e02e185ded`.

`test_results/` contains every unscaled and scaled test metric. The
independent `INDEPENDENT_ROOT_AUDIT.json` checked all original validation
arrays, an alternative one-dimensional optimizer and objective stationarity,
and all 72 test rows. Original full model/checkpoint evidence is retained
separately. The sibling `hpo_selected_test_logits/` stage supplies exact
test member logits; the sibling `validation_tuning/` stage supplies the
official test reference and original validation-selection lock.

`test_results/TEMPERATURE_ALL24_MEANS.csv` is a later deterministic
descriptive aggregation of the full 72-row CSV (all four graphs and six
arms). It gives before/after/delta means and sample standard deviations for
accuracy, NLL, Brier, and ECE, plus inverse temperature. Rebuild it with
`python3 summarize_temperature_all24.py`; output SHA-256 is
`70b8f0affa4890bb08047d47bd8416e8bc265a2e6410572b0cf5554def1befa8`.

From this folder, with NumPy installed and both sibling stages present, run:

```sh
python3 temperature_sensitivity.py verify-lock \
  --validation-stage validation_only \
  --study ../validation_tuning \
  --lock TEMPERATURE_VALIDATION_LOCK.json

python3 temperature_sensitivity.py score \
  --validation-stage validation_only \
  --study ../validation_tuning \
  --lock TEMPERATURE_VALIDATION_LOCK.json \
  --test-companion ../hpo_selected_test_logits \
  --out reproduced_temperature_test_results
```

The second command refuses to overwrite an existing output directory. It
replays the complete validation lock before reading any test input. Compare
the generated CSV and summary with the supplied `test_results/` files.
