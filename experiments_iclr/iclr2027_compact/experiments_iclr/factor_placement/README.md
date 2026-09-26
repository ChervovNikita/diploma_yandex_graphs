# Factor placement and same-runtime TIED control

This compact public package preserves the complete frozen validation record for
two related studies: 72 all-layer-factor cells on Cora, WikiCS, Actor, and
filtered Chameleon, and 36 separately trained same-runtime TIED cells on Cora
and WikiCS. It includes all 108 result records and 1,000-epoch validation
traces, all 108 selected-checkpoint validation hard decisions, and all 36
allowed selected/default test scores with exact pooled float32 logits and four
member hard decisions. The original 432-cell primary validation lock anchors
both studies. The `legacy_tied_default/` projection adds six original TIED
default test arrays for the Actor and filtered-Chameleon comparison rows.

Run from the anonymous code-supplement root, with Python and NumPy:

```sh
python experiments_iclr/factor_placement/verify_factor_compact.py \
  --stage experiments_iclr/factor_placement \
  --primary experiments_iclr/validation_tuning
python experiments_iclr/factor_placement/legacy_tied_default/verify_legacy_default.py \
  --stage experiments_iclr/factor_placement/legacy_tied_default \
  --primary experiments_iclr/validation_tuning
python experiments_iclr/factor_placement/provenance/verify_provenance.py \
  experiments_iclr/factor_placement/provenance
```

The first verifier checks complete cell and score sets, the original records
and freeze hashes, checkpoint epochs recomputed from full validation traces,
validation-selected candidates, validation accuracies from retained hard
decisions, and selected/default test accuracy and cross entropy from retained
pooled logits and official test references. The legacy verifier checks its
six original TIED default projections against the corresponding original
primary score records. The provenance verifier checks byte-level hashes of
the additional audit and scoring records.

`provenance/PLACEMENT_COMPARISON.csv` gives the eight paired three-seed
comparison rows. Selected all-layer minus TIED mean test differences are
+2.233, -0.239, -0.132, and +1.890 percentage points on Cora, WikiCS, Actor,
and filtered Chameleon, respectively. The respective default differences are
-0.633, -0.257, -0.066, and +0.172 points. Actor and filtered Chameleon use
the original same-runtime TIED arms; Cora and WikiCS use the new TIED36
control. The design of the factor-placement extension followed earlier
outcomes, and all-layer factors add parameters. These are sensitivity
comparisons, not a randomized or capacity-controlled causal estimate of
factor position.

`FACTOR_COMPACT_PROTOCOL.md` defines the public projection. Its main
`MANIFEST.json` is a frozen hash index for that projection. The independent
`provenance/FACTOR_FULL_TO_COMPACT_AUDIT.json` checked every one of the 108
validation and 36 test projections directly against the original full arrays
before packaging; `audit_factor_against_full.py` is its source. The original
full arrays, model checkpoints, and individual member float logits are held in
author evidence and are omitted here. Thus the public verifier can
recalculate pooled test metrics and selected hard-decision metrics, but cannot
replay original weights or reconstruct individual member probabilities or
validation cross entropy. Retraining is possible using the frozen source and
public graph data, but does not reproduce omitted original checkpoint bytes.

`provenance/ALL_LAYER_SCORING_AMENDMENT_V2.md` documents the field-name-only
repair to the all-layer scoring gate, made before any all-layer test score
existed. Both original and V2 scoring/verification sources are included. The
two frozen TIED36 design anchor files in `source/` were missing from the
transferred full-author tarball and were restored from their pre-existing
local frozen originals; their hashes matched the TIED36 freeze. The compact
stage and the full-to-compact audit were generated only after that repair.
