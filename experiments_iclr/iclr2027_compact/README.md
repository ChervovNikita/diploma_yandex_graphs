# Audited compact records for the ICLR study

This folder preserves the anonymous supplement's relative directory layout for the completed extension studies. It includes frozen source copies, protocols, hashes, complete validation traces, selected result metadata, and compact decisions derived from audited predictions. Large checkpoints and float32 prediction archives are retained separately by the authors.

From this folder, use a Python environment with NumPy and run:

```text
python experiments_iclr/verify_new_compact.py
python experiments_iclr/verify_roman_budget1000_compact.py
python experiments_iclr/roman_noloop_depth/verify_compact.py
python experiments_iclr/external_depth_sage/verify_compact.py
python experiments_iclr/sharing_position/verify_decisions.py
python experiments_iclr/ogbn_arxiv_sharing/verify_decisions.py
python experiments_iclr/verify_ogb1000_compact.py
```

These checks cover the 40-cell Roman depth grid, its 12-cell longer-budget repeat, 12 no-loop Roman cells, 24 WikiCS/Actor depth cells, 56 four-arm sharing-position records on four graphs, nine 1,000-epoch OGB node runs, and 12 link-prediction runs. Each graph uses the split and seed scope stated in its protocol. Seeds are repeated optimization on one split, not independent graph tasks.

See `experiments_iclr/NEW_STUDIES_README.md` for study-specific details. Its commands assume the same archive layout. Compact hard decisions verify score arithmetic but cannot regenerate raw-logit pooling, cross-entropy, or checkpoint inference. Those checks were performed on the full author artifacts before inclusion. Fresh training requires the public data and dependencies listed by the corresponding runner, and a new output directory. Some runners use the main repository's model or data code, whose frozen hash is recorded in the study manifest.

This public repository identifies its contributors. Use the separately built anonymous archive for conference review.

Frozen source copies retain their original line endings and formatting so that recorded hashes remain verifiable. The local attributes file disables text normalization for this evidence tree.
