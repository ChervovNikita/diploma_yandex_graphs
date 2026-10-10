# Split RNG clarification before model construction

The amended split uses **one** NumPy RandomState seeded190111, separate from every model, optimizer, mask and dropout stream. It draws class permutations sequentially in ascending class order0,1,2. It is not reset per class. The earlier wording about an independent stream meant independence from model training and was ambiguous about the class loop.

The exact recipe is the code in export_train_v4.py: construct rng once, then call rng.permutation for each class. Class quotas remain integer floors of60% and20%, with the remainder assigned to TEST. No model outcome informed this clarification. The original amendment and earlier exporters remain preserved.

The completed custodian confirmed PyG class order0,1,2 as total counts[4103,7739,7875]. The actual role counts in this order are TRAIN[2461,4643,4725], VALID[820,1547,1575] and TEST[822,1549,1575]. The earlier amendment's illustrative count list swapped classes1 and2. The source recipe, class IDs, split seed and resulting node identities are unchanged. DATA_V4_CUSTODY.json is the authoritative role record.
