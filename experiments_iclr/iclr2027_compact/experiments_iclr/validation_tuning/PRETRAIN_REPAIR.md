The first generated freeze (SHA-256
`fe7e142d19a5ca45283b326b978e8c7a1eb435abcf51c20873fe55d220622caa`)
failed the preflight before any training. The comparison in `check_freeze()`
compared Python tuples in the in-memory matrix with JSON arrays in the saved
file, so it always rejected an otherwise matching manifest. We changed only
that comparison to normalize the expected structure through JSON before
comparison. The failed first freeze is preserved as
`FROZEN_STUDY_ABORTED_PRETRAIN.json`. A new prospective freeze will be created
after this source repair, before any training or test scoring.
