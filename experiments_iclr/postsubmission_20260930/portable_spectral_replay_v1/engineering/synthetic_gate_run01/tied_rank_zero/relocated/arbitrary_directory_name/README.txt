Portable selected-checkpoint replay

Use the recorded Python3.11/Torch/PyG CPU environment. No downloading occurs.
From any directory: python /path/to/bundle/code/replay.py --output results/replay_run02.json
The script checks every immutable file, all101 validation metric epochs and checkpoint selection,
then reconstructs the exact frozen bases from the state_dict and executes fresh selected predictions.
All tensor loads use weights_only=True. Temporary/cache/output paths remain inside this bundle.
This is checkpoint/selection replay, NOT independent fresh training reproduction.
The data contains only the two validation graphs; no test split or author calibration records.
Float32 cross-runtime tolerance is atol=rtol=2e-4; threshold changes and fresh metrics are reported.
Both trace-selected and independently recomputed winners must match.
CPU build suffix may differ; Torch core/PyG version and SAGEConv source must match.
Recorded state construction, basis provenance and training validity remain author audit obligations.
