# PPI predictive test of fixed spectral private directions

The gradient audit suggested a restricted hypothesis: under sparse training labels, a few screened member-specific directions might outperform equally small random or uncorrected-gradient directions after training. Its full-label audit favored the uncorrected-gradient control. This new study tests predictions on held-out validation graphs and retains that adverse control.

## Current status

Prepared, source-frozen and syntax-checked. No predictive fit or accuracy claim has been made by the implementation agent. A real-data CPU preflight and runtime freeze are required before GPU fitting. The coordinator runs them using the existing repository environment and supervised launcher. Local provider tensors are unavailable; the implementation agent did not access another server or download data.

Use **version2**:

- `PROTOCOL_v2.md`: purpose, exact100-epoch equal budget,24 fits, mandatory adverse/full-label control, selection, cost and continuation rules.
- `PROSPECTIVE_PROTOCOL_v2.json`: references/hashes of unchanged P25 sources, masks and all existing bases; source/runtime independence is explicit.
- `runner_v2.py`: preflight, runtime freeze, fitting and numerical selected-checkpoint replay.
- `summarize_v2.py`: checks all24 immutable fits/replays, reports all contrasts and applies the prospective continuation gate.
- `prepare_v2.py`: standard-library source/input preparation; does not open checkpoint tensors or predictive results.

V1 sources and its original prospective lock remain unchanged. Its preflight initially failed because a bound prior-evidence Markdown file was missing remotely; the coordinator preserved that supervision. No model fit resulted. Version2 separately repairs selected-model inference profiling: it removes optimizer state and other GPU graphs before measuring memory and explicitly labels the training peak that includes validation/checkpoint serialization. Scientific choices remain identical.

Each validation epoch records pooled micro-F1, pooled BCE, member BCE, equal-graph pooled/member BCE and their Jensen gap. Epoch0 is eligible. The member objective can improve by shrinking disagreement, so only pooled predictive gains pass the continuation gate. Late selected checkpoints with ongoing improvement flag a budget-limited study and block promotion. A positive development gate still requires prospective confirmation on a separate new dataset.

The same small private capacity is used by selected/random/raw; tied is the zero-capacity reference. Rank-zero seed1/10% stays in the study and should provide an identity control. This adapter retains all four full-width graph propagation paths; parameter additions and basis/optimizer buffers are reported without claiming activation or propagation savings.
