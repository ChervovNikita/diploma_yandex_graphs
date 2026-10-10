# Actual full-TRAIN wrapper integration

All nine original GCN/GAT/SAGE × baseline/separable/exchange cases completed on the authorized one-GPU allocation. Each used the full PubMed graph and 11,829 TRAIN labels, performed one backward/AdamW update, and saved/restored its registered model, optimizer and member streams. Native initial-function checks and restored-forward checks passed. No validation or test predictions were scored.

Worker time: 11.05 seconds. Peak reserved GPU memory: 8.15 GB. The exact report and verified process/CUDA closure are in `ACTUAL_REPORT.json` and `ACTUAL_CLOSURE.json`. Nine checkpoints totaling 243,034,066 bytes remain inside the server repository; none was downloaded to this Mac or added to Git.

The first transfer attempt ended before any source transfer or numerical launch. `transfer_and_check_attempt1.py` and `ACTUAL_TRANSPORT.json` preserve it. The reused PTY transport required three positional arguments before its payload-length argument. The corrected transfer, actual child completion and source hashes are in `ACTUAL_TRANSPORT_V2.json`.

This is integration evidence. It establishes no accuracy improvement, methodological novelty or manuscript acceptance.
