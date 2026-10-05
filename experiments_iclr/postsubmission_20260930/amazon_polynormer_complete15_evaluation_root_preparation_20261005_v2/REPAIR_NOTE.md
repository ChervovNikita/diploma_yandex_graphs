# Correct evaluation device

The first evaluation release copied `device=cpu` from the metadata-only closure while binding the CUDA runtime receipt. The unchanged runtime check stopped its worker before numerical evaluation. Original fit checkpoints, original outputs and source remain intact.

This manually reviewed successor changes the declared device to `cuda:0` and uses a fresh immutable release/output. It preserves the failed release and receipts, has no automatic retry, and runs the same original evaluator. No training or original manuscript scores are repeated or replaced.
