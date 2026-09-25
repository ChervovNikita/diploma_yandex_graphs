# Checkpoint-to-prediction inference audit

This read-only audit complements `verify_control_artifacts.py`. The latter checks whether stored predictions and metrics agree with each other. This script asks a separate question: does each stored test prediction come from the selected checkpoint, on the pinned official Roman Empire graph, under the declared SAGE architecture?

From the repository root, use the repository-local environment:

```bash
.venv/bin/python -u experiments_iclr/verify_control_inference.py --threads 2
```

The default is a partial audit of every row present when the CSV is opened. For the final audit after the 50-row matrix and the GNNM split-0 adoption decision are complete:

```bash
.venv/bin/python -u experiments_iclr/verify_control_inference.py --require-complete --threads 2
```

A focused check accepts `--variant gnnm --split 0`; repeat either option to select several variants or masks. `--max-rows 1` is useful for a smoke check. The final command must omit filters and `--max-rows`. The anonymous bundle builder runs this complete check as a required gate before it packages any results and allows up to 1,200 seconds for a read-only verifier.

The script first validates that the CSV has unique keys from the ten declared variants and five official masks. It checks the model, depth, width, learning rate, member count, maximum steps, seed, and selected step. It verifies the local Roman Empire NPZ against the pinned SHA256 in `data_manifest.json`, then uses the repository dataset loader with the same bidirectional graph and self loops as training. Each CSV checkpoint and prediction path must stay inside the expected repository subdirectory and have the expected stem. If a fresh GNNM split-0 checkpoint was adopted, the checker follows that CSV path and verifies the row and both artifact hashes against `gnnm_split0_verification_audit.json`. Final mode requires that audit.

For each row, the script rebuilds the model from the declared variant and configuration, compares its trainable parameter count with the CSV, and strictly loads every checkpoint tensor on CPU. It sets evaluation mode and does one full-graph forward pass per member without gradients. It takes logits at the official test nodes and requires exact node IDs, labels, and shape relative to the saved NPZ. It also checks saved member decisions against saved logits. Every saved logit must meet

`abs(fresh - saved) <= 1e-4 + 1e-4 * abs(saved)`.

This allowance covers small CPU versus CUDA floating point differences. A raw-logit mismatch stops the audit with the variant, mask, number of mismatching logits, and worst coordinate. If CPU and CUDA choose different classes only because two logits are within their combined error allowance, the row passes and prints the count and largest saved margin of these near ties. A wider decision change fails. The saved GPU logits and decisions remain the experiment record. The script prints one JSON line per successful row and a final summary. It writes no result files and does not run training or select checkpoints by test performance.

This check cannot establish that the selected checkpoint was trained exactly as claimed or that the checkpoint was selected without test inspection. The queue's frozen source and split-0 adoption records address those separate provenance questions. The audit also covers only saved test nodes. It does not check logits on train or validation nodes because they were not archived in these NPZ files.

On September 25, a two-thread CPU smoke check took 2.9 seconds of forward inference for a single-member BASE row and about 11.6 seconds per four-member GNNM or independent-projector row. Startup and model loading add time. At that rate, all 50 rows need roughly 8 to 12 minutes when the server load is similar. This is an estimate, not a measured full-matrix runtime.
MD'