# Startup environment recovery

The three v3 launch attempts exited before importing NumPy or constructing a model. Root omitted the existing native dependency-overlay PYTHONPATH from the new owner configuration. They performed zero scientific updates and produced no model outputs. Their exit1 logs and owner receipts are retained.

This explicit v4 launch restores the exact previously used environment, keeps the reviewed scientific source, three seeds,1100epochs and selector unchanged, and uses fresh outputs. It is a recorded startup recovery, not an automatic rerun of a paid training result. Strict FP32 parity remains waived.
