# Normalized Cora matched-parameter inference profile

This separate post hoc profile uses the three validation-selected seed-0
checkpoints from `../cora_matched54`. It ran on GPU0 after both training
partitions and the complete independent score audit finished. Each arm uses
three reordered rounds, 20 warmups per round, and 100 synchronized full-graph
forward timings per round. The JSON retains all 900 samples, model weight
counts, and peak allocated CUDA memory. Timings exclude model loading and
host transfers. They are device-specific measurements, not universal latency
guarantees. The original checkpoints are omitted from the public compact
bundle; the profile source and selected checkpoint hashes are included.
GPU UUID fields in the public JSON are redacted; the redaction audit records
the original author artifact SHA-256 and confirms the retained values.

Run with Python 3:

```sh
python3 verify_cora_matched54_profile.py . ../cora_matched54
```
