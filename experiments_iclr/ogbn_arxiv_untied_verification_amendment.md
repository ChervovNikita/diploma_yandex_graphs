# Verification-only amendment for the ogbn-arxiv tying control

The six-arm, three-seed, fixed-300-epoch training completed under the pre-run source lock with SHA256 `cd2da82e80f489c7558284937a62f9e5eab4c349df68e65553a4f7ff3c4df469`. Its output artifact manifest has SHA256 `78e1aa7af58a343b1cf024413ed1f6f9d4b8081e0209c8dee75b5c8e10808f17`. No training source, selected result, checkpoint, prediction array, protocol, or source lock was changed after the runs.

The original verifier, SHA256 `fa2515f3f0eebafb4ffc0f5f09cf9675e6595c4d5e17ac49c011227cd1effb1e`, failed after checking the result tree, artifact hashes, source lock, dataset, and the first tied arm. It looked for `untied_propagation_rng_start_sha256` in an initialization record that stores the field as `untied_rng_start_sha256`. This was a verifier field-name error, not a training failure. The original failed log remains in `experiments_iclr/logs/ogbn_arxiv_untied_verify.log` on the training system. The original one-shot launcher exited without its completion marker.

A separate verifier copy, `verify_ogbn_arxiv_untied_v2.py`, SHA256 `0388198e3aa8b9924762d61c6eeaa92ff49b384fdfa9fa8d97df5ffa50ddf3cb`, changes only that lookup: when the arm is `untied_propagation`, it reads `untied_rng_start_sha256`; the tied arm continues to read `tied_rng_start_sha256`. The pre-run verifier and source lock remain intact. The corrected verifier ran the full six-arm CPU checkpoint and logit replay and returned `status=complete`, `verified_arms=6`, and `verified_seed_pairs=3` at 2026-09-25 18:31:35 UTC. The separate marker is `experiments_iclr/logs/ogbn_arxiv_untied_v2.complete`. The amended verification is post hoc and should be disclosed with the result.

To repeat the complete read-only gate from the repository root with the official OGB data present, run:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python experiments_iclr/verify_ogbn_arxiv_untied_v2.py
```

The script checks the unchanged pre-run source lock and every recorded artifact hash before restoring all six selected checkpoints and comparing their logits with saved predictions. It does not retrain any model. These three seeds share one temporal graph split; their sample standard deviation measures optimization variation, not variation across graph tasks.
