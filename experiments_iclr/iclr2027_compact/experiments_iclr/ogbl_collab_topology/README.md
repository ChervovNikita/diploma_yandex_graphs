# Deterministic topology baselines on ogbl-collab

Run from the anonymous archive root:

```sh
python experiments_iclr/ogbl_collab_topology/verify_public.py
```

This checks the complete eight raw official-pool score arrays, all four strict Hits@50 metrics, source hashes, frozen graph/split fingerprints, and saved full replay audit. It needs NumPy. It does not regenerate the scores from the graph. The original independent verifier did regenerate every score using a second intersection algorithm before this public copy was made.

For fresh graph regeneration, use the preserved `experiments_iclr/` source subdirectory as the script directory in a fresh copy of this study root, with the official `collab.zip` at `data/ogb/collab.zip`. Use the original runner's dependencies (PyTorch, PyG, OGB, NumPy and SciPy), one CPU thread, and `CUDA_VISIBLE_DEVICES=''`. Verify the download's exact SHA against the freeze. The original runner must see an absent result directory. Retain these submitted results in the evidence copy. Then run `ogbl_collab_topology_baselines.py check-freeze`, `ogbl_collab_topology_baselines.py run`, and `verify_ogbl_collab_topology_baselines.py`. These source scripts are byte-identical to the frozen experiment.

Both Common Neighbors and Adamic–Adar were fixed after the learned link results were known and before these scores were computed. Neither uses training, tuning, features, validation-edge augmentation, or test-edge augmentation. Both results are reported. They are graph-only context, not capacity-matched learned models. A shared negative-pool threshold uses strict greater-than comparison, so ties are misses. Adamic–Adar exceeds the learned tied-model mean in this fixed 400-step study.
