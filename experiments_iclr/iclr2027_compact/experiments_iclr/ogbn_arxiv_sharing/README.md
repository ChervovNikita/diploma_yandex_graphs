# Anonymous ogbn-arxiv sharing-position evidence, decision edition

The official OGB time split, original SAGE BatchEnsemble model, two-block sharing-position arms, and 300-epoch protocol are described in `experiments_iclr/ogbn_arxiv_sharing_protocol.md`. This graph was selected after the Roman-Empire, WikiCS, and Actor outcomes were known. This folder contains all 12 completed arms across three optimization seeds; no subset was selected by test performance.

Every `selected_decisions.npz` stores four member class decisions, the class decision after averaging their raw logits, the official labels, and official node IDs for validation and test. The arrays are deterministically derived from the full audited selected-logit files. `decision_derivation_manifest.json` binds each to its full-logit SHA-256 in the original run result. This compact edition can recalculate accuracy, paired differences, member accuracy, selected validation epoch, and retained hashes. It cannot reconstruct pooled predictions from omitted logits, cross-entropy, or checkpoint outputs. Full logits, checkpoints, and CUDA replay remain in retained author evidence. The public OGB dataset archive is omitted.

Run `python verify_decisions.py` with NumPy. The verifier checks all 12 run identities, source hashes, official node ID fingerprints, paired parameter equality, scores, traces, and stage file hashes.
