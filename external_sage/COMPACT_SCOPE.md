# Compact audited study

This folder contains the frozen training and full-verifier source, the source/data manifests, all 300-step validation traces, selected member logits and labels, and complete GPU replay audit records. Public raw graph files, selected checkpoints, and initial-logit tensors are omitted from this upload. The full verifier requires those omitted files and a CUDA GPU. Re-running the frozen study on the pinned public data creates them. The saved member logits suffice to recalculate all tabulated test accuracies and member-versus-pooling decompositions.
