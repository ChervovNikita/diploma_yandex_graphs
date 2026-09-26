# Roman bridge pooled evidence

All nine fixed Roman rows retain exact float32 pooled logits and member hard classes on validation and test nodes. The public verifier recomputes pooled accuracy and cross-entropy, member accuracy, checkpoint selection from the validation trace, source hashes, and links to the original member-logit hashes. It cannot reconstruct pooling from omitted member float logits or rerun checkpoints. The complete original artifacts remain in the author audit.
