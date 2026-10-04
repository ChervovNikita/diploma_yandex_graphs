# What EMR-GNN ensembles

The primary method combines relation-specific adjacency operators into one learned graph propagation. It alternates a relation-weight update with a representation update, retains original node features through restart, and uses one downstream classifier. Its ensemble terminology does not denote a bank of independently selected classifiers.

This is relevant prior for heterogeneous extensions: relation weighting, decoupled graph propagation and compact learned relation coefficients already exist. Adding separate relation coefficients to GNNM routes would need a useful discriminator beyond those ingredients and a competitive relational backbone. No new research direction is promoted solely from a difference in parameter tying.

The exact primary method scope is blocks14–36 in the retained arXiv2205.12076v1 extraction. Selected introduction, recipe and results paragraphs were also exposed and are listed precisely in READ_SCOPES.json. Numeric results are not adopted. This is one scoped method read, not a full-paper or source audit. Algorithm1 and the appendix pseudocode were not inspected, and the complete theoretical claims were not certified.

The paper uses MUTAG as a relational node-classification graph; it must not be presented as graph-level molecular classification. Existing saved GRE notes and this method scope are distinct operations. No prediction data, test labels, models, training or server execution were used.
