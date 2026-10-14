# Molecular objective: scoped primary reading

15 October 2026. Five relevant primary papers were read in the scopes below.
**Zero full-paper reads, author-code audits or new empirical literature checks.**
Saved HTML is the retrieved primary manuscript, not a review summary; hashes and
URLs are in [RETRIEVAL.json](RETRIEVAL.json). No experimental table from these
papers is evidence for the proposed shared molecular method.

| Primary paper | Abstract scope | Additional section scope actually read | What is adopted / what differs |
| --- | --- | --- | --- |
| Khosla et al., [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | Read | Sections3.1–3.2.2, including Eq2 versus Eq3 and positive/nonself denominator semantics | Average positive log probabilities. Here views are existing member routes, vectors are native graph preheads, and source BCE remains simultaneous; no image augmentation, projector or frozen-encoder classifier stage. |
| Zhu et al., [Balanced Contrastive Learning for Long-Tailed Visual Recognition](https://arxiv.org/abs/2207.09052) | Read | Section3.3, Eq6–11 and framework prose | Class-averaged exponential denominators inform A. Complete BCL also has classifier-derived prototypes/class-complement, its projection branch and logit compensation; none is claimed or copied. Missing classes instead gate A off. Anchor classes are additionally balanced here. |
| Bardes et al., [VICReg](https://arxiv.org/abs/2105.04906) | Read | Section4.1, Eq1–7 and surrounding method description | The standard-deviation hinge informs V. Original VICReg acts on unnormalized expanded image features and includes invariance and covariance terms. Here V is conditional on observed graph class, normalized by its target, and acts on normalized native features without covariance or an expander. |
| You et al., [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902) | Read | Sections3.1–3.2 prose: graph augmentations, graph readout/encoder, projection and minibatch contrastive setup; not appendix derivations | Graph-level targets and augmentation priors are relevant. This extension uses intact molecule graphs, no graph augmentation and no extra projection; it is not GraphCL pretraining or a guarantee that arbitrary graph perturbations preserve molecular labels. |
| Wang et al., [MolCLR](https://arxiv.org/abs/2102.10056) | Read | “MolCLR Framework” and “Molecule Graph Augmentation” including Atom Masking, Bond Deletion and Subgraph Removal | Confirms molecule-level average readout and distinguishes same-molecule augmentation from task-label supervision. Here no unlabeled pretraining corpus, molecule edit, new feature token or fine-tuning head is used. |

An initial locator `2007.08025` retrieved a different paper named “GraphCL:
Contrastive Self-Supervised Learning of Graph Representations.” Only its title
and section inventory were inspected. It is retained as a locator trace and is
**not** counted as a method read or substituted for You et al. `2010.13902`.
Plain-text extracts are reading aids; primary HTML remains authoritative for
equations. The section inventory initially generated for the wrong locator is
also preserved rather than silently represented as the augmentation paper.

## Retained project scopes read

- `contrastive_member_geometry_user_clarification_20261010.md`: within-member
  label alignment versus cross-member negatives; embedding spread is not useful
  complementarity. Saved negative findings are not an exclusion of molecular
  objectives and were not reanalyzed.
- `contrastive_BE_graph_training_rule_synthesis_20261008_v1.md`: fixed
  class-preserving structural credit and shuffled-assignment controls; no degree
  bins or node-specific loss is transferred into this molecule objective.
- `shared_backbone_decision_visible_contrast_assessment_20261007_v1/NOTE.md`:
  classifier-null residual construction and its limitations. This is why V is
  not promoted as functional diversity or a no-harm constraint.
- `P_internal_BE_wrong_class_prediction_contrast_assessment_20261010_v1/METHOD_ASSESSMENT.md`:
  attributed ADP/CDLG ancestry and exact binary degeneration. The proposed
  wrong-class-output geometry cannot supply a MolHIV/MolBACE contrast signal.
- `method_explanation_20261013_v1/method_families_note.md`: selected
  contrastive/residual/GNCL/specialization entries, not an exhaustive reread.
- `portable_internal_be_public_interface_20261007_v2/core/objectives.py`:
  existing alignment/residual pairing and reductions. Those native node/two-view
  loss functions are not imported by the new module.
- Frozen OGB `gnn.py` and `categorical_boundary.py`: native graph pooling/head
  and the live prehead return site. No source modification was made.

The new family is an attributed composition with an explicit small-batch,
binary/missing-label reduction. No priority claim, exhaustive literature search,
MI interpretation, or demonstrated molecular benefit follows from this reading.
