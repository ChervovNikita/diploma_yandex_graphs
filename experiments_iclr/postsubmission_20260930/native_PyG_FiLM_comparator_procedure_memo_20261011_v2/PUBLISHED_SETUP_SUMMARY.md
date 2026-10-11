# Published GNN FiLM setup relevant to the comparator

The official ICML2020 paper is Marc Brockschmidt, GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation, PMLR119, pages1144–1152. The downloaded official PDF is 308971 bytes, SHA256 `c8d4f60b19880ea5118e2612688f1094b616fe2e85ce6c1ac7570424c853b0a3`.

Sections2.1 and3.2 explicitly identify Eq8, with per-message nonlinearity before aggregation, as the evaluated GNN-FiLM formulation. They permit a post-propagation layer combining bounded activation, an affine layer, LayerNorm or their combination. Separate layer weights performed better than propagation with tied layer weights in the inspected setup.

Section3.1 reports experiments on PPI, QM9 and VarMisuse. It describes preliminary citation-graph experiments as sensitive to initialization and warns against transferring good results between tasks. This scope supplies no published WikiCS recipe.

Section3.2 describes a TensorFlow implementation, task defaults found by search in `tasks/default_hypers/`, early stopping after25 unimproved epochs for PPI/QM9 and5 for VarMisuse, and aggregation over separately initialized runs. The original TensorFlow code and task defaults were not audited here.

The inspected PPI setup uses20 training graphs and separate two-graph validation and test sets, with dataset edges plus a fresh self-loop relation. Its selected GNN-FiLM configuration uses four propagation layers, representation width320 and node dropout keep ratio0.9. These PPI settings are not defaults for the proposed WikiCS adaptation. No reported paper score or speed is transferred.

Only PDF pages4–5 and the relevant setup portions of page6 were read for this summary. Those pages were visually inspected; Table1 and nearby results on page6 were incidentally visible, with no table audit or reproduction claim. The complete nine-page PDF and mechanically extracted text are custody carriers, not full-paper reading credit.
