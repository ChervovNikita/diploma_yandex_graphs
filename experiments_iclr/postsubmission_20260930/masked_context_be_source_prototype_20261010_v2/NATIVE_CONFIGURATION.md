# Native PubMed configuration and admission gaps

Source: air029/PolyFormer at d390f39e88d0eaac80318fdc7704bd3bf3cf8b13.
Paper: Ma, He and Wei, KDD2024, DOI10.1145/3637528.3671849. Existing primary
paper/source readings supply this provenance; this packet adds no new paper read.

`node_classification.sh` gives two PubMed configurations:

| Field | Author monomial | Author Chebyshev |
|---|---:|---:|
| K | 2 | 6 |
| hidden width | 256 | 128 |
| blocks | 2 | 1 |
| heads | 8 | 16 |
| FFN width | 32 | 64 |
| q | 1.6 | 2 |
| expansion | 2 | 2 |
| dropout | .5 | .30000000000000004 |
| attention/FFN dropout | .8 | 0 |
| base lr | .005 | .0005 |
| base weight decay | .001 | .001 |
| attention lr | .0005 | .0005 |
| attention weight decay | 1e-8 | .0001 |

Both inherit2000 maximum epochs, patience250 and validation accuracy from
`training.py`. Source uses strict `val_acc > best_val_acc`, initialized at0;
there is no best-checkpoint restoration in its returned-score code. The source
also scores TEST every epoch. A clean prospective runner must instead retain a
strict-first actual selected checkpoint and keep TEST excluded until separately
authorized confirmation. Those are explicit source-port choices to freeze, not
an executed reproduction of the author's training script.

The Planetoid loader normalizes features, makes the graph undirected if needed
and precomputes the polynomial bank. PubMed TRAIN generation chooses
`round(.6*N/3)` targets from each class using NumPy RandomState(split_seed), then
`round(.2*N)` validation rows from the remainder. With N19717 this gives11829
TRAIN,3943 VALID,3945 TEST. The ten author split seeds are1941488137,
4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,
3212139042,2424918363. These are separate from optimizer seeds9101/9203/9307.
No split was generated here. Only the projected TRAIN role enters this qualifier.

The pinned Polynormer loader lacks PubMed, so this packet cannot assert an exact
native PubMed Polynormer recipe. PolyFormer is a documented capable modern
provider; engineering and reference competence remain to be established for
the selected protocol. The author60/20/20 differs substantially from the original
60-label official split and cannot be silently relabeled as that proposal.

Root must freeze before any scientific outcome:

- official split versus the explicit native split amendment and exact split seed;
- feature normalization/decoder target identity, full graph edge policy and hashes;
- ordinary native configuration, factor sites and all optimizer groups including decoder;
- native early-stop versus a complete fixed2000-update budget, checkpoint storage,
  strict-first selector and independent own-member selectors;
- modern reference competence evidence and source/runtime engineering qualification;
- A stage1 quality margin, accuracy and NLL directions, mean/worst member safeguards,
  all9 completion gate and treatment of resource failures;
- the exact stage2 rewire count/rejection cap/canonical graph recipe;
- developmental exposure history, later untouched-task/split confirmation and cost caps.

This source fixes only one monomial recipe. It exposes no width, location,
coefficient, temperature, projection or strength grid. A private-decoder or
independent-start extension would be another frozen generation, not a hidden
change to the27-record proposal.
