# Native link final-serving source audit

No native main or inference definition was executed during this audit. Source
paths below are relative to the phase directory unless prefixed `primary/`.

## NCN and NCNC

`reference_wrapper_redesign_20261013_v1/native_sources/ncn/ncn__ogbdataset.py`,
`loaddataset`: official processed TRAIN graph, weights removed; TRAIN adjacency
symmetrized/coalesced. When `use_valedges_as_input`, concatenate original graph
and raw positive VALID edges; coalesce then symmetrize for `full_adj_t`.
No TEST query edge enters that adjacency.

`.../native_sources/ncn/ncn__NeighborOverlap.py`, `test`: source TRAIN and
VALID predictions use TRAIN adjacency. TEST switches to full adjacency and
recomputes native embeddings, then predicts positive and negative TEST pairs.
`PermIterator(...,False)` retains every official query in order. Hits50 is a
direct OGB evaluator call over full positive/negative vectors. The Collab
author command in the retained README specifies use_valedges_as_input and
testbs131072. The standalone backend preserves the TEST-relevant graph and
predictor operations but does not rerun the source's preceding TRAIN/VALID
diagnostics or claim full-main RNG equivalence.

## BUDDY source recovery

The first filename search found no local BUDDY `src/runners/inference.py`.
`cached_structure_link_ensemble_feasibility_v1/SOURCE_PINS.json` binds repository
`https://github.com/melifluos/subgraph-sketching` at commit
`3562d94a07d1166faa0949030824bf75ad9bb2c4`. Its saved recursive tree contains
the exact inference, utility and evaluation blobs. The three files were fetched
from that commit into `primary/`, with size and Git blob SHA1 independently
checked before writing. Acquisition URLs, original tree pin and SHA256 receipts
are in `primary/ACQUISITION.json`.

`primary/buddy_src_runners_inference.py`, `get_buddy_preds`: read the dataset
from the outer loader, concatenate source positive/negative supervision
population already stored as `data.links`, create a new sequential DataLoader
with eval_batch_size1,000,000, select source cached structural features,
propagated node features and weighted degrees, and call the unchanged model.
The function returns raw logits and splits them by source labels. Dynamic
sampling defaults are disabled by the pinned Collab recipe. Neither sigmoid
nor private posthoc fitting appears in this path.

This corrects the incomplete inference audit underlying the frozen policy:
outer Collab VALID/TEST loaders are configured to shuffle, but this BUDDY
function does not iterate them. It replaces them locally with an unshuffled
range loader. Source TRAIN/VALID/TEST each still invoke prediction and internal
loader construction; skipped calls and saved constructor/RNG differences are
not assumed neutral. The frozen policy and integration files remain untouched.

`primary/buddy_src_evaluation.py`, `evaluate_hits`: set evaluator.K and call
`evaluator.eval` on each role's full raw positive and negative vectors. The
prepared final evaluator performs the same TEST metric operation at K50.

## BUDDY role-specific graph and cache

`.../native_sources/buddy/buddy_src_data.py`, `filter_by_year`: filter TRAIN
edge/year/weight together at2007; construct weighted undirected TRAIN with
reduce add. `get_ogb_data`: TEST only adds independently undirected VALID
edge/weight to filtered weighted TRAIN. The prepared orchestration copies this
narrow concatenation without calling the unrelated negative-cache generator.
Missing VALID weights fail.

`.../native_sources/buddy/buddy_src_datasets_elph.py`, unchanged `HashDataset`:
coalesce Collab graph/weights, weighted CSR degree vector, source SIGN-k0
feature propagation, source MinHash/HLL build and whole-query structural feature
cache. The source cache name includes split and year but cannot by itself
authenticate graph/query identity. The backend therefore requires a fresh,
project-contained cache directory for each role request and records full
source/input/cache provenance. It does not accept a stale cache solely because
its shape matches. Cached TEST features and hashes arise from TEST topology,
not from the TRAIN/VALID graph.

`.../native_sources/buddy/buddy_src_hashing.py`: source MinHash seed1 and
original HLL/minhash routines; full-node role graph; two-hop structural features
retain fixed floor_sf/use_zero_one behavior. No alternate hash, graph view,
query subset or feature approximation is introduced.

## Prepared execution boundaries

The backend loads only byte-pinned own selected checkpoints from completed
native epochs. It restores the exact frozen integration/model/boundary state,
never steps an optimizer, and supports four separately selected native I4
bodies. Input/provider/selected/runtime/source receipts and actual VALID parity
must be supplied before final TEST GO. The current folder has no actual TEST
query receipt, final selected-state manifest or numerical qualification report.
