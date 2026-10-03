# Prepared native HGB-DBLP challengers

This packet provides runnable independent Torch adapters for **GAT, Simple-HGN
and SeHGNN**, plus complete-data development training, source correspondence
fixtures and the exact HGEN author-source qualification. It is **unadopted**.
The author ran only stdlib checks on fabricated records; numerical qualification
remains an independent root CPU step. No original dataset or label bytes were
opened by the author, and no main training or GPU job was launched.

## Native recipes and declared adaptations

GAT and Simple-HGN retain all-type identity features, the released undirected
union with one self loop per node, two hidden convolutions and one output
convolution, hidden width64 per head, heads8/8/1, dropout0.5, ELU, slope0.05,
Adam5e-4/weight-decay1e-4, 300 maximum epochs and native CE/patience30 selection.
Simple-HGN retains edge embeddings64,13 edge categories, feature residuals,
detached residual attentionalpha0.05 between hidden stages, the output-stage
attention reset and final L2-normalized class logits. Ties replace and reset.

SeHGNN retains all provided attributes and venue identity, five native two-hop
feature channels and four native four-hop label channels, embed/hidden512,
two feature projections, three task layers, one-head Transformer with gamma0,
residual features, dropout/input-drop0.5, Adam1e-3/weight-decay0, batch10000,
200 maximum epochs, native CUDA TRAIN AMP and strict best-CE selection.
Its literal stop condition is epoch minus best epoch greater than50:51
nonimproving epochs. CPU execution disables AMP as the author command does.

The model cores are independent pure Torch ports, avoiding the installed GPU
DGL import failure (`libcusparse.so.11`). CPU SciPy sparse products replace
SeHGNN's torch_sparse preprocessing. The fixture qualifies path mathematics
using the actual author extension function with a dense reference operator;
it does **not** assert identity of native PyG kernels. Actual DGL mean-feature
propagation is tested separately. Identity input projection avoids dense identity
matrices, and edge-message temporary tensors are limited to32768 edges per chunk.
Floating-point reduction order on a GPU is not certified by CPU correspondence.

The private root protocol uses the previously proposed paired seeds
131/137/139/149/151 and exactly the same byte-bound split descriptors as the
adopted HGT family. Constructor RNG draws of the independent ports differ from
the released constructors; same-seed numerical reproduction is not claimed.

## Development and TEST separation

The sealed v2 loader opens only `DBLP/node.dat` and `DBLP/link.dat`. This packet
adds one further node pass to retain P/T attributes. Its synthetic ZIP spy check
verifies that neither label member is opened. Development labels are a separate
root-owned `TRAIN_VAL_ONLY` JSON descriptor, binding original label.dat bytes.
All974 TRAIN and243 VAL cases are retained for every seed; classifier4 is checked
from TRAIN. No TEST label member is opened and no TEST diagnostic is computed.

SeHGNN uses TRAIN-only one-hot propagation and removes the diagonal **after each
complete normalized metapath product**, without subsequent normalization.
Native destination-first SeHGNN matrices follow raw file rows and reverse HGT's
source/destination convention. No raw direction is silently substituted.

SeHGNN's final class BatchNorm has `track_running_stats=False`, so evaluation
composition is frozen: sorted TRAIN IDs, then sorted VAL IDs, then the sorted
target complement obtained from topology, all4057 authors in one batch. This
matches the released complete-DBLP batch composition without loading withheld
labels or computing their metrics. Only VAL logits and metrics are saved.
Root must confirm this protocol before execution.

## Qualification and execution

From this directory, the author stdlib checks are:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verify_source.py
PYTHONDONTWRITEBYTECODE=1 python3 stdlib_fixtures.py
```

Root numerical qualification requires Torch2.1.2, NumPy/SciPy and the already
qualified isolated CPU DGL1.1.3. Use root's existing environment with that DGL
site prepended; do not modify the working GPU environment. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python cpu_fixtures.py
```

This executes preserved actual HGB GAT/Simple-HGN models, preserved legacy
DGL0.4.3 GATConv through modern CPU DGL, and the preserved dense DBLP SeHGNN
model. The only model portability replacement is Simple-HGN's hard `.cuda()`
epsilon allocation; the original bytes remain unchanged. SeHGNN's unused
SparseTensor import is removed in memory for its dense branch. Tests include
logits and every parameter/input gradient, matched dropout consumption, native
identity projection, DGL feature means, complete metapath products, diagonal
removal, fixed evaluation composition and24 fabricated optimizer steps with
exact three-step continuation after weights_only model/Adam/global RNG loading.
This does not run the main driver or a released dataset.

Root then fills `FREEZE_TEMPLATE.json` into a separate prospective freeze:
archive path, development descriptor, five split descriptors, GPU UUID,
`paired_HGT_freeze` descriptor and `study_adopted_by_root=true`. Native and HGT
archive/development/split bindings must be identical. Root's admission JSON
must bind `study_freeze_sha256`, `prepared_manifest_sha256`,
`paired_HGT_freeze_sha256`, `execution_authorized=true`,
`CPU_fixture_passed=true`, `run_name`, and `device`.

```sh
python train_native.py --freeze /root/frozen_native.json --admission /root/admission_native.json --run-name native01 --device cuda:0
```

The driver verifies all sources/inputs before importing Torch. It runs all15
cases, retains complete selected model/Adam/AMP/global RNG state and replays
selected VAL logits. Each failed or deferred case remains a terminal record.
No successful subset is summarized as the frozen study. One packet does not
automatically start a replacement study. HGT/native comparison and the later
separate heldout release remain root-owned work.

## HGEN remains a required, unqualified modern challenger

The saved index_v26 and primary metadata point to
[Chrisshen12/HGEN](https://github.com/Chrisshen12/HGEN). A scoped source retrieval
pins commit `3caba805b2c3e16ee2dfd7d56d7e79405f66fd01`; no dataset or masks were
downloaded. The complete repository has five files and no license grant.

`HGEN_QUALIFICATION.json` binds source defects and the intended DBLP recipe:
model.py fails parsing at355, train.py at302, main has no runnable body,
lambda_cov defaults to0, and native loading uses PyG DBLP fixed masks rather than
the paired HGB development protocol. Requirements target Torch2.4/PyG2.6.1 and
CUDA12.4 extensions. No native runtime qualification or recipe repair is claimed.

HGEN constructs independent GCN learners for APA/APTPA/APCPA graphs, fuses them
with learned node-specific member weighting, sums view decoder outputs and
optionally penalizes the Gram matrix of pooled view embeddings. There is no
shared HGT core or member-by-raw-relation residual factor table. Its source also
contains an unused extra learner pass and an unguarded min-max fusion denominator.
The qualification records a separate prospective repair/HGB adaptation plan;
scientific choices are not silently repaired or dropped from the challenger set.

## Source and publication scope

Preserved author references retain their original unresolved license scope.
No HGB NC, SeHGNN or HGEN redistribution right is inferred. In particular,
`hgen_source/` contains author source and repository metadata for private review
and must remain outside publication artifacts. No fitted published score is
used to select a model, setting or experimental result.
