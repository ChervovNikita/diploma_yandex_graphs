# HGEN code-literal model and paired-HGB source proposal

This packet supplies an independent runnable **model/preprocessing adapter** for
the finite function defined by the pinned HGEN author code, plus an actual-author
synthetic CPU fixture for later root qualification. It supplies **no training
driver** and adopts no HGEN execution or training recipe. The author performed
only source/metadata/stdlib checks, with no Torch, SciPy model, original arrays,
labels, remote model, GPU or training execution.

## Settled source behavior

The author repository is `Chrisshen12/HGEN`, commit
`3caba805b2c3e16ee2dfd7d56d7e79405f66fd01`. Private original files are preserved
in the earlier native challenger source packet. Removing only orphaned model
lines355–359 makes its354-line model prefix parse without changing any active
class. The original training main is missing and remains unresolved.

`hgen_adapter.py` retains nine independent GCN cores: three members for each of
APA/APTPA/APCPA, author feature dropout, two native GCN layers, per-view learned
node-specific member fusion, summed view decoders and the pooled view Gram.
Unused `GCN_embed.dec` and `AttentionH.att` parameters are retained. The observed
code constructor defaults imply312525 total parameters for HGB author width334
and classifier4, including39000 unused parameters; this is stdlib-derived
architecture accounting pending root's actual-model count check.

The only settled objective is the documented code default **lambda_cov0**.
The code's squared L1 Gram value is returned, with zero objective weight. A
nonzero coefficient is rejected until a separately declared regularized profile
is qualified. Constructor defaults are observed source values, not a selected
published DBLP configuration or an adopted experimental recipe.

The min-max attention has no source-defined epsilon or zero-range fallback.
The adapter raises on a nonfinite or zero range. On the source's finite,
positive-range domain its predictor is unchanged. The source's asymmetric
residual is retained: member0 gets its min-max coefficient; later members also
get1/k. No unsupported smoothing or uniform residual is silently introduced.

The source executes an unused first pass before its used learner pass. The
adapter preserves every first-pass learner call and dropout draw under
`no_grad`, avoiding unused autograd graphs. GCN has no mutable running statistics.
The future actual-author fixture checks the finite outputs, all gradients and
global dropout state against the literal source's original two-pass behavior.

## Full HGB preprocessing and current paired splits

`hgen_inputs.py` converts the complete six-direction A/P/T/V graph into native
binary closed-metapath support. APCPA's conferenceC maps to HGB type3 venue.
Path counts and weights are discarded as in PyG `AddMetaPaths(weighted=False)`;
no nodes or edges are sampled. Boolean reassociation avoids a dense intermediate
while preserving exact reachability. Only author-provided features are used,
as defined by HGEN; all node types still contribute to metapath topology.

GCN adjacency uses destination rows, one unit loop per node and native
source-to-target incoming-degree normalization. Torch CSR multiplication avoids
E-by-hidden message tensors. Native PyG COO primitive correspondence and GPU
resource/runtime qualification remain separate root tasks.

`PAIRED_HGB_PROPOSAL.json` binds all five existing seeds131/137/139/149/151 and
their exact974 TRAIN/243 VAL descriptors to root's adopted native15 freeze,
SHA `274b293cefa8dc8257ede17a0e5e6b1fb2fa1b53392ce0a34227bf87fc46b87a`.
Its native adoption does not adopt HGEN. The future HGEN input scope is the same
complete archive and explicit development-only input, with classifier4 from
TRAIN. Model forward takes no labels; its objective takes only TRAIN IDs/labels.
No TEST label member, copied native PyG labels or TEST diagnostic is needed.

## Paper/code differences remain explicit

Saved method conclusions and exact saved equations establish:

| Choice | Paper | Pinned code |
|---|---|---|
| Fusion residual | Eq6 adds1/k for every member | Missing for member0 |
| Gram penalty | Eq9 uses unsquared L1 | Squared L1; default lambda0 |
| DBLP settings | Implementation states a hyperparameter grid was searched | CLI defaults exist; runnable main and selected DBLP recipe are missing |
| Selection | No complete recipe settled here | Pre-update train-mode VAL accuracy selects post-update weights; initial random fallback |

`PAPER_CODE_CONCLUSIONS.json` and `SOURCE_REPAIRS.json` record the exact mechanical
repair, finite-function decisions, methodological HGB/runtime adaptations and
unresolved choices. A paper-aligned regularized model, missing main invocation,
selected DBLP hyperparameters or changed validation rule must be a separately
named, prospectively frozen adaptation. No published score selects any value.
No full-paper-read or native numerical reproduction claim is made.

## Qualification

The stdlib source checks are:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verify_source.py
PYTHONDONTWRITEBYTECODE=1 python3 stdlib_checks.py
```

Future root CPU qualification uses an isolated environment with Torch2.1.2,
PyG2.6.1 and NumPy/SciPy:

```sh
PYTHONDONTWRITEBYTECODE=1 python cpu_fixtures.py
```

The fixture executes the actual354-line repaired author model prefix and actual
PyG `GCNConv`/`AddMetaPaths` on fabricated data. It compares binary supports,
normalization, isolated-node loops, logits/Gram/embeddings, every parameter and
input gradient, unused-parameter accounting, matched dropout consumption and
zero-range failure behavior. It performs **zero optimizer steps** and opens no
original arrays/labels. PyG2.6.1 source is pinned at
`7ca40d635c4757f53ec2a1260e4a9e6505f8ad49`; dependency source and MIT license are
saved. A compatible pure Python PyG installation with Torch sparse support may
avoid native extensions, but root must establish that environment and receipt.

The existing root native3-model CPU correspondence receipt is only a bound
metadata input. It is not reused as HGEN qualification and is not rerun here.
Full-graph native3-model CPU resource qualification remains root-owned next work.

## Private source scope

No HGEN license grant was located. Raw author files remain private and
unpublished in the preserved earlier packet; this new packet stores references
and independently authored code. No original external HGEN file is copied here.
