## Expanded benchmark and representative training — 15 October 2026

The scientific target is improved prediction quality with a shared backbone.
Separate paper-reference single-model reproductions, matched single models,
base GNNM and genuine independent ensembles remain the common comparisons.
The 2,480-fit proposal below is preserved history while the method catalogue
is broadened; it is not the final expanded count.

**A frozen 24-fit molecular pilot is running on the authorized one-GPU
allocation.** It compares base GNNM, private nonlinear input corrections,
class-aware graph embedding supervision, and their combination, plus four
independent native GINs per trial. Three paired seeds use the official MolBACE
split and unchanged OGB model/LR/WD settings. Each fit has a 1,000-epoch cap,
minimum100 and patience50, with whole-VALID selection and no TEST scoring.
No new accuracy improvement is claimed before complete paired results.

Complete-epoch node/molecular training and independent checkpoint ownership
are implemented. Graph-level contrastive and private-adapter code is prepared;
query-specific link contrast/responsibility code still needs native full-epoch
integration. These are attributed hypotheses, not established novelty.
The broader campaign remains disabled pending its exact design and user GO.
Only the dedicated allocation is used; large research artifacts stay there.

- [Current methods, comparisons, pilot and remaining work](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/README.md)
- [Frozen 24-fit pilot](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_molbace_v1/SPEC.json)
- [Full-epoch runtime](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/runtime/API.md)
- [Saved literature conclusions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/literature/SCOPED_CONCLUSIONS.md)

The dated sections below record earlier states and proposals.
