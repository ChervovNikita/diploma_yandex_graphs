# Original boundary GNNM on binary molecular graphs

This is a source-based architecture/training-law adaptation to new graph
tasks, not replication of the original published node tasks or their scores.
The original five GNNM datasets remain excluded. Existing molecular v3 and
all of its provider/split bytes remain immutable.

`adapters/molecular_boundary.py` implements explicit arm
`original_gnnm_binary_graph`. It instantiates the pinned original `BEBlock`
from `common_wrapper_native_source_20261010_v1/models.py`:
10,977 bytes, SHA256
`07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`.
The pinned native member-own function is in `source_snapshot/run_tabm.py`:
11,240 bytes, SHA256
`b1d748e50c854802b93f4d4fa7485f096daa1cd36d3dd3bd1c89fbfa63cbd124`.

## Retained source laws

- Four private R/S/B rows at the input and output boundaries only; shared
  bias-free Xavier W. First input R is Rademacher; later R and all S are1;
  member-private B starts0. These are the actual source constructors.
- One shared plain slow GINE body, with no internal FactorLinear maps or
  member message factors. Four factual member forwards propagate their
  different boundary inputs through this shared nonlinear body.
- One own BCE forward per member/minibatch, each divided by4 and separately
  backpropagated before one AdamW update with weight_decay0. The source native
  training function already includes the binary BCE branch.
- Inference averages the four raw logits, then applies sigmoid. Fixed scalar
  CAL is applied to this pooled logit afterward.

The existing `be_init` molecular arm factors internal dense maps and message
channels and uses a different bias law. It is not relabeled as this reference.

## Declared graph-task changes

The input boundary operates after the shared OGB AtomEncoder, before the
shared residual bond-aware GINE stack. Raw categorical atom/bond indices are
never sign-flipped or transformed outside their vocabulary. The output
boundary follows graph pooling and has one scalar binary output per graph.
Shared bond encoders, residual GINE, virtual node, LayerNorm and graph mean
pooling come from the exact archived molecular body. Its actual readout is
`global_mean_pool`; `global_add_pool` is used only in virtual-node updates.
The catalogue's readout add-pool wording is superseded by this source binding.

Construction first builds the archived molecular body and then instantiates
its two native boundary blocks; discarded ordinary input/head maps are not
optimizer parameters. This constructor order, private dropout streams,
graph minibatches, official scaffold TRAIN/TEST, separate SELECT/CAL and
100/300 full-epoch horizons are declared adaptations. No bitwise source-RNG,
published-task/environment or score-match claim is made.

## Finite prospective domain

The original reference gets its own40 source-scoped GINE configurations:
LR in [.00025,.0005,.001,.002,.004], hidden in [256,512], layers in [5,6],
dropout in [.5,.7], fixed AdamW/betas(.9,.999)/eps1e-8/weight_decay0.
Batch/eval size128, members4, one own view per member, no auxiliaries;
ineffective temperature is fixed.2. Search seeds70101/70103 and comparative
seeds70201/70203/70207 match the existing declared paired roster. The strict
first SELECT ROC-AUC selector and source/raw/CAL graph authorities are bound.

Each of four studies adds80 search fits,3 comparative fits,1 selector and3
exports/CAL/evaluations each:83 paid fits and93 records. V4 therefore declares
24,892 total model fits,25,396 records,36 selectors and156 exports/CAL/
evaluations each. Existing48 genuine I4/fixed-body0 reuse identities remain
preserved. New original banks have one physical body and four routes.

## Exact optimizer-call accounting

Every v4 fit binds positive integer `config.train_minibatches_per_epoch`
and a complete `molecular_cost` receipt contract: TRAIN graph count, actual
batch size, include-tail policy, last batch, one optimizer call/minibatch,
TRAIN pack path/bytes/hash and full-budget call maximum. The actual adapter
checks TRAIN graph count against the bound pack and each completed epoch's
optimizer calls against that declared count.

| Dataset | TRAIN graphs | Batch size | Minibatches/epoch | Last batch |
| --- | ---: | ---: | ---: | ---: |
| MolHIV | 32,901 | 128 | 258 | 5 |
| BACE | 1,210 | 128 | 10 | 58 |

Existing v3 declarations imply658,208,000 backbone optimizer calls. The
original reference adds8,897,600, giving **667,105,600** v4 calls across the
four100/300-epoch studies and5,044,800 full TRAIN epoch passes. Scalar CAL
calls are separate and excluded. V3's frozen metadata remains unchanged.
The master can use `budget.maximum * train_minibatches_per_epoch` for v4
molecular fits; the field is absent in older unresolved v3 contracts.

## Qualification and release

`test_molecular_boundary.py` checks exact tail metadata, source R/S/B init,
absence of internal factors, live shared/private gradients and exact updated
state against the pinned native four-member BCE/4 training function on tiny
CPU graph fixtures. It verifies selected state/private streams restore and
four-route graph-only export while both heldout target files are absent.
Existing molecular tests still verify prior geometry/objective/I4/role laws.
These are engineering checks, not benchmark or acceptance evidence.

`compile_molecular_boundary_matrix.py` builds a fresh disabled v4 from v3
metadata, rehashes existing custody/source bytes without decoding heldout
tensors, preserves old learning contracts, adds the reference/actual cost law,
and writes new compressed studies/PLAN/SUMMARY. No source model acquisition,
scientific fit, score, approval or phase-release receipt is created. GPU
memory/time reservations remain provisional until root qualifies them and
the complete master roster receives explicit human GO.
