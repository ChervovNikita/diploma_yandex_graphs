# Literal WikiCS author adapter assessment

Date: 2026-10-10. Scope: the already retrieved author text and the completed common-wrapper source. This is a source assessment, with no model, data, prediction or checkpoint execution.

## Decision

The retrieved source is insufficient to implement a literal author GCN or GAT adapter while preserving its forward semantics, graph preparation and original Parameter objects. No adapter code is admitted from these files. No official author SAGE implementation or preset is established by the retrieved evidence.

`shared_fast_graph_model_interface_20261010_v1` remains unchanged. Its `NativeModelAdapter` explicitly supports the original repository's PyG `models.Model` residual/FFN GCN, GAT and SAGE bodies. It does not certify a DGL author model, and it does not make those residual/FFN models published WikiCS references.

## Available primary evidence

Sources are saved in `../live_route_common_wrapper_pilot_protocol_20261010_v1/sources/`, with retrieval URLs and hashes in that folder's `PRIMARY_SETTING_RETRIEVAL.json`.

The saved commit response identifies `f5207315d649377f936edb66d7d93f5342f01d81`. The three training files were retrieved at that immutable commit. The README was retrieved from `master`; its settings are evidence from that retrieval, without a separately established README-at-commit identity. `wikics_tree.json` is truncated and its retrieval record reports a JSON decoding failure. It cannot establish repository-wide presence or absence of an implementation.

| Backbone | README reproduction preset | Constructor evidence in saved training entry point |
|---|---|---|
| GCN | `--self-loop --n-hidden-layers 1 --n-hidden-units 33 --dropout 0.25 --lr 0.02` | Passes `F.relu`, hidden layer count, width and dropout to missing `GCN`. |
| GAT | `--self-loop --n-hidden-layers 1 --n-hidden-units 14 --in-drop 0.5 --attn-drop 0.5 --n-heads 5 --lr 0.007` | Passes `F.elu`, input/attention dropout, slope and residual flag to missing `GAT`; output heads default to 1, slope to 0.2, residual to false. |

The README says GCN/GAT/APPNP were taken from DGL examples. The saved common trainer uses Adam with default weight decay `5e-4`, cross entropy and patience 100. Its checkpoint selection is strict improvement in **stopping loss**, despite the argument help mentioning accuracy. Training, stopping, validation and test masks have separate roles. The README specifies all 20 splits; the trainer defaults to five runs per split. Its README commands include `--test`, which is not authorization to use TEST in the current studies.

The saved GAT entry point constructs `heads` from `args.n_layers`, while its parser registers `n_hidden_layers`. This discrepancy must be resolved from primary source or documented author correction before claiming a literal runnable author preset. It must not be silently repaired.

## Exact missing source

Retrieve these files at the author commit above before writing an adapter:

1. `experiments/node_classification/gcn/gcn.py`: the actual GCN layer stack, dropout placement, convolution normalization, weight layout, activation order and output operation.
2. `experiments/node_classification/gat/gat.py`: projection and attention Parameter layouts, head concatenation or averaging, attention normalization/dropout, residual behavior, activation placement and output operation.
3. `experiments/node_classification/load_graph_data.py`: graph direction, duplicate handling, self-loop removal/addition, feature preparation, graph-held normalization and mask orientation. A CLI `--self-loop` flag alone does not establish its operation.

If those models delegate equations to DGL modules, their dependency pin and the corresponding convolution implementation are also required to settle version-sensitive behavior and Parameter orientation. The README's mutable DGL-examples link does not settle these details.

For SAGE, an actual author model, training entry point and reproduction preset are missing from the retrieved evidence. The README lists `svm`, `mlp`, `gcn`, `gat` and `appnp`. This supports the narrow statement that no official SAGE reference is established here; the failed tree does not support a claim that none exists anywhere in the repository.

## Consequences for common-wrapper reuse

The existing `wrap_shared` provides one stored body, original Parameter preservation checks, fast-factor installation, per-member execution and an optional live prefix/tail site. An author adapter would still need an explicit native forward mapping and a complete intermediate site followed by the author's remaining graph operation. The missing source prevents identifying those mappings and checking whether its dense maps can use the existing factor installer without replacing or transposing native Parameters.

The current PyG prototype uses input linear → dropout → GELU, normalized residual graph/FFN modules, and output normalization/head. The retrieved author entry points instead supply ReLU for GCN and ELU for GAT to unavailable DGL bodies. Matching a model name or copying hidden widths does not establish equivalent architecture or published competence. The current native runner's 5000-step cap and 300-step patience also differ from the author stopping-loss procedure. Quality of the existing prototype and any prospective SAGE settings requires its own complete measurements under the frozen study protocol.

No legacy432/54 winner, score, quality estimate or timing was used in this assessment. No trainer, launcher, ownership, provenance or verification framework was added. All existing sealed sources, scores and study criteria remain unchanged.
