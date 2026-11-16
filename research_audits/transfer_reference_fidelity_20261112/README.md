# Reference fidelity of the two added tasks

Verified 12 November 2026. The prepared native backbones and official data match pinned author code and preserved official download archives. The complete campaign is not yet an exact reproduction of every paper protocol.

## Verified matches

- All five vendored molecular source files match their manifest. Fresh pinned author downloads independently confirm the convolution, GNN and example runner; the GNN's only patch is a relative package import.
- All 24 pinned BUDDY files match the same files in the author's Git commit, without source patches.
- All 18 raw data and official split files for MolBACE and Collab match both their acquisition receipts and the preserved official archives.
- MolBACE uses its one official scaffold split: 1,210 training, 151 validation and 152 test molecules. The molecular paper specifies ten repeated runs on fixed splits; the prepared ten cohorts match that count, without establishing identical published seed IDs.
- Collab uses its one official temporal split and original node features. Its reference year-2007 filter retains 811,405 of the 1,179,052 official training positives. The filter matches the pinned author function; validation queries and official validation/test partition files remain unchanged. Validation edges enter the test message graph only under the reference Collab rule.
- Five-layer, width-300 molecular GCN and GIN wrappers reproduce native logits on four real TRAIN molecules with identity factors, across all four paths. The width-1,024 BUDDY wrapper reproduces native logits exactly on synthetic inputs with the official 128-feature dimension.
- Six focused CPU checks passed, covering model identity, scaffold handling, native sketch-feature parity and official ensemble metric calculation. This audit performed no optimizer updates or scientific held-out metric scoring.

## Protocol differences that remain

| Item | Current status |
| --- | --- |
| Molecular architecture | Nonvirtual GCN is a supported published baseline. The upstream command default is GIN with a virtual node; the chosen variant must remain explicit. |
| BUDDY paper recipe | The author's public repository calls itself a reimplementation, including its paper-era README. Current Collab width 1,024 and LR 0.02 differ from the paper's stated search ranges of 64 to 512 and 0.0001 to 0.01. Exact reproduction of the published selected recipe is unresolved. |
| BUDDY seeds and repetitions | Author runner seeds repetitions consecutively from zero. Prepared cohorts use multiples of 1,000 plus member offsets. The paper's exact repetition count was not established; five prepared cohorts are not certified as paper-identical. |
| Training duration | Author examples use 100 epochs. The campaign continues with patience and caps up to 2,000, while retaining a separate first-100-epoch readout. Extended results must be distinguished from reference-horizon results. |
| Runtime and randomness | The author loader has obsolete PyG imports and an optional logging dependency absent here. Its small year filter was verified and executed in isolation from literal source with matching syntax trees. No author or environment files were patched. Resumable shuffling is not certified to recreate the exact author RNG trajectory. |
| New contrastive arms | Molecular contrastive-only and combined arms, and BUDDY contrastive-only, still need implementation and qualification. |

Strict paper replication must resolve or explicitly disclose these differences before the owner authorizes final packaging. Native-model identity and official-data verification do not establish a reproduction of published predictive scores.

## Evidence and sources

- [Machine readable audit](audit.json), [verification program](verify.py), [successful run](run_attempt3.log), and [six CPU checks](cpu_checks.log).
- Failed import attempts and their exact program versions remain in this directory.
- [OGB paper](https://arxiv.org/abs/2005.00687), [pinned molecular source](https://github.com/snap-stanford/ogb/tree/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/graphproppred/mol).
- [BUDDY paper](https://arxiv.org/abs/2209.15486), [pinned BUDDY source](https://github.com/melifluos/subgraph-sketching/tree/3562d94a07d1166faa0949030824bf75ad9bb2c4).

Existing training source, generated queue and packages remain unchanged.
