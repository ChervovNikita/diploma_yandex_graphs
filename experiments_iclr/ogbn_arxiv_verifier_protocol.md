# ogbn-arxiv post-run artifact verifier

`verify_ogbn_arxiv_results.py` is a read-only CPU audit of the fixed
`ogbn_arxiv_pilot.py` run. It is optional while the queue is producing results.
It is **not** called by the pilot or `run_after_sage.sh`. Run it after the
queue has finished, or when a partial result set is idle. An in-progress
seed/variant directory intentionally fails as a partial artifact.

From the repository root:

```bash
.venv/bin/python experiments_iclr/verify_ogbn_arxiv_results.py
.venv/bin/python experiments_iclr/verify_ogbn_arxiv_results.py --require-complete
.venv/bin/python experiments_iclr/test_verify_ogbn_arxiv_results.py
```

The first command accepts zero to nine finished results, but fails if any
present result is malformed or if a result directory contains files without
`selected.json`. The second command additionally requires exactly one result
for each of BASE, ENS, and GNNM at seeds 0, 1, and 2. It exits nonzero until
all nine pass. The third command creates a tiny temporary fixture inside
`experiments_iclr/.tmp/`, uses CPU only, and checks the verifier's rejection
of a wrong selected epoch, missing NPZ, changed checkpoint, and partial run.

The verifier prints a JSON report. `artifact_sha256` contains the observed
hashes of each result's `selected.json`, `epochs.csv`, checkpoint, and NPZ.
The checkpoint hash is compared to the hash recorded in `selected.json`.
The other observed hashes are useful for retaining a later audit record.

## Producer schema and checks

The pilot's `run_config.json` records the fixed hyperparameters, all three
seeds and variants, package versions, checkpoint rule, and SHA256 hashes of
`ogbn_arxiv_pilot.py` and `models.py`. The verifier compares those source
hashes to the current files and requires the fixed two-layer SAGE protocol.
It reloads only the already processed OGB data under `experiments_iclr/data/`
and independently recomputes the tensor fingerprints, graph dimensions, and
official split sizes in `dataset_manifest.json`. Missing local files fail
before the OGB loader is called; the verifier has no download step.

For each canonical `seed_<n>/<variant>/` directory, the producer writes:

| File | Verification |
| --- | --- |
| `selected.json` | Identity, schema, validation selection rule, score and runtime fields, checkpoint SHA256. |
| `epochs.csv` | Exact columns, contiguous epochs, evaluation schedule, early stopping, and replay of pooled validation accuracy, CE, and earliest-epoch tie break. The replay must choose `selected_epoch`. |
| `selected_checkpoint.pt` | Recorded SHA256, CPU deserialization, seed/variant/epoch metadata, finite state tensors, strict load into the declared model, and parameter counts. No GPU or graph inference is used. |
| `selected_predictions.npz` | Exact six arrays, float32 member logits and int64 labels/indices, expected 1 or 4 members, official split order, labels against the local OGB data, and finite logits. |

The NPZ contains `valid_member_logits`, `valid_labels`, `valid_indices`,
`test_member_logits`, `test_labels`, and `test_indices`. The verifier averages
the saved **logits**, recomputes validation and test accuracy and cross
entropy, and compares those values with `selected.json`. CPU versus CUDA
floating-point reductions can differ slightly; tolerances are 1e-7 for
accuracy and 2e-5 for cross entropy. It also reports per-member accuracy,
mean member accuracy, pooled gain, pairwise prediction disagreement, fractions
with all members right or wrong, pooled rescues with no individually correct
member, and counts by number of correct members. These diagnostics are
descriptive. Only the validation history chooses the selected epoch; test
labels and test scores are read after that choice has been verified.

## Limits of the available artifacts

The producer records a checkpoint SHA256, but no expected SHA256 for the NPZ,
CSV, or JSON. The verifier reports their observed hashes and cross-checks
their contents, but cannot prove that an altered NPZ with the same validated
labels and scores is byte-identical to the original. It also does not replay
full-graph inference from the checkpoint, so the saved logits are checked for
internal consistency rather than matched to fresh model outputs.

The pilot exposes `--no-save-logits`, but that flag is absent from
`run_config.json`. This verification protocol assumes the queued default,
which saves logits. A resumed result made with `--no-save-logits` fails here
even if the pilot considers its `selected.json` complete. This is deliberate:
without the arrays, independent score and diagnostic checks are impossible.

## Audit note, 2026-09-25

The read-only default check passed against the queued result root: the current
pilot source hashes, fixed configuration, and local OGB tensor fingerprints
match their recorded manifests. At this check, `verified_runs` was 0 and all
nine pairs were absent. As intended, `--require-complete` exited with status 1.
The synthetic CPU check passed for nine complete fixture pairs and rejected a
wrong selected epoch, a missing prediction archive, a changed checkpoint, and
an unfinished pair. A separate tiny CPU round trip ran the actual pilot for
BASE, ENS, and GNNM, then passed all three saved results through the full
verifier, including strict checkpoint model loading. The verifier and optional
bundle gate are ready; neither
the active pilot nor the GPU queue was edited.

Final read-only check after the nine runs exist, from the remote repository:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python \
  experiments_iclr/verify_ogbn_arxiv_results.py --require-complete
```

This check will exit 0 only after all nine canonical results pass. The
remaining evidence limits are the unrecorded expected NPZ hash, the omitted
`--no-save-logits` flag in the run-config fingerprint, and the lack of a fresh
full-graph inference comparison between checkpoint weights and saved logits.
