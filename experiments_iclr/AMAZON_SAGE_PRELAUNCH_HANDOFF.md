# Amazon Ratings SAGE paired control: prelaunch handoff

Prepared 2026-09-25 UTC. **No Amazon GPU arm has been launched and no paper file was edited.** All paths below are inside `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs`. The existing GAT-frozen source, queue scripts, queue logs, and source locks were not modified.

## Prepared files and SHA-256

| Repository-relative file | SHA-256 |
| --- | --- |
| `experiments_iclr/amazon_sage_tied_untied.py` | `c5f57fc72b583cbe1784e2e78a6f8b8663e0ff08879094778dc9c900b5b3c5ac` |
| `experiments_iclr/verify_amazon_sage_tied_untied.py` | `e3a7e396a5493cb9b484877292584aa3333be6257c2f1d2c6bc768321a938a76` |
| `experiments_iclr/amazon_sage_tied_untied_protocol.md` | `981128f237f792c8571762f1de65e83c94cdf2bc1172dffc797d81fbcf5d6b4f` |
| `experiments_iclr/run_amazon_sage_tied_untied_once.sh` | `feb91cf104960aef030c9306cd6cc463e0829013f9e55fa28688b17aaf294030` |
| `experiments_iclr/amazon_sage_tied_untied_results/source_manifest.json` | `e0a3d8f9e8fcc38e14447d58ebe7d2a1e6e358141bc71d9a77a18ba1a95804e1` |
| `experiments_iclr/build_anonymous_bundle.py` | `2f074f7699beda9f6601900d10c04757d63061fb0600f1b2fe497ea140a59ec3` |
| `data/amazon_ratings.npz` (existing corrected public graph) | `4c3a3e3b9d9f6cba0fede4625a00aad8c5721c1a36ed771367f446763241c7dd` |

The source manifest was declared at `2026-09-25T19:10:55Z` and pins the corrected NPZ, public data manifest, `datasets.py`, `models.py`, `run_common.py`, `ablation/models_ablation.py`, `projector_controls.py`, and all four new protocol/runner/verifier/launcher files. It fixes protocol `amazon_sage_tied_untied_masks0_1_2_v1`, official masks `[0,1,2]`, and seed equal to mask index. The bundle builder is intentionally outside this training source lock.

## Fixed study and schedule

Exactly six fresh arms run in order: for each official mask 0, 1, 2, GNNM then initially matched `untied_backbone`, using optimization seed equal to that mask index. Both use four members, SAGE depth 5, width 512, LayerNorm, dropout 0.2, AdamW learning rate `3e-5`, zero weight decay, and the mean member cross-entropy objective. Checkpoints are selected from pooled validation accuracy at step 1 and every ten steps, with strict improvement, earliest ties, 300-step patience, and a 5,000-step cap. The test mask is scored only after checkpoint restoration.

Roman SAGE tied/untied timing was 0.335/0.340 seconds per step. Its masks 0-2 would total about 18,680 effective steps across six arms. Amazon has 24,492 nodes and 93,050 raw edges; projected Amazon cost is about 0.45-0.60 seconds per step, or **2.3-3.1 GPU hours** plus 15-30 minutes for independent replay. This is an estimate without Amazon GPU profiling. All-step-cap behavior could exceed the fixed limit. The six checkpoints need about 398 MB raw state storage, with roughly 0.45-0.5 GB total repository-local artifacts. NFS reported 2.4 TB available at preparation.

Start is refused at or after **2026-09-26 03:00 UTC**. The dormant launcher requires the core coordinator to be terminal, MC Dropout and layerwise completion, the optional OGB1000 completion marker and summary, no queued process, the free shared GPU lock, an idle GPU, the unchanged source lock, and a result root containing only `source_manifest.json`. It has a 255-minute training timeout followed by a 30-minute verifier timeout; a just-in-time start ends by 07:45 UTC, leaving over four hours before the 11:59 UTC deadline. A timeout leaves an incomplete result and cannot yield a completion audit. If any prerequisite is unmet at cutoff, skip this extension; no smaller or shorter substitute is authorized by this protocol.

From the remote repository root, after a separate prelaunch audit and launch decision:

```bash
bash experiments_iclr/run_amazon_sage_tied_untied_once.sh --status
bash experiments_iclr/run_amazon_sage_tied_untied_once.sh --execute
```

The second command is **not** scheduled or invoked by this preparation. The current read-only status was `PRISTINE_RESULT_ROOT`, `SOURCE_PREFLIGHT_PASS`, `CORE_PENDING`, `MC_PENDING`, `LAYERWISE_PENDING`, `OGB1000_PENDING`, `NO_QUEUED_PROCESS`, and `GPU_BUSY_OR_UNAVAILABLE`.

## Gates exercised without GPU training

- Runner source/data preflight passed against the corrected NPZ and all eleven pinned files.
- Runner CPU smoke passed: canonical initial state, CPU RNG, all four initial member logits, and one optimizer step for both arms.
- Independent verifier source/data preflight and validation-trace self-test passed.
- Full verifier **rejected** the pristine root with `Expected exactly six complete runs; found []`; no `completion_audit.json` exists.
- Dormant launcher shell syntax and read-only `--status` passed. It did not start training.
- `build_anonymous_bundle.py` has a default-off `--include-amazon-sage-pair` option. Its direct negative gate rejected the incomplete six-arm tree. The optional compact bundle includes only source, source manifest, six selected result JSONs, validation traces, initialization audits, and completion audit. It omits checkpoints, raw predictions, initial-logit arrays, logs, and the public Amazon NPZ. No bundle was rebuilt during this preparation.

## Required independent completion audit after any launch

1. Confirm the manifest and corrected NPZ hashes, exact three official mask IDs, five labels, per-mask partition/counts, seed rule, and six canonical arm directories with no partial arm.
2. Reconstruct each seed's canonical initial parameters and CPU/CUDA RNG state; check tied/untied hash equality, distinct untied parameter storage, and saved paired initial logits within `1e-6`.
3. Replay each arm's validation trace, strict earliest-tie winner, selected checkpoint hash and strict load, then fresh full-graph inference on the same device.
4. Recompute pooled validation and test metrics and match saved official test node IDs, labels, member logits, and pooled predictions for all six arms.
5. Require the verifier's `completion_audit.json` with all six arms before reading or reporting the three paired percentage-point differences, mean, and sample SD. State that three masks remain correlated observations from one graph. An incomplete or unfavorable outcome is not silently omitted or replaced.
