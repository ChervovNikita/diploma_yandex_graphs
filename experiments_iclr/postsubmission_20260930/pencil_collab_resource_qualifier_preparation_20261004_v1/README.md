# Disabled PENCIL Collab resource qualifier

This packet contains an executable source preparation for one complete native epoch 0 and all official VALID queries on authorized 18.77 GPU 0. It is **disabled, unstaged and unexecuted**. Server actions were the authorized read-only metadata/file-presence preflight and pip resolver dry-run through the existing MacLink `run_gpu77_v3` transport and explicitly pinned RAPIDS Python. No packages were installed, no Torch numerical modules were imported by those checks, and no graph arrays or TEST file were opened. Pip planning allowed temporary resolver downloads, with no persistent download cache or target staging.

## Observed readiness

The preflight confirms the interpreter SHA256 and the sizes/presence of the exact TRAIN, VALID, raw features and raw edges. The pinned CPU ego/saint extension files exist with the expected sizes; their behavioral compatibility is untested. The native import closure also needs scikit-learn, observed separately as version 1.7.0 without importing it.

**Transformers, wandb and rootutils are absent** in the authorized RAPIDS plus existing project PYTHONPATH. Their author versions are 4.46.2, 0.18.7 and 1.0.7. Rich 14.0.0, PyYAML 6.0.2, tqdm 4.67.1, networkx 3.5, OGB 1.3.6 and SciPy 1.16.0 are present. `DEPENDENCY_READINESS.json` distinguishes observations from prospective dependency admission and unexecuted native operator/model compatibility.

The pip 25.1.1 dry-run successfully resolves those three roots plus 13 absent transitive dependencies. `DEPENDENCY_INSTALL_PLAN.json` pins all 16 selected wheels, including their SHA256 hashes, and records reused installed dependencies. `dependency_requirements.txt` contains only those new wheel URLs/hashes. The proposed install command uses `--target` in repo-owned `.gnnm_runtime/pencil_extra_v1/site`, `--no-deps`, `--require-hashes`, `--no-compile`, `--no-user` and no cache. It has no upgrade or implicit dependency resolution. Existing Torch, NumPy, PyG, sparse/scatter and CUDA builds are preserved; the resolver report contains no replacement core or CUDA wheel. Root installation authorization, a fresh target, version checks and complete observed installed-file admission are still required. This is metadata compatibility evidence, not an import/numerical PASS.

The installation plan is also independently sealed in the sibling `pencil_repo_owned_dependency_install_plan_20261004_v1` packet, so its review/installation can proceed while this larger numerical source is reviewed. That packet's requirements bytes are identical; its proposed command points to the standalone installer packet's copy.

## Exact workload

`worker.py` calls byte-identical author `build_loaders`, `get_model`, `train_loop` and `evaluate_loop` helpers from commit `2d32e29dbed533288d9d758138e07547a0a7d8a9`. Twenty native Python files are copied with verified Git blob identities. The official YAML is copied unchanged; the author-supported feature/early-fusion/seed0 flags are applied by the owned worker.

The worker retains scratch BERT with hidden size 512, 8 layers, 8 heads and intermediate size 2048; 75 one-hop neighbors per endpoint; no replacement; native target-edge removal after sampling; batch 1024; 12 loader workers; AdamW learning rate 1e-4 and weight decay .01; total accumulation 8; bf16 autocast; and native TF32/matmul-medium. One DDP/NCCL rank makes accumulation 8 per rank. Native 50% cyclic positive sampling and global negatives use the filtered year>=2007 TRAIN graph. The partial final update retains the native divisor 8. No debug/sample cap or alternative recipe is available.

The existing selective loader authenticates complete TRAIN/raw/VALID bytes and tensor digests, retains all 235868 nodes and 128 features, then calls native `filter_by_year`. The consumed graph attributes and native float32 weight cast are preserved. Only native TRAIN and VALID dataset constructors are called. The native post-constructor seed reset and real `get_feature_dim` query sample remain in their original order before model initialization.

The actual tokenizer bound is 152 nodes, structural width 306 and at most 154 positions; the model positional capacity remains 2048. Actual batch shapes are checked. The full VALID loader must serve 60084 positives plus 100000 negatives, 157 batches and sequential indices. `evaluator=None`, `compute_loss=False` and no evaluator construction prevent predictive metrics or selection. Native per-batch BCE is still constructed because the native helper supplies labels; it is not aggregated or reported. Training progress text is silenced because it contains losses. Scores are checked for shape/finite status and discarded.

`RESOURCE.json` retains counts, cryptographic data/query/order identities, shapes, phase timings and CUDA allocator peaks. Neither score arrays nor losses/metrics/checkpoints are published. Model and Adam state are discarded. This resource attempt cannot donate initialization, fit state or a selected checkpoint to a later scientific fit. It does not complete the official 20-epoch recipe.

## Protocol differences

`PLAN.json` lists every intentional change: selective file custody replaces the native OGB startup/cache/split loader; unused processed graph attributes and all TEST construction are omitted; a pinned570-byte non-weight BertConfig replaces HF lookup; the owned source marker resolves native definitions without repository changes; one authorized rank replaces automatic eight-GPU selection; only one epoch is run; no checkpoint is saved; monitoring/finite checks/custody are charged; and the contemporary admitted runtime differs from the author requirement versions. These are resource qualification adapters, with no numerical reproduction or published-score claim.

## Ownership and limits

`supervise.py` reuses the proven minimal-gradient v4 held direct-child `waitid/WNOWAIT`, birth/session/group guards, owned group cleanup, physical terminal before collection, persistent spent-attempt lock and final custody/publication checks. Its eight ownership/utility function ASTs are unchanged. `SUPERVISOR_ADAPTATION.diff` makes the scope/collection/torchrun command changes reviewable. The supervised session includes torchrun, its numerical rank and all data-loader descendants. It is not a process-escape sandbox.

The predeclared candidate ceilings are 7200s inclusive parent/child elapsed, 64GiB sampled aggregate owned-session RSS, 70GiB CUDA allocated, 75GiB reserved and 64MiB output. RSS and allocator monitoring use .25s observations. RSS is not an instantaneous OS limit; CUDA allocator statistics exclude non-allocator driver allocations. Phase timings include the instrumentation within them. The final finite receipt/hash/fsync/stdio tail has the same disclosed limits as the inherited supervisor. A failure, incomplete epoch/VALID traversal, cap breach, unresolved cleanup or custody drift cannot produce resource adoption. No retry, subset, precision/shape change or automatic continuation exists.

## Remaining concrete steps

Root must review these exact sealed bytes and the selective adapter. The proposed installation plan must be authorized and executed into the fresh owned overlay, then its complete observed source/binary/metadata file inventory and versions must be bound in an external dependency admission. This packet supplies a disabled installation command and makes no installation or staging change. The exact qualifier closure must then be staged, independently source-reviewed and bound by a concrete external root release; both provided templates are disabled. `PROPOSED_COMMAND.json` specifies the eventual supervisor command, environment and one-rank child command.

`AUTHOR_SOURCE_CHECK.json` is source-only author evidence: AST parsing, byte/Git identity, unchanged ownership functions, selected TRAIN/VALID constructors, no native main/split accessor/checkpoint call, disabled templates and unchanged bound inputs. It is not independent review, numerical PASS or a resource result.
