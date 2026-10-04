# Disabled PENCIL Collab resource qualifier

This v2 packet repairs the source ownership issue identified in the fresh independent v1 review. The numerical worker is now launched directly by the inherited supervisor with `start_new_session=True`; explicit rank 0/world size 1 and a private `file://` rendezvous replace the elastic launcher. It remains **disabled, unstaged and unexecuted**. No numerical, network, staging, launch or package operation occurred while preparing v2. Immutable v1 and its review verdict are preserved.

## Actual dependency readiness

Root has completed the repo-owned overlay installation. `ACTUAL_DEPENDENCY_BINDING.json` binds the existing root monitor, installation result, distribution admission and installed inventory: 16 new packages, 3361 files, 125404681 bytes, unchanged core versions, inventory SHA256 `f5052723eeff9ae66616413fda3f30ef6cee69a6ade3d682379243fa83162d7a`, distribution admission SHA256 `3c52edf196945f7610ff7b243bc709d3264b8ae40106224d87121be744f41d65`. Observed distribution versions exactly match the unchanged PLAN. No package installation or environment change is requested by v2.

Inherited v1 preflight/resolver/installer records remain historical evidence. They do not describe packages as currently absent and are not an instruction to reinstall. The existing external dependency-admission gate is unchanged; root can bind its already produced inventory/origins into that schema. Operator, private SDPA import, model and full-workload numerical compatibility remain unobserved.


## Exact workload

`worker.py` calls byte-identical author `build_loaders`, `get_model`, `train_loop` and `evaluate_loop` helpers from commit `2d32e29dbed533288d9d758138e07547a0a7d8a9`. Twenty native Python files are copied with verified Git blob identities. The official YAML is copied unchanged; the author-supported feature/early-fusion/seed0 flags are applied by the owned worker.

The worker retains scratch BERT with hidden size 512, 8 layers, 8 heads and intermediate size 2048; 75 one-hop neighbors per endpoint; no replacement; native target-edge removal after sampling; batch 1024; 12 loader workers; AdamW learning rate 1e-4 and weight decay .01; total accumulation 8; bf16 autocast; and native TF32/matmul-medium. One DDP/NCCL rank makes accumulation 8 per rank. Native 50% cyclic positive sampling and global negatives use the filtered year>=2007 TRAIN graph. The partial final update retains the native divisor 8. No debug/sample cap or alternative recipe is available.

The existing selective loader authenticates complete TRAIN/raw/VALID bytes and tensor digests, retains all 235868 nodes and 128 features, then calls native `filter_by_year`. The consumed graph attributes and native float32 weight cast are preserved. Only native TRAIN and VALID dataset constructors are called. The native post-constructor seed reset and real `get_feature_dim` query sample remain in their original order before model initialization.

The actual tokenizer bound is 152 nodes, structural width 306 and at most 154 positions; the model positional capacity remains 2048. Actual batch shapes are checked. The full VALID loader must serve 60084 positives plus 100000 negatives, 157 batches and sequential indices. `evaluator=None`, `compute_loss=False` and no evaluator construction prevent predictive metrics or selection. Native per-batch BCE is still constructed because the native helper supplies labels; it is not aggregated or reported. Training progress text is silenced because it contains losses. Scores are checked for shape/finite status and discarded.

`RESOURCE.json` retains counts, cryptographic data/query/order identities, shapes, phase timings and CUDA allocator peaks. Neither score arrays nor losses/metrics/checkpoints are published. Model and Adam state are discarded. This resource attempt cannot donate initialization, fit state or a selected checkpoint to a later scientific fit. It does not complete the official 20-epoch recipe.

## Protocol differences

`PLAN.json` lists every intentional change: selective file custody replaces the native OGB startup/cache/split loader; unused processed graph attributes and all TEST construction are omitted; a pinned570-byte non-weight BertConfig replaces HF lookup; the owned source marker resolves native definitions without repository changes; one authorized rank replaces automatic eight-GPU selection; only one epoch is run; no checkpoint is saved; monitoring/finite checks/custody are charged; and the contemporary admitted runtime differs from the author requirement versions. These are resource qualification adapters, with no numerical reproduction or published-score claim.

## Ownership and limits

`supervise.py` reuses the proven minimal-gradient v4 held direct-child `waitid/WNOWAIT`, birth/session/group guards, owned group cleanup, physical terminal before collection, persistent spent-attempt lock and final custody/publication checks. Its eight ownership/utility function ASTs are unchanged. `SUPERVISOR_ADAPTATION.diff` makes the scope/collection/direct-worker command changes reviewable. The held direct child is the numerical rank itself. Its native data-loader descendants inherit the worker session/group; the existing parent identity and owned-session sampling/cleanup checks remain in force. The private rendezvous file is inside the newly created child directory and is removed after process-group destruction. This is a source topology repair; no new OS/process behavior has been observed. It is not a process-escape sandbox.

The predeclared candidate ceilings are 7200s inclusive parent/child elapsed, 64GiB sampled aggregate owned-session RSS, 70GiB CUDA allocated, 75GiB reserved and 64MiB output. RSS and allocator monitoring use .25s observations. RSS is not an instantaneous OS limit; CUDA allocator statistics exclude non-allocator driver allocations. Phase timings include the instrumentation within them. The final finite receipt/hash/fsync/stdio tail has the same disclosed limits as the inherited supervisor. A failure, incomplete epoch/VALID traversal, cap breach, unresolved cleanup or custody drift cannot produce resource adoption. No retry, subset, precision/shape change or automatic continuation exists.

## Remaining concrete steps

Root must review these exact sealed bytes and the selective adapter. The already installed overlay inventory/origins must be bound in an external dependency admission using the unchanged schema. No new installer or environment closure is needed. Historical installation command files are retained unchanged for provenance. The exact qualifier closure must then be staged, independently source-reviewed and bound by a concrete external root release; both provided templates are disabled. `PROPOSED_COMMAND.json` specifies the eventual supervisor command, environment and directly supervised one-rank child command. The resource worker initializes NCCL/DDP with `file://` under its fresh execution root and does not use a port rendezvous.

`AUTHOR_SOURCE_CHECK.json` is source-only author evidence: AST parsing, byte/Git identity, unchanged ownership functions, selected TRAIN/VALID constructors, no native main/split accessor/checkpoint call, disabled templates and unchanged bound inputs. It is not independent review, numerical PASS or a resource result.


## V2 repair evidence

`V1_TO_V2.diff` and `V1_TO_V2_PROVENANCE.json` expose the exact payload delta. The numerical source changes are restricted to direct launch, explicit rank environment, private file rendezvous/cleanup and launch identity metadata. The common gate, data adapter, native files, config assets, workload, caps, precision and scientific/data adapters are byte-identical to v1. Source checks verify these identities and the unchanged ownership helper ASTs. They do not revise the v1 review or establish an independent v2 PASS. Root will obtain a fresh review of the repaired sealed bytes.
