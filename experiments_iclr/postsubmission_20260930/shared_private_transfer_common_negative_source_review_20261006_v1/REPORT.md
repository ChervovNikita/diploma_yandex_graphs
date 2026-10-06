# Saved common-negative diagnostic: source readiness

## Verdict

The existing source is ready for root-authorized execution of its fixed twelve-bank common-error diagnostic. No blocking correctness defect was found. Source SHA is `2a7aa40400fb944dc83144f49575653b32c404a749bd21f34184d09142ce0b53`; binding SHA is `99981bad6baf3d6e1596d9064695fc07588917fc6f25eb74cabf9b1d2264524a`. Manifest closure and all twelve D2 path/hash/member-count/selected-state/input/runtime bindings pass. All targets are the selected allocation E_joint, E_live, S_joint and J4_joint in b0/b1/b2; excluded old77 donors are absent.

The reader hashes all twelve bank files before Torch deserialization, uses safe CPU tensor loading, checks exact finite FP32 bank shapes and state/input custody, and checks that saved means equal member raw means. It computes strict error masks and signed margins from saved values. It does not execute models or recompute the original MRR/Hits scores, select checkpoints, read feature/positive/candidate input files, or contact77.

## What identifies the mechanism

Each `(query_slot, negative_slot)` identifies the same stored candidate across members/cells/blocks under the original evaluator order and common positive/pool hashes. These are candidate-slot identities, not deduplicated graph edges. Node-pair annotation would require a separate authorized pool-input read and is unnecessary for the requested overlap analysis.

For each query, C is the set of candidates strictly outranking the positive in every member; U is the set doing so in any member. The observed saved-pool survivors L distinguish strict serving obstructions from FP32 pooling ties. At least L strict outrankers gives reciprocal rank at most `1/(1+L)`; L at least 10 excludes Hits10 without recomputing full MRR. A high C/L burden means same-candidate errors are shared and a mean pool cannot remove them. Low C with high pooled error burden points toward disagreement, calibration or insufficient useful complementarity, rather than unanimous strict error.

The exact convex-aggregation statement applies when the same nonnegative normalized member weights are used for positive and negative logits. If every member margin is negative, their weighted margin is negative. A router choosing one fixed member for the query has the same obstruction. Arbitrary different weights for different candidate edges are a different rule and are not ruled out by this pairwise statement.

E-common slots strictly corrected in the J4 or S saved pool identify mistakes specific to these selected shared-bank states, not universally unsolvable candidates. J4 changes both encoder and predictor, so this cannot isolate sharing as the cause. Compare the three blocks separately; E_live remains a characterization of the stopped arm.

## Routing scope limit

`U minus C` includes ties and does not guarantee that any member is strictly correct. Different offending negatives may require different members, preventing a single query-level router from recovering them all. The existing output supplies each query's member strict-error counts, common count, pooled strict-error count and pooled tie count. Comparing the minimum member error count with C and the pooled count is useful: a minimum far above C shows that few common errors can coexist with a substantial obstruction for every fixed member. It is a diagnostic of available complementarity, not a deployable router or an oracle performance claim.

The source exports individual margin vectors only for common candidates. Strict-correct member masks among noncommon pooled errors are not exported. If root wants the narrower strict rescue diagnostic after the first result, it can be added using these same retained logits and no forward; the current source should not be described as having already measured that routing result.

## Exact allocation execution

`EXECUTION_COMMAND_DISABLED.json` gives the exact argv and environment. Use the allocation native interpreter at `PHASE/native_ncn_runtime_20261005_v1/.venv/bin/python`, with the reviewed repository `.venv/lib/python3.11/site-packages` as PYTHONPATH, matching the successful D2 CPU reader. The native interpreter's bare Torch spec was absent in that prior operation; the existing reviewed overlay supplied Torch2.1.2+cu118. No installation or77 change is required.

The root release artifact root is the existing allocation phase `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930`. Root sets only `enabled` and `source_review_approved` true in the provided preview after authorizing the exact source; TEST/model/checkpoint reselection remain false. Stage the exact source and BINDINGS at the existing preparation path. Capture stdout to a fresh `COMMON_NEGATIVE_RESULTS.json`; the reader accepts `--release` and has no `--output` argument.

Reuse the existing allocation CPU ownership contract with fresh command/config/output. The proposed bounds are 600 seconds, 8 GiB RSS, one attempt, 1 GiB stdout diagnostic and 8 MiB stderr. This is a full per-slot margin export: stdout is a result artifact and can exceed the 8 MiB training-log allowance. Prior D2 completed39 bank diagnostics in10.55 seconds and approximately425 MB RSS, but this diagnostic's full export cost has not been measured. No new process framework is prepared.

## Execution and interpretation status

Only source, metadata, AST and hashes were inspected. No prediction/model/data/checkpoint payload or numerical framework was opened/imported, no new score was calculated, and no server or77 command was executed. Existing source banks were not checked on the host in this assignment; the reader itself authenticates all twelve before doing analysis.

These are retrospective selected VALID states at differing selected cycles on one graph/split/pool. Repeated queries and candidates are dependent. The outputs can establish factual error overlap and cross-model correction at fixed identities. They cannot establish historical training causality, graph-information erasure, method novelty, independent heldout performance or a useful learned router without further evidence.
