# Serialized checkpoint replay source review — passed

The exact source manifest SHA256 is `966ddaa6def6bd6a1f9c195e257b652b851e47460984d3166e8a61e5ba077fd8` (1504 bytes). All eight payload hashes/sizes match; the replay source AST and all sealed JSON parse. All 16 unchanged source and 90 metadata bindings verify locally, with the canonical comparison checked through its declared byte-identical metadata mirror. No concrete source blocker was found. `REVIEW.json` approves the source for the checkpoint-replay gate and binds the exact remote manifest descriptor expected by the admission template. Separate root execution release and physical custody remain required.

This is an implementation review of reproducibility-only replay for an already scientifically closed variant. It does not reopen selection, validate new scientific results, review a manuscript, or request an acceptance verdict.

## All thirty cases

The fixed cohort covers Photo and Squirrel; seeds17,29,43; and graph, common_only, random_tangent, topology_permuted, warm_copy at cfg0/K4. All six original context hashes match the registry. Every case's fit/selected-checkpoint/saved-logit descriptors match the closed comparison and V10 saved-array source-VALID audit. JSON-only checks confirmed all thirty fit/selection/matched-warm/preprocessing/role/qualification-certificate links. Numerical payloads remained descriptors only.

## Native restoration and deterministic inference

The replay uses the unchanged original driver's guarded runtime and canonical raw graph reader, recomputes full native preprocessing, and requires the retained qualified receipt to equal the fit receipt. It constructs the matched cfg0 native `TeacherFamily`, strictly restores complete warm tensor state plus mode/stage, and uses the original boundary clone operation. It then installs each arm's complete selected state. Exact key inventory, tensor shapes/dtypes/finiteness and logical-byte hashes are checked before and after loading. All private R/S/B factors overwrite clone identity factors for every arm; both active and inactive Photo heads remain restored. No reset or factor mutation follows selected loading. The selected Photo global branch is explicitly enabled. Fixed non-state attributes are deterministically rebuilt by the same pinned native constructor/specification.

All modules enter eval mode and the complete four-member forward runs under no_grad. The pinned native dropout calls are conditional on training; there is no random eval operation. Constructor RNG consumption therefore cannot alter inference after complete selected tensors are installed. Training RNG and an optimizer need not be restored for this scope. Ordinary named Adam state and alias/parameter coverage are checked as metadata without constructing/updating an optimizer. No label reader, training, initializer, scoring or checkpoint-selection function is called.

## Full logits and original comparison

The inferred tensor must be finite FP32 `[4,N,C]`. The complete tensor is compared to saved selected logits as reference using the original integration's `torch.allclose(atol=1e-6, rtol=1e-5)`; the runtime checks these values against the unchanged `TOLERANCES`. Tensor payload byte equality and NPY file byte equality remain separate recorded diagnostics. FP64 max absolute/relative error diagnostics must be finite. The selected state is byte-checked again after inference.

A finite logit mismatch is preserved while the remaining cases run, and the child exits unsuccessfully after all thirty comparisons. A construction/runtime error retains completed rows and the next case and stops the run. No tolerance widening, replacement case, score recomputation or selection change is introduced.

## Route, dependencies and physical closure

The disabled template requires a separate source-reviewed root release before importing numerical code. The pinned original driver/source guard binds unchanged sources and native recipes. The interpreter, repository, sole retained GPU and entire original per-context environment are fixed, including cudnn_allow_tf32=true, matmul TF32=false and CUBLAS workspace4096:8. Safe numerical readers use weights_only=True or allow_pickle=False. Outputs are fresh, dedicated and exclusive. There is no launcher, SSH client, installation, filesystem isolation, ledger operation or canonical-results write in this replay source.

The child cannot certify physical success: its result keeps accepted_checkpoint_verification=false and physical_closure=false. The stdlib closure operation requires root's own wait4 reap, successful exit, matching child identity/command/source/admission/output, real dispatch/reap timestamps, released wall/host/GPU bounds, sole-GPU route probes and stdout/stderr hashes. It rechecks code/metadata/raw dependency hashes and all thirty checkpoint/logit/fit/warm/selection/inferred-output descriptors before issuing the final replay receipt. The declared resource bounds are forecasts; root must enforce the real bounded process and provide actual measurements. A successful source review does not substitute for those measurements or actual numerical replay.

Only source, AST, text and JSON were read. No numerical library or source module was imported/executed, no dataset/checkpoint/array payload was opened or hashed, no science/SSH/network/subagent action ran, and no existing source, result or canonical ledger changed. Only this fresh review directory was written.
