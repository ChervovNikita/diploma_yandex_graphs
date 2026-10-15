# Queued MolHIV GIN virtual resource check retarget to portablev3

Source-only prelaunch successor of the existing one-epoch MolHIV GIN-VN
resource check. JOB bytes, full native input/recipe, callback/extension, policy,
seed, resource envelope and native work limits are preserved. Runtime target
is `portable_dispatcher_v3/runtime_node_molecular`. The copied driver verifies
that actual runtime module directory and calls the exact portable host/root/
descriptor guards before Torch import. Its existing original resource sampler
remains source-bound; no portable generation or native math is changed.

Work remains one complete native TRAIN epoch: 1029 Adam maps, selection plus
ordinary selected finalization, two whole VALID calls, and one actual/mean
selected export. Original finite forward/update/restore bounds are unchanged.
The original queued output must still be absent before a new attempt. The
successor uses a fresh output namespace and genuine current-source binding,
without rewriting old states or repeating this lifecycle automatically.

Root must review and bind actual source/publication/current GPU-peer/resources
in a separate ROOT_ADMISSION.json. The template is not permission. Existing
slot, free-memory, host/disk, native-role-loader and watchdog checks remain;
no third CUDA worker or TEST work is admitted by this preparation. The declared
resource envelope is not presented as a new measurement.

Source/free-name checks pass; no models, tensors, datasets, checkpoints, GPU,
TEST or remote commands were executed here. Successful exports can later
supply genuine current-source selected-state VALID-serving qualification.
