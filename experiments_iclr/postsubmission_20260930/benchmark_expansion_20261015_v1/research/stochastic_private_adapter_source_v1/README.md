# Stochastic private MolBACE adapters — source for root review

**Prepared, unexecuted.** This package adds two narrow arms to the exact paid
residual-only/private-BN producer. PubMed remains next; this package changes no
queue, launch admission, original score, or campaign roster.

## Files

- `extension.py`: exact parent composition, scalar objective, ownership and checkpoint binding.
- `noise.py`: stochastic forward on the original residual object, immutable prior, four independent CPU noise streams.
- `qualify_cpu.py`: guarded full-data two-update/save-reload-resume command, one arm per invocation.
- `check_source.py`: stdlib source, scalar derivative and RNG lifecycle mocks. No Torch import or model/data execution.
- `CONTROL_COMPATIBILITY.json`: descriptor/source comparison and original reference identities, scores and receipts.
- `METHOD.json`, `PROTOCOL.md`, six `JOB_seed*_*.json`: finite scientific definition for later root admission.
- `SOURCES.json`, `MANIFEST.json`: retained dependency and package hashes. `SOURCE_CHECKS.json` is a separate generated source-check receipt.

## Implementation

The existing down Parameter is `mu`; its name remains `input_residual.down`.
The existing up Parameter is `U`. Their identity, initial values, native model,
optimizer, complete AtomEncoder hook and private-BN v3 are retained. Learned
`rho` adds 9,600 coordinates to the same Adam group. Fixed sigma adds none.
All native parameters remain trainable. No BE factors, extra gates or losses
are introduced beyond the declared Gaussian KL.

One 8×300 FP32 Gaussian matrix is drawn per route per eligible TRAIN minibatch,
then reused throughout that route. Each route owns its own CPU generator with
seed `cohort_seed + 170000003 + 104729*route`. The fixed and learned arms begin
with the same paired streams. Draws move to the original parameter device;
native construction, dropout and loader generators are never used for noise.
Saving is allowed only between completed updates. All four generator byte
states and draw counts are restored explicitly. Checkpoints retain the parent
model/Adam/private-BN/sampler state; the existing runner owns global RNG state.
Optimizer checks compare live Parameters, group layout, options and defaults,
allowing group dictionary replacement by genuine `Adam.load_state_dict`.

Mean evaluation uses `mu`, the original private BN buffers and raw-logit mean,
without drawing noise. Prior centers are detached copies of the exact initial
QR frames; restore rejects a different center. The scale is native FP32:
`prior_log_sigma = log(FP32(.1/sqrt(300)))`, and sigma is its exponential.
Both arms use this same represented prior. The learned KL uses `expm1` in the
variance term, so value and gradient at initialization are exactly zero.

## Checks already run and their limit

See `SOURCE_CHECKS.json` for the source-only receipt. It exercises the actual
noise lifecycle with stdlib generator stand-ins and the production scalar
reparameterization/residual/KL kernels with scalar dual numbers. It checks
saved-state replay, independent route streams, no aliasing of saved state,
coherent draw reuse, lifecycle rejection, and derivative activation after U
moves. It also checks six descriptor identities, source hashes and syntax.

These checks do **not** qualify Torch autograd, native forwards, real data,
checkpoint serialization, CUDA, runtime cost or scientific performance.
The CPU qualifier below has not been run. It uses the full native 1,210 TRAIN /
151 VALID roles; two updates reuse the first genuine shuffled batch. This is
not a full training epoch. Both arms require separate successful qualification.

## Prepared qualification commands

Root first reviews source and records an admission copied from the matching
`ADMISSION_fixed_TEMPLATE.json` or `ADMISSION_learned_TEMPLATE.json`, with true
approval/review flags, the actual manifest hash, and its own admission hash.
No template is an approval. From the already verified allocation repository,
run one of the following commands after replacing the admission hash/path and
choosing a fresh output. The qualifier independently checks hostname, sole GPU
UUID, original virtualenv, source/data integrity and admission before importing
Torch or constructing a model.

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/stochastic_private_adapter_source_v1/qualify_cpu.py --project-root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs --arm fixed --output /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/qualification/stochastic_private_fixed_v1 --admission /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/stochastic_private_adapter_source_v1/ROOT_ADMISSION_fixed.json --admission-sha256 ROOT_ACTUAL_ADMISSION_SHA256

CUDA_VISIBLE_DEVICES='' .venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/stochastic_private_adapter_source_v1/qualify_cpu.py --project-root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs --arm learned --output /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/qualification/stochastic_private_learned_v1 --admission /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/stochastic_private_adapter_source_v1/ROOT_ADMISSION_learned.json --admission-sha256 ROOT_ACTUAL_ADMISSION_SHA256
```

Each command allows exactly two native constructions, two total optimizer
transitions, three full VALID calls and 68 native forwards: update1 + VALID,
save, fresh construction/load + bit-exact VALID replay, reload + resumed
update2 + VALID. It checks initial zero down/rho gradients and nonzero U
gradient, their later activation, noise/global/sampler RNG separation and
exact next-noise replay. It records failures/costs without retry. A third
uninterrupted update2 comparison is deliberately not executed or claimed.

SSH remains the literal authorized destination only:
`ssh -tt -p 2222 -i /Users/alex/.ssh/mlspace__private_key_anogena.txt anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru`.
Verify `anogena-2-0` and sole UUID
`GPU-44039938-fd82-41d2-fefd-de71514e2fac` before remote project reads/writes.
No SSH or remote publication was performed in this preparation.
