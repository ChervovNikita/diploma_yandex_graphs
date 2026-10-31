# Spectral neighbor prototype v2

Status: source implementation and real torch verifier prepared on 30 September 2026. **Torch/PyG execution has not been performed by this agent.** Local standard-library AST checks pass. The implementation gate is a successful coordinator-run synthetic CPU verification. No dataset, server, GPU, or manuscript operation was performed here.

Completed v1 `PROPOSAL.md`, `sanity_spectral.py`, and `SANITY_REPORT.json` were not changed. Their exact bytes are locked by `V1_PRESERVED_MANIFEST.json`. The new implementation manifest is `PROTOTYPE_MANIFEST.json`; source changes after a server check require another manifest/check identity. Outputs use exclusive creation and never replace earlier evidence.

## APIs

| Module | Purpose |
| --- | --- |
| `adapters.py` | `SourceSAGEEnsemble`, original source SAGE neighbor maps plus frozen-output centered corrections; `tied_source_copy()` exports a plain source checkpoint only when private coefficients are zero. |
| `spectral_core.py` | Float64 output covariance, signed eigenvectors, fixed discovery/screen rank rule, and random/raw-gradient bases at matching ranks. |
| `collector.py` | `GradientUnit`, P10/D4/E4/F2 partition manifests, observed-node BCE, and named matrix collection with state/grad/mode/RNG preservation. |
| `contracts.py` | Canonical JSON seals and raw float64 tensor hashes, without pickle; exclusive artifact writes. |
| `replay.py` | Exact reconstruction of statistics, selected bases/ranks, and controls from hashed matrix evidence, with no labels. |
| `verify_torch.py` | Synthetic CPU torch/PyG source-forward, gradient, covariance, collector, and replay checks. |

The package supports imports such as `from spectral_privacy.v2.adapters import SourceSAGEEnsemble` when the phase directory is on `sys.path`. The scripts also run directly from `v2`. No new training runner is provided.

## Source adapter contract

Input is the original `TABMModel` with `model_name='SAGE'`, three residual blocks, hidden width128, multiplier1, LayerNorm, and four members. Other settings require an explicit compatibility study. The constructor clones the source by default, consumes no RNG, and never calls source weight initialization. It targets exactly

    residual_modules.{0,1,2}.module.conv.lin_l.weight

Each original PyG `lin_l` object becomes the adapter's `base`. Calling `base(features)` keeps the source neighbor bias in its original location and applies it once. Root weights, normalization, FFNs, source boundaries, source graph propagation, and hidden-state width remain the original operations. Corrections are

    Delta_lm = U_l sum_q C_mq A_lq

with frozen orthonormal output basis U, zero initialized A, and deterministic Helmert C stored once by the ensemble. Each target contains `(M-1)*rank*h` trainable coefficients and `rank*h` frozen basis scalars. C contains12 additional scalars. `added_storage()` reports the actual counts. No activation or propagation saving is asserted.

`forward_member(graph, features, member)` sets and restores temporary member context around the original source forward. It supports serial calls; simultaneous calls to the same instance are outside the contract. The member context never enters `state_dict`. A rank-zero target returns its source base operation directly.

The native mean-member objective gives the coefficient gradient

    grad(A_q) = (1/M) sum_m C_mq U.T G_m

At zero A, SGD therefore produces `Delta_m=-(eta/M) U U.T (G_m-mean G)`. There is no hidden member-gradient multiplier. Adam's coordinatewise moments need not satisfy the same projected-SGD trajectory.

## Collection and selection contract

`GradientUnit` stores one complete training graph, its feature/target tensors, and an optional immutable observed-node or observed-entry boolean mask. The collector accepts exactly D4/E4 units with `split='train'`. The partition also carries P10 and F2 **IDs only**. P/F targets cannot be supplied to this collector. Actual probe provenance remains a caller attestation: a content hash does not prove that a model was fitted on P labels alone.

Each own-member own-graph gradient uses the mean BCE over finite observed entries, with no division by members or scoring graphs. Different graph sizes do not receive different weights in the score. Each named gradient has source layout `[out,in]`; raw evidence has `[units,members,out,in]` shape.

Member centering occurs separately within each graph. For n scoring graphs,

    Sigma = sum_(i,m) (D_im-Y_m)(D_im-Y_m).T / [M(n-1)]
    S = mean_m Y_m Y_m.T - Sigma/n
    N = Sigma/n0
    T = S-N

`S` equals the ordered off-diagonal graph-pair U-statistic. The covariance acts on output rows, never input columns or the member axis. The first PPI rule fixes n0=1 because updates use one complete graph. Other n0 values are supported only by the algebra helper, not the frozen pilot selector/record. `choose_bases` uses discovery eigenvalue>0 and screening `u.T T_E u>0`, orders by the smaller score, and admits at most16 directions per fixed target. Rank zero is allowed.

Eigenvectors receive a deterministic sign convention. Eigenspace degeneracy, finite sampling, and near-zero eigenvalues remain genuine limitations; this rule is not a confidence interval or an unbiased selected-subspace estimate. The record saves all scores, selected indices, ranks, basis bytes, and graph identities.

`random_matched_bases` uses an isolated CPU generator, preserving global RNG. `raw_gradient_matched_bases` uses the leading output basis of the concatenated discovery mean centered gradients. Both match selected ranks exactly, including rank zero. They share the same adapter and coefficient optimizer parameterization.

The collector snapshots parameter/buffer values, existing `.grad` object identities and values, every module's mode, CPU/Python RNG, already initialized CUDA RNG if any, and NumPy RNG when that library is loaded. Eval/autograd collection must not change these states. A detected mutation raises; the `finally` path restores them even on failure. The prepared verifier is CPU only and does not initialize CUDA.

## Minimal integration

Use the coordinator's already fitted P-only original source model. Keep its source/checkpoint identity immutable.

```python
from spectral_privacy.v2.adapters import SourceSAGEEnsemble
from spectral_privacy.v2.collector import collect_record, partition_manifest
from spectral_privacy.v2.contracts import state_hash, unpack_float64
from spectral_privacy.v2.replay import replay_record

probe = SourceSAGEEnsemble(fitted_source)  # Rank zero, independent source clone.
partition = partition_manifest(P_ids, D_ids, E_ids, F_ids)
record = collect_record(
    probe, D_units + E_units, partition,
    source_sha256=frozen_models_py_sha256,
    dataset_identity=frozen_new_study_data_and_mask_identity,
    optimizer_seed=seed,
)
replay_record(record, expected_source_sha256=frozen_models_py_sha256,
              expected_probe_state_sha256=state_hash(probe))
bases = {key: unpack_float64(payload)
         for key, payload in record['selection']['basis'].items()}
candidate = SourceSAGEEnsemble(probe.tied_source_copy(), bases)
```

If the caller fits a wrapped rank-zero probe directly, `tied_source_copy()` exports its fitted plain source before constructing candidate and controls. It refuses to discard nonzero private coefficients. All final arms must use the same exported common checkpoint, optimizer reset, graph masks, and budget. The collector does not fit the probe, train final arms, select validation checkpoints, or access audit/test labels.

## CPU verifier command

Copy `v2` and the three locked v1 files together to the server's corresponding phase directory. Keep the original `source_snapshot/models.py` there or pass its explicit path. In the existing torch/PyG environment:

```bash
python spectral_privacy/v2/verify_torch.py \
  --source-models source_snapshot/models.py \
  --output spectral_privacy/v2/CPU_VERIFICATION_run01.json \
  --record-output spectral_privacy/v2/SYNTHETIC_MATRIX_RECORD_run01.json \
  --artifact-root spectral_privacy/v2
```

This constructs 20 tiny synthetic graphs only. It verifies exact zero-correction source outputs in eval and with the same dropout stream; source/bias preservation; native coefficient and shared gradients; eta/M correction scaling; member rank1 versus output rank3; the explicit graph-pair covariance identity; a positive-recurrence noisy-direction example; random/raw matching ranks; empty bases; mixed mode and preexisting-grad preservation; restoration after RNG-mutating forwards; validation-unit rejection; JSON/raw-byte round trips; exact replay; and five tamper cases. It also checks that v1 bytes remain unchanged.

It sets CPU torch intra-op threads to1. Failure exits without claiming PASS. Save a new report filename on reruns; preserve the failed trace. No dependency installation is requested by this prototype.

For a later separate replay process, use the recorded thread configuration:

```bash
python spectral_privacy/v2/replay.py \
  spectral_privacy/v2/SYNTHETIC_MATRIX_RECORD_run01.json \
  --threads 1 \
  --output spectral_privacy/v2/SYNTHETIC_REPLAY_run01.json \
  --artifact-root spectral_privacy/v2
```

If its inter-op count differs, pass the value saved in the record with `--interop-threads`. Public `replay_record` does not alter thread settings. By default it checks code hashes, torch/PyG/Python/build/thread environment, tensor hashes, source target names, shape/dtype, graph order, loss scale, and exact reconstructed choices. Explicitly disabling environment comparison still requires byte-exact derived results. It never treats a tolerance comparison as an exact replay success. Bitwise eigenbasis identity across different BLAS/torch builds, or arbitrary degenerate eigensystems, is not promised.

Raw float64 arrays use checked base64 byte payloads. A full D4/E4 record for the three h128 matrices carries about12.6 MB of gradient bytes before base64/metadata, plus covariance and basis evidence. The encoding avoids pickle and preserves exact IEEE values. Outer hashes authenticate recorded content, not biological independence, statistical unbiasedness, or source/data provenance.

## Gate and claim boundary

Run the synthetic CPU verifier before integration. Audit the new collector against the coordinator's immutable source/data/mask manifests before any real task run. The fixed-projector gradient-target denoising proof from v1 is the only theoretical interpretation. There are no new dataset results, final optimizer guarantees, novelty claims, or resource-speed claims in v2.
