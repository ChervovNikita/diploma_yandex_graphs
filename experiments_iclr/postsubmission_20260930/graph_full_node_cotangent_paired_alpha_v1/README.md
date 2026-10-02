# Paired full-node cotangent shared-alpha path v1

Source-only extension of the sealed support-v1 packet. The original v1 source/default API and every active Round17 file remain unchanged. This packet adds one separately named helper and fixtures; it includes no driver, optimizer, continuation, coordinator or launch.

## Interface and result

```python
arm_slices, report = initialize_paired_four_arms(
    logits_fn, theta0, S, S_permuted, target_nodes,
    train_rows, train_labels, homogeneous_full_node_outputs=True,
)
```

Four fixed arms are `common_only`, `train_remasked`, `full_node`, and `full_node_permuted`. All begin at the same deterministic common identity-factor warm boundary, use the same compact TRAIN mean-CE gradient, private slice, projection/centering, Frobenius cap and TRAIN acceptance constants. Both graphs must cover the same full homogeneous output universe; `target_nodes` is its complete row permutation. `S_permuted` must be a genuine caller-supplied `Pi S Pi^T` alignment-null control while feature/label order stays fixed. Shape checks cannot prove that semantic contract; the caller must save its permutation and topology equality receipt.

One grid is used for every arm:

```text
alpha0 = RELATIVE_FACTOR_RADIUS*sqrt(d)/(sqrt(1+CAP^2)*||g||)
alpha[k] = alpha0/2^k, k=0..5
```

Orthogonality and the common Frobenius bound imply `||-g-t_m|| <= sqrt(1+CAP^2)||g||`, so this grid respects the shared factor radius, subject to the retained cast tolerance. It is conservative; it does not make every arm's maximum radius identical. The first alpha accepted by **all four arms** returns a dict of four `[4,d]` factor tensors and `status='joint_accepted'`. Exhausting the grid returns **None**, `status='joint_failure'`, and complete per-alpha/per-arm records. No failed graph branch becomes a common-descent or warm-copy arm. Zero common gradient is an explicit early joint failure with no positive-alpha grid.

Every trial evaluates all four members of all four arms before deciding joint acceptance, even if a geometry/separation failure is already known. Member/pooled TRAIN CE and TRAIN functional separation retain the v1 expressions and thresholds. Full-logit trial finiteness retains the v1 requirement. Full-output Gram and signed diagnostics remain diagnostic-only. Their q is the shared full-node q for that arm's topology; the remasked arm is marked cross-support. The already evaluated common-arm logits supply the exact same-alpha common baseline, with no extra forward.

Common-only is explicitly exempt from both source-tangent and finite pair-separation guards, as in v1. Its four routes are identical and it is accepted by the finite member/pooled TRAIN CE guard. The three graph arms must pass separation; their failure cannot be replaced by common-only.

Trial closure exceptions are charged before each call and retained per member. Geometry AD/graph-call exceptions and failed post-primal certificates raise `PairedGeometryError`; its `.report` retains attempted counters, failure stage and no accepted alpha. Invalid input contracts before the first model call raise the original validation errors. The caller should retain every exception; none authorizes a replacement arm.

## Verification status

**107 stdlib checks passed.** They verify syntax, v1 primitive bytes, projection/CE/Armijo expression preservation, exact rational radius identities, native source/descriptor bindings, and unchanged bytes for all 16 sealed v1 files and all 11 active Round17 files. The paired source was not imported by these checks.

Four substantive Torch CPU tests are authored and syntax-checked, **not run by this Mac author**:

- Independent analytic full-output Jacobian and factored dense Bernstein reference; checks nonzero support and topology deltas, shared alpha, all arm decisions, returned steps, Grams and signed slopes.
- Forced zero-tangent joint failure with valid common descent; all six trials and all 96 forwards remain recorded.
- Forced trial-forward exceptions; all 96 attempted calls are charged and retained.
- Geometry-primal exception receipt.

The root's earlier support-v1 CPU qualification passed all five original tests under Torch 2.1.2+cu118 with CUDA hidden; its receipt is bound in `SOURCE_BINDINGS.json`. That result does not execute or qualify this new paired helper.

Executed stdlib command:

```sh
'/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' -B '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_full_node_cotangent_paired_alpha_v1/fixtures/stdlib_checks.py'
```

Future CPU command, not executed here:

```sh
python -B '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_full_node_cotangent_paired_alpha_v1/fixtures/torch_paired_checks.py'
```

`NATIVE_QUALIFICATION_PATH.md` provides the concrete next source/AD path without a GPU launch. `COSTS_AND_LIMITS.md` records operations, memory and scientific limits. The manifest/seal describe a source snapshot and are not execution admission.
