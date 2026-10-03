# Exact V5 root sequence

Preparation transfers source and metadata only. Run these commands sequentially after inspecting the physical upload receipt. Every command preserves a unique client terminal; no automatic restart or predictive fit is authorized.

```sh
python3 -B postsubmission_research_20260930/amazon_polynormer_paired_family_execution_root_20261003_v3/run_root_release_client_v1.py --mode cpu --id CLIENT_RUNTIME_CPU_V5_20261003_v1
```

After authoritative CPU success:

```sh
python3 -B postsubmission_research_20260930/amazon_polynormer_paired_family_execution_root_20261003_v3/run_root_release_client_v1.py --mode gpu --id CLIENT_RUNTIME_GPU_V5_20261003_v1
```

After both exact V5 runtime FREEZE/TERMINAL successes:

```sh
python3 -B postsubmission_research_20260930/amazon_polynormer_paired_family_execution_root_20261003_v3/run_general_v5_release_client_v1.py --prepare-stage after-runtime --id CLIENT_PREPARE_AFTER_RUNTIME_V5_20261003_v1
```

Inspect the materialized registration, then register:

```sh
python3 -B postsubmission_research_20260930/amazon_polynormer_paired_family_execution_root_20261003_v3/run_general_v5_release_client_v1.py --kind register --release amazon_polynormer_paired_family_execution_root_20261003_v3/releases/register_authorized_v1.json --output amazon_polynormer_paired_family_execution_root_20261003_v3/registry --id CLIENT_REGISTER_V5_20261003_v1
```

After authoritative registration success:

```sh
python3 -B postsubmission_research_20260930/amazon_polynormer_paired_family_execution_root_20261003_v3/run_general_v5_release_client_v1.py --prepare-stage after-register --id CLIENT_PREPARE_AFTER_REGISTER_V5_20261003_v1
```

Inspect the materialized qualifier, then full qualification:

```sh
python3 -B postsubmission_research_20260930/amazon_polynormer_paired_family_execution_root_20261003_v3/run_general_v5_release_client_v1.py --kind qualify --release amazon_polynormer_paired_family_execution_root_20261003_v3/releases/qualify_authorized_v1.json --output amazon_polynormer_paired_family_execution_root_20261003_v3/qualification_block0_cuda0 --id CLIENT_QUALIFY_V5_20261003_v1
```

Captures have900s and16GiB RSS/CUDA allocated/reserved caps. Registration/qualification have3600s,32GiB RSS and75GiB CUDA allocated/reserved caps. The interpreter is `.venv/bin/python`, SHA256 `6ff97f602038740073dca96714310a30e303332326268e0f1bb2767edc820944`,21338208bytes. Each remote client checks GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`.

Initially disabled files: `releases/register.json`, `releases/qualify.disabled.json`, `release_specs/qualify.template.json`. Authorized registration and qualifier releases materialize only after their physical gates. Prior V2/V4 costs and both zero-fit registries remain in custody. Fresh V5 capture is required. No predictive fit, heldout-control scoring, TEST labels, installation or environment change is released.
