# Root release schema v1

Required schema: `ncnc-collab-predictive-driver-root-release-v1`.

Required fields are `driver_manifest_sha256`, `design_root`, `prototype_root`, `resource_root`, `data_authority_file`, `data_authority_sha256`, `runtime_authority_file`, `runtime_authority_sha256`, `root_authorization_reference`, `family_id`, `family_lock_output_directory`, `authorized_stages`, `authorized_invocations`, and `cuda_visible_devices` for numerical stages. All file/directory paths are absolute on the execution host.

Each invocation row has exactly `stage`, `unit`, `base_seed`, and `output_directory`. Only an exact row grants that invocation. Stage choices are `synthetic`, `valid_engineering`, `resource_engineering`, `fit`, and `family_lock`; units and seeds are frozen by the accepted design. The lock destination is part of family identity and must remain constant across every release and receipt. The `family_lock` CLI unit/seed fields identify its authorized invocation; closure itself checks all units/seeds.

Engineering stages require `engineering_rng_seed=20261003` and an explicit integer `engineering_factor_sign_seed`. Engineering state is never a scientific donor. The example fixes both to 20261003.

Fit releases additionally require:

```json
{
  "fit_qualification": {
    "synthetic": {"path": "/absolute/synthetic/QUALIFICATION.json", "sha256": "ACTUAL_SHA256"},
    "resource_engineering": [
      {"unit": "native_bank4", "path": "/absolute/native_bank4/QUALIFICATION.json", "sha256": "ACTUAL_SHA256"},
      {"unit": "factor_private4", "path": "/absolute/factor_private4/QUALIFICATION.json", "sha256": "ACTUAL_SHA256"},
      {"unit": "factor_pooled4", "path": "/absolute/factor_pooled4/QUALIFICATION.json", "sha256": "ACTUAL_SHA256"},
      {"unit": "native70", "path": "/absolute/native70/QUALIFICATION.json", "sha256": "ACTUAL_SHA256"}
    ]
  }
}
```

The qualification gate verifies the same driver/design/prototype/resource/data/runtime/family identities, finite full native TRAIN and VALID coverage, no metric/donor scope, and both F4 routes' identical initial state/streams/final RNG. Reusing a qualification receipt from another family or successor is rejected. `--resume` is accepted only for an explicitly released fit row's own unclosed output, journal and selected device.

Family lock releases additionally bind `family_inputs`: exactly 20 rows covering every unit and base seed once. A complete row has `unit`, `base_seed`, `output_directory`, `disposition="COMPLETE"` and `complete_sha256`. A terminal-failed row has those identity/path fields plus `disposition="TERMINAL_FAILED"`, `failure_sha256`, `terminal_failure_immutable=true` and a nonempty `terminal_failure_authorization_reference`. The latter binds the driver's `FAILED.json`; its bound attempt ledger must remain unchanged since the failure. Root must not retire an already-resumed stale failure receipt.

Every complete unit and terminal failure is verified before private complete selections are parsed. Selected checkpoint bytes and private selection metadata are hash-bound by each complete receipt. Missing cells are preserved. Terminal-failed closures omit subset summaries/primary contrast. Neither closure path grants TEST authority.

`ROOT_RELEASE_EXAMPLE.json` is an executable-shape qualification example with only the manifest SHA and root authorization reference left for root to fill. It authorizes no fit or family lock. Copy it externally before editing; it is part of the sealed source packet.
