# Coordinator commands

No command was remotely launched by this agent. Start all numerical commands
through the project's existing cache/environment wrapper and
`protocols/run_authorized_v2.py`; expose exactly GPU0 from the supplied one-GPU
`anogena-2` allocation. Do not use the seven-GPU account.

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process creation and keep all cache/
temporary paths inside the phase as required by `protocols/repo_env.sh`.
The following are **child argument lists**, to be supervised from the exact
authorized repository cwd with its repository interpreter. Replace only the
exclusive output suffix and current supervisor directory, not the recipe.

```sh
python objective_lookahead_molhiv_predictive_v1/runner.py prepare \
  --output objective_lookahead_molhiv_predictive_v1/PREPARATION_authorized_run01.json

python objective_lookahead_molhiv_predictive_v1/verify_synthetic.py \
  --output objective_lookahead_molhiv_predictive_v1/SYNTHETIC_authorized_run01.json

python objective_lookahead_molhiv_predictive_v1/runner.py runtime-freeze \
  --supervisor protocols/logged_runs/primary_runtime_authorized_run01 \
  --output objective_lookahead_molhiv_predictive_v1/RUNTIME_authorized_run01.json
```

The runtime freeze performs data-free constructed model counts and environment
capture. It needs the fresh authorized GPU guard; it opens no dataset or model
checkpoint. Qualify its completed supervisor afterward.

These documented real commands deliberately refuse before reading their binding:

```sh
python objective_lookahead_molhiv_predictive_v1/runner.py preflight128 \
  --binding objective_lookahead_molhiv_predictive_v1/FUTURE_ROOT_APPROVED_BINDING.json \
  --output objective_lookahead_molhiv_predictive_v1/REAL128_future.json

python objective_lookahead_molhiv_predictive_v1/runner.py train \
  --binding objective_lookahead_molhiv_predictive_v1/FUTURE_ROOT_APPROVED_BINDING.json \
  --seed 0 --objective own --layout tied \
  --output objective_lookahead_molhiv_predictive_v1/fits/seed0_own_tied
```

After root supplies the qualified locks and separately seals/approves an admitted
entry point, that entry point must expose the same128-training-graph preflight
and the exact30-cell schedule via the engine functions in `README.md`. Run the
complete preflight before the first predictive fit. After each fit's wrapper
exits, run independently supervised full selected and epoch100 checkpoint
replay with the **matching completed fit supervisor**. When selection is epoch100
the same full forward covers both endpoints. Only then can root's primary
analysis count that cell. There is no mutable opening flag in this source packet.
