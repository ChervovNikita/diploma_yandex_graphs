# What to run

From the repository root, with `R` on storage shared by all nodes. Data goes in `./data`.
Start one worker per GPU and restart workers freely; workers share the queue through `--root`
and pick the configurations that are not finished yet. Progress: add `--status` to a run command.

```sh
for g in 0 1 2 3; do python -m ablation.p01_placement.run --root $R/p01 --device cuda:$g & done; wait
```

| Point | Jobs | Run | Results |
|---|---|---|---|
| Placement (p01) | 1750 | `python -m ablation.p01_placement.run --root $R/p01 --device cuda:0` | `python -m ablation.common.collect --root $R/p01 --out $R/out/p01` |
| Ablations (p02) | 1050 | `python -m ablation.p02_full_ablations.run --root $R/p02 --device cuda:0` | `python -m ablation.common.collect --root $R/p02 --out $R/out/p02` |
| RMS init (p03) | 350 | `python -m ablation.p03_rms_init.run --root $R/p03 --device cuda:0` | `python -m ablation.common.collect --root $R/p03 --out $R/out/p03` |
| Heads (p04) | 700 | `python -m ablation.p04_shared_heads.run --root $R/p04 --device cuda:0` | `python -m ablation.common.collect --root $R/p04 --out $R/out/p04` |
| GAT grid (p05) | 7650 | `python -m ablation.p05_hparam_grid.run --root $R/p05 --device cuda:0` | `python -m ablation.p05_hparam_grid.collect --root $R/p05 --out $R/out/p05` |
| Diversity (p06) | 175 | `python -m ablation.p06_member_diversity.run --root $R/p06 --device cuda:0` | `python -m ablation.p06_member_diversity.analyze --root $R/p06 --out $R/out/p06` |
| Cost (p07) | 2000 | `python -m ablation.p07_cost_small_setup.run --root $R/p07 --device cuda:0` | `python -m ablation.p07_cost_small_setup.collect --root $R/p07 --out $R/out/p07` |
| Steps (p08) | none | uses the p07 runs and `results_merged` | `python -m ablation.p08_training_steps.estimate --cost_root $R/p07 --out $R/out/p08` |
| Depth table (p10) | none | uses `results_merged` | `python -m ablation.p10_depth_metrics.depth_table --out $R/out/p10` |

p07 measures time: its GPU must run nothing else, and all its jobs should use one GPU model.

Environment: Python 3.11, torch 2.1.2+cu118, PyG 2.7.0, DGL 2.4.0, torchdata 0.7.1, packaging,
pyyaml, pydantic, psutil, pandas, scikit-learn. The test server used DGL 2.1.0 because
data.dgl.ai was unreachable.

Retry a job that failed three times: delete `failures.log` in its folder, then restart a worker.
