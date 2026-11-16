# Automatic rebuttal cluster campaign

The current reviewed campaign entry point is documented in
[campaign/DESIGN.md](campaign/DESIGN.md) and
[campaign/SLURM.md](campaign/SLURM.md). Its immutable queue is
[campaign/plans/rebuttal.json](campaign/plans/rebuttal.json), with a sortable
[campaign/plans/tasks.csv](campaign/plans/tasks.csv) index.

It contains 27,009 potential training fits, including 9,270 fits in a
validation-gated expansion. Additional DAG tasks perform selection, scoring,
diagnostics, profiling and reporting. Scientific training is pending.
Prepared data and the conda environment support offline transfer.

On the destination, follow the supplied short filesystem/CUDA checks, set
GNNM_ENV to the relocated environment, and use one launcher per Slurm node
against the same shared campaign root. The launcher supports more than 32 GPUs
and up to 4 concurrent fits per GPU, with memory and CPU admission.

The training command from the repository is

    python -B -m ablation.campaign.launch --plan ablation/campaign/plans/rebuttal.json --root rebuttal_campaign --gpus auto --jobs-per-gpu 4

The Slurm guide specifies resource arguments and environment setup. Preserve
the frozen plan and every task result, including failures. Review the current
capacity scenarios in campaign/evidence/resource_estimate.json before choosing
a GPU allocation. Original study-specific commands remain below.

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
| Molecular transfer (p09) | 150 | `python -m ablation.p09_molecular.run --root $R/p09 --device cuda:0` | `python -m ablation.p09_molecular.collect --root $R/p09 --out $R/out/p09` |
| Collab BUDDY (p11) | 25 | `python -m ablation.p11_collab_buddy.run --root $R/p11 --device cuda:0` | `python -m ablation.p11_collab_buddy.collect --root $R/p11 --out $R/out/p11` |
| Collab NCN (p12) | 25 | `python -m ablation.p12_collab_ncn.run --root $R/p12 --device cuda:0` | `python -m ablation.p12_collab_ncn.collect --root $R/p12 --out $R/out/p12` |


p07 measures time: its GPU must run nothing else, and all its jobs should use one GPU model.

Environment: Python 3.11, torch 2.1.2+cu118, PyG 2.7.0, DGL 2.4.0, torchdata 0.7.1, packaging,
pyyaml, pydantic, psutil, pandas, scikit-learn. The test server used DGL 2.1.0 because
data.dgl.ai was unreachable.

Retry a job that failed three times: delete `failures.log` in its folder, then restart a worker.

## Additional datasets and paired comparisons

Install the transfer dependencies into the environment described above, then prepare the official OGB data.

```sh
python -m pip install -r ablation/transfer_requirements.txt
python -m ablation.download_transfer_data --data_dir data
```

The downloader records archive and input hashes in `data/ogb_transfer_manifest.json`.
Molecular tasks use official scaffold splits and ROC-AUC. Collab uses the official time split and Hits@50.

| Dataset | Backbone | Cohorts | Fits per cohort | Training fits | Derived independent ensembles |
|---|---|---:|---:|---:|---:|
| ogbg-molbace | GCN | 10 | 5 | 50 | 10 |
| ogbg-molbace | GIN | 10 | 5 | 50 | 10 |
| ogbg-molhiv | GIN | 10 | 5 | 50 | 10 |
| ogbl-collab | BUDDY | 5 | 5 | 25 | 5 |
| ogbl-collab | NCN | 5 | 5 | 25 | 5 |
| Total | | 40 | | 200 | 40 |

Each cohort contains four independently trained native models and one four-route GNNM.
The first native member supplies the single-model result. The collector averages the selected
predictions of all four native members to form the ordinary ensemble. Member seeds are
`1000 * cohort + member`; GNNM uses `1000 * cohort`.

All five recipes train for the reference 100-epoch horizon and select the first best validation checkpoint.
The molecular recipes use the official nonvirtual GCN/GIN models with five layers, width 300,
dropout 0.5, mean graph pooling, Adam at 0.001, batch size 32 and zero weight decay.
BUDDY and NCN use their pinned published Collab recipes, documented in their study README files.

The GNNM adaptations share the native backbone and add the original input/output factor mechanism.
Native encoder and readout weights are retained at initialization. Input R factors use random signs,
output R/S factors start at one, and private output biases start at the native bias.
Training averages member losses and inference averages raw logits.

The existing and additional workers total 11,875 training fits and 2,000 short cost jobs.
The transfer collectors derive 40 independent-ensemble evaluations from those fits.

## Multiple workers on one GPU

Use a separate root per study inside this repository on the shared filesystem.
Workers for the same study share one root and claim different jobs.

```sh
R="$PWD/rebuttal_runs"
for worker in 1 2; do
    python -m ablation.p09_molecular.run --root "$R/p09" --device cuda:0 &
done
wait
```

The same pattern applies to p11 and p12. The existing p07 cost measurements use exclusive GPU time.

For engineering checks, p09 accepts `--smoke_updates 2 --cohorts 0`,
p11 accepts `--smoke --smoke_max_batches 2 --cohorts 0`, and p12 accepts
`--smoke-updates 2 --cohorts 0`. Use dedicated smoke roots. Each smoke evaluates the complete
official validation set. Collectors require `--include-smoke` to process engineering outputs.

New source pins and adaptation details are in
`ablation/p09_molecular/README.md`, `ablation/p11_collab_buddy/README.md` and
`ablation/p12_collab_ncn/README.md`.
