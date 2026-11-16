# Final rebuttal campaign: how to run it

Read `PROTOCOL.md` first. It is binding: the protocol, the stages and the output
locations below must not change without the owner's request.

The campaign uses the existing launcher (`ablation/campaign`), environment and
data of the earlier campaign. Nothing of the earlier campaign root is reused or
written to.

## 1. Check the code (CPU, a minute)

```sh
python -m unittest ablation.fixed_protocol.test_fixed_protocol ablation.p14_collab_ogb.test_p14 -v
```

## 2. Build the plan

```sh
python -m ablation.campaign.plan_fixed_protocol --out ablation/campaign/plans/fixed_protocol_20261116.json
```

It prints the number of fits per stage (23,642 at most; 9,690 of them belong to
the conditional expansion of the search and run only if the validation rule
allows it). The plan file is about 50 MB and is generated, not committed.

## 3. Smoke one fit per new kind (GPU, minutes)

Use a separate smoke root; it is deleted afterwards and never collected
(the collector refuses smoke results).

```sh
P=ablation/campaign/plans/fixed_protocol_20261116.json
S=ablation/campaign/runs/fixed_protocol_smoke
python -m ablation.campaign.tasks --plan $P --root $S --device cuda:0 --smoke-updates 20 \
    --task transfer/collab_gcn/cohort0/gnnm_contrast
```

The node stages were smoke-tested on CPU before hand-off (main-table ensemble,
`gnnm_m1`, `xavier_rms`, `independent_projectors`, node contrast, scoring).

## 4. Launch

One launcher per node, all nodes with the same plan and root:

```sh
python -m ablation.campaign.launch --plan ablation/campaign/plans/fixed_protocol_20261116.json \
    --root ablation/campaign/runs/fixed_protocol_20261116 --jobs-per-gpu 4
```

Queue order (lower first): parity and main-table reruns (1), the staged search
(2, expansion 5), Collab GCN/GraphSAGE (3), Roman ablation splits 0-4 (4), splits
5-9 (6), node contrast (9). If time runs out, the ablation is complete on five
splits before any sixth split starts.

## 5. Check parity as soon as it finishes

The parity stage has ten fits (Roman Empire, GCN, depth 2, splits 0-4, BASE
and GNNM) that stopped by patience in the archive. Collect and compare:

```sh
python -m ablation.fixed_protocol.collect --root ablation/campaign/runs/fixed_protocol_20261116 \
    --out results_fixed_protocol_parity
```

In `parity.csv`, validation and test scores should be within a few tenths of a
point of the archived ones, and every run must stop by patience well before the
old limit. Training is not bitwise deterministic, so the stopping step may move
by a few hundred steps (a CPU check before hand-off: archived stop 1171, val
72.73, test 72.33; rerun stop 1010, val 72.85, test 72.47). If the scores are
systematically different or the runs stop much later, stop the campaign and
tell the owner.

## 6. Recompute Hits@20/50/100 of the earlier BUDDY runs (CPU, no training)

```sh
python -m ablation.p14_collab_ogb.rescore \
    --root ablation/campaign/runs/production_owner_20261112_giga --pair collab_buddy \
    --out ablation/p14_collab_ogb/fixed_protocol_buddy_hits
```

It only reads the saved prediction files of the earlier campaign.

## 7. Collect

Use a new output folder every time (the name must contain `fixed_protocol`):

```sh
python -m ablation.fixed_protocol.collect --root ablation/campaign/runs/fixed_protocol_20261116 \
    --out results_fixed_protocol_$(date +%Y%m%d_%H%M)
```

The staged search and Collab reports are written by the campaign itself under
`ablation/campaign/runs/fixed_protocol_20261116/tasks/report/`.
Send back the collector folder, the `report/` folder and
`ablation/campaign/runs/fixed_protocol_20261116/{plan_identity.json,events.jsonl}`.
