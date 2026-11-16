# Frozen training protocol of the final rebuttal runs

Read this before launching, changing or extending any experiment of the final
campaign (`iclr-rebuttal-fixed-protocol-20261116`). It is binding for everyone
who runs the jobs, including automated agents.

## The protocol

Every node-classification fit (main table, ablations, hyperparameter search,
contrastive variant) trains exactly as the submitted paper did, with one change:

| Field | Value | Paper |
|---|---|---|
| Patience (steps without validation improvement) | 300 | 300 |
| Minimum number of steps | 0 | 0 |
| Checkpoint selection | best validation score, kept only on steps divisible by 10 | same |
| Step limit | **40000** | 5000 (20000 on Tolokers) |
| Width, learning rate, dropout, attention dropout, heads | 512, 3e-5, 0.2, 0.2, 8 | same |
| Seeds | BASE and GNNM: the split; ensemble member k: 1000 x split + k | same |

The search stage changes only the hyperparameters it searches (learning rate,
weight decay, dropout, attention dropout, heads), exactly as designed earlier.
The Collab GCN/GraphSAGE runs follow the official OGB example (see
`ablation/p14_collab_ogb/protocol.py`).

The values live in `ablation/fixed_protocol/protocol.py` (`FROZEN`,
`MAIN_TABLE`, `PROTOCOL_ID`). Every fit carries the protocol identifier, and the
campaign refuses to train a fit whose protocol fields differ.

## Why it is frozen

1. **Comparability with the paper.** The main table is reused from the archive;
   only runs that stopped at the old limit (or are missing) are retrained. This
   is valid only because the protocol is identical except for the limit: a run
   that stopped by patience before step 5000 would stop at the same step again.
   Any other change (patience, minimum steps, checkpoint rule) makes every
   archived number incomparable and forces retraining the whole main table.
2. **The earlier campaign is the cautionary example.** It switched to a minimum
   of 5000 steps, patience 1000 and per-step selection. Its ablations and search
   then could not be put next to Table 1, and the change helped the baselines
   more than GNNM (GNNM - ENS on Roman Empire fell from 1.09 to 0.80 points)
   for reasons unrelated to the method. Those results had to be redone.
3. **No selection on results.** Choosing or adjusting a protocol after seeing
   which methods it favours is selection on outcomes. The protocol was fixed
   before these runs; it is not tuned afterwards, whichever way results go.
4. **Ablations need one protocol.** A variant is compared with BASE, ENS and
   GNNM from the main table. That comparison measures the variant only when all
   of them were trained the same way.

## Rules

1. Do not edit `FROZEN`, `MAIN_TABLE` or `PROTOCOL_ID`, and do not pass protocol
   fields on the command line or in a hand-edited plan.
2. If a change looks necessary (for example, fits reach step 40000, or a model
   diverges), **stop and ask the owner**. Do not patch it locally. An approved
   change needs a new `PROTOCOL_ID`, a new campaign identifier and output
   roots, and a rerun of everything, including the main table.
3. Do not add, remove or reorder datasets, backbones, depths, splits, seeds,
   variants or search configurations without an explicit request of the owner.
4. A failed fit is retried with the same specification. Never change
   hyperparameters to make a fit finish.
5. Decisions use validation only. Test scores are read after selection, and
   every outcome, including failures and fits that hit the limit, is reported.
6. Outputs go only into new folders whose name contains `fixed_protocol`
   (for example `ablation/campaign/runs/fixed_protocol_20261116` and
   `results_fixed_protocol_<date>`). The code refuses the earlier locations
   (`results_merged`, `results*`, the earlier campaign root), refuses a root
   that belongs to another protocol or plan, and refuses to reuse a non-empty
   collector folder. Never delete or move earlier results.
7. Report the parity stage first: its ten runs stopped by patience in the
   archive and must reproduce their stopping step and score closely. If they do
   not, stop and ask before anything else is trained.
