# All-layer BatchEnsemble SAGE control

This is a post hoc Roman Empire control for the question of whether adapting
hidden graph layers helps the projector-only GNNM. It runs after the original
GPU queue and after the optional GAT failure-pair launcher has exited. The
original queue must have its final success marker. The GAT study may have
completed, stopped at its cutoff, or failed; the runner requires that neither
its launcher nor a GAT training job is active and freezes the observed GAT
status and artifact hashes in its protocol manifest. During training it holds
the GAT launcher's `.run.lock`, so a later GAT restart cannot take the GPU.

## Fixed comparison

- Official Roman Empire masks 0–4, with seed equal to mask index.
- Five residual SAGE blocks, width 512, LayerNorm, GELU, dropout 0.2.
- Four members, AdamW at learning rate `3e-5` and zero weight decay.
- Mean of the four training cross entropies, at most 5000 steps.
- Pooled validation logit accuracy at step 1 and every 10 steps; stop after
  300 steps without improvement. Save one checkpoint at the best pooled
  validation step, with the earliest step winning ties.
- The official test mask is scored after the checkpoint has been restored.

The control starts from the same `TABMAblationModel` used by `gnnm`. Its input
and output BatchEnsemble projectors stay unchanged. It adds member-specific
rank-one input and output factors and offsets to each SAGE neighbor and root
linear map and to both feed-forward linear maps in every residual block. Each
new hidden factor starts as an identity (`R=S=1`, `B=0`). The existing shared
weights are retained. The CPU audit checks exact equality of all four initial
member logits and equality of the post-construction random-number state for
all five seeds. The expected parameter count is 6,737,728 for projector-only
GNNM plus 133,120 hidden-factor parameters, or **6,870,848 total**.

The runner imports the existing training, validation, and prediction export
routine from `projector_controls.py`; it does not modify that active runner.
Its fixed result root is `experiments_iclr/all_layer_be_sage_results/`. Before
training, it freezes the protocol, runtime source and dataset SHA256 hashes,
selected comparator files, queue/GAT state, and Python/Torch/PyG versions.
The complete selected-control artifact verifier runs on the CPU, including
checkpoint reads, before the comparator set is frozen. Completed rows must
form an ordered prefix of the five masks. Each saved checkpoint and prediction
archive is checked against an artifact hash before a restart skips it. The
manifest also freezes the completed SAGE comparator CSV and split-0 adoption
audit hashes. CPU initialization/count checks and CUDA availability are
verified before a new manifest is written.

## Run after the queue and GAT study

From the repository root:

```bash
.venv/bin/python experiments_iclr/test_all_layer_be_sage.py
.venv/bin/python -u experiments_iclr/all_layer_be_sage.py
.venv/bin/python experiments_iclr/all_layer_be_sage.py --verify-only --require-complete
```

The second command also serves as the resume command. It exits before
creating a result directory while the original queue or GAT launcher is
active, or if the queue lacks its final success marker. An incomplete GAT
study is recorded in `protocol.json` and does not block this control. No
all-layer GPU run is scheduled by this document or by the CPU test. The third
command is read-only and requires an existing frozen protocol; it creates no
result directory or lock file.

Compare the five paired test accuracies and member disagreement values with
projector-only GNNM, `untied_backbone`, and pooled ENS. A gain shows the effect
of adding rank-one adaptation within the hidden SAGE maps under this fixed
protocol. It does not isolate parameter count or prove a general advantage
across datasets.
