# Route quality after private BatchNorm running state

The completed three-seed follow-up improved selected validation AUC over base
GNNM in all three seeds. This analysis asks whether that change accompanies
better individual route predictions or a larger benefit from averaging them.
It also records BCE and representation geometry, including unfavorable changes.

Use every completed follow-up seed (101, 202, 303) and its own already selected
checkpoint. Run the existing, unchanged `analysis/selected_molecular.py` once
per checkpoint. Its `run_member` call enters the candidate's registered private
BN context, and the checkpoint restores those buffers. Compare its reconstructed
mean logits against the saved selected export with the analysis's existing
diagnostic tolerance (rtol 1e-4, atol 1e-5). Preserve warnings without retrying.

The fixed work is three constructions, fifteen complete validation batches and
sixty native route forwards across 151 validation graphs per seed. There is no
training, recalibration, optimizer update, checkpoint selection or test scoring.
Keep tensor outputs on the allocation. Fetch compact reports only.

Compare mean member AUC, pooled AUC, the difference between these quantities,
member and pooled BCE, and recorded error overlap with the existing base-GNNM
and independent-ensemble diagnostics. Each of the three seeds is one training
replicate. Graphs, routes and positive-negative pairs are not independent
training replicates. The validation set already selected the checkpoints, so
these are explanatory development results.

Different selected epochs and training histories prevent attribution of a
difference solely to the inference buffers. The known BN technique is a
normalization control, not a novelty claim. This analysis neither adds a recipe
to the 6,460-fit roster nor selects another pilot automatically.

Execute sequentially on the dedicated allocation, with two CPU threads and one
temporary GPU slot. Require at least 12 GiB cache-aware host memory after the
parallel provider gate's reservation and 8 GiB free GPU memory. Each child has
a 180-second timeout and no retry. Preserve any failed attempt and its work.
