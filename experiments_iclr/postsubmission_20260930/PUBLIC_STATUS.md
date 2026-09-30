# GNNM post-submission research

We are investigating when ensemble members should receive private graph transformations. The original benchmark scores are unchanged. No new methodological superiority claim has been established.

## Completed development experiment

The first prospective predictive screen compared four PPI ensembles: fully shared graph weights, training-derived private directions, equal-rank random directions, and equal-rank raw-gradient directions. Every arm began from the same supplied training checkpoint, with zero private coefficients and fresh AdamW. The study used all 20 official training graphs, both validation graphs, three paired seeds, and both 10% and 100% observed training labels. Each fit received 2,000 additional graph updates. Test predictions were not scored.

All 24 fits and 24 numerical checkpoint replays completed without a recorded failure. The selected directions **failed the fixed development continuation gate**. At 10% labels, they trailed all three controls in mean validation F1 and had worse pooled BCE. Most trajectories were still improving at the 100-epoch cap. This screen does not justify promoting the criterion or making a convergence claim. Complete results and adverse controls are retained in the research archive; a separate paired uncertainty audit is in progress.

## Mechanism and next experiments

A separate six-cell training-gradient diagnostic found that recurring improvements for individual member losses need not agree with the loss of the averaged prediction. This used two already examined training graphs, took no optimizer step, and supplies no generalization evidence. A new hypothesis will judge the actual optimizer update on separate training graphs using pooled loss, with direct pooled-training controls.

The molecular direction transfers block-sharing decisions to bond-aware GINE on MolHIV. Its data preparation retains all original graph features and checks every training graph's topology. Export repairs and failures are recorded; chemical predictive fits have not yet started. A declared trainable low-rank comparator is being prepared because low-rank same-task ensembles and gradient-based sharing already have close precedents.

## Reproducibility limits

The portable harness passed synthetic relocation/tamper checks and a prospectively chosen completed checkpoint export. On the recorded server runtime, it reproduced all 101 validation metric epochs, selection, and fresh selected-checkpoint logits. This is checkpoint replay, not fresh training or independent hardware replication. Large binary evidence has not yet been publicly released.

Parameter storage, latency, activation memory and method-selection overhead are reported separately. The completed study ran on CPUs; its unmeasured allocator peaks and concurrent timings establish no GPU efficiency advantage. Independent task confirmation and fresh manuscript review remain required.
