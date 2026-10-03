# Exact retained prospective comparison

Source: amazon_ratings_baseline_context_20261003_v1/REPORT.md, section4. This is retained specification text, not launch/adoption.


**Compare a four-member boundary GNNM Polynormer-r against four independently trained native Polynormer-r models on the existing Amazon 80%-TRAIN protocol.** This is an empirical extension of the saved backbone layout, not a new method claim. It tests whether GNNM's sharing preserves useful predictive quality on a source-qualified local-to-global Amazon backbone, beyond the current fixed-token PolyFormer transfers.

| Item | Fixed choice |
|---|---|
| Data roles | Existing official splits 0/1/2 and exact hash80% fit/20% TRAIN-control derivation; official VAL is the checkpoint selector; official TEST remains outside this proposed comparison |
| Block/member seeds | Blocks 17/29/43; independent member seeds `block_seed + 1009*m`, m=0,1,2,3; member 0 and the GNNM common-body initialization use the block seed; archive factor and dropout RNG states |
| Backbone | Pinned author ReLU model and Amazon release parameters above; raw features; bidirected edges; native self-loops; no LPF or extra structural channel |
| GNNM sharing | Saved `PolynormerBoundaryFamily` layout: shared native interior; private R/S/B only at `lin_in`, `pred_local`, `pred_global`; native biases copied to private B; Rademacher stem R, S=1, other R=1; complete private local hidden, GAT scores, global Q/K/V and reductions for every member |
| Training | Mean of four member TRAIN CE losses for GNNM; each independent model uses its own TRAIN CE. Same 200-local +2,500-global source schedule and Adam settings; no additional diversity/KD/error-allocation objective; no recipe search |
| Stage/selector | Strict official-VAL accuracy, earliest tie, spanning both stages as in native source. GNNM uses arithmetic-mean member probabilities for its VAL accuracy; each independent fit uses its own native selector. Restore the selected local-stage model/optimizer at the transition; explicitly serialize local/global flag. Report chosen stage for every fit |
| Pooling | Arithmetic mean of four class probability vectors for both arms; no learned weighting, temperature fitting, or member exclusion |
| Primary endpoint | Paired TRAIN-control NLL difference at selected checkpoints, reported per block and mean; report control accuracy, Brier score, and each member's NLL as descriptive companions |
| Cost evidence | Trainable/stored bytes, total preparation and fit time, peak memory, all four inference trajectories, and local/global stage count. Same recipe/trajectory count does not imply equal wall time or parameter count |
| Interpretation | Three block pairs give a bounded diagnostic, not a universal efficacy claim. No published-result pass threshold or numerical competence gate. Do not choose a new recipe or change the objective after control outcomes |

The saved adapter is an **unexecuted source draft**. Before an independently authorized fit, source parity and stage/checkpoint custody must be qualified; this packet does not certify them. This comparison is not launched or queued here.

Closest priors: Polynormer's local-to-global polynomial architecture; BatchEnsemble's shared W with member input/output scaling ([2002.06715v2](https://arxiv.org/abs/2002.06715v2)); the existing GNNM boundary-projector layout and its saved Photo backbone amendment. PolyFormer already supplies node-specific hop/order filters; ATLAS supplies topology/neighbor/label channels; Spexphormer supplies learned structural sampling. The remaining empirical question is the quality/cost effect of **parameter sharing within four complete trajectories on this Amazon-specific native backbone and fit protocol**. None of these inspected single-predictor tables measures that contrast. No global absence or scientific novelty is claimed. This does not duplicate NCNC decoder-completion work or reopen the FoRDE engineering packet.

