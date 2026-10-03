# Closed DBLP native and HGT development comparison

Descriptive comparison of recorded, checkpoint-selected validation scores. All 35 HGT and all 15 native fits were closed in the supplied root evidence; the existing native15 audit reports complete selected-state restoration/replay and original custody. This tool does not run a model or recalculate a model score.

## Shared input custody

The frozen studies have exactly equal archive, development-label and all five split descriptors, including paths, SHA256 and byte counts. Member names/hashes, source label-member hash and node counts also agree. The native paired-HGT freeze equals the closed35 audit freeze, and all shared input descriptors occur in native audit custody. Only those descriptors were inspected; archive, label and split payloads remained closed.

Five seeds (131, 137, 139, 149, 151) reuse one DBLP graph and its 1217-node development pool: 974 TRAIN and 243 VAL per split. The splits overlap; this is not five independent datasets.

## Means of selected raw scores

NLL is in nats per VAL node; F1 values are fractions. Means average the five recorded split scores.

| Recipe | Raw NLL | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|
| global_BE | 0.209826 | 0.931687 | 0.925515 |
| native_HGT | 0.206615 | 0.925103 | 0.917648 |
| untied_HGT | 0.184060 | 0.935802 | 0.929649 |
| native_GAT | 0.193320 | 0.939095 | 0.934039 |
| native_Simple_HGN | 0.738552 | 0.942387 | 0.937775 |
| native_SeHGNN | 0.368496 | 0.939918 | 0.935523 |

## Full per-seed recorded scores

Display values are rounded to nine decimal places; SUMMARY.json preserves every parsed source value. Epoch is the selected post-update checkpoint, not the paid training budget.

| Recipe | Seed | Selected epoch | Raw NLL | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| global_BE | 131 | 82 | 0.183041990 | 0.946502058 | 0.941145104 |
| global_BE | 137 | 48 | 0.274466455 | 0.909465021 | 0.902899640 |
| global_BE | 139 | 49 | 0.263223976 | 0.913580247 | 0.898580984 |
| global_BE | 149 | 52 | 0.172778279 | 0.938271605 | 0.935518471 |
| global_BE | 151 | 65 | 0.155618474 | 0.950617284 | 0.949428480 |
| native_HGT | 131 | 47 | 0.179731697 | 0.950617284 | 0.943563306 |
| native_HGT | 137 | 45 | 0.281309426 | 0.905349794 | 0.899526683 |
| native_HGT | 139 | 46 | 0.200101450 | 0.921810700 | 0.908814896 |
| native_HGT | 149 | 52 | 0.170787826 | 0.925925926 | 0.923490437 |
| native_HGT | 151 | 47 | 0.201144487 | 0.921810700 | 0.912842879 |
| untied_HGT | 131 | 50 | 0.145204082 | 0.958847737 | 0.953796588 |
| untied_HGT | 137 | 44 | 0.245011732 | 0.921810700 | 0.915759751 |
| untied_HGT | 139 | 42 | 0.211115539 | 0.925925926 | 0.914481296 |
| untied_HGT | 149 | 43 | 0.173388943 | 0.925925926 | 0.922507834 |
| untied_HGT | 151 | 43 | 0.145580769 | 0.946502058 | 0.941698516 |
| native_GAT | 131 | 196 | 0.126169965 | 0.950617284 | 0.945071010 |
| native_GAT | 137 | 102 | 0.227638617 | 0.934156379 | 0.930206354 |
| native_GAT | 139 | 92 | 0.219427854 | 0.930041152 | 0.920423138 |
| native_GAT | 149 | 77 | 0.275707662 | 0.913580247 | 0.909555569 |
| native_GAT | 151 | 117 | 0.117656142 | 0.967078189 | 0.964938420 |
| native_Simple_HGN | 131 | 98 | 0.722288668 | 0.946502058 | 0.940956614 |
| native_Simple_HGN | 137 | 76 | 0.752509654 | 0.934156379 | 0.931216330 |
| native_Simple_HGN | 139 | 80 | 0.747012734 | 0.934156379 | 0.924711126 |
| native_Simple_HGN | 149 | 93 | 0.748074889 | 0.942386831 | 0.940068310 |
| native_Simple_HGN | 151 | 98 | 0.722872734 | 0.954732510 | 0.951924186 |
| native_SeHGNN | 131 | 38 | 0.345203191 | 0.946502058 | 0.939360328 |
| native_SeHGNN | 137 | 43 | 0.397772163 | 0.930041152 | 0.928157926 |
| native_SeHGNN | 139 | 36 | 0.387205780 | 0.930041152 | 0.922092558 |
| native_SeHGNN | 149 | 56 | 0.370777696 | 0.942386831 | 0.941014938 |
| native_SeHGNN | 151 | 43 | 0.341520339 | 0.950617284 | 0.946989041 |

## Paired differences and illustrative t95 intervals

Each cell is mean HGT minus native recipe, followed by [illustrative t95 lower, upper]. Negative NLL means smaller recorded HGT raw NLL; positive F1 means larger recorded HGT F1. All 27 metric contrasts are descriptive. No interval is used as a significance or continuation gate.

For each metric, differences are matched by seed; sample SD uses denominator 4, SE=SD/sqrt(5), and the interval is mean +/-2.7764451051977987*SE (t quantile 0.975, df=4). These intervals assume independent approximately normal differences, which overlapping splits do not establish. They cannot quantify graph-population generalization or remove checkpoint-selection optimism.

| HGT recipe | Native recipe | Raw NLL difference | Micro-F1 difference | Macro-F1 difference |
|---|---|---:|---:|---:|
| global_BE | native_GAT | +0.016506 [-0.066831, +0.099843] | -0.007407 [-0.031483, +0.016668] | -0.008524 [-0.034787, +0.017738] |
| global_BE | native_Simple_HGN | -0.528726 [-0.585463, -0.471988] | -0.010700 [-0.024505, +0.003106] | -0.012261 [-0.029375, +0.004854] |
| global_BE | native_SeHGNN | -0.158670 [-0.201478, -0.115862] | -0.008230 [-0.020214, +0.003753] | -0.010008 [-0.026774, +0.006757] |
| native_HGT | native_GAT | +0.013295 [-0.081303, +0.107893] | -0.013992 [-0.042579, +0.014595] | -0.016391 [-0.048308, +0.015526] |
| native_HGT | native_Simple_HGN | -0.531937 [-0.580769, -0.483104] | -0.017284 [-0.035494, +0.000926] | -0.020128 [-0.040152, -0.000104] |
| native_HGT | native_SeHGNN | -0.161881 [-0.204110, -0.119652] | -0.014815 [-0.031214, +0.001584] | -0.017875 [-0.036388, +0.000637] |
| untied_HGT | native_GAT | -0.009260 [-0.075991, +0.057471] | -0.003292 [-0.020393, +0.013808] | -0.004390 [-0.023340, +0.014559] |
| untied_HGT | native_Simple_HGN | -0.554492 [-0.593718, -0.515265] | -0.006584 [-0.020390, +0.007221] | -0.008127 [-0.023221, +0.006968] |
| untied_HGT | native_SeHGNN | -0.184436 [-0.209387, -0.159484] | -0.004115 [-0.017143, +0.008912] | -0.005874 [-0.021308, +0.009560] |

## Recipe and output differences

| Recipe | Inputs and prediction | Optimizer and selection |
|---|---|---|
| global_BE | HGT feature2: provided author attributes, identity for other types; four members share a core with private BE factors; raw member logits are averaged | AdamW defaults/decay 1e-4 + OneCycle 300/max_lr 1e-3; TRAIN mean member CE, select mean-logit VAL CE; latest ties, patience 30/max 300 |
| native_HGT | Same HGT features/backbone; one factor-off member with an unconstrained final linear classifier | Same AdamW/OneCycle; latest ties, patience 30/max 300 |
| untied_HGT | Same HGT features; four separate cores, raw logits averaged; joint mean member CE with one selected ensemble checkpoint | Same AdamW/OneCycle; latest ties, patience 30/max 300; not four separately selected native fits |
| native_GAT | Identity inputs for every type, including authors; homogeneous support plus self loops; no final per-node L2 normalization | Adam lr 5e-4/decay 1e-4; latest ties, patience 30/max 300 |
| native_Simple_HGN | Same identity/homogeneous inputs; typed edges, feature/attention residuals; final four-class vector divided by its L2 norm (clamp 1e-12) | Adam lr 5e-4/decay 1e-4; latest ties, patience 30/max 300 |
| native_SeHGNN | All provided A/P/T attributes, venue identity, normalized metapath feature products; TRAIN-only one-hot labels propagated with diagonal removed after complete products; final class BatchNorm with affine=False/track_running_stats=False uses all 4057 authors | Adam lr 1e-3/decay 0; earliest tied minimum via strict improvement; 51 nonimproving epochs/max 200 |

Source anchors: `native_models.py:128-129` (Simple-HGN normalization), `native_models.py:200-201` (SeHGNN BatchNorm), `native_inputs.py:84-173` (metapaths, TRAIN-only labels and batch membership), `train_native.py:45-57,124-165` (selectors and original predictor), `train_dblp.py:97-135` (HGT logits, objective and selector), `families.py:22-67` (one/shared/untied members), and `hgt_private.py:144-163` (final HGT classifier). Their exact source hashes appear below.

**Simple-HGN's raw probability scale is bounded by its output rule.** In exact arithmetic, for four logits with L2 norm at most 1, any softmax class probability is at most 0.514018388, so each raw NLL is at least 0.665496241 nats (up to floating-point rounding). Write the target logit as a and the mean of the other three as b. The norm gives a^2+3b^2<=1, hence a-b<=sqrt(4/3); convexity gives sum(exp(other-a))>=3exp(-sqrt(4/3)). This is an analytic source invariant, not a calculation from predictions. Positive per-row scaling preserves argmax while limiting confidence. Its large raw NLL gap therefore cannot support a pure classification-quality inferiority claim.

SeHGNN also serves logits on a different scale: final class BatchNorm standardizes using the whole 4057-author evaluation batch, with no learned affine scale. The exact TRAIN/VAL/topology-complement composition was retained; no complement labels are inputs. Its label propagation, supplied attributes, model size (10,842,378 parameters versus 1,971,464 GAT and 2,308,872 Simple-HGN), optimizer and budget all differ. HGT hidden LayerNorm does not impose Simple-HGN sample-unit-norm or SeHGNN final-class-BatchNorm constraints on its final classifier.

Raw NLL evaluates the probabilities served by each recorded recipe and mixes calibration/scale with discrimination. F1 describes argmax classification at the NLL-selected checkpoint, yet remains affected by feature, architecture and selection differences. No temperature calibration, shared feature ablation, optimizer matching, reselection or causal isolation is performed here.

## Conditional reading

The recorded means show untied_HGT with the smallest raw NLL among these six recipes; the three native recipes have larger mean micro/macro-F1 than the three HGT recipes in this development comparison. Those observations summarize this fixed graph, these overlapping splits and their selected checkpoints. The paired tables retain the per-seed variation and do not establish broad superiority. In particular, Simple-HGN can have high F1 and high raw NLL because its normalization caps confidence.

The 243 VAL nodes both select the checkpoint and provide the reported scores, so this is development reuse rather than an independent final test. Macro-F1 uses the fixed four-class schema; micro-F1 is single-label accuracy. Five seeds cannot identify dataset variation or provide strong uncertainty calibration. All nine recipe pairs and three metrics are displayed without selecting a favorable subset; no multiplicity-adjusted testing or new gate is introduced. There is no heldout or ACM confirmation here, and no manuscript acceptance verdict.

## Reproducibility and exact inputs

Run `python3 summarize.py` from this directory to create fresh SUMMARY.json and REPORT.md; existing outputs are protected. Run `python3 summarize.py --check` to regenerate in memory and verify both existing outputs. INPUTS.json pins only the authorized closed score JSON, frozen/admission metadata and inspected source text. The script imports only the Python standard library and never opens data/labels/tensors or follows remote descriptors.

| Input role | Phase-relative path | SHA256 | Bytes |
|---|---|---|---:|
| hgt_decision | graph_heterogeneous_dblp_execution_root_v1/CLOSED_DEVELOPMENT_DECISION_ROOT_v1.json | fbd2e25cd9582260f2017a03ceae2d617f87ef732353489357b23e0fc6a0e9e5 | 108939 |
| hgt_driver | graph_heterogeneous_dblp_training_preparation_20261003_v2/train_dblp.py | 642c3478c2772d9b146fdea7e9b0cfbd97e2d319b38e428281d0441b1b8f77e8 | 19430 |
| hgt_execution_binding | graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v2/BINDINGS.json | 07f5c9933a7119f25692f83f58fb98d35a773410fd31257c326d6e5da3aed925 | 15987 |
| hgt_families | graph_heterogeneous_dblp_training_preparation_20261003_v2/families.py | 8f55059b017d91739e961b66dce5e489a5e24cd5e300da61343597bdbe81abec | 3117 |
| hgt_freeze | graph_heterogeneous_dblp_execution_root_v1/FROZEN_STUDY.json | 29885a100527226e9d182c54567e748f17f384254d23e009d9089fb18352d352 | 8461 |
| hgt_implementation | graph_heterogeneous_private_modulation_implementation_20261003_v1/hgt_private.py | 97a401fd289bbea9a21ae98d3b488ecea91a69547a2ec71d9d7392adb0195042 | 15378 |
| hgt_provenance | graph_heterogeneous_dblp_training_preparation_20261003_v2/PROVENANCE.json | bd836437858a9f0de5035bd26d5ec273e802f9bf572d59d05f40236f693578d3 | 8134 |
| hgt_reader | graph_heterogeneous_dblp_training_preparation_20261003_v2/dblp_inputs.py | 2f29f5839f53f711c9823c4f2671f4950e9c0e8e655000a0e5a5704944a9f866 | 9508 |
| hgt_recorded_score_auditor_source | graph_heterogeneous_dblp_execution_root_v1/audit_completed_development.py | 4fde2b2ba7527fa7e80ee27b1fb3b5cf8131a717874f5872f4dbaaafdd9e4a79 | 9463 |
| native_audit | graph_heterogeneous_dblp_native15_audit_execution_root_20261003_v1/NATIVE15_AUDIT_run01.json | c697a571abcc9bdef6f5d543ba8cda58cb394fa4784405653138d75e690f01ce | 72174 |
| native_driver | graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2/train_native.py | 5df1397e1f7aa518e5d8cb821e9d55f2449d246d40998deb096d9e16916e2bf9 | 19792 |
| native_freeze | graph_heterogeneous_dblp_native_execution_root_v1/FROZEN_NATIVE_STUDY.json | 274b293cefa8dc8257ede17a0e5e6b1fb2fa1b53392ce0a34227bf87fc46b87a | 7594 |
| native_inputs | graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2/native_inputs.py | 5951274a99d15e923c41875a3fec09f9775a5f8256901ea8b7cb18e1dfb95a68 | 8710 |
| native_models | graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2/native_models.py | 5a15de713edf2a5f185320f701877270b53ff1a5e7c42146513f88c32917404e | 10718 |
| native_study | graph_heterogeneous_dblp_native15_audit_execution_root_20261003_v1/NATIVE_STUDY_original.json | cf0c6a3f28a442f61e04b5072f409954b80de84bb26ebb201af8ea139b951fb4 | 11490 |

All 10 earlier review files retained their exact pinned hashes. Only the new comparison directory was written.
