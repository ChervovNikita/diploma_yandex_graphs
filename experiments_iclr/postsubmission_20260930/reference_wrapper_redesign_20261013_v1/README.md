# Revised GNNM campaign: unchanged reference backbones

## The comparison this campaign makes

For each fixed paper/source preset, train the native single, a four-member
native-preserving GNNM wrapper, and four genuinely independent native models.
The latter have separate weights, optimizers, training streams and checkpoint
selectors. Each model uses the same native architecture, data protocol,
learning rate, weight decay, dropout and optimizer groups. The single is
exactly member zero of the independent ensemble, avoiding a duplicate fit.
Published scores are a separate reference column; paired comparisons use our
newly trained models under the same complete experimental setting.

The former v29 80,850-fit proposal is withdrawn for new execution. Its sources,
decisions and evidence remain preserved. No scientific GO exists.

**This is a frozen proposed design, not an executable or completed benchmark.**
The portable boundary wrapper, training rules and native TRAIN/VALID smoke
adapters are implemented. Real-data checks are recorded in
`smoke_20261014/`; they include optimizer updates, native parameter ownership,
checkpoint restoration and a physical two-job memory probe. Individual failures
and source repairs remain in the receipts. Short checks do not qualify complete
epochs, independent checkpoint selectors, stopping, final TEST serving or the
three-GPU launcher. The 18.77 runtime remains pending. No benchmark fits have
been launched; the authorized engineering updates are separate from the study.

## Fixed experimental coverage

| Task | Native backbones / source presets | Datasets | Presets |
| --- | --- | --- | ---: |
| Node classification | PolyFormer Mono and Chebyshev paper/release recipes | CiteSeer, PubMed, Coauthor CS, Coauthor Physics | 8 |
| Molecular graph classification | OGB GCN, GCN with virtual node, GIN, GIN with virtual node; full features; separately reported published dropout 0 and 0.5 presets | MolHIV, MolBACE | 16 |
| Link prediction | Original NCN, NCNC and BUDDY source recipes | OGBL Collab | 3 |
| **Total** | **8 native architectures, including two fixed PolyFormer polynomial presets** | **7 datasets** | **27** |

There is no hyperparameter search or selection of a winning preset. Each
paper-declared setting is a separately reported experiment. OGB's paper says
it selected between dropout 0 and 0.5 but does not identify the selected value
for every published cell. Both values are therefore frozen and reported;
neither is falsely called the exact paper-winning setting.

Eight Polynormer paper/release rows are deferred until their head-count,
width interpretation and training-horizon differences are resolved. The two
HGB rows are excluded from this campaign because the paper's official TEST is
hidden online; local public-label validation cannot replace that benchmark.
These ten dispositions are recorded in `DEFERRED_CELLS.csv` and contribute no
registered fits. They were not removed because of measured outcomes.

## Models and training-only GNNM hypotheses

Every main comparison includes native M1, original-style boundary GNNM adapted
to preserve the native model, and ordinary independent I4. There are no F1,
internal/all-map placement, jointly trained untied bodies, equipped references
or capacity controls in this campaign. Graph, attention, memory, retrieval,
new-head and structural-operator modifications are also outside this study.

The four additional GNNM hypotheses use the same boundary wrapper:

1. **Mixed individual/ensemble loss.** Half the mean member task loss and half
   the task loss of the served ensemble prediction train all parameters.
2. **Private ensemble credit.** Shared native parameters receive mean member
   task gradients; boundary-private parameters receive the same half mixture.
3. **Fixed private example responsibility.** Shared parameters receive ordinary
   task gradients, while private factors receive fixed positive TRAIN-example
   weights, normalized within class. Included for node and binary molecular
   classification; no unsupported link-loss substitution is invented.
4. **Hidden supervised contrastive loss.** Fixed coefficient 0.05 and
   temperature 0.2 at a passive native prehead capture. Included only for the
   supported single-label node tasks; molecular minibatches and link targets
   are not silently given a different auxiliary objective.

The auxiliary definitions are fixed method rules from retained implementations,
not tuned native hyperparameters. Their portable training helpers and the
eligible native boundary smoke ports exist. These are separately declared
candidate methods, not a claim that every candidate is already better or novel.
The common inference rule for GNNM and I4 is the arithmetic mean of native raw
logits, followed by the native output interpretation. No learned aggregator,
calibration fit, test-selected ensemble member or pooled epoch reselection is
added. Historical GNNM scores and aggregation remain unchanged.

The historical generic TABM scaffold cannot be copied into a paper model: it
recreates matrices, changes native bias ownership and imposes its own scaffold.
The new port must retain every native module and W/b object and add only the
specified boundary factors. Categorical IDs remain categorical; input encoders,
normalization, samplers, losses and regularizers remain native. Each source
recipe must explicitly bind its input/output interface before execution.
See `methods/PRESERVATION_CONTRACT.md`.

The baseline is labelled **native-preserving boundary GNNM**. It implements the
original shared-backbone boundary-factor idea in the actual reference model;
it is not the historical generic scaffold substituted for that model. This
adaptation and its bias/aggregation rules are disclosed, and original-paper
scores are unchanged. Architectures with several categorical encoders, reused
heads or custom affine leaves require explicit interface adapters; an arbitrary
internal layer is never selected to fill a missing boundary mapping.

## Training adequacy and source-score checks

There are two explicitly separated tracks:

- **Source-horizon native check:** native M1 only, with its source cap and
  stopping rule. This checks our implementation against the reference score.
  Source-seed/runtime limitations remain disclosed; a successful check is not
  asserted before training. Published confidence intervals and standard
  deviations retain their original meanings.
- **Main matched comparison:** native M1, independent I4 and the chosen GNNM
  procedures all train with the same larger cap and declared patience. Native
  LR, WD, optimizer, dimensions, inputs and output rule remain fixed.

| Source family | Source horizon | Main maximum | Main minimum | Main patience |
| --- | --- | ---: | ---: | ---: |
| PolyFormer | 2,000 epochs, native patience 250 | 10,000 epochs | 2,000 epochs | 100 validation checks |
| OGB molecular | 100 complete epochs, no native early stop | 1,000 epochs | 100 epochs | 50 validation checks |
| NCN, NCNC, BUDDY | 100 complete source epochs | 1,000 epochs | 100 epochs | 50 validation checks |

Patience starts after the minimum and every native warmup/phase gate. All
primary presets use constant native Adam groups, so these larger horizons do
not silently rescale a fixed-horizon OneCycle schedule. Each independent member
uses its own selector and stopping state. GNNM selects its complete shared bank
using the same source validation metric. Curves, selected/completed epochs,
stop reasons and costs are retained. Reaching the cap without an adequate
plateau is recorded as inconclusive, not evidence of a weak method.

The extended-horizon results are explicitly different training procedures from
the source-horizon checks. A higher number after longer training is not claimed
to be a matched improvement over a historical published number; the matched
M1/I4 results establish that comparison.

## Source protocols preserved

- PolyFormer uses its ten declared 60/20/20 split seeds; these are not replaced
  by default Planetoid masks or a CAL subdivision.
- OGB molecular tasks retain official scaffold TRAIN/whole VALID/TEST splits,
  full atom/bond features, native batch size 32 and ROC-AUC evaluator.
- NCN/NCNC preserve their original Collab input masking, dropped-tail policy,
  complete official positive/negative pools and validation-edge TEST input.
- BUDDY preserves its source year-2007 filtering, source batches, sketch
  configuration and TRAIN+VALID test-time topology.
- There is no extra dataset, new split, simplified backbone or feature variant
  introduced to fill an empty cell.

The paper/release roster and directly read historical score cells are in
`SOURCE_CELLS.csv`, `source_protocols/PAPER_SETTINGS.md` and `references/`.
The source audit retains its initial candidate rows; final adoption is recorded
separately in `source_protocols/FINAL_DISPOSITIONS.md`. Nineteen exact saved
author-source files are copied under `native_sources/`, with hashes and original
paths in its manifest. These snapshots bind the model/training evidence without
overwriting historical source captures. They are not complete executable
dependency/provider trees.
Original-paper GNNM datasets Roman Empire, Amazon Ratings, Minesweeper,
Questions and Tolokers are excluded for all new methods.

## Exact finite work

Every fixed preset has ten paired seed/split trials. PolyFormer's ten source
split seeds form ten trials, not a ten-seed by ten-split Cartesian product.

| Work | Fits |
| --- | ---: |
| Source-horizon native checks | 270 |
| Main node classification: 80 trials x (4 native bodies + 5 GNNM procedures) | 720 |
| Main molecular classification: 160 trials x (4 native bodies + 4 GNNM procedures) | 1,280 |
| Main link prediction: 30 trials x (4 native bodies + 3 GNNM procedures) | 210 |
| **Total cold model fits** | **2,480** |

The 540 evaluation rosters and 1,940 logical prediction banks do not add
training. Member-zero reuse is exact within each main cohort. Source-horizon
fits are separate because their stopping contracts differ. No historical
weights, logits or completed scores satisfy a new fit obligation.

Proposed three-GPU assignment is 793 / 785 / 902 fits to allocation/GPU0,
77/GPU0 and 77/GPU1. Whole cohorts stay on one worker. Placement uses labelled
relative epoch/route-work proxies; it is not a measured GPU-hour balance or
ETA. At most two jobs per GPU may run, subject to their combined memory and
host storage reservations. Shared RAM/disk admission on 77 accounts for both
GPUs. Current preparation uses only the allocation; no 77 access has occurred.

## Reporting and remaining preparation

The report will show published native score, newly trained source-horizon M1,
matched longer M1, matched longer GNNM and matched longer I4, with the fixed
preset and horizon visible. New results retain per-trial values and paired
mean differences with 95% bootstrap intervals. The 226 predeclared contrasts
(each wrapper versus M1 and I4) receive Holm multiplicity correction for the
two-sided paired sign-flip tests, with their sign-symmetry assumption stated.
These intervals are marginal, not simultaneous confidence guarantees.
Ten seeds on one molecular scaffold split describe training randomness;
they do not measure variation across different scaffold splits. Incomplete
presets are reported rather than analyzed as a surviving subset.
All declared cells, failures, costs and unfavorable outcomes are
preserved. Claims about reliability follow the complete matrix rather than a
test-selected set of winning datasets or presets.

`PLAN.json`, `ROSTER.json`, `SUMMARY.json`, `FIT_ROSTER.csv` and `FREEZE.json`
bind this proposed design. Before any scientific launch: complete native
factories/providers, exact boundary/optimizer maps, source checkpoint capture,
full evaluation adapters, resource admission and executable three-GPU launch
integration. Existing v29 commands are retired. No GO is created by this file,
the source audit, metadata checks or compilation of the run list.

The public repository stores `PLAN.json.gz` to avoid committing the bulky
expanded JSON. The allocation keeps the raw plan as well. `build_design.py`
reproduces both from the fixed source roster and method definitions.
