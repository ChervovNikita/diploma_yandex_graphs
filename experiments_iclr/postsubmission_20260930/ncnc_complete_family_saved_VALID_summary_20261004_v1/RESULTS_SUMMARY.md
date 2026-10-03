# Complete NCNC development comparison

The complete saved validation results support a promising structural-design observation: maintaining each member's completion response improves prediction over pooling those responses before the decoder. All five paired seeds favor private completion. Its mean Hits@50 closely matches the independent ensemble. These values are validation-selected development results; selected-checkpoint numerical replay and heldout confirmation remain pending.

All 35 fits and 25 served cells are included. Existing five-seed, 100-epoch recipes, selectors, and complete official validation queries remain unchanged. TEST was not opened.

| Model | Mean validation Hits@50 (%) | Seed SD (pp) |
| --- | ---: | ---: |
| Native single (64) | 65.7496 | 0.6266 |
| Independent ensemble (4) | 66.4150 | 0.2152 |
| Shared ensemble, private completion | 66.4130 | 0.3501 |
| Shared ensemble, pooled completion | 65.5699 | 0.3353 |
| Capacity control (70) | 65.7433 | 0.2954 |

| Private completion minus control | Mean difference (pp) | Descriptive 95% interval (pp) | Positive / negative / tied seeds |
| --- | ---: | ---: | ---: |
| Native single (64) | +0.6634 | [-0.0142, +1.3410] | 5 / 0 / 0 |
| Independent ensemble (4) | -0.0020 | [-0.3118, +0.3078] | 3 / 2 / 0 |
| Shared ensemble, pooled completion | +0.8432 | [+0.0353, +1.6510] | 5 / 0 / 0 |
| Capacity control (70) | +0.6697 | [+0.2729, +1.0665] | 5 / 0 / 0 |

Only private-versus-pooled completion is the frozen primary contrast. Baseline differences are exploratory. The intervals describe variability across five training seeds on one fixed graph/time split; they assume independent approximately normal seed effects. They do not quantify transfer to new graphs. VALID also selected checkpoints, and the exploratory intervals have no multiple-comparison adjustment. N64 reuses the independently selected first native member; its fit cost is charged once to the independent bank.

The difference from the independent ensemble is approximately zero. These results therefore support investigating shared ensembles with private graph-completion responses; they establish neither superiority over independent ensembles nor a general state-of-the-art claim. The likelihood and relevant architectural ancestry remain prior work. The already frozen joint-pattern versus marginal-supervision pilot addresses the next methodological question.

Every seed, selected checkpoint identity, actual closed-attempt costs, native64 alias and failed-attempt history remains in the original family lock/ledgers. This report does not create a continuation threshold, authorize a new fit, change a selector, or release TEST.
