# Member quality and the remaining pooling gap

The completed MolBACE pilot suggests two separate targets: improve each route's
ranking and preserve useful differences that averaging can exploit. This note
uses existing selected validation predictions only. It performs no new inference
or training and changes no recorded scores.

| Model | Mean route AUC (%) | Pooled AUC (%) | Gain from averaging (points) |
| --- | ---: | ---: | ---: |
| Independent four | 73.410 | 74.847 | 1.438 |
| Base GNNM | 62.005 | 67.778 | 5.772 |
| Private BN statistics | 72.628 | 73.614 | 0.986 |

Base GNNM obtains a large gain from averaging weak routes. Private normalization
raises individual route quality, while the gain from averaging becomes smaller.
The final 1.233-point gap to independent four decomposes
arithmetically into 0.781 points of mean route gap and
0.452 points of smaller pooling gain. The latter is
an observed difference between ensemble summaries. It is not an identified
causal effect of diversity or a quantity that can be optimized independently.

This motivates the already proposed nine-fit follow-up. It tests geometry,
private correction and their combination with private BN, reusing the completed
BN-only and original ingredient controls. We will retain member quality, pooled
AUC and BCE together. A method that only spreads representations apart, or only
improves AUC while badly worsening probability loss, needs that limitation stated.
The known normalization correction is a reference mechanism, not new methodology.

All values average three paired seeds on the same151validation molecules.
Checkpoint selection used these labels. Independent and shared fits selected
different epochs. These observations do not establish causality, statistical
significance, new-data generalization or an advantage over independent ensembles.
Every seed and the exact arithmetic are in RESULT.json.
