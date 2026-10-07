# WikiCS closed24 selected development accuracy readout

Development accuracy on union of official validation and stopping masks, split0 (5274 nodes). All8 arms and3 seeds are retained. Accuracy differences below are in percentage points.

| Arm |6101|6203|6307| Mean | Sample SD | Range |
|---|---:|---:|---:|---:|---:|---|
| single | 81.2287 | 81.7596 | 81.4372 | 81.4752 | 0.2675 | 81.2287–81.7596 |
| single_contrastive | 81.0011 | 81.7027 | 82.0250 | 81.5763 | 0.5235 | 81.0011–82.0250 |
| independent4 | 82.0440 | 82.0440 | 82.0440 | 82.0440 | 0.0000 | 82.0440–82.0440 |
| independent4_contrastive | 81.9113 | 82.0061 | 82.0629 | 81.9934 | 0.0766 | 81.9113–82.0629 |
| be_unit | 81.0580 | 81.2666 | 81.0580 | 81.1275 | 0.1204 | 81.0580–81.2666 |
| be_init | 80.2048 | 80.4323 | 80.9822 | 80.5398 | 0.3997 | 80.2048–80.9822 |
| be_unit_contrastive | 81.5889 | 81.8544 | 81.4562 | 81.6332 | 0.2027 | 81.4562–81.8544 |
| be_init_contrastive | 80.4892 | 80.8495 | 80.8115 | 80.7167 | 0.1980 | 80.4892–80.8495 |

## Paired contrasts (declared and DESCRIPTIVE)

- be_init_contrastive minus be_init (primary): seeds 6101=0.2844, 6203=0.4171, 6307=-0.1706; mean 0.1770; sample SD 0.3083; signs +/−/0=2/1/0; exploratory paired95% df2 interval [-0.5888, 0.9428].
- be_init minus be_unit (declared_secondary): seeds 6101=-0.8532, 6203=-0.8343, 6307=-0.0758; mean -0.5878; sample SD 0.4435; signs +/−/0=0/3/0; exploratory paired95% df2 interval [-1.6894, 0.5138].
- be_unit_contrastive minus be_unit (declared_secondary): seeds 6101=0.5309, 6203=0.5878, 6307=0.3982; mean 0.5056; sample SD 0.0973; signs +/−/0=3/0/0; exploratory paired95% df2 interval [0.2639, 0.7473].
- be_init_contrastive minus single (DESCRIPTIVE): seeds 6101=-0.7395, 6203=-0.9101, 6307=-0.6257; mean -0.7584; sample SD 0.1432; signs +/−/0=0/3/0; exploratory paired95% df2 interval [-1.1140, -0.4028].
- be_init_contrastive minus independent4 (DESCRIPTIVE): seeds 6101=-1.5548, 6203=-1.1945, 6307=-1.2325; mean -1.3273; sample SD 0.1980; signs +/−/0=0/3/0; exploratory paired95% df2 interval [-1.8190, -0.8355].

## Failures and costs

Complete cells: 24/24. Failed or unlaunched: 0/24.

Actual family driver elapsed: 40202.135s. Separate metadata extraction: 19.617s wall, 17.936s process CPU.

REPORT.json retains every failure reason, selected development member metric, selected epoch, original terminal custody, reused/fresh resource cost and unavailable cost field.


## Interpretation limits

- The reported metric is development accuracy on union of official validation and stopping masks, split0 (5274 nodes). Stored source fields selected_VALID/member_VALID name this merged development population.
- This merged development population is repeatedly used for strict first-best checkpoint selection over1100 epochs, including the100-epoch local restoration. Selected development accuracy is optimistic for generalization.
- Published WikiCS test scores use a different held-out TEST population and evaluation context; direct comparison to these development scores is inappropriate.
- Three paired optimizer seeds share official split0 and the same graph. Seed dispersion and df2 t intervals are exploratory; they do not quantify uncertainty over graphs, splits, or independent query populations.
- No TEST, confirmation-seed results, checkpoint reselection, iid-node bootstrap, multiplicity-adjusted inference, or quality/novelty conclusion is supplied.
- Ordinary independent4 serves an evaluation-only bank from four independently selected epochs; other arms use a coherent joint selected checkpoint. Member metrics have those same selection identities.
- For ordinary independent4, final CLOSED_OWN_BEST_BANK member metrics are table authority. Earlier own-checkpoint member metrics, exact deltas, epochs and modes are retained as diagnostics from different original evaluation events; discrepancies do not discard final-bank scores.
- be_init_contrastive minus single and minus ordinary independent4 are DESCRIPTIVE baseline comparisons, not new declared primary or secondary contrasts.
- Any missing required seed makes the full three-seed summary unavailable. Observed seed values remain explicit; no complete-case aggregate or imputed score is calculated.
- Pool minus mean member accuracy is a descriptive score difference, not evidence about shared error support or a causal diversity mechanism.
