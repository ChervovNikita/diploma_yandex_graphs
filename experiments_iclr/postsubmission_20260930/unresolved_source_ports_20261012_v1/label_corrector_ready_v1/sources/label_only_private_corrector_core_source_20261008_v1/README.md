# Label-only private corrector core

8 October 2026. **Disabled callable prototype only.** No native capture adapter, full experiment driver, selector, owner, queue, runtime qualification or scientific fit is supplied. No data, checkpoint, server or partial score was accessed. Existing sources, single8 preparation and canonical records remain unchanged. This is an attributed composition of masked-label graph learning, label propagation/correction, BatchEnsemble factors, attention and stop-gradient treatment; no novelty or utility claim follows.

The external feature-only native backbone must supply current full-population `H[N,F]` and base logits `[N,C]` in the same node order. The core explicitly detaches both. It owns only four prediction correctors and their Adam. Native feature supervision, optimizer, RNG/buffers and native parameter updates remain caller-owned.

## Fixed operation

Each route uses shared bias-free Q/K/V/output matrices with pinned private BE `r/s` arithmetic. Q/K scores depend only on detached H and learned parameters. A single scaled dot-product attention hop normalizes over **all supplied incoming nonself edge records**, including hidden-label and unlabeled neighbors. Input duplicates retain their multiplicity/order; self records are removed. Selecting query rows retains every neighbor of each selected row, rather than sampling or removing its hidden-label neighbors.

Message values contain only embeddings of permitted TRAIN labels. All other nodes have exactly zero values. Value and output maps are bias-free and label-linear; there is no post-aggregation activation, raw-H value path, additional propagation or cached label-derived state. Empty neighborhoods and neighborhoods with zero visible label values produce exactly zero correction. Four corrected logits serve the fixed mean of class probabilities.

For `n>=2` permitted anchors, each step draws one uniform `k=floor(n/2)` TRAIN query set Q with a private CPU mask generator. The owner-issued immutable token is shared by every route. Every Q label is excluded from all four literal value fields; only CE target lookup reads those hidden labels. Values are rebuilt after masking. Training multiplies visible label values by `(n-1)/(n-k)` and minimizes mean-four own corrected CE on all Q. There is no pooled loss or auxiliary. Heldout inference requires query IDs disjoint from TRAIN anchors, uses all permitted TRAIN labels with **scale one**, and accepts no heldout truths.

The conditional first-moment identity applies to one-hop messages/residual logits at fixed parameters and mask-independent H/scores. **Probabilities, CE, pooled NLL and masked-loss gradients are not unbiased full-context estimates.** Shared Q correlates routes. A confident native TRAIN fit can provide weak correction CE feedback; the prototype adds no rescue loss.

With sparse label anchors, most attention mass may fall on zero-valued neighbors and Q/K fitting may receive little useful signal. The assessed all-neighbor denominator remains fixed. A future denominator over fixed potential-anchor neighbors A, still including masked Q anchors, would require an explicit separately defined successor. It is not an option or score-based rescue in this prototype; currently visible-only normalization is also absent.

## Initialization, updates and interface

Attention/value widths are fixed at 64, routes at four. Shared Linear/Embedding reset occurs under an isolated CPU initializer state; ambient CPU, Python and already-loaded NumPy RNG states are restored. No CUDA random draw or global manual-seed call is made. Q/K input factors use the pinned standard Rademacher reset with seeds `initializer_seed+900001/+900003`; all other factors start at one. Final output W starts at zero, so every initial correction is zero despite different feature attention. This is established diverse-factor/zero-residual initialization, not a new initialization claim.

The callable factory is:

```python
core = make_core(
    nodes=N, feature_width=F, classes=C,
    edge_index=observed_graph, train_ids=A_ids, train_labels=A_labels,
    initializer_seed=declared_initializer_seed, mask_seed=declared_mask_seed,
    device="cpu", later_execution_authorized=False,
)
```

The default refuses before torch import. A later separately authorized caller can draw `mask = core.draw_common_query_mask()` **before** its mask-independent native forward, then call `core.train_corrector_step(H, base_logits, mask)`. All four CE gradients accumulate at old corrector parameters before one fixed corrector Adam step (.001, eps1e-8, weight decay0). Its parameter versions and finite gradient/Adam state are checked. The core performs zero native updates. `core.serve(H, base_logits, heldout_ids)` or `core(H, base_logits, heldout_ids)` returns legal unscaled member logits and mean probabilities. No data loader, capture helper or scheduling interface exists.

## Native integration remains unresolved

The current public WikiBackbone interface returns H only for batch IDs; this prototype requires full `[N,F]` states. Its source at `core/models.py:72` already captures the full active prediction-head input and computes full logits before indexing batch IDs. A later separate read-only active-head hook can capture both during the same native forward, without an extra backbone forward or RNG advance. This is a feasible, unqualified integration route. It must preserve the chosen native feature recipe and prove that current H/scores do not read Q labels or depend on Q. No capture adapter or complete-cell readiness is claimed here.

Stop-gradient alone does not preserve the native training trajectory. For Polynormer, replacing the native own-local selector with an ensemble local selector can change the restored native model/Adam and its later trajectory. Future integration must define **native own-local model/Adam restoration**, restore the coherent corrector model/Adam snapshot from that same epoch, and preserve declared live end-local native/mask streams. Whole-ensemble final checkpoint selection can still differ from the native single's final selected epoch. Native baseline competence, snapshot custody, final serving and all selection costs require separate integration and full qualification. This core supplies no selected-state or exact-resume contract.

`ROLES_AND_SHAPES.json`, `MASKING_UPDATE_CONTRACT.json` and `WORK_COUNTER_CONTRACT.json` specify the interfaces and partial-work counters. `SOURCE_BINDINGS.json` pins the two authoritative notes, root constraints and exact factor/native source interfaces. `STATIC_CHECKS.json` contains source/AST/JSON inspection only. No numerical import, mini-training, synthetic fixture or runtime check was performed; peak memory, gradients and initializer/RNG behavior remain unqualified at runtime.
