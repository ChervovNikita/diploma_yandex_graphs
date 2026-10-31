# Executable source-only correction screen

These three Python files were authored and inspected as text. They have **never been imported or executed** in this phase. No datasets/models were accessed and no fit, score, scientific test, GPU work or pilot occurred. `ast.parse` establishes syntax only. Runtime, gradients, native-baseline parity and resource sufficiency remain unverified.

## Entry point and inputs

`correction_screen_driver.py` supplies `prepare`, `environment`, `forecast`, `qualify`, `fit`, `report`, and `summarize`. It integrates the exact original wrapper `coordinate_ensemble_source_v1/models.py` (SHA256 `a74a87dc26b7675a2d3fa0aaf0e7734b4786fa6c49411ef52f9bb21c0516dbc1`) and verifies its immutable `propagation_cost_impl_v1/_pinned/frozen_models.py` dependency (SHA256 `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`). No Stage1 outputs, selection or development logits are inputs.

Fill `GRAPH_INPUT_TEMPLATE.json` separately for revised Squirrel and Photo with exact source release/preprocessing, graph dimensions, FP32 features `[N,F]`, and unique lexicographically sorted int64 undirected edges `[2,E]`, `0<=u<v<N`. Squirrel additionally needs the revised release's published masks as an NPZ with Boolean `[S,N]` arrays named `train`, `validation`, `pool`, at least three splits. Every released node/edge is retained. Feature processing and correction-edge canonicalization are completed and disclosed before input freeze; teacher propagation receives both edge directions and no added self loops.

`prepare` accepts no labels. It saves the exact three role partitions, FP32 APS uniforms, int64 member permutations, twenty final calibration/test partitions, and label-free NetworkX asynchronous label-propagation communities. Every cell is fingerprinted by `ROLE_FREEZE.json`. Rounding and mask transformation are in the driver/protocol. Label extraction must occur after this freeze and produce **five separate compact source NPZ files** containing only sorted `nodes` and int64 `labels` for train, validation, A, B and D. No full label vector or final-pool pack is accepted by `fit`.

Complete `ADMISSION_TEMPLATE.json` per cell with exact role/source-label hashes, the actual `environment` command output, exposure disclosure, source provenance and scientific authorization. Its teacher schedule is fixed: K4 original boundary-factor residual SAGE, width128, three blocks, affine LayerNorm, hidden multiplier1, dropout.5; mean member CE; AdamW(.001,.01), betas(.9,.999), eps1e−8; max2000/min300/patience200; source predictor-validation NLL of mean raw logits, earliest strict improvement. This is a **new prospective architecture/schedule/role setting using the qualified original wrapper**, not an original published-result rerun.

## Future executor commands — not run by this author

From this directory, with an admitted scientific runtime (`torch`, `torch_geometric`, `numpy`, `networkx`):

```sh
python correction_screen_driver.py environment --device cuda:0 --output /admitted/runtime.json
python correction_screen_driver.py forecast --input /admitted/squirrel_input.json --output /admitted/squirrel_forecast.json
python correction_screen_driver.py prepare --input /admitted/squirrel_input.json --output /admitted/squirrel_roles
python correction_screen_driver.py prepare --input /admitted/photo_input.json --output /admitted/photo_roles
python correction_screen_driver.py qualify --admission /admitted/squirrel_seed17_admission.json --output /admitted/qualification_squirrel_seed17
python correction_screen_driver.py fit --admission /admitted/squirrel_seed17_admission.json --output /admitted/fitted_squirrel_seed17
python correction_screen_driver.py report --frozen /admitted/fitted_squirrel_seed17/SCORE_FREEZE.json --pool-labels /admitted/squirrel_seed17_pool_labels_manifest.json --output /admitted/reported_squirrel_seed17
python correction_screen_driver.py summarize --reports /admitted/reported_squirrel_seed17/FINAL_REPORT.json /admitted/reported_squirrel_seed29/FINAL_REPORT.json /admitted/reported_squirrel_seed43/FINAL_REPORT.json /admitted/reported_photo_seed17/FINAL_REPORT.json /admitted/reported_photo_seed29/FINAL_REPORT.json /admitted/reported_photo_seed43/FINAL_REPORT.json --output /admitted/SIX_CELL_SCREEN.json
```

Repeat qualification prospectively for the other graph if its graph dimensions/edge regime create a concrete resource risk. Run full fits for all six paired source-split/seed cells; the example above expands identically to the other five cells. Qualification does three full teacher updates, two updates per gate/config, and two CF updates per backbone exercising epoch1 CE and epoch1001 CE+TPS branches. The latter is **not a 1001-epoch trajectory**. Qualification outputs have no score freeze and cannot enter reporting; no short-run utility conclusion or checkpoint reuse is permitted. Runtime qualification must verify all attempted arms, gradients, array contracts, cost and source-only label access before normal fits.

The source author executed only this static syntax pattern, without importing prototype/teacher/author modules:

```sh
python -c 'import ast,pathlib; [ast.parse(p.read_text(),filename=str(p)) for p in pathlib.Path(".").glob("*.py")]'
```

## Six primary methods and reporting boundary

Four identically sized signed gates: aligned, complete nodewise marginal, fixed nodewise alignment-shuffled, pooled. Each gets learning rates .001/.0003, max2000/min200/patience200. HeAD selects from APS/DAPS/signed/edge/v3 using A thresholds and D hard coverage/set size. CF-GNN gets two normal two-layer width64 choices (GCN/GraphSAGE sum), native dropout.5, Adam(.001,5e−4), TPS tau.1/target0, 1000-epoch CE warm-up, max5000/min2000/patience200. It is a declared fixed-teacher/source-role/harmonized-APS port. APS is eligible in every arm. Config/checkpoint selection is source-only; exact ties prefer APS, then config order, then earlier epoch. All traces, failed configurations, costs and states are kept, including methods emitting APS for source infeasibility.

`fit` saves complete member logits, score tables, selected states, environment, source traces and costs, then writes `SCORE_FREEZE.json`. `report` first verifies this freeze and role payload, then requires a separately supplied `POOL_LABELS_MANIFEST_TEMPLATE.json` bound to its exact hash. It reads final labels only as a compact pool pack, calibrates each already frozen method on twenty prepared uniform half-pool allocations, and computes reporting only. An exclusive `REPORT_STARTED.json` in the frozen directory blocks a second report even to another output directory; unsuccessful reports retain failure receipts.

Primary CF uses randomized APS; CF-selected unrandomized APS under shared kth calibration, APS/DAPS and published HeAD selector are separately frozen secondary results. The selector uses the declared mean over nonisolated nodes (all-isolated mean0); this unresolved native averaging convention prevents an exact-reproduction claim. No final-outcome oracle selects among secondary/primary wrappers.

Canonical point predictions are always argmax(softmax(mean raw logits)). Score rankings, minimum-score classes and singleton decisions can change. Reporting supplies class/degree/community/disagreement strata, all counts including small groups, rank change frequency, singleton hit/conditional accuracy and point disagreement cases. CF corrected-probability point metrics are separate from canonical teacher metrics. Saved logits support direct CPU64 point-metric recomputation; fresh-serving replay logit differences are recorded without a bitwise-determinism claim or inherited Stage1 gate.

## Timing and resource qualification

The cold serving measurement includes local graph/random-array/checkpoint reads, construction/loading/moving the teacher/corrector, and one full request. Each of ten warm samples recomputes all four teacher trajectories plus the selected correction; cached teacher logits do not substitute for serving. Covariance is computed only for the aligned/shuffled serving arms; marginal/pooled do not pay it. GPU synchronization and allocated/reserved peaks are explicit. CPU `ru_maxrss` is process-lifetime metadata, not an isolated method peak. Cold is a local loading measurement, not new-graph generalization or network acquisition latency.

Across six cells, caps are **12,000 teacher +96,000 gate +60,000 CF =168,000 full-graph training epochs** (distinct operation costs). Qualification provides actual per-update extrapolation inputs, not a convergence or duration guarantee. Input arrays cost 4NF bytes, canonical/directed edges16E/32E, logits16NC, uniforms4NC, permutations32N; complete node features4N(6C+1), both a/q tables16E, one score table4NC. Twenty final masks cost160 times pool count bytes (≤160N). Model weights/optimizer/autograd/selected states/traces/ten output score tables are additional. Gate edge chunks retain autograd state; no low-memory or matched-compute training claim. The `forecast` command uses manifest dimensions only and invents no time estimate.

Before admission, the independent executor still needs actual released artifacts, completed manifests, runtime gradient/selection/native-control checks, full-graph memory sufficiency and current scientific authorization. Source syntax success does not establish any of these.

The source-selected HeAD serving path evaluates only its frozen variant and needs pstar/APS/degree, without member-multiset sorting or unused family scores. Admission binds the exact SHA256 of all three prototype files; entry, score freeze and reporting-before-labels check them. The originally read admission hash is preserved through completion. Residual-inclusive fit wall receipts include setup/construction/checkpoint/selection work; final reporting includes per-method/allocation calibration and whole-report wall timing.

CF's secondary deliberately uses the shared kth order-statistic calibration. The author implementation's np.quantile(method='higher') can select a different order statistic; this secondary is not named literal native calibration.

Economical pooled serving retains pstar/pbar and degree with zero multiset/covariance slots, without member-vector sorting or centered covariance work. Marginal serving sorts the full multiset but does not construct centered a/q. APS bypasses correction features; CF uses pstar plus its selected correction only. Runtime fingerprints include TF32/matmul precision, deterministic-algorithm state and CUBLAS workspace configuration; numerical replay remains diagnostic with saved logits.
