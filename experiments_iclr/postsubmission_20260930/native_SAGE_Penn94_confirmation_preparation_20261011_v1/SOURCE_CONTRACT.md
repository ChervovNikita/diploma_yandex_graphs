# Bounded native Penn94 source contract

## Status and attribution

This is a reviewed source proposal, not an admitted executable study. Root chooses and freezes the finite acquisition/selector and binary scorer before execution. Ten fresh paired model seed blocks 0–9 are a proposal. No historical study is reopened. The original binary branch correction below is prospective and precedes any Penn94 outcome.

Two distinct comparisons are declared:

1. **Placement:** current internal-factor shared4 versus matched native boundary-factor shared4. Both use native Model CE2, identical common native weight/bias/normalization constructor seeds, first-input factor seed rows, persistent route dropout streams, mean own CE, AdamW, acquisition ceiling, patience and every-update SELECT checkpoint law. Additional internal factor parameters and their training trajectories are part of the placement procedure.
2. **Whole original recipe:** current internal/probability versus saved TABMModel boundary/logit. The original retains BCE1, Xavier projector W, no W bias, private zero-initial B, global sequential dropout, sequential scaled backwards and its checkpoint law. Its saved 5,000-update default is costed as a proposal, separate from the matched 1,000-update control. This contrast cannot be labeled a factor-placement or pooling-only effect.

For the same selected current and matched-boundary states, both logit mean and probability mean are predeclared secondary readouts. Cached member logits suffice. They isolate serving at fixed acquired predictions; they are not separate complete pipelines selected under logit mean.

## Data authority and public view

The selected opportunity is Penn94 published fixed split **0**, chosen by first index and public metadata. CUAI revision `82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6` lists the MAT blob `10f95de35049269ee41778c4aeb8e71bc98427bb` (7,021,246 bytes) and official split blob `29097c1d9d7637d8b0ce037ad9631cf8efeeb77b` (581,866 bytes). These are repository metadata, not acquired payload SHA-256s. No payload contents or exact D/E/role counts have been measured.

Retain all 41,554 public nodes and their original row identities, including unknown-target nodes. Custodian only: validate integer `local_info`, remove column 1 before any public model export, set private y to that column minus one, accept known classes 0/1 and missing −1. Missing targets are excluded from every supervised role and metric; they remain in message passing. No other metadata column or node is dropped by outcome.

Public features use the saved PyG LINKX loader's per-column unique/inverse and one_hot of the non-target metadata. Bind caller PyG one_hot implementation/provider hash and float32 convention. This differs from the author's `label_binarize`, which can encode a binary column with one output. **The chosen pipeline is PyG feature encoding plus author undirected, coalesced unweighted support**, consistently shared by all seven arms; it is not an exact reproduction of either stock complete loader. Freeze exact feature dimension/category order and canonical tensor hashes before models.

MAT adjacency coordinates use the saved PyG path `A.tocsr().tocoo()` row/col, converted by the custodian into complete int64 COO. Record any stored zero-weight coordinates explicitly; this policy keeps the extracted coordinates rather than silently switching to the author's `A.nonzero()` path. Symmetrize and coalesce directed support once, sort source/destination codes, retain original self loops once and add no loops. Weights are discarded, as native SAGE uses support only. Custodian must bind MAT sparse extraction, node mapping, original coordinate hash, final X/E shape/dtype/hash and decoded role hash. The helper itself cannot prove a caller supplied the complete source graph or removed the true target column; admitted provenance supplies that authority.

## Role custody

Official TRAIN, VALID and TEST memberships must be nonempty, disjoint and cover exactly all eligible known-target nodes. Preserve official TRAIN/TEST. Sort VALID ids, use **NumPy RandomState MT19937 seed 11709**, take its permutation's first floor half for CAL and remainder for SELECT; sort each resulting role. This fixes the previously unspecified permutation implementation without using targets/predictions for the choice. Any source/domain mismatch is a qualification failure, not an opportunity to silently choose a different split or population.

Custodian owns raw payloads and all private targets. Model-facing native data gets exact int64 TRAIN ids/targets only, converting y to float32 only for BCE1; other full-graph y entries are constant zero, with TRAIN mask. Public wrapper batches contain X/E and a graph with the same E object; they contain no y, role labels or heldout masks. SELECT truth is in a separate selector bundle; CAL truth is in a later calibrator bundle; evaluation truth remains in a separate scorer bundle. Evaluation ids are public membership, not evaluation targets. A future LINKX view reuses the same public X/E through `graph['node_feat']`, `graph['edge_index']`, `graph['num_nodes']`; it must likewise expose TRAIN labels only.

Role generation is custodial processing, not an assertion that no person ever accessed publicly released labels. Eligibility requires root's known-project scoring-history and current admission decision. The checked ledger/CONTEXT/README/reviewer tracker previously had zero Penn94 scoring identifiers. Do not assume unused WikiCS TEST/new masks or replace Penn94 with another population after outcomes.

## Seven primary banks

| Bank | Body/output/loss | Primary serving/selector proposal | Body fits per seed |
| --- | --- | --- | ---: |
| Current internal shared4 | Model; every native dense map wrapped by existing FactorLinear; CE2; mean own CE | Probability mean; common coherent-bank SELECT | 1 |
| Original boundary shared4 | TABMModel/BEBlock; input/output only; BCE1; sequential member BCE/4 | Raw logit mean then sigmoid; original checkpoint law with explicit accuracy override | 1 |
| Matched boundary shared4 | Model; FactorLinear at input_linear/output_linear only; CE2; mean own CE | Probability mean; same common coherent-bank SELECT | 1 |
| Ordinary native M1 | Unmodified Model; CE2; full own CE | Single softmax; common own-body SELECT | 1 |
| Internal-factor M1 | Model plus existing factors; CE2; full own CE | Single softmax; common own-body SELECT | 1 |
| Genuine internal-factor I4 | Four complete Model/factor bodies; own CE2, optimizers, RNG streams and selectors; no shared parameter storage | Probability mean of the four independently selected bodies | 4 |
| Fixed LINKX context | Saved author LINKX, two logits, native log_softmax/NLL path | Single softmax; first strict SELECT maximum; fixed 500 full epochs proposal | 1 |

Common SAGE proposal: depth 2, hidden 128, multiplier 1, LayerNorm, GELU/native residual stack, dropout 0.2, num_heads 8 (unused by SAGE), AdamW lr 0.001/weight decay 0, 1,000 updates maximum and 300 stale updates. At every update evaluate raw SELECT accuracy; save first strict improvement; ties are stale and preserve the earliest saved state. Shared banks optimize mean member CE with one combined backward. Singles and each independent body optimize unscaled own CE. No pool CE, SupCon, affinity, teacher or other auxiliary term contributes gradient.

Common seeds reuse the saved F law: for block b and member m, native seed `b + 1000003*m`, factor seed `native + 2000003`, dropout seed `native + 3000007`. Shared bodies use native member 0, route factor/dropout member m. Both singles use member 0. I4 body m uses native/factor/dropout member m, separate complete ownership and own selector. Current and boundary first-input Rademacher rows use the same per-row factor seeds; all later r/s start at one. Native W/b/normalization initialization is retained, and shared native biases remain shared. CPU construction/factor installation precedes device transfer and optimizer creation. No adapter replaces native SAGEConv/residual/FFN computation.

Original proposal: set paired block seed b once before native TABMModel construction; preserve its construction order, first Rademacher R draw, Xavier boundary W, S/later R=1 and private B=0. Retain global sequential four-route dropout rather than substituting common persistent streams. For each update call four complete forwards, backpropagate each mean TRAIN BCE/4 separately, then one AdamW step. Original native `output_dim=1`; ordinary/shared/matched controls use `output_dim=2`. Saved original ceiling 5,000 and PATIENCE 3 × 100 = 300 are proposed and openly unequal.

The exact original checkpoint law is unusual: initialize saved best to −infinity; if current SELECT accuracy exceeds **saved** best, reset stale even at a non-ten step, but update best/selected step/save only when step is divisible by ten; otherwise increment stale. It does not track the best of every update. A future implementation must preserve this law for the original native-recipe benchmark or explicitly revise/freeze it before fits. Preserve complete traces and failed checkpoint records. Do not call stock run_tabm, which evaluates TEST automatically.

Fixed LINKX proposal: hidden 128/layers 2 from one published-script grid point, parser defaults dropout 0.5, lr 0.01, decay 0.001, 500 epochs, adjacency/feature initialization layers 1, inner activation/dropout false, sampling false, directed false, default AdamW and native NLL. Bind paired block seed, constructor/reset sequence and first strict SELECT maximum before execution. Native final MLP BatchNorm buffers are retained and must be saved/restored. Source forward builds sparse adjacency on every pass and subtracts row.min(); qualification must establish this is identity on the complete admitted graph, or report a source integration issue for root review before freeze. This single point is source-backed, not demonstrated optimized paper performance or qualified predictive competence. Stock author evaluator accesses TEST per call and cannot be used.

## Binary scoring, calibration and complete closure

Original BCE1 primary probability is `[1-sigmoid(mean z), sigmoid(mean z)]`; raw accuracy follows the source's **strict mean logit > 0** rule, zero tie to class 0. Freeze the scorer to preserve this exact logit decision rather than accidentally change it through rounded float32 probability ties. The stock unknown-binary task default selects ROC-AUC; Penn94 must explicitly select **raw accuracy** according to its task benchmark. Current CE2 uses mean softmax probabilities, class argmax (first-index tie), with alternate softmax(mean logits); fixed LINKX follows argmax. No threshold is fitted. `raw_binary_probability_pool` returns probabilities only and is not an accuracy/NLL scorer.

The earlier prospective all-CE2 original design is superseded only for this new preparation by source-grounded original BCE1. No old artifact is modified. The distinct binary representations remain part of the original whole recipe contrast; the matched placement contrast is CE2 in both arms.

After all native selected states are frozen, the earlier design proposes one positive scalar temperature per fixed primary/secondary fusion on CAL only: for BCE1 sigmoid(z/T), for CE2 softmax(z/T), with the declared order of fusion. One T per whole I4/bank, no per-member T. Equal scalar capacity and label exposure. Proposed log T=0 initialization, 500 CPU float64 Adam updates, lr 0.01, betas (0.9,0.999), epsilon 1e−8, no decay, fixed final endpoint, no retries/grids. Seventy primary plus twenty same-state alternate CAL fits = 90 if ten seeds are frozen. Raw float32 accuracy remains primary. Metric precision, NLL/clamping, Brier and failure rules must be source-bound by root before execution; nonfinite endpoints remain failures.

The earlier assessment's proposed accuracy rule is mean +0.2 pp and Holm familywise 0.05 across candidate versus original, ordinary M1, factor M1 and genuine I4. Exact paired test/interval/symmetry sensitivity implementation and secondary attribution family remain root freeze items. Ten optimizer seeds are not ten independent graph replications. Confidence screens (+0.02 mean/+0.05 each-seed calibrated NLL harm) remain separately reported; failed protection accompanies an accuracy/confidence tradeoff. A weak member is diagnostic, not an extra accuracy veto. LINKX strength limits the headline to the SAGE-family claim where appropriate.

All selected models, failures, CAL endpoints, serving modes and hashes close before one separately admitted complete evaluation release. Retain every seed/cost/negative result. No TEST-selected seed top-up, repair grid, alternative population, fixed-key port or quadratic TRAIN dictionary follows.

## Qualification and next boundary

Use QUALIFICATION_CONTRACT.json for a meaningful representative complete Penn94 graph operation on each distinct integration. Bind data/provider/source/config/role hashes; verify full TRAIN loss and upstream gradients, actual parameter ownership/counts/seeds, role safety, native save/restore and selected readout, native original cadence, and LINKX sparse/BatchNorm behavior. Measure peak GPU and host memory, synchronized training/selector/restore/serving times and actual operation counts under normal host execution. No small favorable subset, toy numerical parity, proof, collapse/Gram test or synthetic dictionary is required.

Source availability, static parsing and finite gradients do not establish predictive utility. Qualification does not select a configuration by TEST or provide an ETA; admitted full-development competence diagnostics and failures must remain visible. Root binds current eligibility/input custody, provider/scorer/driver sources, final finite common acquisition/selector and seed/statistical proposal, then qualifies resources before scheduling. This packet does not admit or schedule jobs.
