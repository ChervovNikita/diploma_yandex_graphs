# Original SAGE reference admission for the GNCL and SupCon factorial

10 October 2026. **Reference map completed; root's guarded metadata/hash admission now passes.** The new [factorial protocol](../SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1/PROSPECTIVE_PROTOCOL.json) requires exactly 15 archived own-only banks and 33 selected-fit records. This map identifies those records and the admission contract. No model, checkpoint, data/prediction payload, remote host or TEST target was accessed by this agent, and no canonical state or frozen criterion was changed.

## Exact denominator and roles

| Original arm | Banks across three seeds | Fits per bank | Selected-fit records | Role in the factorial |
| --- | ---: | ---: | ---: | --- |
| ordinary_M1 | 3 | 1 | 3 | Contextual ordinary single |
| ordinary_genuine_I4 | 3 | 4 | 12 | Required original independent reference |
| factorized_allmap_M1 | 3 | 1 | 3 | Required capable own-only single for A |
| factorized_allmap_genuine_I4 | 3 | 4 | 12 | Required capable own-only I4 |
| shared4_unchanged | 3 | 1 joint shared fit | 3 | Base cell and required unchanged shared reference |
| **Total** | **15** | | **33** | |

Use seeds **7301 / 7403 / 7507** in every arm. `exchange` and `separable_equal_size` are not anchors. Their six fits remain part of the complete original **21-bank / 39-fit** history and costs, not the 15-bank admission denominator. Do not deduplicate separately acquired M1 and I4-member0 fits even when their selections agree.

The fresh six-arm family has **18 banks / 27 fits**. The joined comparison therefore has **33 banks / 60 fit records**, followed by the protocol's **165** bank-level scalar calibration fits. Fresh completion alone explicitly has `full_comparative_family_complete=false`. All anchors and fresh terminals must pass admission before any factorial result is interpreted.

## Local evidence already available

- Original and new CONFIG bytes are identical, SHA256 `c0b9c49b903af625755892a0761e89be73938cb48f21896c766cb0d7b3bf55a6`.
- [ACTUAL_COMPLETE_FAMILY.json](../common_wrapper_SAGE_root_20261010_v1/ACTUAL_COMPLETE_FAMILY.json) matches the original custody hash `33f3e394b2dace9bd0bab07be6ede3637ab2d1b4cab7f86498f51e48a3a2bd9c`. Original FREEZE matches the launch hash `c3cc6a37393c99bd44a33e8cf21b633b9d5bbedcafa9bf0ef5c2eefd1dbaa471`.
- The original launch records hostname `anogena-2-0`, sole GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`, source commit `005dea0ed713df7276cb20e556753f364a89dcc5`, direct child closure, complete family and TEST access false. This is the authorized SAGE family, not an excluded earlier allocation.
- `Family.make`, `seeds`, `scope`, `logits`, `metrics`, `progress` and `synchronize` have identical static ASTs in the original and factorial sources. Their full program hashes intentionally differ because the learning rules and fresh roster change. The preserved source review covers the new loss path; this comparison is not numerical qualification.
- All 15 original bank `selected_VALID.npz` paths and SHA256 hashes are available in the closed distribution reader's archive custody partition. Each of the 33 fits has a selected checkpoint path, byte size, step, completed updates, seed streams, selection/restoration metrics and costs.
- **No historical per-checkpoint SHA256 is present in the original local fit records.** Archive hashes authenticate bank outputs, not checkpoint files. [REFERENCE_INVENTORY.json](REFERENCE_INVENTORY.json) is preserved byte-for-byte as the pre-admission expected-record snapshot, with null/pending current hashes. The root receipt below now supplies the first current checkpoint identities.

## Root's exact admission checklist

| Check | Exact expected contract | Evidence root must bind |
| --- | --- | --- |
| Complete owner and roster | Original 21 / 39 closed; five anchor arms × all three seeds = 15 banks / 33 fits; no favorable subset | Original completion/custody/FREEZE bytes, authorized launch and closed owner identity; every bank/fit in the inventory |
| Native dependencies | Original fit program `12859e58…a8f3`; common routes `84ff13b2…0773`; factors `9f185dfe…f9c3`; native models/run_base/run_common hashes in inventory | Actual exact native files or immutable preserved source snapshot, constructor dependency identities and fresh source dependency bindings. A changed full factorial program is expected; changed native math is not silently accepted |
| Config and data | SAGE depth 2, width 128, dropout 0.2, LayerNorm, FFN multiplier 1; constructor `Model("SAGE",2,300,128,10,1,8,"LayerNorm",0.2)`; AdamW 0.001 / decay 0; maximum 1,000 / patience 300 | Identical config digest; exact TRAIN/VALID archive hashes and ordered numeric fingerprints. Preserve full graph and prepared edge order, no extra loop/feature transform. Rank 16 is inherited but not an operative baseline rank grid |
| Label roles | Complete 11,701 × 300 features, ten classes, TRAIN 580 and VALID 5,274; VALID is `(val_mask OR stopping_mask)[:,0]`; TRAIN keys x/edge_index/ids/y, VALID keys ids/y | Authorized allocation data receipt and the inventory's archive hashes, dtypes and ordered IDs/labels. VALID drives selection; TEST truth is absent from trainer roles. The fresh SC support check requires two TRAIN examples per class |
| Constructor and starts | Ordinary native bodies versus all-map factorized bodies; shared baseline is one common body with four routes; each I4 body independently constructed | Preserve native constructor/map/bias/norm behavior and baseline `kind`; no quartile/decoder head or side descriptor. Factorized input r starts as rowwise CPU-generator ±1 signs; every other FactorLinear r/s starts at one |
| Seeds and ownership | For route/member m: native = s + 1,000,003 m; factor = native + 2,000,003; dropout = native + 3,000,007; block seed = s + 4,000,037 | Every stored seed triple in all 33 records, independent I4 optimizers/streams and shared route scopes. Shared native weights start from s; I4 bodies beyond member0 use their own native seeds. Do not claim all-member equal-function initialization |
| Own-only update law | Shared: flattened mean CE over four routes and all TRAIN labels. Singles/I4: native own CE, separately optimized, no 1/4 divisor on an I4 body's objective | Exact old source and RESULT-to-complete equality; one factual forward/backward/Adam step per update; no GNCL, SupCon, teacher or auxiliary in reused anchors. Boolean-mask ordering for singles and ordered TRAIN IDs for banks remain aligned |
| Selector and restore | Post-update complete VALID float32 probability-mean accuracy; strict `>` only, earliest exact tie, no NLL tie-break; initialization ineligible; stale counter breaks at 300 | Original source defines model + optimizer + persistent CPU/CUDA stream restoration; exact RESULT equality binds selected step, completed updates and selection/restored metrics. Each I4 body keeps its own selected state; no joint I4 checkpoint reselection. No new model/native replay is required for this metadata admission |
| Selected state and raw archive custody | 33 exact selected.pt paths/byte sizes; 15 exact bank archive hashes; source-declared M1 logits [1,5274,10], shared/I4 [4,5274,10] | Record first current SHA256 for every stored checkpoint, labelled current admission identity; verify exact RESULT JSON equality to its old complete-fit record, recorded checkpoint path and byte size. Those hashes are not original historic-hash attestations. Original complete-owner/source/config receipts and already pinned bank archives supply the existing historical chain; no checkpoint deserialization or native replay |
| Serving and complete costs | Original float32 mean of member softmax, not mean logits; original float32 error/count authority; reported FP64 log-softmax/logsumexp NLL; historical work retained | Original source-declared archive keys ids/y/raw_logits/probability_mean/member_errors/pooled_errors and ordering; actual archive hashes; exact RESULT equality retains all 33 fit costs plus original whole-owner cost, with overlap/RSS scope. No payload decoding is added by this map. Reuse is not free fresh training or isolated timing evidence |
| Joined opening and calibration | 27 fresh fits + all 33 admitted original fits; equal fixed calibration of all 33 banks, five folds seed 11709 / final 500 scalar updates each | Root's complete source/reference/calibration freeze before launch and joined readout after complete admission. No original raw archive replacement, unused-validation claim, TEST access or outcome-selected replacement |

The exact data SHA256 values are TRAIN `88c36e1983f17a9e53d3f13a473baaf9eff86f52bc4be56aba012321983fdc6d` and VALID `591fe3a05d4bb80b9bfaf047292c04f91163abf60c43106d13f98046258268ad`. The machine inventory contains full native dependency digests, ordered-role fingerprints, archive hashes, all 33 state paths and per-fit metadata. A failed reuse condition requires a separately frozen successor roster, not selective substitution. The archived selected states were not loaded here.

## Completed root admission

Root's [REFERENCE_ADMISSION.json](../SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1/REFERENCE_ADMISSION.json), SHA256 `6942e2ecbc11f63c53f0fc4ef3fba6fff91c8081f95b7d22351bde0f9eef7b2d`, records `admitted=true`, exact source/config/role/seed/selector/custody compatibility, all 15 archive hashes and 33 checkpoint records, qualification passed, no new outcomes seen and TEST access false. Root reports exact current RESULT-to-original-complete equality and checkpoint path/size checks. Every recorded checkpoint digest is explicitly a **first current identity**, not a retrospective historic-hash attestation. No model deserialization or native forward was used. [ROOT_ADMISSION_STATUS.json](ROOT_ADMISSION_STATUS.json) binds this later authority; it supersedes the pre-admission inventory's pending status without rewriting that snapshot.

## Scope of a later result

The factorial tests one SAGE learning comparison on encountered WikiCS development. A+B, then B, then A is the predeclared later-confirmation priority only when its complete rule qualifies. The ordinary single cannot replace the capable single or genuine I4. A jointly optimized untied GNCL bank is absent, so the roster does not isolate a causal tying-by-objective interaction. One supported backbone is sufficient prospectively; it does not relax any original frozen gate. The three current history implications are saved separately in [IMPLICATIONS.md](IMPLICATIONS.md).
