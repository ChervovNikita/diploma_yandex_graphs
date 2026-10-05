# Independent D2 v2 repair review

5 October 2026. Reviewed `shared_private_transfer_fixed39_d2_analysis_preparation_20261005_v2` against preserved v1 and its independent review. Local source/schema/proposal reads, hashing, source diff and AST analysis only. No target import/function execution, numerical fixtures/statistics, real outcome/history/FREEZE/checkpoint/feature payload reads, server/MacLink contact or existing-file edits. This report is the only new artifact.

## Verdict

**The concrete v1 history-access ordering defect is repaired in the documented D2 consumer path. No additional concrete source defect was found in this bounded review.** This is a source/design verdict, not approval to collect outcomes or execute D2. The exact9 collector and retrospective inventory producer remain proposed, unimplemented and unauthenticated; the source remains hard disabled.

## Exact identity and preservation

| File in v2 | Bytes | SHA256 |
|---|---:|---|
| MANIFEST.json | 1180 | `518d3c1f1b5f5b9d0282a4c6ae044fd18991e371306bab8d5b1efaf4193da7fc` |
| SEAL.json | 698 | `50d27c2b3c3df8b20d9d8d249f4a6bc8fff9121f60f6d3fe83e95620307557d3` |
| analyze_d2.py | 37669 | `bd7c44b8a10b2bb8d3e8456ed8d8c8577252fd858cf012771723679a22e00fab` |
| ANALYSIS_RELEASE_DISABLED.json | 1577 | `9f16bcf9cf3f9266c8245eb1e61ec15b35e0771dcf85fd8de4b875c1e71623e6` |
| REGISTRY_INTERFACE_PROPOSAL.json | 9036 | `250eba1882bee5549d09faea7873286bb690e306ff9e8cba57b39ca0bd373500` |
| INPUT_BINDINGS.json | 6375 | `68c849ee2eb80a6afa16a8da4c24c209526769b6523d521075db08f06650cbe0` |
| REPORT.md | 4914 | `97f9cd64d6bfa0a6def6d00ea8e60d6b4c4b978e11742295c481b09a88bb146f` |
| CHANGES.patch | 14477 | `f42bc8994e7b65de3b9958585092e46f46fc9e37debbccdad069580f9a518f03` |
| CONTROL_FLOW_EVIDENCE.json | 3515 | `c0529f844b0dae58c2eb83da56d14940583a139d229d4abac5aad310caaa293b` |

Assigned v2 manifest/seal hashes match. All seven manifest entries and all26 INPUT_BINDINGS records match declared bytes/hashes. All six preserved-v1 manifest entries still match; its manifest/seal remain `3f90040012a0d2052c93eb526f1b258970da8e5972f7104f3006717aecaefcfe` / `c281417f684ece95b90bbd4180f537d7e89b9434f77156a86e427977ab2886e7`. Prior independent review remains bound at `e22fcc7cb89aa40186588af5ab7e062aa5703298b476646f47b9e4e51bd4a665`. No v1/v2 bytes were changed.

## Repair and noncircular inventory

The v1 call to `existing_selector()` inside each per-fit custody iteration is removed. V2's sequential main path is `release_scope:493 → populations:494 → custody:495 → selected_state_custody:496 → selector_audits:497 → Torch:502 → retained-logit load:507`.

- `populations:130–163` authenticates both exact complete ordered populations/common inputs before the per-fit pass. `custody:232–322` then checks both collection releases/plans/specs, all39 jobs/terminals/provider/source/input metadata, and every FREEZE/checkpoint/logit/history **byte** binding. It reads no retained FREEZE or history JSON. Every history binds at313 and is required to be `VALID_HISTORY.jsonl` beside its original `FREEZE.json` at314–315. The pass must return all39 at321.
- `history_inventory:214–229` requires a root-bound retrospective inventory, exact registry bindings, all39 ordered unique IDs, honest post-completion creation and explicit no-history/no-FREEZE-parsing/no-scores/no-pre-fit-authority flags. Every record copies its original registry FREEZE path/hash exactly (311–312). The proposal derives the history path from that FREEZE **path**, then observes a hash directly; it does not obtain an expected hash by early FREEZE deserialization. The hash is retrospective byte custody, with no invented pre-fit authority. Therefore its proposed creation needs no outcome-bearing JSON prerequisite.
- `selected_state_custody:325–352` starts only after the all39 byte pass. Its first retained FREEZE read is333. All39 semantic source/job/input/identity and checkpoint/logit/history-reference checks finish before any selector audit. FREEZE.history is compared to the already authenticated retrospective history hash at340–343; mismatched late companion or original references fail before earlier history-score audits. Available companion CONFIG observations are checked against the same FREEZE at344–349, without selecting states.
- `selector_audits:355–364` then audits every original frozen choice at360. `existing_selector:109–123` rehashes the admitted history against source FREEZE.history before parsing scores; it preserves the original cadence, rounded4 values and first maximum. Torch/logits still wait until all selector audits return; checkpoints remain hash-only.

The original static counterexample (valid original30 with a late bad companion spec/artifact) now fails inside the all39 byte pass, before any retained FREEZE/history score parsing. A wrong history hash/path also fails there. A history hash differing from source FREEZE fails in the all39 semantic pass, before selector audits. These are control-flow conclusions; no counterexample payload or numerical test was executed.

The inventory's actual creation order cannot be proven by its boolean declarations. Root's future collector/inventory review must authenticate that provenance and exact records. The current proposal explicitly leaves that implementation unresolved; this review makes no actual completeness/custody assertion.

## Metadata helper closure and hidden-outcome check

I independently rebuilt the AST call closure reachable from `custody`: **binding, custody, history_inventory, phase_file, provider_custody, read, require, sha**. It has no path to `selected_state_custody`, `selector_audits`, `existing_selector`, D2 functions or Torch. The nine JSON read sites are the science contract(235), collection release(242), plan(249), job(275), available-input manifest(284), source manifest(300), history inventory(216) and original/companion provider descriptors(169/180). Other bound objects, including prospective review/gate/runtime/cost evidence and CONFIG observation references, are hashed in this pass, not parsed.

This conclusion also checks producer schemas rather than relying on helper names:

1. The original30 registry is produced by bound `collect_cohort.py:183–210` (SHA256 `9e016f7f4318a81f22a0858d30a3c1e69895489dfc8761ec74f38f14e004966c`). It emits per-fit identities, source/runtime provenance, input/artifact hashes and an operational donor receipt; it does not emit selected cycles, MRR/Hits10, VALID histories or prediction values. The nested receipt is constructed by `run_queue.py:111–118`, augmented only with FREEZE path/hash at166–167, and copied by the collector. Its fields are process identity/exit/signals, resource/time observations, log hashes and custody/scope flags; stdout/stderr contents are not embedded. Source `run_queue.py` SHA256: `8a4dc8dcc8dd8b89c0198248b27d36780388033f05b8efffbbd15e9568a11f6b`. The historical collector itself parses FREEZE **metadata** at171; this review does not claim that prior collection never parsed FREEZE. Its emitted registry contains no quality fields, so the D2 consumer's pre-custody registry read is not a hidden score read under that producer.
2. Original jobs are prospective qualified base metadata plus fixed cell/schedule/source/authority/path/bounds fields (`freeze_queue.py:104–112`, SHA256 `5e73dd7223eba6728544c01ba47c6aaefd9daee1816fa884e5ecc8b936746bfb`). The reviewed original/companion base templates contain authority/runtime/input fields and references, without selected-state or VALID quality fields. Plans/science/provider/release/source manifests similarly carry fixed protocol/custody metadata. External-anchor values are not parsed by v2; the anchor file is hashed only.
3. The actual CONFIG producers (`original run.py:117–135`; `companion run.py:119–137`) write before training cycles and contain job/schedule/runtime/input, architecture/optimizer/dropout/serving/geometry and deterministic coverage metadata, without VALID scores or selected-state outcomes. CONFIG JSON is read only in the later semantic pass at348; its runtime observation is not an early outcome channel.
4. The proposed exact9 descriptor, registry and history inventory schemas list provenance/hash/operational fields, with no quality values. The companion queue receipt producers also add only FREEZE path/hash, not scores (`singleton/run_queue.py:111–118,168–171`; `gpu77/run_queue.py:113–120,177–180`). The future exact9 collector remains unimplemented: its actual output must be reviewed against this contract before release. This source review cannot authenticate arbitrary unreviewed JSON or its creation chronology.

## Statistics, gates and disabled interface preserved

Independent AST comparisons show byte-structure-identical function ASTs for **defined, undefined, error_correlation, d2, block_mean, block_summaries**. `release_scope`, `populations`, `provider_custody` and `packet` ASTs also match v1. All eight immutable authority/metric constants and `COMPANION_COLLECTION_INTERFACE_REVIEWED=False` match. Thus the fixed Citeseer shapes227/500, native tie-aware float32 quality checks, exact rational query relations, all member pairs/undefined cases, one-member identity, separate BCE normalization/Jensen reporting and three-block summaries preserve the previous reviewed formulas. No quality/promotion/selector gate is changed. The revised selector helper only receives the preauthenticated history path in place of constructing it from FREEZE.parent; its score logic is unchanged.

The false guard at16/489–490 still precedes the release read492; JSON alone cannot enable this source. The disabled release has null registry/history-inventory references, false root/custody approval, no fit/TEST/model/reselection authority and the unchanged selector. No actual numerical correctness, complete39 availability or scientific result is established here.

**Disposition:** v1's concrete ordering finding is resolved in v2 source. Preserve both packets. Actual exact9/inventory integration, complete custody, provenance/chronology and independent numerical verification remain requirements for a later separately sealed enabled source/release; they are not defects that this disabled preparation claims to have completed.
