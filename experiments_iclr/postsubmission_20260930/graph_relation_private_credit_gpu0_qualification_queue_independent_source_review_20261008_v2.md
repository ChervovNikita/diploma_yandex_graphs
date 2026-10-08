# Independent targeted review: GPU0 qualification queue v2

Date: 2026-10-08. Reviewed packet: `graph_relation_private_credit_gpu0_qualification_queue_source_20261008_v2`, MANIFEST SHA256 `7531d5ee0968c63251f56a9ed41774c449dfe80559f4468932f87c7cdad907de`; `queue_owner.py` SHA256 `45110c2e938823d196dd95f1edf0c0a53aaf2bb302dddb906a5e9fb1c24121a9`.

**The concrete v1 blocker B1 is resolved in the reviewed v2 source. No additional execution blocker was found in this targeted delta.** This is source clearance of the correction; root still supplies the exact enabled release and checks actual remote custody/resources. It is not a launch, successful qualification, fit completion, or comparative opening claim.

Scope: the exact `V1_V2_RUNWAY_GATE.diff`, the new runway/terminal gates and their call sites, their correspondence to the already inspected controller scheduling/closure receipts, and verification that other execution bindings remained unchanged. The scientific gradient review, native numerical paths, historical custody, and supervisor algorithm were not re-audited.

## Correction checked

- `family_live` now rejects either permanent `LANE_0_CLOSURE.json` or `FAMILY_CLOSURE.json`, regardless of the parent's continuing liveness (`queue_owner.py:84–100`). A closed lane cannot be reopened by a queue admission.
- `lane0_runway` reads and hashes one scheduling payload, requires a timestamp at most60 seconds old and not future dated, and checks the first unconsumed `6203_alphaF` receipt with no lane admission or child start. Finite nonnegative cumulative use and positive remaining time are required. Its conservative usable time is `min(reported remaining,108000-cumulative use)-receipt age-30 seconds` (`queue_owner.py:103–134`). This agrees with the controller's earlier cumulative lane budget and its reported family/next-fit reserve; it does not allocate a fresh lane window.
- The initial absolute deadline is capped by that existing runway. Every later scheduling receipt can only tighten it. Queue waiting also retains the existing600-second envelope reserve. Immediately before invoking the unchanged qualification engine, the queue requires the entire330-second qualification envelope plus60 seconds for commit/controller observation (`queue_owner.py:274–334`).
- Lane preparation and the last pre-replacement check each require a fresh60-second usable observation allowance. The final check also retains the queue deadline/stop test, rechecks parent/terminal custody, and then publishes the sole lane0 addition (`queue_owner.py:193–248`). Existing index hashes, exclusive reservations, consumed lane1 equality, and fresh pending files are retained.

## Preservation and limits

The v1 manifest still matches its frozen hash. All ten v2 payload hashes/sizes match. `SOURCE_BINDINGS.json` differs only by the declared30/60/60-second runway constants. Qualification supervisor, its constants-only diff, launcher, and both disabled templates are byte-identical to v1. The scientific source, controller, family release42916409…, consumed lane1daa2cb10…, pending cell roster/order, activation names, no-retry rule, and denied anchor/comparative authority remain bound as before. AST compilation was performed without executing any source module.

Actual scheduling receipt freshness, owner liveness, free memory, dependency cleanup, qualifier completion, and controller consumption were not observed remotely. A parent may stop or fail after index publication; the new commit metadata explicitly declines to guarantee subsequent liveness or consumption. The sole-writer reservation remains an operational requirement for root. This limit does not recreate B1: v2 no longer accepts an already terminal or expired lane based solely on a live parent.

No numerical imports, fixtures, data/checkpoint/outcome reads, remote commands, source mutation, or live index mutation occurred. Only this review artifact was written.
