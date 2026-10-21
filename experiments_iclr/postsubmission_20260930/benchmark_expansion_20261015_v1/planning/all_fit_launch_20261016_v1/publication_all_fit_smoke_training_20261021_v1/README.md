# Verified campaign launch source publication

This compact successor preserves the 21 October launch safeguards, process
names, readable coordinator, activation helper and dated evidence on
`codex/all-fit-smoke-training-20261016`. Its exact parent is
`6adbcf17b706f73a1cd878c20e046ba3f22fee39`. The live pilot checkout stays at
`48329b19941ea869c439e5a220a3df0e30aa1237` with its existing Git index.

[Campaign progress](../../../PROGRESS_20261021.md) describes the actual training
state. The original scientific design, 6,460 fits, 270 exports and three-GPU
assignments remain unchanged. Historical readiness records, the coordinator v1
staging timeout and the partially delivered v2 release remain preserved.

The snapshot contains source and compact receipts. Full smoke inventories,
candidate payload, raw command caches, datasets, checkpoints and predictions
stay in their existing evidence locations. Their identities are retained in
the selected receipts. `COMPACT_AUDITS.json` retains every scalar and receipt
field from the complete allocation and reconnected server51 audits, replacing
the long token arrays with counts and SHA256 hashes.

## Local preparation

`SELECTED_SCOPE.json` names every selected file relative to the phase directory.
Run `prepare_snapshot.py` once after the final launch receipt and continuation
are saved. It validates text, parses Python and JSON, writes read-only snapshot
files and records exact source hashes in `SELECTION.json`, `STAGING_MAP.json`
and `PACKAGE_MANIFEST.json`. It refuses to overwrite an existing snapshot.
Preparation performs no remote operation or Git mutation.

## Plaintext staging

`stage_plaintext.py --manifest-sha256 HASH` verifies the frozen local package
and prints its exact commands. Add `--stage` only for the authorized root
operator's dispatch. The hash must be the actual SHA256 of
`PACKAGE_MANIFEST.json`. The helper pins the reviewed coordinator v2 transport,
checks the allocation hostname and complete GPU identity, and uses ordinary
SCP to a fresh incoming directory. It reads back the verifier source before
invocation, verifies every staged file and then moves the verified directory
to the new publication path. Command and result records are saved beside this
package. A failed or timed-out delivery is preserved; inspect its evidence
before deciding how to reconcile it. Legacy encoded staging is excluded.

## Isolated commit and push

After staging, invoke the named remote `publish_source_successor.py` with
`--root` set to the verified allocation project, `--selection` set to this
package's `SELECTION.json` and `--receipt` set to a fresh `COMMIT_RECEIPT.json`.
The publisher requires the exact branch parent and live HEAD, uses a separate
Git index, commits only the selected snapshot bytes and advances the source
branch with compare-and-swap. It does not check out the publication branch.

`push_source_successor_v8.py` accepts `--commit-receipt` and `--push-receipt`.
It requires the same exact parent, checks the Git origin, uses a private
temporary SSH agent and the existing GitHub key through port 443, then verifies
the exact published ref. Credentials stay in the normal SSH input path.
The publisher and push helper preserve the existing live HEAD and index;
neither starts training. Root owns staging, commit and push, and their receipts
determine whether those actions actually occurred.
