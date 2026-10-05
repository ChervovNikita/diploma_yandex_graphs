# Disabled retrospective collector v2: complete inventory and confinement

5 October 2026. V2 is the focused successor to the preserved sealed v1 packet. It closes the final owned-root inventory/confinement prerequisite. It remains disabled before imports/reads; no contact, staging, collection, target import/execution, fixture, rerun or signal occurred. Independent assessment and root review are pending.

## Repair

The fixed project repository and execution root must resolve to their exact expected absolute paths. ROOT must have the expected project phase as its parent and remain inside the repository. Every file and directory, including every ancestor inside the project, is checked for confinement and absence of symlinks.

The collector inventories the complete root path set, including directories. Every ordinary file contributes a compact relative path, type, actual hashed byte count and SHA256. Directory entries have zero payload bytes and null SHA256. Special files and unknown payload paths fail before reading; the permitted ordinary inventory is precisely the original staged source/metadata plus files that the preserved qualifier/supervisor can generate. Raw input/model/checkpoint payload paths cannot be read through this inventory. Known logs and partition/partial RESULT outputs are hashed, never parsed or exported.

Every ordinary file is rehashed and the full path set is checked again before returning the inventory. Total owned file bytes remain bounded at64MiB, entries at512, and the single compact stdout receipt at64KiB. Inventory byte accounting uses the bytes actually hashed. The collector writes no remote files and exports no raw payloads.

## Preservation

All assertions in the existing `collect` function are AST-identical to v1, including original `physical_ownership_observation`, exact saved failure and transport1 handling, PID448525/start ticks6001542095, actual RESULT hash `9cd5f486bc0002353e980d2476aac15f218d0b92666cd5c63b921ebbbf3b4743`, and unchanged numerical/source/input/runtime/resource predicates. The v1 source proof of all27 original post-child assertion atoms remains preserved. V2 changes only the inventory/confinement helpers, their compact receipt fields, fixed repo literal and input-binding pin.

Original workflow success stays false. Original final successful-response checks remain unreached; the physical-observation gap remains unresolved by the collector. Retrospective assertion pass is separately classified evidence, with qualification and fit admission false. The source has no transport/client, process or allocation probe, subprocess, signal or model APIs.

INPUT_BINDINGS retains the44 permitted v1 inputs and adds exact v1 source/manifest/seal pins (47 total). V1 remains seven read-only files unchanged. V1 MANIFEST: `08c9255dca0c17acfa9e8d1cb9bf552f54488bd36af875127d7598907391e601`; SEAL: `5612073b396c8131423c99cb826b0b52dd4f7d9a7e1b5f4e3da43589ce0d21d5`. Full original diagnostic facts and limitations remain in its REPORT.

Only the separate v2 preparation directory was written. Verification was local source/AST/JSON/hash inspection. Original files/receipts, launcher, numerical source, science, canonical/index/publisher and all predecessors were preserved. Raw RESULT remains remote. Allocation contact and18.77/MacLink access did not occur; withdrawn access remains unchanged.
