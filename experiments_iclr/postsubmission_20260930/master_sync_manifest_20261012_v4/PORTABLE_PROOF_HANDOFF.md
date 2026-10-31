# Portable master sync proof and environment recipe handoff

Status: **metadata proof complete; wait for root's final plan/controller before
building one final bundle**. No tar has been created. No server77/MacLink
access, package install/download, credential creation, tensor deserialization,
scientific fit, CUDA pilot, GO or phase-release receipt occurred. Existing
packets/common controller/running allocation environment are unchanged.

## Actual master proof

- Master: `massive_practical_master_20261012_v2/PLAN.json`.
- Master canonical hash: `96acea7bdb8bb8be5f5928c348136f3a6e187bff528ad01e3245916768cf9611`.
- Accepted manifest: `master_sync_manifest_20261012_v4/MANIFEST.json`.
- Manifest canonical hash: `7e479b8e06b9cf6cb201bdb16f2ee136a25049cd7f8160185dda3223662c86ed`.
- Exact union:1025 files,1603820194 bytes (1.494GiB),13 studies,18746 records,
  8 allocation-only author CLI groups. Every required source/config/role/spec
  binding is included and opaque bytes were checked. The layout/spec IDs are
  preserved.26 optional provider-provenance copies totaling144084110 bytes
  are excluded, with original receipts/possible relative aliases/reasons
  retained as metadata. No absolute runtime input is silently relocated.
- Eight allocation stdlib QA tests passed in0.979s (1.181s process), including
  actual frozen `launcher.load_plan` with an absent provenance-only absolute
  cache and the repeated-input hashing regression. Actual union audit4.271s.

Source tooling: `master_sync_tooling_20261012_v4/runtime/prepare_sync_bundle.py`.
It supports a fresh minimal manifest, optional verified regular-file tar,
no-clobber installation, exact manifest/plan/spec/file hashing, physical host
checks before project reads, exact environment metadata and optional CPU native
imports without CUDA initialization. Original five datasets are hard excluded.
Git/credential/history/quarantine/scientific-output/checkpoint paths and any
traversal/symlink/conflicting receipt fail. Current startup needs an existing
Git checkout and its plan-bound private remote-URL digest; no .git/URL/key is
bundled, and no alternate scientific launcher is introduced.

Local mirror: `preparation/master_sync_tooling_allocation_20261012_v4/allocation_artifacts/`.
Preserved attempts: v1 rejected absolute provenance; v2 superseded overbroad
provider copying/repeated hashing; v3 rejected heterogeneous evidence metadata.
Their source bytes/receipts remain. None produced an archive or scientific fit.

## Exact isolated environment recipe

- Allocation environment snapshot: `master_sync_tooling_20261012_v4/ENVIRONMENT.json`.
- Environment canonical hash: `96ca1fc422b22e985e1ac0ee47af919140087eb4034ad898911d9b681bf955b7`.
- Recipe: `master_sync_environment_tooling_20261012_v1/RECIPE.json`.
- Recipe canonical hash: `143707269a5c61e68ca6f71d378afc254f27566738a0b1ec46b90acddbb4f7e9`.
- Script: `master_sync_environment_tooling_20261012_v1/runtime/prepare_sync_environment.py`.
- Capture produced44 pinned installed distributions/57 dependency edges;
  metadata constraints are consistent. Syntax/capture/dry-run passed; no wheel
  was downloaded or installed. Local mirror is
  `preparation/master_sync_environment_allocation_20261012_v1/allocation_artifacts/`.

The source base is CPython3.11.14/Linux x86_64/glibc2.35/SQLite3.50.4/
OpenSSL3.5.5. This exact base must be supplied and qualified; pip cannot recreate
the interpreter/system libraries. Torch2.1.2+cu118/PyG2.7.0/
scatter2.1.2+pt21cu118/sparse0.6.18+pt21cu118 are pinned with the full observed
dependency closure. Future wheels are binary-only from public Torch cu118,
PyG Torch2.1.0+cu118 ABI and PyPI indices, with all transitive versions fixed,
offline installation and wheel SHA receipts. Setup creates fresh project-local
named venv/wheelhouse and rejects current/default/existing environments. Its
`recreate` mode requires later reauthorized server77 physical bindings and
the final manifest; it starts no scientific worker.

## Remaining root-owned integration

After root finishes resource qualification and the final minor startup fix,
regenerate one manifest from the **new final plan**, adding environment recipe,
script, helper and documentation receipts as explicit tooling dependencies.
Do not build/reuse a tar tied to this older proof hash. Preserve the v4 proof.
Only then build/hash-verify one immutable server-side archive and mirror its
compact roster/size/hash. No server77 connection occurs during this preparation.

All-study source bindings are validated on both hosts, so allocation-only
author source/raw/private dependency files must be mirrored for hash validation.
Their absolute Python/PYTHONPATH/LD environment is retained, and author groups
remain allocation-only. Server77 requires later authorization, frozen hostname/
GPU UUIDs, Git remote digest, exact environment/hash checks and resource
qualification before GO. Once synchronization and explicit human GO are
complete, the unchanged `start_massive.sh` remains one command per host.

One-second timing placeholders are **not an ETA**. The proof/recipe does not
qualify throughput, serialization, headroom or two-slot contention; only later
concrete resource measurements support an accurate experiment window.
