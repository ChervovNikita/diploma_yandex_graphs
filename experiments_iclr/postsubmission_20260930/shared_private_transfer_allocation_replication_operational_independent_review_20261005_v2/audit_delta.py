"""Narrow stdlib static V2 repair review; no source import or operational execution."""
import ast
import difflib
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V1 = BASE / 'shared_private_transfer_allocation_replication_preparation_20261005_v1'
V2 = BASE / 'shared_private_transfer_allocation_replication_preparation_20261005_v2'
R1 = BASE / 'shared_private_transfer_allocation_replication_operational_independent_review_20261005_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def emit(name, value):
    with (HERE / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


assert sha(V1 / 'MANIFEST.json') == '5a667680152ff17b0e19aff2beb367f84b064013b5bc48d2b55d01318f59f4e9'
assert sha(V1 / 'SEAL.json') == '4fa4418626f02670cd9e09472673252445b60babadf47f6970f899a377b8e8b6'
assert sha(V2 / 'MANIFEST.json') == 'c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'
assert sha(V2 / 'SEAL.json') == '074f75fb8d42ecb9614a1ad04607aa2ad16ee395364989fcc2653606f0988130'
assert sha(R1 / 'MANIFEST.json') == 'ea3c45758cb86ce2b231f9efce5f3e780aaf5bae1bcad7d312b9cf4d18eec219'
assert sha(R1 / 'SEAL.json') == '3979f077588ba18727abecb2b6fd95aadc9778f0e7de0e1067a45fabe7bc06bb'
inputs = [folder / name for folder in [V1, V2, R1] for name in ['MANIFEST.json', 'SEAL.json']]
members = {}
for folder in [V1, V2]:
    manifest = read(folder / 'MANIFEST.json')
    assert read(folder / 'SEAL.json')['manifest_sha256'] == sha(folder / 'MANIFEST.json')
    members[folder.name] = {q['path']: q for q in manifest['files']}
    for entry in manifest['files']:
        path = folder / entry['path']
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        inputs.append(path)
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
old, new = members[V1.name], members[V2.name]
assert len(old) == 52 and len(new) == 54
assert set(new) - set(old) == {'COLLECTOR_CUSTODY_REPAIR.patch', 'REPAIR_DELTA.json'}
assert not (set(old) - set(new))
changed = [name for name in old if old[name]['sha256'] != new[name]['sha256']]
assert set(changed) == {'REPORT.md', 'STATIC_VERIFICATION.json', 'collect39_metadata.py'}
unchanged = sorted(set(old) - set(changed))
assert len(unchanged) == 49
repair = read(V2 / 'REPAIR_DELTA.json')
assert sorted(repair['unchanged_files']) == unchanged
for record in [repair['preserved_v1'], repair['preserved_independent_finding']]:
    for ref in record.values():
        assert sha(BASE / ref['path']) == ref['sha256']
assert repair['prepared_source_import_execution_remote_contact'] is False
before = (V1 / 'collect39_metadata.py').read_text()
after = (V2 / 'collect39_metadata.py').read_text()
functions = lambda value: {q.name: ast.dump(q, include_attributes=False) for q in ast.parse(value).body if isinstance(q, ast.FunctionDef)}
old_functions, new_functions = functions(before), functions(after)
assert set(old_functions) == set(new_functions)
assert [q for q in old_functions if old_functions[q] != new_functions[q]] == ['new_prefit']
non_functions = lambda value: ast.dump(ast.Module(body=[q for q in ast.parse(value).body if not isinstance(q, ast.FunctionDef)], type_ignores=[]), include_attributes=False)
assert non_functions(before) == non_functions(after)
patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=str((V1 / 'collect39_metadata.py').relative_to(BASE)), tofile=str((V2 / 'collect39_metadata.py').relative_to(BASE))))
assert patch == (V2 / 'COLLECTOR_CUSTODY_REPAIR.patch').read_text()
assert "launch['launch_intent']['path'] == donor['donor_directory_relative']+'/LAUNCH_INTENT.json'" in after
assert "c.donor_binding(donor,launch['launch_intent'])" in after
assert "if d['family'] == descriptor_before['family'] and d['block'] == descriptor_before['block']" in after
assert "len(selected_predecessors) == 1" in after
assert "chosen_predecessor['donor_directory_relative'] == descriptor_before['execution_directory_relative']" in after
assert "predecessor['block_freeze']['path'] == descriptor_before['execution_directory_relative']+'/BLOCK_FREEZE.json'" in after
assert "predecessor['block_freeze'] == chosen_predecessor['block_freeze_binding']" in after
assert "c.donor_binding(chosen_predecessor,predecessor['block_freeze'])" in after
assert "c.binding(launch['launch_intent'])" not in after and "c.binding(predecessor['block_freeze'])" not in after
old_static, new_static = read(V1 / 'STATIC_VERIFICATION.json'), read(V2 / 'STATIC_VERIFICATION.json')
assert all(new_static[k] == v for k, v in old_static.items())
assert set(new_static) - set(old_static) == {'v2_minimal_repair'}
for filename, field in [('ROOT_RELEASE_DISABLED.json', 'execution_authorized'), ('ROOT_JOB_REVIEW_DISABLED.json', 'all26_generated_jobs_reviewed'), ('B0_CUSTODY_DISABLED.json', 'root_authenticated_all13_terminal_and_artifact_bytes'), ('COLLECTION_RELEASE_DISABLED.json', 'root_collection_approved'), ('HISTORY_INVENTORY_RELEASE_DISABLED.json', 'root_history_inventory_approved')]:
    assert read(V2 / filename)[field] is False
inputs.append(R1 / 'FINDINGS.json')
bindings = [{'path': str(q.relative_to(BASE)), 'bytes': q.stat().st_size, 'sha256': sha(q)} for q in inputs]
checks = {'V1_and_review_preserved': True, 'V2_members_authenticated': 54, 'unchanged_V1_members': unchanged, 'changed_V1_members': changed, 'added_members': sorted(set(new) - set(old)), 'only_changed_Python_function': 'new_prefit', 'all_non_function_collector_AST_unchanged': True, 'patch_matches_actual_bytes': True, 'current_intent_exact_original_path_and_current_donor_replica_join': True, 'predecessor_unique_family_block_selected_donor_lookup': True, 'predecessor_exact_original_directory_and_path': True, 'predecessor_reference_equals_collector_selected_donor_freeze_binding': True, 'predecessor_replica_join_uses_predecessor_donor': True, 'all26_jobs_amendment_history_and_other_operational_source_byte_unchanged': True, 'disabled_state_preserved': True, 'unresolved_findings': []}
emit('CHECKS.json', checks)
emit('SOURCE_BINDINGS.json', {'inputs': bindings, 'prepared_source_imported_or_executed': False, 'numeric_result_prediction_model_label_or_history_payloads_opened': False})
emit('REVIEW.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_NARROW_V2_REPAIR_DISABLED', 'preserved_finding': {'review': R1.name, 'finding_id': 'R1', 'priority': 2, 'sites_in_V1': [73, 111], 'status': 'resolved in separately sealed source V2'}, 'unresolved_concrete_findings': [], 'source_manifest_sha256': sha(V2 / 'MANIFEST.json'), 'root_execution_approval': False, 'manuscript_certification': False, 'scientific_execution': False, 'remote_contact_or_current_custody_checks': False, 'scope': 'Only concrete donor-relative custody repair and preservation checks; prior V1 unchanged science/operational checks inherited.', 'next_gate': 'Root current b0/GPU/source/runtime custody and separate enabled release, then bounded review of all26 actually generated job/queue metadata before the first replication fit.'})
with (HERE / 'REPORT.md').open('x') as handle:
    handle.write('''# Independent operational allocation replication review — narrow V2 repair

**PASS for the concrete V2 repair; disabled state preserved.** The P2 finding from the separately sealed V1 review remains recorded and is resolved at both sites. No unresolved concrete defect was found in this bounded delta. This is not execution approval or manuscript certification.

The collector now requires the emitter-fixed current donor LAUNCH_INTENT path and resolves it with that donor's authenticated replica mapping. For each predecessor, it finds exactly one predetermined donor by family/block, checks its fixed original directory and BLOCK_FREEZE path, requires the emitted reference to equal the collector's selected predecessor freeze binding, and resolves through that predecessor donor. This removes the original-path lookup error for legitimate separately located replicas and preserves exact donor/path/hash custody.

All54 V2 packet members authenticate. Only the Python function `new_prefit` changes; every other collector function and all non-function AST nodes are unchanged. The patch exactly matches the source byte delta. Forty-nine existing packet files are byte-identical, including all26 disabled job previews, the amendment, chosen attempt IDs, old-attempt/cost history, all disabled releases and the queue/freezer/launcher/history wrapper sources. The two additions are the repair patch and preservation record; report/static metadata only add repair documentation. V1 and its independent finding manifest/seal hashes remain intact.

The prior V1 checks of exact science, all39 requirements, predetermined b0 donors, genuine old b0 custody interfaces, original77 unknown status/cost preservation, no old-versus-new outcome choice, source/runtime bindings, inherited owned supervision and fixed resource/horizon/selector/stream contracts remain applicable because their bytes are unchanged. They were not rerun as numerical tests.

No prepared-source import or execution, numerical/result/tensor/model/label/history payload read, SSH, launch, current route/physical custody check, cleanup, canonical edit or manuscript edit occurred. Root still owns actual b0 all13 terminal/artifact and current allocation custody, staging/partial-staging disposition and separate activation. All26 generated jobs and four queues must receive their bounded metadata review before the first replication fit. No scientific fit is admitted by this report.
''')
assert all(sha(BASE / q['path']) == q['sha256'] for q in bindings)
emit('MANIFEST.json', {'files': [{'path': q.name, 'bytes': q.stat().st_size, 'sha256': sha(q)} for q in sorted(HERE.iterdir()) if q.is_file()], 'status': 'PASS_NARROW_V2_REPAIR_DISABLED', 'scientific_execution': False, 'source_manifest_sha256': sha(V2 / 'MANIFEST.json')})
emit('SEAL.json', {'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'status': 'SEALED_INDEPENDENT_V2_NARROW_REPAIR_PASS'})
for q in HERE.iterdir():
    os.chmod(q, 0o444)
print(json.dumps({'status': 'PASS_NARROW_V2_REPAIR_DISABLED', 'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'seal_sha256': sha(HERE / 'SEAL.json')}))
