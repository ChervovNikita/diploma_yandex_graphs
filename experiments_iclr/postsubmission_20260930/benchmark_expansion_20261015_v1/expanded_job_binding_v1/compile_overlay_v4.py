"""Guarded metadata-only BUDDY production successor overlay; preserve ancestors."""
from __future__ import annotations
import ast
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT / 'experiments_iclr/postsubmission_20260930'
HERE = PHASE / 'benchmark_expansion_20261015_v1/expanded_job_binding_v1'
PARENT_READY_SHA = 'dacf157ffa7bebc52acafbe1021cac1eaeffc1eac2d1326c9c821b0bb48eb374'
OVERLAY_SHA = 'e9671777106b2e3e34ea187d6293bb83c455b3458cb0101c0bdb3352c114a5ab'
FORBIDDEN = ('torch', 'numpy', 'scipy', 'torch_geometric', 'ogb')


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    assert path.is_relative_to(PHASE) and path.suffix in {'.py', '.json', '.md', '.txt', '.html', '.xml'}
    assert not {'results', 'datasets', 'working_data', 'processed', 'raw'} & set(path.relative_to(PHASE).parts)
    raw = path.read_bytes()
    value = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if expected:
        for k in ('bytes', 'sha256'):
            if k in expected: assert value[k] == expected[k], str(path)
    return value


def load(path, expected=None):
    pin(path, expected)
    return json.loads(Path(path).read_text())


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()
    with path.open('xb') as out: out.write(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def module(path, name):
    pin(path)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    assert not any(n in sys.modules for n in FORBIDDEN)
    return result


def scientific_contract(value):
    result = copy.deepcopy(value)
    result.pop('source_registry'); result.pop('binding_entry_sha256')
    return result


def main():
    import socket, subprocess
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert Path(__file__).resolve().parent == HERE
    output, parent = HERE / 'generated_v4', HERE / 'generated_v3'
    assert not output.exists(), 'Fresh generation only; never overwrite or retry into prior output'
    ready_pin = pin(parent / 'READINESS.json', {'sha256': PARENT_READY_SHA})
    ready = load(ready_pin['path'])
    parent_verification = pin(parent / 'VERIFICATION.json')
    prior_verification = load(parent_verification['path'])
    assert prior_verification['status'] == 'pass' and prior_verification['readiness'] == ready_pin
    overlay_pin = pin(HERE / 'OVERLAY_v4.json', {'sha256': OVERLAY_SHA})
    overlay = load(overlay_pin['path'])
    assert overlay['runnable'] is False and overlay['campaign_GO'] is False
    source_receipts = [pin(PHASE / r['path_relative_to_phase'], r) for r in overlay['source_files']]
    accepted_pin = overlay['accepted_representative_qualification']
    accepted = load(PHASE / accepted_pin['path_relative_to_phase'], accepted_pin)
    assert accepted['status'] == accepted['supervisor_status'] == 'passed' and accepted['full_campaign_GO'] is False
    assert (accepted['optimizer_updates'], accepted['whole_VALID_calls'], accepted['constructions'], accepted['forwards']) == (2, 3, 2, 32)
    assert accepted['TEST_accessed'] is False
    registry = load(PHASE / overlay['parent_registry']['path_relative_to_phase'], overlay['parent_registry'])
    load(PHASE / overlay['parent_overlay']['path_relative_to_phase'], overlay['parent_overlay'])
    # Authenticate the complete inherited source receipt chain, without tensors.
    source_seen = set()
    def authenticate_sources(receipt):
        if receipt['path'] in source_seen: return
        source_seen.add(receipt['path'])
        data = load(receipt['path'], receipt)
        for r in data.get('source_and_context_metadata', []): pin(r['path'], r)
        if data.get('parent_sources'): authenticate_sources(data['parent_sources'])
    authenticate_sources(ready['artifacts']['sources'])
    bindings = {(r['cell_id'], r['method_id']): r for r in overlay['entries']}
    assert len(bindings) == 10 and all(r['supported_seeds'] == list(range(10)) for r in bindings.values())
    for row in bindings.values():
        callback = PHASE / row['callback']['path_relative_to_phase']
        pin(callback, row['callback'])
        assert row['callback']['function'] in {n.name for n in ast.parse(callback.read_text()).body if isinstance(n, ast.FunctionDef)}
        assert not row.get('session_transform') and row['method_settings']['provider_mode'] == 'production_native'
        assert not {'context_manifest', 'context_manifest_sha256'} & set(row['method_settings'])
        assert row['qualification_scope']['all_seeds_executed'] is False
    index = load(ready['artifacts']['index']['path'], ready['artifacts']['index'])['fits']
    assert len(index) == len({r['id'] for r in index}) == 6460
    indexed = {r['id']: r for r in index}
    templates = {(r['cell_id'], r['cohort_seed']): r for r in index if r['method_id'] == 'gnnm_boundary_native'}
    original_compiler = module(HERE / 'compile_descriptors.py', '_v3_ancestor_constants')
    ancestor_pin = pin(original_compiler.PREDECESSOR / 'DESCRIPTOR_INDEX.json', {'sha256': original_compiler.PREDECESSOR_INDEX_SHA})
    ancestors = {r['id']: r for r in load(ancestor_pin['path'], ancestor_pin)['fits']}
    assert len(ancestors) == 2480
    runtime_folder = PHASE / registry['runtime_package']
    sys.path.insert(0, str(runtime_folder))
    runtime = module(runtime_folder / 'run_fit.py', '_v3_metadata_validator')
    new_rows, changed, closed = [], [], []
    output.mkdir(); (output / 'jobs').mkdir()
    for old in index:
        binding = bindings.get((old['cell_id'], old['method_id']))
        previous = old.get('descriptor') or old.get('blocked_record')
        prior = load(previous['path'], previous)
        if binding is None:
            new_rows.append(copy.deepcopy(old)); continue
        assert old['cell_id'] == 'collab-buddy' and old['cohort_seed'] in range(10)
        if old['status'] == 'blocked':
            contract = copy.deepcopy(prior['expanded_contract'])
            template_row = templates[(old['cell_id'], old['cohort_seed'])]
            template = load(template_row['descriptor']['path'], template_row['descriptor'])
            job = copy.deepcopy(template)
            if old['ancestor']:
                origin = load(ancestors[old['id']]['descriptor']['path'], ancestors[old['id']]['descriptor'])
                for key in ('source_plan_entry', 'source_plan_contract_hash'):
                    job[key] = copy.deepcopy(origin[key])
            else:
                job.pop('source_plan_entry', None); job.pop('source_plan_contract_hash', None)
        else:
            assert old['status'] == 'descriptor_prepared_not_admitted' and digest(prior) == old['runtime_job_hash']
            job, contract = copy.deepcopy(prior), copy.deepcopy(prior['expanded_contract'])
        assert digest(contract) == old['expanded_contract_hash']
        old_science = scientific_contract(contract)
        contract.update(source_registry=overlay_pin, binding_entry_sha256=digest(binding))
        assert scientific_contract(contract) == old_science
        job.pop('session_transform', None)
        job.update(id=old['id'], body=contract['body'], scientific_method_id=old['method_id'],
            procedure=binding['runtime_procedure'], base_procedure=binding['base_procedure'],
            callback_module=str(PHASE / binding['callback']['path_relative_to_phase']), callback_sha256=binding['callback']['sha256'],
            method_settings=copy.deepcopy(binding['method_settings']), expanded_contract=contract, expanded_contract_hash=digest(contract))
        job['descriptor_preparation'].update(registry=overlay_pin, previous_generation_record=previous,
            ancestor_descriptor=None if not old['ancestor'] else ancestors[old['id']]['descriptor'],
            ancestor_contract_preserved=bool(old['ancestor']),
            runtime_binding_migration={'from_previous_generation_status': old['status'], 'scientific_method_id': old['method_id'],
                'to_runtime_procedure': job['procedure'], 'source_context_successor_only': True},
            source_context_qualified=True, native_CPU_gate=binding['native_CPU_gate'],
            native_qualification_scope=copy.deepcopy(binding['qualification_scope']),
            accepted_representative_qualification=accepted_pin, source_equivalence_review=binding['source_review'],
            runtime_selected_serving_qualified=False, all_native_epoch_lifecycle_qualified=False,
            resource_admission=False, resource_binding=None, human_GO_received=False, scientific_launch_enabled=False,
            pending=copy.deepcopy(binding['pending']))
        assert job['seed'] == job['cohort_seed'] == old['cohort_seed'] and job['split'] == 0
        assert job['recipe']['_seed'] == job['recipe']['seed'] == job['seed']
        assert job['recipe']['native_fixed_config'] == contract['native_settings']['native_config']
        assert job['recipe']['official_split_bindings']['0'] == contract['role_bindings']
        assert job['recipe']['architecture'].upper() == 'BUDDY'
        assert job['policy'] == original_compiler.make_policy('link', False)
        assert original_compiler.check_fixed_controller(job['policy'], contract['native_settings'], False) == contract['native_controller']
        if old['ancestor']:
            assert job['source_plan_contract_hash'] == old['ancestor_contract_hash']
            assert digest(job['source_plan_entry']['contract']) == old['ancestor_contract_hash']
        runtime.validate_job(job, ROOT)
        row = copy.deepcopy(old)
        row.pop('blocked_record', None); row.pop('blocked_reason', None)
        name = hashlib.sha256(row['id'].encode()).hexdigest() + '.json'
        row.update(status='descriptor_prepared_not_admitted', descriptor=write(output / 'jobs' / name, job),
            runtime_job_hash=digest(job), expanded_contract_hash=digest(contract), previous_generation_record=previous,
            runtime_metadata_validation='pass_metadata_only', launch_admitted=False)
        new_rows.append(row); changed.append(row)
        if old['status'] == 'blocked': closed.append(row)
    assert len(changed) == 100 and len(closed) == 0 and len(new_rows) == 6460
    statuses = Counter(r['status'] for r in new_rows)
    assert statuses == {'descriptor_prepared_not_admitted': 6460}
    assert {r['id'] for r in new_rows} == set(indexed)
    assert all(r['ancestor_contract_hash'] == indexed[r['id']]['ancestor_contract_hash'] for r in new_rows)
    assert sum(r['ancestor'] for r in new_rows) == 2480
    new_index = {r['id']: r for r in new_rows}
    banks = load(ready['artifacts']['banks']['path'], ready['artifacts']['banks'])
    controls = load(ready['artifacts']['controls']['path'], ready['artifacts']['controls'])
    for bank in banks['logical_banks']:
        if 'fit_status' in bank and len(bank['fit_ids']) == 1: bank['fit_status'] = new_index[bank['fit_ids'][0]]['status']
    for control in controls['entries']:
        for row in control['controls']:
            if 'fit_status' in row and len(row['fit_ids']) == 1: row['fit_status'] = new_index[row['fit_ids'][0]]['status']
    assert len(banks['logical_banks']) == 5920 and len(controls['entries']) == 5110
    artifacts = copy.deepcopy(ready['artifacts'])
    artifacts['index'] = write(output / 'DESCRIPTOR_INDEX.json', {'schema': 'expanded-fit-overlay-index-v4', 'parent_index': ready['artifacts']['index'], 'fits': new_rows})
    artifacts['blocked'] = write(output / 'BLOCKED_FITS.json', {'schema': 'expanded-blocked-fit-index-v4', 'fits': []})
    artifacts['banks'] = write(output / 'LOGICAL_BANKS.json', banks)
    artifacts['controls'] = write(output / 'CONTROL_LINKS.json', controls)
    artifacts['sources'] = write(output / 'SOURCE_RECEIPTS.json', {'overlay': overlay_pin, 'parent_sources': ready['artifacts']['sources'],
        'source_and_context_metadata': source_receipts, 'data_receipts_refreshed': False, 'model_source_imports': False,
        'accepted_representative_qualification': accepted_pin})
    artifacts['delta'] = write(output / 'DELTA.json', {'schema': 'expanded-BUDDY-shared-body-source-overlay-delta-v4',
        'parent_readiness': ready_pin, 'overlay': overlay_pin, 'same_fit_intention_ids': 6460,
        'changed_descriptor_rows': changed, 'closed_source_context_rows': closed, 'source_context_blocks_closed': 0,
        'rebound_existing_source_rows': 100, 'new_descriptor_files': 100, 'inherited_unchanged_descriptor_files': 6360,
        'parent_files_rewritten': 0, 'native_settings_or_scientific_method_changes': False,
        'other_recipes_individually_executed': False, 'source_only': True, 'resource_admission': False, 'campaign_GO': False})
    result = copy.deepcopy(ready)
    result.update(schema='expanded-6460-BUDDY-shared-body-source-overlay-readiness-v4', UTC=datetime.now(timezone.utc).isoformat(),
        status='all6460source descriptors prepared; no campaign/runtime/resource admission', remote_output=str(output),
        registry={'base': ready['registry'], 'overlay': overlay_pin}, parent_readiness=ready_pin,
        concrete_descriptors=6460, blocked_fit_intentions=0, blocked_by_reason={},
        runtime_metadata_validation_passes=6460, additional_metadata_validation_passes=100, inherited_validation_passes=6360,
        source_receipt_files=len(source_receipts), descriptor_storage={'reused_parent_descriptors': 6360, 'new_descriptors': 100, 'closed_source_context_blocks': 0},
        native_BUDDY_interaction_CPU_gate='accepted actual conditional interaction seed1; other production recipes source-equivalent only',
        accepted_BUDDY_qualification=accepted_pin, all_native_method_preset_source_bindings_exist=True,
        source_binding_is_runtime_qualification=False, all_selected_serving_lifecycles_qualified=False,
        remaining_global_gaps=['Actual native whole-epoch/selected/export/resume and protocol/device coverage outside accepted scoped gates',
            'Production runtime bindings remain unadmitted; NCNC v6 evidence is separate and not transferred to BUDDY',
            'Measured resource/cost/reciprocal concurrency admission', 'Explicit human campaign GO'],
        expanded_statistical_decisions_frozen=True, expanded_statistical_decisions_scope='root-approved exploratory TRAIN/VALID descriptive protocol',
        reporting_ROOT_REVIEW=overlay['accepted_reporting_ROOT_REVIEW'], runtime_link_source_successor=pin(PHASE/'benchmark_expansion_20261015_v1/runtime_link_v9/MANIFEST.json'),
        resource_bindings=None, resources_measured=False, runtime_admitted=False, scientific_launch_enabled=False,
        human_GO_received=False, artifacts=artifacts)
    readiness_pin = write(output / 'READINESS.json', result)
    for row in changed:
        job = load(row['descriptor']['path'], row['descriptor'])
        assert digest(job) == row['runtime_job_hash'] and scientific_contract(job['expanded_contract']) == scientific_contract(
            load(indexed[row['id']]['descriptor']['path'], indexed[row['id']]['descriptor'])['expanded_contract'] if indexed[row['id']]['status'] != 'blocked'
            else load(indexed[row['id']]['blocked_record']['path'], indexed[row['id']]['blocked_record'])['expanded_contract'])
    assert not any(n in sys.modules for n in FORBIDDEN)
    verification_pin = write(output / 'VERIFICATION.json', {'schema': 'expanded-BUDDY-shared-body-source-overlay-verification-v4', 'status': 'pass',
        'readiness': readiness_pin, 'fit_intentions': 6460, 'descriptor_prepared': 6460, 'source_context_blocks': 0,
        'closed_source_context_blocks': 0, 'new_metadata_validation_passes': 100, 'reused_descriptor_receipts_authenticated': 6360,
        'all2480ancestor_ID_contract_hashes_preserved': True, 'scientific_contracts_unchanged_except_source_binding': True,
        'logical_bank_and_paired_control_fit_IDs_preserved': True, 'I4_M1_source_track_and_alias_contracts_unchanged': True,
        'accepted_actual_BUDDY_representative_only': ['interaction_input_conditionalJ', 1], 'other_recipes_actual_native_passes_claimed': False,
        'native_models': 0, 'native_forwards': 0, 'optimizer_updates': 0, 'data_or_tensor_files_opened': 0,
        'TEST_accessed': False, 'parent_output_rewritten': False, 'runtime_admission': False, 'resource_admission': False, 'campaign_GO': False})
    print(json.dumps({'readiness': readiness_pin, 'verification': verification_pin, 'descriptor_prepared': 6460,
        'source_context_blocks': 0, 'closed_source_context_blocks': 0, 'new_descriptors': 100, 'reused_descriptors': 6360,
        'native_work': 0, 'campaign_GO': False}, sort_keys=True))


if __name__ == '__main__': main()
