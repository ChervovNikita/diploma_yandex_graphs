"""Root-invoked stdlib adoption only; never enables or launches collection."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time

PHASE = Path(__file__).resolve().parent
SOURCE = PHASE / 'Wiki12_SupCon15_union_collection_source_20261008_v1'
COLLECTION_SHA = '098db46113da63b64f869f898dd45cf8b89f34103fdbcf179a3f72f062153b90'
MANIFEST = 'Wiki12_SupCon15_root_metadata_adoption_MANIFEST_20261008_v1.json'
ACTIVATION = 'Wiki12_SupCon15_union_closure_adoption_root_20261008_v1'
OUTPUT = 'Wiki12_SupCon15_union_collection_execution_root_20261008_v1'
MEMORY = (
    dict(path='Wiki24_selected_analysis_after_closure_execution_root_20261007_v1/predictions/compact/COST.json',
         sha256='7c7ce9068cb4e587d5bebc623af900c404df6a9a5f4281bdc4d53ebd297f4349', bytes=48209),
    dict(path='context9_whole_family_collection_execution_root_20261008_v1/compact/COST.json',
         sha256='bdd58d7b1cbeae3b3ed731f065c76e6063d0ea5b3caf5149ed32c08c31f03f09', bytes=10148))
GIB = 1024**3


def bootstrap():
    path = PHASE / 'internal_BE_Wiki12_closed_attribution_analysis_source_20261007_v1/gate.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != 'cc6b110eed4bab38c5fc7fe555404773a53fd9f0141469e718b790e127ddb745':
        raise ValueError('Exact reused stdlib metadata helpers required')
    spec = importlib.util.spec_from_file_location('_union15_adoption_metadata', path)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    return gate


def fresh(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def pid(identity):
    # Keep the saved object unchanged; only select its source-native lookup key.
    key = 'PID' if 'PID' in identity else 'pid'
    value = identity[key]
    if type(value) is not int or value <= 0 or type(identity['start_ticks']) is not int:
        raise ValueError('Exact saved process identity required')
    return value


def terminal(gate, pins):
    original = gate.inside(pins['original_activation']); unused = gate.inside(pins['unused2_output'])
    supcon = gate.inside(pins['supcon_output'])
    union_path, family_path = unused / 'UNION_CLOSURE.json', supcon / 'CLOSURE.json'
    union, family = gate.read(union_path), gate.read(family_path)
    union_binding, family_binding = gate.binding(union_path), gate.binding(family_path)
    gate.require(union['schema'] == 'Wiki12-old10-plus-never-started2-union-closure-v1'
        and union['complete'] is True and union['original_fixed_cells'] == 12 and union['original_completed_cells'] == 10
        and union['old_terminal_custody'] == pins['old_terminal_custody'] and union['fatal_error'] is None
        and all(union[key] is True for key in ('original_complete_false_closure_immutable',
            'original_completed_cells_not_repeated', 'never_started_cells_only'))
        and all(union[key] is False for key in ('automatic_retry', 'quality_scores_read', 'TEST_access')),
        'Actual complete old10 + unchanged unused2 union required')
    gate.require(family['schema'] == 'canonical-SupCon-full3-owned-family-closure-v1'
        and family['family_accounted'] is True and family['all_new_fits_complete'] is True and family['fatal_error'] is None
        and all(family[key] is False for key in ('family_hard_cap_exceeded', 'automatic_retry', 'quality_or_prediction_opening_performed'))
        and family['controller_manifest_sha256'] == pins['reuse']['supcon_controller_manifest']['sha256']
        and family['protocol_sha256'] == pins['reuse']['supcon_protocol']['sha256']
        and all(union['supcon_terminal_custody'][key] == family_binding[key] for key in ('path', 'sha256'))
        and union['supcon_terminal_custody']['all_new_fits_complete'] is True,
        'Actual complete original SupCon3 and identical union custody required')
    old = gate.read(gate.bound(pins['old_terminal_custody']))
    gate.require(old['old_owner_absent'] is True and old['quality_scores_read'] is False and old['checkpoint_loaded'] is False,
        'Original old10 metadata custody required')
    for path, row in old['files'].items():
        gate.bound(dict(row, path=path))
    gate.require(gate.read(original / 'FAMILY_CLOSURE.json')['complete'] is False
        and gate.read(original / 'LANE_0_FAILURE.json')['error']
            == 'TimeoutError: Fixed fresh-GPU resource window exhausted before child launch', 'Preserve failed original owner')
    failed = union['preserved_original_failed_preflight']; failed_path = gate.bound(failed)
    preflight = gate.read(failed_path)
    gate.require(failed_path == original / 'logs/6307_residual_only.PREFLIGHT.json' and preflight == failed['observation']
        and preflight['scientific_child_started'] is False and preflight['elapsed_seconds'] >= 1800, 'Preserve original paid failed admission')
    evidence = gate.read(SOURCE / 'TERMINAL_EVIDENCE_TEMPLATE_DISABLED.json')
    evidence.update(Wiki12_union_closure=union_binding, SupCon3_closure=family_binding, template_only=False)
    owner_files = dict(original_Wiki12=original / 'PARENT_OWNER.json', unused2=unused / 'OWNER.json', SupCon3=supcon / 'OWNER.json')
    identities = {}
    for row in evidence['owners']:
        role = row['role']; saved = gate.read(owner_files[role]); identity = saved if role == 'original_Wiki12' else saved['owner']
        expected = pins['owners'][role]
        gate.require(pid(identity) == expected['pid'] and identity['start_ticks'] == expected['start_ticks'], 'Exact saved owner: ' + role)
        if role == 'unused2':
            gate.require(saved['admission_sha256'] == pins['reuse']['unused2_admission']['sha256']
                and saved['protocol'] == pins['reuse']['unused2_protocol'], 'Original unused2 owner admission')
        if role == 'SupCon3':
            gate.require(saved['admission_sha256'] == pins['reuse']['supcon_admission']['sha256']
                and saved['controller_manifest_sha256'] == pins['reuse']['supcon_controller_manifest']['sha256']
                and family['owner'] == identity, 'Original SupCon owner admission/closure')
        row.update(identity=identity, owner_receipt=gate.binding(owner_files[role]))
        identities[role] = identity
    old_rows = {row['cell_id']: row for row in old['completed']}
    new_rows = {row['cell_id']: row for row in union['new_completed']}
    supcon_rows = {row['seed']: row for row in family['rows']}
    roster = {row['cell']: row for row in pins['roster']}
    gate.require(len(old_rows) == 10 and set(new_rows) == {'6307_residual_only', '6307_combined'}
        and len(family['rows']) == len(supcon_rows) == 3 and set(supcon_rows) == {6101, 6203, 6307}
        and len(roster) == len(evidence['children']) == 15 and {row['cell'] for row in evidence['children']} == set(roster), 'All15 original slots')
    for row in evidence['children']:
        item = roster[row['cell']]; cell = item['cell']
        if item['custody_partition'] != 'SupCon3':
            closed = old_rows[cell] if item['custody_partition'] == 'old10' else new_rows[cell]
            path = gate.bound(closed['exit']) if item['custody_partition'] == 'old10' else unused / 'logs' / (cell + '.EXIT.json')
            receipt = gate.read(path); identity = receipt['raw_identity_observation']; pid(identity)
            argv = [pins['original_source_pins']['python'], '-B', str(gate.bound(pins['reuse']['original_adapter'])),
                '--release', str(gate.bound(item['release'])), '--release-sha256', item['release']['sha256']]
            gate.require(closed['release'] == item['release'] and identity['argv'] == argv
                and identity['PID'] == identity['pgid'] == identity['sid'] and receipt['exit_code'] == 0
                and receipt['reason'] is None and receipt['terminal_wait_observed'] is True and receipt['signals_sent'] == []
                and receipt['job_sha256'] == item['release']['sha256'], 'Original Wiki12 child wait/identity: ' + cell)
            if item['custody_partition'] == 'old10':
                gate.require(closed['exit']['pid'] == identity['PID'] and closed['exit']['child_absent'] is True
                    and closed['exit']['child_no_CUDA_rows'] is True, 'Original old10 receipt custody')
            else:
                gate.require(not (original / 'logs' / (cell + '.CHILD_STARTED.json')).exists()
                    and receipt == closed['exit_receipt'] and gate.sha(path) == closed['exit_sha256']
                    and closed['complete'] is True and closed['child_absent'] is True and closed['child_no_CUDA_rows'] is True,
                    'Unchanged never-started2 successful receipt')
            source_receipt = gate.binding(path)
        else:
            closed = supcon_rows[item['seed']]; identity = closed['owner']; pid(identity)
            started_path = supcon / 'entry' / ('seed' + str(item['seed']) + '_STARTED.json')
            entry_path = supcon / 'entry' / ('seed' + str(item['seed']) + '_TERMINAL.json')
            started, entry = gate.read(started_path), gate.read(entry_path); job_path = gate.inside(item['job'])
            argv = [pins['original_source_pins']['python'], '-B', str(gate.bound(pins['reuse']['supcon_entry'])),
                '--job', str(job_path), '--job-sha256', closed['job_sha256']]
            gate.require(closed['status'] == 'complete' and closed['exit_code'] == 0 and closed['actual_exit_and_reap'] is True
                and closed['owned_child_after'] is None and closed['hard_cap_exceeded'] is False
                and identity['pid'] == identity['group'] == identity['session'] and closed['argv'] == argv
                and gate.sha(job_path) == closed['job_sha256'] and gate.sha(entry_path) == closed['entry_terminal_sha256']
                and entry['exit_code'] == 0 and entry['error'] is None and entry['job_sha256'] == closed['job_sha256']
                and started['owner'] == identity and started['parent_owner'] == identities['SupCon3']
                and started['source_manifest_sha256'] == pins['reuse']['supcon_manifest']['sha256']
                and started['physical_gpu_uuid'] == pins['original_source_pins']['GPU_per_seed'][str(item['seed'])], 'Canonical child saved wait/identity')
            source_receipt = family_binding
        gate.require(row['source_receipt']['path'] == source_receipt['path'], 'Supplied terminal template receipt mapping')
        row.update(identity=identity, source_receipt=source_receipt, wait_and_reap_observed=True)
    return evidence, dict(original_failed_closure=gate.binding(original / 'FAMILY_CLOSURE.json'),
        original_lane_failure=gate.binding(original / 'LANE_0_FAILURE.json'), original_failed_preflight=gate.binding(failed_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077); began = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_SELF)
    gate = bootstrap(); gate.require(gate.sha(args.release) == args.release_sha256, 'Exact separate root metadata-adoption release')
    cfg = gate.read(args.release)
    gate.require(cfg.get('schema') == 'Wiki12-SupCon15-root-metadata-adoption-release-v1'
        and all(cfg.get(key) is True for key in ('enabled', 'root_metadata_adoption_authorized', 'source_review_approved'))
        and all(cfg.get(key) is False for key in ('TEST_access', 'training', 'score_opening', 'collection_launch', 'automatic_retry')), 'Disabled metadata-only preparation')
    gate.verify(dict(path=MANIFEST, sha256=cfg['helper_manifest_sha256']))
    gate.verify(dict(path=str((SOURCE / 'MANIFEST.json').relative_to(PHASE)), sha256=COLLECTION_SHA))
    pins = gate.read(SOURCE / 'SOURCE_BINDINGS.json'); runtime = pins['runtime']
    activation = gate.inside(ACTIVATION); gate.require(activation.is_dir() and args.release.resolve().parent == activation, 'Separate root activation')
    names = ('TERMINAL_EVIDENCE_ROOT.json', 'RUNTIME_EVIDENCE_ROOT.json', 'RESOURCE_READINESS_ROOT.json', 'COLLECTION_RELEASE_ROOT.json')
    gate.require(all(not (activation / name).exists() for name in names) and not gate.inside(OUTPUT).exists(), 'Fresh evidence/output; no overwrite')
    gate.require(platform.system() == 'Linux' and socket.gethostname() == runtime['hostname']
        and Path.cwd().resolve() == Path(runtime['repository']).resolve() and str(Path(sys.executable).absolute()) == runtime['python']
        and os.environ.get('PYTHONPATH', '') == runtime['PYTHONPATH'], 'Actual ordinary normal77 host/interpreter')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')}
    gate.require(versions == {name: runtime[name] for name in versions}, 'Actual original provider metadata without numerical imports')
    gate.bound(cfg['external_owner_manifest']); gate.verify(cfg['external_owner_manifest']); gate.bound(cfg['external_owner_review'])
    evidence, preserved = terminal(gate, pins)
    helper = gate.module(gate.bound(pins['original_source_pins']['reviewed_ownership_helper']), '_union15_adoption_ownership')
    identities = [row['identity'] for row in evidence['owners']] + [row['identity'] for row in evidence['children']]
    for identity in identities:
        gate.require(helper.identity(pid(identity)) is None and not Path('/proc', str(pid(identity))).exists(), 'Saved owner/child is not actually absent')
    inventory = helper.query(['--query-gpu=uuid', '--format=csv,noheader'], 10)
    compute = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], 10)
    resources = helper.query(['--query-gpu=uuid,memory.free,memory.total', '--format=csv,noheader,nounits'], 10)
    gate.require(inventory == runtime['physical_gpu_inventory'], 'Actual normal77 physical inventory')
    saved_pids = {pid(identity) for identity in identities}
    gate.require(not any(len(parts) > 1 and parts[1].strip().isdigit() and int(parts[1]) in saved_pids
        for parts in (row.split(',') for row in compute)), 'Saved owners/children still have CUDA rows')
    gpu = runtime['physical_gpu_inventory'][0]; matched = [row.split(',') for row in resources if row.split(',')[0].strip() == gpu]
    gate.require(len(matched) == 1 and len(matched[0]) == 3 and all(value.strip().isdigit() for value in matched[0][1:]), 'Actual GPU0 resource telemetry')
    free_bytes, total_bytes = [int(value.strip()) * 1024**2 for value in matched[0][1:]]
    gate.require(free_bytes >= 10 * GIB and total_bytes >= 8 * GIB, 'Fixed prospective8GiB reader cap/10GiB fresh headroom')
    memory = []
    for row in MEMORY:
        value = gate.read(gate.bound(row))
        gate.require(value['status'] == 'complete' and value['attempted_member_forwards'] == value['completed_member_forwards'] == value['maximum_member_forwards']
            and value['peak_CUDA_reserved_bytes'] < 8 * GIB and all(value[key] == 0 for key in ('TRAIN_updates', 'backward_calls', 'optimizer_constructions')), 'Existing complete inference-only memory evidence')
        memory.append(dict(binding=row, metadata={key: value[key] for key in ('status', 'maximum_member_forwards', 'attempted_member_forwards',
            'completed_member_forwards', 'peak_CUDA_allocated_bytes', 'peak_CUDA_reserved_bytes', 'TRAIN_updates', 'backward_calls', 'optimizer_constructions')}))
    # Recheck absence after all telemetry; never replace saved identities with observations.
    for identity in identities:
        gate.require(helper.identity(pid(identity)) is None and not Path('/proc', str(pid(identity))).exists(), 'Process appeared during metadata adoption')
    stamp = datetime.now(timezone.utc).isoformat()
    for row in evidence['owners']: row.update(absent=True, no_CUDA_rows=True)
    for row in evidence['children']: row.update(child_absent=True, child_no_CUDA_rows=True)
    evidence.update(complete=True, root_observed=True, owners_and_all15_children_terminal=True, observed_UTC=stamp,
        observed_absent_PIDs=sorted(saved_pids), CUDA_process_rows=compute, preserved_failed_original_costs=preserved)
    observed_runtime = dict(schema='Wiki12-SupCon15-root-runtime-evidence-v1', observed_UTC=stamp, actual_platform=platform.system(),
        actual_hostname=socket.gethostname(), actual_repository=str(Path.cwd().resolve()), actual_python=str(Path(sys.executable).absolute()),
        actual_PYTHONPATH=os.environ.get('PYTHONPATH', ''), actual_provider_versions=versions, physical_gpu_inventory=inventory, numerical_imports=False)
    readiness = dict(schema='Wiki12-SupCon15-root-resource-readiness-v1', observed_UTC=stamp, physical_gpu_uuid=gpu,
        observed_free_GPU_bytes=free_bytes, observed_total_GPU_bytes=total_bytes, owned_GPU_cap_bytes=8 * GIB,
        minimum_fresh_GPU_free_bytes=10 * GIB, external_active_seconds=300, external_cleanup_seconds=10, external_hard_seconds=310,
        raw_GPU_resource_rows=resources, memory_evidence=memory, prospective_limit_not_future_peak_guarantee=True,
        preserved_failed_original_costs=preserved, metadata_adoption_seconds=time.monotonic() - began,
        metadata_CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime - usage.ru_utime,
        metadata_CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime - usage.ru_stime, quality_scores_read=False, TEST_access=False)
    for row in evidence['owners']: gate.bound(row['owner_receipt'])
    for row in evidence['children']: gate.bound(row['source_receipt'])
    gate.bound(evidence['Wiki12_union_closure']); gate.bound(evidence['SupCon3_closure'])
    fresh(activation / names[0], evidence); fresh(activation / names[1], observed_runtime); fresh(activation / names[2], readiness)
    collection = gate.read(SOURCE / 'RELEASE_TEMPLATE_DISABLED.json')
    collection.update(Wiki12_union_closure=evidence['Wiki12_union_closure'], SupCon3_closure=evidence['SupCon3_closure'],
        terminal_evidence=gate.binding(activation / names[0]), runtime_evidence=gate.binding(activation / names[1]),
        resource_readiness=gate.binding(activation / names[2]), external_supervision=cfg['external_owner_manifest'],
        external_owner_review=cfg['external_owner_review'], collection_manifest_sha256=COLLECTION_SHA,
        entire_Wiki12_union_complete=True, entire_SupCon3_complete=True, owners_and_all15_children_terminal=True,
        runtime_resource_readiness_confirmed=True, owned_GPU_cap_bytes=8 * GIB, minimum_fresh_GPU_free_bytes=10 * GIB,
        external_active_seconds=300, external_cleanup_seconds=10, external_hard_seconds=310, output_directory=OUTPUT,
        template_only=False, instruction='Actual metadata populated; root must explicitly enable and authorize this separate release before owned collection.')
    gate.require(collection['enabled'] is False and collection['root_execution_authorized'] is False
        and collection['trusted_checkpoint_deserialization_authorized'] is False and collection['external_owned_bound_confirmed'] is False,
        'Metadata helper never supplies collection authority')
    fresh(activation / names[3], collection)
    print(json.dumps(dict(metadata_adopted=True, collection_release=gate.binding(activation / names[3]),
        collection_enabled=False, scientific_or_collection_children_launched=0, quality_scores_read=False)), flush=True)


if __name__ == '__main__':
    main()
