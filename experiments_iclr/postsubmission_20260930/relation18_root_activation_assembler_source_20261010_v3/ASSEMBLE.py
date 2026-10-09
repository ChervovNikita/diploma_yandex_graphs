"""Disabled root metadata assembler; never launches owner or numerical reader."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
READER_NAME = 'graph_relation18_selected_readout_source_20261009_v3'
OPEN_FLAGS = ('enabled', 'root_execution_authorized', 'source_review_approved', 'whole_relation12_complete',
    'exact_original12_union_complete', 'historical_whole24_and_collection_closed', 'all_owners_and_children_terminal',
    'trusted_checkpoint_deserialization_authorized', 'historical_prediction_archive_opening_authorized',
    'runtime_resource_readiness_confirmed', 'collection_and_analysis_cost_charged', 'external_owned_bound_confirmed')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(relative):
    item = Path(relative)
    require(not item.is_absolute() and '..' not in item.parts, 'Phase-relative root custody')
    path = (PHASE / item).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Root path inside current phase')
    return path


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE) and path.is_file(), 'Root metadata file within phase')
    return dict(path=str(path.relative_to(PHASE)), bytes=path.stat().st_size, sha256=sha(path))


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'Exact root metadata hash')
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Exact root metadata byte size when recorded')
    return path


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['runtime_disabled'] is True and seal['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Exact disabled assembler seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Assembler source unchanged')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(sha(bound(pins['reader_seal'])) == pins['reader_seal']['sha256'], 'Exact sealed V3 reader')
    require(read(bound(pins['reader_seal']))['manifest_sha256'] == pins['reader_manifest']['sha256'], 'Same exact V3 reader seal/manifest')
    for row in read(bound(pins['reader_manifest']))['files']:
        path = PHASE / READER_NAME / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Reviewed V3 reader source unchanged')
    return pins


def observation(rows, key, identity, group=False, hostname='peptide'):
    require(key in rows, 'Root current terminal slot: ' + key)
    row = rows[key]
    require(row['identity'] == identity and row['hostname'] == hostname
        and row['absent'] is True and row['no_CUDA_rows'] is True,
        'Exact root current PID/start/identity absence and no owned CUDA: ' + key)
    if group:
        require(row['process_group_absent'] is True, 'Root current detached process group absent')
    return row


def terminal_records(reader_pins, cfg, observations):
    terminal = read(PHASE / READER_NAME / 'TERMINAL_EVIDENCE_TEMPLATE_DISABLED.json')
    family = read(bound(cfg['relation_family_closure']))
    identity = family['parent_identity']
    observed = observation(observations['relation_owner'], 'relation12', identity, group=True)
    terminal.update(complete=True, root_observed=True, all_owners_and_children_terminal=True,
        relation_family_closure=cfg['relation_family_closure'], relation_parent_owner=cfg['relation_parent_owner'],
        relation_parent_launch=cfg['relation_parent_launch'], original12_union_closure=cfg['original12_union_closure'],
        root_current_observations=cfg['current_terminal_observations'])
    terminal['relation_owner'].update(identity=identity, owner_receipt=cfg['relation_parent_owner'],
        launch_receipt=cfg['relation_parent_launch'], absent=observed['absent'],
        process_group_absent=observed['process_group_absent'], no_CUDA_rows=observed['no_CUDA_rows'],
        wait_and_reap_observed=False, exit_code=None)
    relation_rows = {}
    for lane in family['lane_results']:
        for row in lane['rows']:
            cell = row['cell_id']
            receipt = read(bound(row['exit_receipt']))
            require(receipt == row['actual_exit_receipt'] and receipt['raw_identity_observation'] == row['child_identity']
                and receipt['terminal_wait_observed'] is True and receipt['exit_code'] == 0,
                'Actual original relation child direct wait/exit receipt')
            observation(observations['relation_children'], cell, row['child_identity'])
            relation_rows[cell] = dict(cell_id=cell, identity=row['child_identity'], source_receipt=row['exit_receipt'],
                child_absent=True, child_no_CUDA_rows=True, wait_and_reap_observed=True)
    require(set(relation_rows) == {row['cell_id'] for row in terminal['relation_children']}, 'Entire relation12 terminal dictionary')
    terminal['relation_children'] = [relation_rows[row['cell_id']] for row in terminal['relation_children']]
    p = reader_pins['union_pins']
    oldroot, fresh = inside(p['original_activation']), inside(p['unused2_output'])
    old, union = read(bound(p['old_terminal_custody'])), read(bound(cfg['original12_union_closure']))
    old_rows = {row['cell_id']: row for row in old['completed']}
    new_rows = {row['cell_id']: row for row in union['new_completed']}
    original = {}
    for item in p['roster']:
        cell = item['cell']
        exit_path = bound(old_rows[cell]['exit']) if item['custody_partition'] == 'old10' else fresh / 'logs' / (cell + '.EXIT.json')
        receipt = read(exit_path); identity = receipt['raw_identity_observation']
        require(receipt['terminal_wait_observed'] is True and receipt['exit_code'] == 0, 'Actual original12 direct waited child')
        if item['custody_partition'] != 'old10':
            require(sha(exit_path) == new_rows[cell]['exit_sha256'], 'Exact original never-started2 final EXIT hash')
        observation(observations['original12_children'], cell, identity)
        original[cell] = dict(cell_id=cell, identity=identity, source_receipt=binding(exit_path),
            child_absent=True, child_no_CUDA_rows=True, wait_and_reap_observed=True)
    terminal['original12_children'] = [original[row['cell_id']] for row in terminal['original12_children']]
    for row in terminal['original12_owners']:
        role = row['role']
        receipt_path = oldroot / 'PARENT_OWNER.json' if role == 'original_Wiki12' else fresh / 'OWNER.json'
        saved = read(receipt_path); identity = saved if role == 'original_Wiki12' else saved['owner']
        observed = observation(observations['original12_owners'], role, identity)
        row.update(identity=identity, owner_receipt=binding(receipt_path), absent=True, no_CUDA_rows=True,
            wait_and_reap_observed=False, exit_code=None)
    custody = read(bound(cfg['historical_original_route_custody']))
    owner = read(bound(custody['historical_child_owner']))
    receipt = read(bound(custody['historical_child_terminal']))
    require(owner['identity'] == receipt['identity'] and receipt['reaped'] is True and receipt['exit_code'] == 0,
        'Actual historical collector is the directly waited child, separate from watcher')
    observed = observation(observations['historical_prediction_collector'], 'collector', owner['identity'], hostname='anogena-2-0')
    terminal['historical_prediction_collector'].update(identity=owner['identity'], command=owner['command'],
        child_owner_receipt=custody['historical_child_owner'], child_terminal_receipt=custody['historical_child_terminal'],
        release=custody['historical_collector_release'], collection=custody['historical_collection'], cost=custody['historical_cost'],
        absent=observed['absent'], no_CUDA_rows=observed['no_CUDA_rows'], wait_and_reap_observed=True, exit_code=0)
    return terminal


def stage_record(reader_pins, cfg):
    stage = read(PHASE / READER_NAME / 'HISTORICAL_ARCHIVE_STAGING_TEMPLATE_DISABLED.json')
    receipt = read(bound(cfg['historical_archive_transfer_receipt']))
    require(receipt['id'] == 'relation18_historical_six_exact_server_only_stream_20261009_v1' and receipt['exit_code'] == 0
        and receipt['current_Mac_payload_copy_created'] is False and receipt['linked_Mac_payload_copy_created'] is False,
        'Actual successful six-file server-only stream receipt')
    expected = []
    for row in reader_pins['reference_pointers']['records']:
        old = row['raw_prediction_archive_inherited_binding']
        expected.append(dict(source_relative='experiments_iclr/postsubmission_20260930/' + old['path'],
            target_relative='experiments_iclr/postsubmission_20260930/' + reader_pins['historical_archive_stage_directory'] + '/' + Path(old['path']).name,
            bytes=old['bytes'], sha256=old['sha256']))
    require(receipt['files'] == expected, 'Actual stream exact six original/staged paths and hashes')
    destination_rows = [json.loads(line) for line in receipt['stdout'].splitlines() if line.startswith('{')]
    require(len(destination_rows) == 1, 'Unique actual destination verification metadata')
    destination = destination_rows[0]
    require(destination['status'] == 'EXACT_STREAM_DESTINATION_HASHES_VERIFIED'
        and destination['hostname'] == reader_pins['runtime']['hostname']
        and destination['repository'] == reader_pins['runtime']['repository']
        and destination['payloads_decoded'] is False and destination['existing_files_overwritten'] is False
        and destination['scientific_jobs_untouched'] is True and len(destination['files']) == 6,
        'Original exact stream verified fresh destination without decoding or overwritten files')
    for observed, slot in zip(destination['files'], stage['records']):
        staged = slot['staged_archive']
        require(observed['path'] == str(inside(staged['path'])) and observed['bytes'] == staged['bytes']
            and observed['sha256'] == staged['sha256'] and observed['created_by_this_stream'] is True
            and observed['preexisting_exact_file_preserved'] is False, 'Actual archive created by the fixed fresh stream')
        bound(staged)  # Bytes/hash only, never archive decode.
    stage.update(complete=True, root_observed=True, fresh_scope_created=True, server_only=True, byte_preserved=True,
        original_paths_unchanged=True, all_original_and_staged_hashes_verified=True,
        original_route_custody=cfg['historical_original_route_custody'],
        actual_transfer_receipt=cfg['historical_archive_transfer_receipt'], actual_transfer_and_hash_cost=None)
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--root-input', type=Path)
    parser.add_argument('--root-input-sha256')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, metadata_gate_executed=False, numerical_reader_invoked=False)))
        return
    started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    pins = source_gate()
    require(args.root_input is not None and args.root_input_sha256 == sha(args.root_input), 'Exact separately bound root assembler admission')
    cfg = read(args.root_input)
    require(cfg['schema'] == 'relation18-root-activation-assembler-input-v1'
        and all(cfg[key] is True for key in ('enabled', 'root_metadata_assembly_authorized', 'root_metadata_gate_only_authorized',
            'root_opening_release_fields_adopted', 'root_owns_existing_finite_owner_and_later_numerical_launch'))
        and all(cfg[key] is False for key in ('numerical_reader_invocation', 'training', 'automatic_retry', 'TEST_access')),
        'Disabled until actual root metadata-only preparation and release-field adoption')
    require(cfg['assembler_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
        and cfg['assembler_entry_sha256'] == sha(__file__), 'Exact reviewed root assembler')
    reader = PHASE / READER_NAME; reader_pins = read(reader / 'SOURCE_BINDINGS.json')
    require(set(cfg['resources']) == {'physical_gpu_uuid', 'external_active_seconds', 'external_cleanup_seconds',
        'external_hard_seconds', 'owned_GPU_cap_bytes', 'minimum_fresh_GPU_free_bytes', 'maximum_output_bytes'}
        and set(cfg['owner_resources']) == {'owned_RSS_cap_bytes', 'combined_child_log_cap_bytes'}, 'Exact root resource fields, no release-field overrides')
    require(socket.gethostname() == reader_pins['runtime']['hostname'] and Path.cwd().resolve() == Path(reader_pins['runtime']['repository'])
        and str(Path(sys.executable).absolute()) == reader_pins['runtime']['python'], 'Actual same ordinary normal77 runtime for metadata gate')
    adoption = read(bound(cfg['root_reader_adoption']))
    require(adoption['schema'] == 'relation18-root-reviewed-reader-v3-adoption-v1' and adoption['approved'] is True
        and adoption['source_review_approved'] is True and adoption['unresolved_blockers'] == []
        and adoption['reader_manifest'] == pins['reader_manifest'] and adoption['reader_seal'] == pins['reader_seal'],
        'Actual root exact V3 reader review/adoption, never assumed')
    bound(adoption['independent_review'])
    observations = read(bound(cfg['current_terminal_observations']))
    require(observations['schema'] == 'relation18-root-current-terminal-observations-v1'
        and observations['complete'] is True and observations['root_observed'] is True
        and observations['hostname'] == reader_pins['runtime']['hostname'] and observations['TEST_access'] is False,
        'Bound root current process/group/CUDA observations')
    runtime = read(bound(cfg['runtime_evidence'])); readiness = read(bound(cfg['resource_readiness']))
    require(runtime['schema'] == 'relation18-root-saved-runtime-evidence-v1' and runtime['root_observed'] is True
        and runtime['runtime'] == reader_pins['runtime']
        and readiness['schema'] == 'relation18-root-saved-resource-readiness-v1'
        and readiness['root_observed'] is True and readiness['ready'] is True
        and readiness['runtime_evidence'] == cfg['runtime_evidence']
        and readiness['physical_gpu_uuid'] == reader_pins['runtime']['physical_gpu_inventory'][0]
        and cfg['resources']['physical_gpu_uuid'] == readiness['physical_gpu_uuid']
        and readiness['minimum_fresh_GPU_free_bytes'] == cfg['resources']['minimum_fresh_GPU_free_bytes']
        and readiness['actual_free_GPU_bytes'] >= readiness['minimum_fresh_GPU_free_bytes'], 'Actual same-runtime root physical resources')
    for row in pins['historical_eight_metadata_files']:
        bound(row)
    require(cfg['historical_original_route_custody'] == pins['original_route_attestation']
        and read(bound(cfg['historical_original_route_custody']))['original_gate'] == pins['rechecked_original_gate']
        and cfg['historical_archive_transfer_receipt'] == pins['historical_archive_stream_receipt'], 'Exact actual original-route attestation and successful stream receipt')
    activation = inside(cfg['activation_directory']); output = inside(cfg['numerical_output_directory'])
    frozen = {HERE, reader, *(inside(Path(row['path']).parts[0]) for row in reader_pins['source_files']),
        *(inside(Path(row['path']).parts[0]) for row in pins['historical_eight_metadata_files']),
        bound(cfg['relation_family_closure']).parent,
        *(inside(Path(reader_pins['data_source_pins'][key]['path']).parts[0]) for key in ('train', 'development', 'polynormer')),
        inside(reader_pins['union_pins']['original_activation']), inside(reader_pins['union_pins']['unused2_output']),
        inside(reader_pins['historical_archive_stage_directory']).parent}
    require(not activation.exists() and activation.parent.is_dir() and not output.exists() and output.parent.is_dir()
        and activation != output and not activation.is_relative_to(output) and not output.is_relative_to(activation)
        and all(not item.is_relative_to(root) for item in (activation, output) for root in frozen), 'Fresh separate metadata activation and later numerical outputs')
    os.umask(0o077)
    activation.mkdir(mode=0o700)
    record = dict(schema='relation18-root-activation-assembly-report-v1', status='started', complete=False,
        numerical_reader_invoked=False, numerical_imports=False, archive_decoding=False, checkpoint_deserialization=False,
        training=False, owner_launched=False, TEST_access=False, automatic_retry=False, root_input=binding(args.root_input),
        actual_current_observations=cfg['current_terminal_observations'], root_reader_adoption=cfg['root_reader_adoption'])
    write(activation / 'ASSEMBLY_REPORT.json', record)
    try:
        terminal = terminal_records(reader_pins, cfg, observations)
        terminal_path = activation / 'TERMINAL_EVIDENCE.json'; write(terminal_path, terminal)
        stage_path = activation / 'HISTORICAL_ARCHIVE_STAGING.json'; write(stage_path, stage_record(reader_pins, cfg))
        release = read(reader / 'RELEASE_TEMPLATE_DISABLED.json')
        release.update({key: True for key in OPEN_FLAGS})
        for key in ('relation_family_closure', 'relation_parent_owner', 'relation_parent_launch', 'relation_lane_closures',
                'original12_union_closure', 'historical_original_route_custody', 'runtime_evidence', 'resource_readiness'):
            release[key] = cfg[key]
        custody = read(bound(cfg['historical_original_route_custody']))
        release.update(historical_prediction_release=custody['historical_collector_release'],
            historical_archive_staging=binding(stage_path), terminal_evidence=binding(terminal_path),
            readout_manifest_sha256=pins['reader_manifest']['sha256'], output_directory=cfg['numerical_output_directory'])
        release.update(cfg['resources'])
        release_path = activation / 'RELEASE.json'
        supervision = read(reader / 'EXTERNAL_SUPERVISION_TEMPLATE_DISABLED.json')
        supervision.update(enabled=True, root_execution_authorized=True, finite_owned_bound_confirmed=True,
            root_retains_actual_exit_reap_cleanup_transfer_cost=True,
            owned_entry_program=binding(reader / 'collect_relation18.py'),
            argv_prefix=[reader_pins['runtime']['python'], '-B', str(reader / 'collect_relation18.py')],
            release_argument_path=str(release_path))
        supervision.update({key: cfg['resources'][key] for key in ('physical_gpu_uuid', 'external_active_seconds',
            'external_cleanup_seconds', 'external_hard_seconds', 'owned_GPU_cap_bytes', 'maximum_output_bytes')})
        supervision.update(owned_RSS_cap_bytes=cfg['owner_resources']['owned_RSS_cap_bytes'],
            combined_child_log_cap_bytes=cfg['owner_resources']['combined_child_log_cap_bytes'])
        supervisor_path = activation / 'EXTERNAL_SUPERVISION.json'; write(supervisor_path, supervision)
        release['external_supervision'] = binding(supervisor_path); write(release_path, release)
        write(activation / 'NUMERICAL_COMMAND_NOT_LAUNCHED.json', dict(enabled=False, owner_launched=False,
            numerical_reader_invoked=False, requires_separate_root_launch=True,
            argv=supervision['argv_prefix'] + ['--release', str(release_path), '--release-sha256', sha(release_path)],
            release=binding(release_path), existing_run_fit_helper=supervision['existing_run_fit_helper'],
            existing_ownership_helper=supervision['existing_ownership_helper']))
        # The only entry called is the original stdlib consume custody gate.
        gate = module(reader / 'readout_gate.py', '_relation18_root_assembler_metadata_gate')
        _, _, records, checked_output, _, _ = gate.consume(release_path, sha(release_path))
        require(len(records) == 18 and checked_output == output, 'Actual original metadata gate exact eighteen banks')
        record.update(status='complete', complete=True, metadata_gate_passed=True, release=binding(release_path),
            terminal_evidence=binding(terminal_path), external_supervision=binding(supervisor_path),
            historical_archive_staging=binding(stage_path), exact_bank_slots=18,
            numerical_output_created=output.exists(), actual_numerical_member_forwards=0)
        require(record['numerical_output_created'] is False, 'Metadata gate creates no numerical output')
    except BaseException as error:
        record.update(status='failed', complete=False, metadata_gate_passed=False,
            failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        errors = []
        for key, function in (('seconds', lambda: time.monotonic() - started),
                ('CPU_user_seconds', lambda: resource.getrusage(resource.RUSAGE_SELF).ru_utime - usage.ru_utime),
                ('CPU_system_seconds', lambda: resource.getrusage(resource.RUSAGE_SELF).ru_stime - usage.ru_stime),
                ('RSS_process_lifetime_highwater_bytes', lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)))):
            try:
                record[key] = function()
            except BaseException as error:
                record[key] = None
                errors.append(dict(observation=key, type=type(error).__name__, message=str(error)))
        if errors:
            record.update(status='failed', complete=False, resource_observation_failures=errors)
        record['finished_UTC'] = datetime.now(timezone.utc).isoformat()
        write(activation / 'ASSEMBLY_REPORT.json', record)
        write(activation / 'COMPLETE.json', {key: record[key] for key in ('status', 'complete', 'numerical_reader_invoked', 'owner_launched', 'training', 'TEST_access')})
    require(record['complete'], 'Metadata assembly incomplete, preserved once-only attempt')
    print(json.dumps(dict(status='complete', activation=str(activation), release=record['release'], metadata_gate_only=True, numerical_reader_invoked=False)))


if __name__ == '__main__':
    main()
