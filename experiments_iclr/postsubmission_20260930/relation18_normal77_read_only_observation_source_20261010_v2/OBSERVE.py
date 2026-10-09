"""Disabled once-only stdlib observation; writes false-admission drafts only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
READER = 'graph_relation18_selected_readout_source_20261009_v3'
ASSEMBLER = 'relation18_root_activation_assembler_source_20261010_v2'
NUMERICAL_MODULES = ('torch', 'numpy', 'torch_geometric', 'torch_sparse', 'torch_scatter', 'dgl', 'ogb')


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
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Phase-relative metadata custody')
    path = (PHASE / value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Current phase custody only')
    return path


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE) and path.is_file(), 'Existing metadata file inside phase')
    return dict(path=str(path.relative_to(PHASE)), bytes=path.stat().st_size, sha256=sha(path))


def bound(row):
    path = inside(row['path'])
    require(path.suffix == '.json' and path.is_file() and sha(path) == row['sha256'], 'Exact bound JSON metadata only')
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Exact JSON byte size')
    return path


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['runtime_disabled'] is True and seal['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Exact disabled observation seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Observation source unchanged')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in pins['immutable_packets']:
        path = bound(row['manifest'])
        require(read(bound(row['seal']))['manifest_sha256'] == row['manifest']['sha256'], 'Exact source packet seal/manifest')
        for item in read(path)['files']:
            payload = path.parent / item['path']
            require(payload.stat().st_size == item['bytes'] and sha(payload) == item['sha256'], 'Immutable reader/assembler payload unchanged')
    return pins


def proc_stat(pid):
    """Stat metadata only. Never read any process cmdline or executable path."""
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(PID=pid, start_ticks=int(fields[19]), state=fields[0],
        ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]))


def snapshot_groups(group_ids):
    result = {key: [] for key in group_ids}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        row = proc_stat(int(directory.name))
        if row is not None and row['pgid'] in result:
            result[row['pgid']].append(row)
    return {key: sorted(rows, key=lambda row: row['PID']) for key, rows in result.items()}


def query(arguments):
    return subprocess.check_output(['nvidia-smi', *arguments], text=True, timeout=10).splitlines()


def owned_handles(cfg, reader_pins, source_reads):
    def metadata(row):
        path = bound(row)
        source_reads.append(binding(path))
        return read(path)
    family = metadata(cfg['relation_family_closure'])
    parent = metadata(cfg['relation_parent_owner'])
    launch = metadata(cfg['relation_parent_launch'])
    require(family['complete'] is True and family['fixed_scientific_cells'] == 12 and len(family['lane_results']) == 2
        and family['schema'] == 'graph-relation-full12-normal77-family-closure-v1'
        and parent['identity'] == family['parent_identity']
        and all(launch['owner_identity'][key] == family['parent_identity'][key]
            for key in ('PID', 'start_ticks', 'pgid', 'sid', 'argv'))
        and launch['owner_identity']['observation_complete'] is True
        and family['parent_identity']['observation_complete'] is True
        and launch['argv'] == family['parent_identity']['argv']
        and parent['release'] == family['root_release'] == launch['release']
        and launch['detached'] is True and launch['parent_poll'] is None,
        'Bound complete relation family and actual detached parent identities')
    expected_parent = inside(cfg['relation_family_closure']['path']).parent / 'PARENT_OWNER.json'
    require(bound(cfg['relation_parent_owner']) == expected_parent, 'Exact original relation parent receipt')
    handles = [('relation_owner', 'relation12', family['parent_identity'])]
    relation = {}
    for index, lane in enumerate(family['lane_results']):
        saved = metadata(cfg['relation_lane_closures'][str(index)])
        fixed = reader_pins['relation_controller_pins']['lanes'][str(index)]
        require(saved == lane and lane['complete'] is True and lane['failure'] is None
            and lane['completed'] == fixed['cells'] and [row['cell_id'] for row in lane['rows']] == fixed['cells'],
            'Entire complete original relation lane dictionaries')
        for row in lane['rows']:
            receipt = metadata(row['exit_receipt'])
            require(receipt == row['actual_exit_receipt'] and receipt['raw_identity_observation'] == row['child_identity']
                and receipt['terminal_wait_observed'] is True and receipt['exit_code'] == 0
                and receipt['reason'] is None and receipt['signals_sent'] == [] and row['complete'] is True,
                'Real original relation child wait/reap and successful closure')
            require(row['cell_id'] not in relation, 'Unique fixed relation child')
            relation[row['cell_id']] = row['child_identity']
            handles.append(('relation_children', row['cell_id'], row['child_identity']))
    require(len(relation) == 12, 'All twelve relation child handles')
    p = reader_pins['union_pins']
    union = metadata(cfg['original12_union_closure'])
    old = metadata(p['old_terminal_custody'])
    require(union['complete'] is True and union['schema'] == 'Wiki12-old10-plus-never-started2-union-closure-v1'
        and union['old_terminal_custody'] == p['old_terminal_custody'] and union['fatal_error'] is None
        and union['original_fixed_cells'] == 12 and union['original_completed_cells'] == 10,
        'Bound complete old10 plus never-started2 union')
    old_rows = {row['cell_id']: row for row in old['completed']}
    new_rows = {row['cell_id']: row for row in union['new_completed']}
    require(len(old_rows) == 10 and set(new_rows) == {'6307_residual_only', '6307_combined'}, 'Exact original union partitions')
    for role, root, filename in (('original_Wiki12', p['original_activation'], 'PARENT_OWNER.json'),
            ('unused2', p['unused2_output'], 'OWNER.json')):
        owner_binding = cfg['original12_owner_receipts'][role]
        require(bound(owner_binding) == inside(root) / filename, 'Exact original owner JSON receipt')
        saved = metadata(owner_binding); identity = saved if role == 'original_Wiki12' else saved['owner']
        require(identity['PID'] == p['owners'][role]['pid'] and identity['start_ticks'] == p['owners'][role]['start_ticks'],
            'Frozen original owner PID/birth identity')
        handles.append(('original12_owners', role, identity))
    for item in p['roster']:
        cell = item['cell']
        if item['custody_partition'] == 'old10':
            closed = old_rows[cell]; row = closed['exit']
        else:
            closed = new_rows[cell]
            row = dict(path=p['unused2_output'] + '/logs/' + cell + '.EXIT.json', sha256=closed['exit_sha256'])
        receipt = metadata(row); identity = receipt['raw_identity_observation']
        require(receipt['exit_code'] == 0 and receipt['terminal_wait_observed'] is True
            and receipt['reason'] is None and receipt['signals_sent'] == [], 'Real original12 successful direct child wait')
        if item['custody_partition'] != 'old10':
            require(closed['exit_receipt'] == receipt and closed['complete'] is True, 'Exact never-started2 final exit history')
        handles.append(('original12_children', cell, identity))
    return handles


def observe_handles(handles, CUDA_rows):
    observations = dict(schema='relation18-root-current-terminal-observations-v1', complete=False,
        root_observed=True, hostname=socket.gethostname(), TEST_access=False,
        observation_UTC=datetime.now(timezone.utc).isoformat(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        relation_owner={}, relation_children={}, original12_owners={}, original12_children={}, historical_prediction_collector={})
    group_ids = {identity['pgid'] for _, _, identity in handles if 'pgid' in identity}
    groups = snapshot_groups(group_ids)
    for kind, key, identity in handles:
        pid, ticks = identity['PID'], identity['start_ticks']
        require(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0, 'Exact saved process birth tuple')
        current = proc_stat(pid)
        same = current is not None and current['start_ticks'] == ticks
        reused = current is not None and current['start_ticks'] != ticks
        matching_CUDA = [row for row in CUDA_rows if row['PID'] == pid]
        # A different birth is an unrelated identity even when the PID number
        # has CUDA rows. No unrelated cmdline is read and no process is touched.
        no_own_CUDA = not matching_CUDA or reused
        row = dict(identity=identity, hostname=socket.gethostname(), absent=not same,
            no_CUDA_rows=no_own_CUDA, current_stat=current, original_birth_present=same,
            reused_PID_observed=reused, unrelated_reused_PID_CUDA_rows_ignored=bool(reused and matching_CUDA),
            original_OS_exit_code=None, direct_wait_observed=False)
        if 'pgid' in identity and 'sid' in identity:
            members = groups[identity['pgid']]
            new_group = bool(reused and current['pgid'] == identity['pgid'] and current['sid'] == identity['sid']
                and all(value['start_ticks'] >= current['start_ticks'] and value['sid'] == current['sid'] for value in members))
            row.update(process_group_absent=not members or new_group, numeric_group_reused=new_group,
                numeric_group_members=members, group_identity_scope='saved pgid/sid with saved leader birth; unrelated replacement group is excluded')
        else:
            row.update(process_group_absent=None, group_identity_scope='not available in original saved identity')
        require(key not in observations[kind], 'Unique exact terminal observation slot')
        observations[kind][key] = row
    return observations


def historical_observation(cfg, pins, source_reads):
    require(cfg['historical_original_route_custody'] == pins['original_route_attestation'], 'Exact already-observed original route')
    path = bound(cfg['historical_original_route_custody']); custody = read(path); source_reads.append(binding(path))
    require(custody['schema'] == 'relation18-original-route-historical-custody-attestation-v1'
        and custody['root_observed'] is True and custody['hostname'] == 'anogena-2-0'
        and custody['collector_child_direct_wait_reap'] is True and custody['own_CUDA_rows'] == []
        and custody['raw_payloads_decoded'] is False and custody['TEST_access'] is False, 'Actual original-route collector observation')
    owner_path, terminal_path = bound(custody['historical_child_owner']), bound(custody['historical_child_terminal'])
    source_reads.extend((binding(owner_path), binding(terminal_path)))
    owner, terminal = read(owner_path), read(terminal_path)
    identity = owner['identity']
    require(terminal['identity'] == identity and terminal['reaped'] is True and terminal['exit_code'] == 0
        and terminal['timed_out'] is False and terminal['cap_exceeded'] is False
        and terminal['error'] is None and terminal['cleanup_error'] is None, 'Actual historical collector direct child receipt')
    matches = [row for row in custody['actual_observations'] if row['PID'] == identity['PID']
        and row['expected_start_ticks'] == identity['start_ticks'] and row['present'] is False]
    require(len(matches) == 1, 'Historical collector absent on its own original route')
    return dict(hostname='anogena-2-0', identity=identity, absent=True, no_CUDA_rows=True,
        source_current_attestation=cfg['historical_original_route_custody'],
        observation_scope='actual original-route attestation; no normal77 PID lookup',
        collector_command_separate_from_identity=owner['command'], wait_and_reap_observed=True, exit_code=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--root-input', type=Path)
    parser.add_argument('--root-input-sha256')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, observations_executed=False, numerical_imports=False)))
        return
    started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    pins = source_gate()
    require(args.root_input is not None and sha(args.root_input) == args.root_input_sha256, 'Exact separate root read-only admission')
    cfg = read(args.root_input)
    require(cfg['schema'] == 'relation18-normal77-read-only-observation-input-v1'
        and cfg['enabled'] is True and cfg['root_read_only_metadata_observation_authorized'] is True
        and cfg['observation_entry_sha256'] == sha(__file__) and cfg['observation_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
        and all(cfg[key] is False for key in ('scientific_opening', 'metadata_assembly_invocation', 'numerical_reader_invocation',
            'training', 'automatic_retry', 'TEST_access')), 'Explicit read-only observation only')
    reader_pins = read(PHASE / READER / 'SOURCE_BINDINGS.json'); expected = reader_pins['runtime']
    require(socket.gethostname() == expected['hostname'] and Path.cwd().resolve() == Path(expected['repository'])
        and str(Path(sys.executable).absolute()) == expected['python'], 'Actual pinned normal77 host/repository/interpreter')
    require(not any(name in sys.modules for name in NUMERICAL_MODULES), 'No numerical provider imports')
    output = inside(cfg['admission_output_directory'])
    protected = {HERE, PHASE / READER, PHASE / ASSEMBLER,
        bound(cfg['relation_family_closure']).parent,
        *(inside(Path(row['path']).parts[0]) for row in reader_pins['source_files']),
        inside(reader_pins['union_pins']['original_activation']), inside(reader_pins['union_pins']['unused2_output']),
        inside(reader_pins['historical_archive_stage_directory']).parent,
        *(inside(Path(row['path']).parts[0]) for row in pins['historical_eight_metadata_files'])}
    require(not output.exists() and output.parent.is_dir()
        and all(not output.is_relative_to(path) for path in protected), 'Fresh separate root admission, no sources/histories modified')
    os.umask(0o077); output.mkdir(mode=0o700)
    source_reads = []
    record = dict(schema='relation18-normal77-read-only-observation-report-v1', status='started', complete=False,
        root_input=binding(args.root_input), source_metadata_reads=source_reads, numerical_imports=False,
        checkpoint_or_array_reads=False, outcomes_opened=False, process_cmdlines_read=False,
        signals_sent=False, providers_or_GPU_models_called=False, metadata_assembly_invoked=False,
        numerical_reader_invoked=False, training=False, TEST_access=False, automatic_retry=False)
    write(output / 'OBSERVATION_REPORT.json', record)
    try:
        handles = owned_handles(cfg, reader_pins, source_reads)
        raw_CUDA = query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
        CUDA_rows = []
        for line in raw_CUDA:
            if not line.strip():
                continue
            pieces = [part.strip() for part in line.split(',')]
            require(len(pieces) == 3 and pieces[1].isdigit(), 'Actual NVML compute row format')
            CUDA_rows.append(dict(GPU_uuid=pieces[0], PID=int(pieces[1]), used_memory_MiB=pieces[2]))
        observations = observe_handles(handles, CUDA_rows)
        observations['historical_prediction_collector']['collector'] = historical_observation(cfg, pins, source_reads)
        normal_rows = [row for key in ('relation_owner', 'relation_children', 'original12_owners', 'original12_children')
            for row in observations[key].values()]
        observations['complete'] = all(row['absent'] and row['no_CUDA_rows']
            and (row['process_group_absent'] is True if row['process_group_absent'] is not None else True) for row in normal_rows)
        observation_path = output / 'CURRENT_TERMINAL_OBSERVATIONS.json'; write(observation_path, observations)
        inventory = query(['--query-gpu=uuid', '--format=csv,noheader'])
        packages = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')}
        actual = dict(hostname=socket.gethostname(), repository=str(Path.cwd().resolve()), python=str(Path(sys.executable).absolute()),
            PYTHONPATH=os.environ.get('PYTHONPATH', ''), physical_gpu_inventory=inventory, **packages)
        require(not any(name in sys.modules for name in NUMERICAL_MODULES), 'Version metadata reads import no numerical package')
        runtime = dict(schema='relation18-root-saved-runtime-evidence-v1', root_observed=True,
            observation_UTC=datetime.now(timezone.utc).isoformat(), runtime=actual,
            package_version_source='importlib.metadata distribution metadata only', providers_imported=False,
            CUDA_VISIBLE_DEVICES_during_metadata_observation=os.environ.get('CUDA_VISIBLE_DEVICES'))
        runtime_path = output / 'RUNTIME_EVIDENCE.json'; write(runtime_path, runtime)
        require(inventory == expected['physical_gpu_inventory'], 'Exact original normal77 GPU inventory')
        free = query(['--id=' + expected['physical_gpu_inventory'][0], '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        require(len(free) == 1 and free[0].strip().isdigit(), 'Actual GPU0 free memory metadata')
        free_bytes = int(free[0].strip()) * 1024**2
        draft = read(PHASE / ASSEMBLER / 'ROOT_INPUT_TEMPLATE_DISABLED.json')
        minimum = draft['resources']['minimum_fresh_GPU_free_bytes']
        readiness = dict(schema='relation18-root-saved-resource-readiness-v1', root_observed=True,
            observation_UTC=datetime.now(timezone.utc).isoformat(), runtime_evidence=binding(runtime_path),
            physical_gpu_uuid=expected['physical_gpu_inventory'][0], minimum_fresh_GPU_free_bytes=minimum,
            actual_free_GPU_bytes=free_bytes, ready=observations['complete'] and actual == expected and free_bytes >= minimum,
            resource_source='single actual nvidia-smi metadata snapshot; no model or allocation call',
            source_review_or_opening_approved=False)
        readiness_path = output / 'RESOURCE_READINESS.json'; write(readiness_path, readiness)
        for key in ('relation_family_closure', 'relation_parent_owner', 'relation_parent_launch', 'relation_lane_closures', 'original12_union_closure'):
            draft[key] = cfg[key]
        draft.update(assembler_manifest_sha256=pins['assembler_manifest']['sha256'],
            assembler_entry_sha256=pins['assembler_entry']['sha256'], current_terminal_observations=binding(observation_path),
            runtime_evidence=binding(runtime_path), resource_readiness=binding(readiness_path), root_reader_adoption=None)
        require(all(draft[key] is False for key in ('enabled', 'root_metadata_assembly_authorized', 'root_metadata_gate_only_authorized',
            'root_opening_release_fields_adopted', 'root_owns_existing_finite_owner_and_later_numerical_launch',
            'numerical_reader_invocation', 'training', 'automatic_retry', 'TEST_access')), 'All actual ROOT_INPUT draft approvals remain false')
        draft_path = output / 'ROOT_INPUT_DRAFT.disabled.json'; write(draft_path, draft)
        record.update(status='complete', complete=True, terminal_absence_verified=observations['complete'],
            runtime_matches_exact_reader=actual == expected, physical_resource_ready=readiness['ready'],
            same_route_saved_handles=len(handles), historical_collector_from_original_route_only=True,
            current_terminal_observations=binding(observation_path), runtime_evidence=binding(runtime_path),
            resource_readiness=binding(readiness_path), disabled_ROOT_INPUT_draft=binding(draft_path),
            all_admission_approval_flags_false=True, no_framework_or_launch_added=True)
        require(observations['complete'], 'Exact original process/group/CUDA absence not established; draft stays disabled')
        require(actual == expected, 'Actual runtime distribution metadata differs; draft stays disabled')
    except BaseException as error:
        record.update(status='failed', complete=False, failure=dict(type=type(error).__name__, message=str(error)))
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
        write(output / 'OBSERVATION_REPORT.json', record)
        write(output / 'COMPLETE.json', {key: record[key] for key in ('status', 'complete', 'numerical_reader_invoked', 'training', 'TEST_access')})
    require(record['complete'], 'Read-only observation incomplete, retain this bounded attempt')
    print(json.dumps(dict(status='complete', output=str(output), draft=record['disabled_ROOT_INPUT_draft'], all_approval_flags_false=True)))


if __name__ == '__main__':
    main()
