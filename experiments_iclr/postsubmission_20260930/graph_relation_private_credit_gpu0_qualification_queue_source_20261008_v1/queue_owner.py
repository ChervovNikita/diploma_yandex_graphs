"""Once-only root GPU0 qualification queue; atomically admit the fixed pending lane."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
STOP = [False]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative root custody')
    path = (PHASE / relative).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Root path leaves project phase')
    return path


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256']
            and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Bound file changed: ' + row['path'])
    return path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path), bytes=path.stat().st_size)


def fresh_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


def index_guard(pins):
    bound(pins['family_release']); bound(pins['lane1_admission'])
    for row in pins['lane1_cell_releases']:
        bound(row)
    path = inside(pins['index_path']); value = read(path)
    require(value['schema'] == 'graph-relation-full12-root-lane-index-v1'
        and value['family_release_sha256'] == pins['family_release']['sha256']
        and value['controller_manifest_sha256'] == pins['controller_manifest']['sha256']
        and value['root_lane_binding_authorized'] is True and set(value['lanes']) == {'0', '1'}
        and value['lanes']['0'] is None and value['lanes']['1'] == pins['expected_lane1_binding'],
        'Exact existing family/index; lane0 still pending and consumed lane1 unchanged')
    return path, value, sha(path)


def family_live(pins, control_pins, helper):
    expected = pins['family_owner']; actual = helper.identity(expected['pid'])
    require(actual is not None and actual['start_ticks'] == expected['start_ticks'] and actual['state'] != 'Z',
            'Exact existing full12 owner must remain live for pending lane admission')
    family = read(bound(pins['family_release']))
    owner_path = inside(family['owner_output_directory']) / 'PARENT_OWNER.json'
    owner = read(owner_path)['identity']
    require(owner['PID'] == expected['pid'] and owner['start_ticks'] == expected['start_ticks']
            and actual['pgid'] == actual['sid'] == expected['pid'], 'Exact detached existing family identity')
    for cell in control_pins['lanes']['0']['cells']:
        preview = read(bound(control_pins['previews'][cell]))
        require(not inside(preview['output']).exists(), 'No previously launched/trained6203 cell admitted or retried')
    return binding(owner_path)


def qualification_release(pins, cfg, control_pins):
    path = bound(cfg['qualification_release']); value = read(path)
    original = read(bound(pins['existing_GPU1_qualifier_release']))
    changes = {'physical_gpu_uuid', 'output'}
    require(all(value[key] == entry for key, entry in original.items() if key not in changes)
        and value['physical_gpu_uuid'] == pins['GPU0'] and value['output'] == pins['qualification_output']
        and path == inside(pins['worker_release_path']), 'Unchanged enabled8-update qualifier; only GPU/output differ')
    require(value['source_manifest_sha256'] == pins['scientific_manifest']['sha256']
        and value['policies'] == pins['policies'] and value['seed'] == 6101
        and value['VALID_scores_read'] is False and value['TEST_access'] is False
        and value['automatic_retry'] is False, 'Qualification-only fixed scope')
    return path, value


def validate_pass(pins, cfg, activation, output):
    exit_path = activation / 'EXIT.json'; exit_record = read(exit_path)
    result_path = output / 'QUALIFIED.json'; result = read(result_path)
    child = read(activation / 'CHILD_OWNER.json')
    owner = read(activation / 'OWNER.json')
    queue_owner = read(activation / 'QUEUE_OWNER.json')['identity']
    require(exit_record['child_identity'] == child['identity']
        and child['identity']['pid'] == child['identity']['group'] == child['identity']['session']
        and child['argv'] == [read(bound(cfg['qualification_release']))['runtime_python'], '-B',
            str(bound(pins['scientific_manifest']).parent / 'qualify.py'), '--release',
            str(bound(cfg['qualification_release'])), '--release-sha256', cfg['qualification_release']['sha256']]
        and owner['owner']['pid'] == queue_owner['PID'] and owner['owner']['start_ticks'] == queue_owner['start_ticks']
        and owner['release_sha256'] == cfg['qualification_release']['sha256'],
        'Exact original supervisor parent/child/argv/qualification release custody')
    require(exit_record['exit_code'] == 0 and exit_record['reason'] is None and exit_record['signals'] == []
        and exit_record['actual_exit_and_reap'] is True and exit_record['child_after'] is None
        and exit_record['child_no_CUDA_rows'] is True and exit_record['scientific_fits'] == 0
        and exit_record['quality_scores_read'] is False and exit_record['automatic_retry'] is False
        and exit_record['source_manifest_sha256'] == pins['scientific_manifest']['sha256']
        and exit_record['release_sha256'] == cfg['qualification_release']['sha256']
        and exit_record['inclusive_seconds'] <= pins['qualification_hard_seconds']
        and exit_record['peak_observed_child_RSS_bytes'] <= 32 * 1024**3
        and exit_record['peak_observed_child_GPU_bytes'] <= 32 * 1024**3,
        'One actual complete/reaped/absent/noCUDA bounded qualification pass')
    require(result['complete'] is True and result['source_static_only'] is False
        and result['source_manifest_sha256'] == pins['scientific_manifest']['sha256']
        and result['physical_gpu_uuid'] == pins['GPU0'] and result['policies'] == pins['policies']
        and result['real_complete_TRAIN_updates'] == result['native_models_constructed'] == 8
        and result['scientific_training_fits'] == 0 and result['VALID_scores_read'] is False
        and result['TEST_access'] is False and result['fabricated_graph_or_numerical_fixture_used'] is False,
        'Actual unchanged native qualification; no metrics or fixture credit')
    work = dict(shadow_member_forwards=8, replay_member_forwards=8, output_cotangent_collections=2,
                member_reverse_collections=16, optimizer_bank_updates=1, exact_member_RNG_endpoint_checks=1)
    require(len(result['rows']) == 8 and [(row['policy'], row['global_stage']) for row in result['rows']]
        == [(policy, stage) for policy in pins['policies'] for stage in (False, True)]
        and all(row['complete'] is True and row['VALID_metrics_read'] is False and row['work'] == work
                and row['fullgraph_nodes'] == 11701 and row['TRAIN_labels'] == 580 for row in result['rows']),
        'Exactly the eight original policy/stage TRAIN updates; no qualification rerun')
    # Gradient norms/results are not used to choose policies or numerical benchmark outcomes.
    return binding(result_path), binding(exit_path), exit_record


def admit_lane0(pins, control_pins, controller, helper, qualification, queue_release, deadline):
    require(not STOP[0] and time.monotonic() < deadline, 'Stopped/expired queue cannot admit a new lane')
    family_live(pins, control_pins, helper)
    path, before, before_sha = index_guard(pins)
    activation = inside(pins['full_family_activation']); cells = []
    lane_path = activation / 'LANE_0_ADMISSION.json'
    targets = [activation / 'cells' / (cell + '.json') for cell in control_pins['lanes']['0']['cells']]
    require(not lane_path.exists() and all(not target.exists() for target in targets), 'Fresh pending lane artifacts; no overwrite/retry')
    for cell, target in zip(control_pins['lanes']['0']['cells'], targets):
        # Same fixed source/recipe/seeds/order/outputs/caps. Only approved admission fields are filled.
        value = read(bound(control_pins['previews'][cell]))
        value.update(enabled=True, root_execution_authorized=True, source_review_approved=True,
            native_qualification_approved=True, source_manifest_sha256=pins['scientific_manifest']['sha256'],
            native_qualification=qualification, physical_gpu_uuid=pins['GPU0'], GPU_assignment_requires_root_freeze=False)
        require(value['seed'] == 6203 and value['policy'] in pins['policies'] and value['risk_beta'] == .5
            and value['epochs'] == 1100 and value['auxiliary'] is False
            and value['anchor_reuse_authorized'] is False and value['comparative_opening_authorized'] is False
            and value['original12_union_closure'] is None and value['original12_union_pending'] is True,
            'Root continuation is the four unchanged training-only6203 policies')
        fresh_json(target, value); cells.append(dict(binding(target), cell_id=cell))
    admission = dict(schema='graph-relation-full12-root-lane-admission-v1', enabled=True,
        root_fit_launch_authorized=True, source_review_approved=True, native_qualification_approved=True,
        controller_manifest_sha256=pins['controller_manifest']['sha256'], source_manifest_sha256=pins['scientific_manifest']['sha256'],
        lane=0, physical_gpu_uuid=pins['GPU0'], native_qualification=qualification, cells=cells,
        TEST_access=False, automatic_retry=False, template_only=False,
        root_conditional_authority=queue_release)
    fresh_json(lane_path, admission); lane_binding = binding(lane_path)
    latest_path, latest, latest_sha = index_guard(pins)
    require(latest_path == path and latest_sha == before_sha and latest == before,
            'Root index changed during preparation; preserve files and stop without overwriting')
    require(not STOP[0] and time.monotonic() < deadline, 'Stopped/expired queue cannot commit index')
    family_live(pins, control_pins, helper)
    after = dict(before, lanes=dict(before['lanes'], **{'0': lane_binding}))
    temporary = path.with_name(path.name + '.GPU0_QUEUE.tmp')
    fresh_json(temporary, after)
    require(sha(path) == before_sha, 'Index changed before atomic replacement; no overwrite')
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    committed = read(path)
    require(committed == after and committed['lanes']['1'] == pins['expected_lane1_binding'], 'Exact lane0-only index commit')
    return dict(lane0_admission=lane_binding, cell_releases=cells, index_before_sha256=before_sha,
                index_after=binding(path), lane1_binding_preserved=pins['expected_lane1_binding'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    began = time.monotonic(); usage_start = resource.getrusage(resource.RUSAGE_SELF)
    require(sha(args.release) == args.release_sha256, 'Exact separate root queue release')
    cfg = read(args.release)
    require(cfg.get('schema') == 'graph-relation-GPU0-queue-root-release-v1'
        and all(cfg.get(key) is True for key in ('enabled', 'root_queue_launch_authorized', 'source_review_approved',
            'one_unchanged_native_qualification_authorized', 'automatic_lane0_admission_after_exact_pass_authorized',
            'finite_wait_and_supervisor_bounds_authorized')),
        'Queue preparation is disabled pending root conditional admission')
    require(cfg['TEST_access'] is False and cfg['automatic_retry'] is False
        and cfg['anchor_reuse_authorized'] is False and cfg['comparative_opening_authorized'] is False, 'No numerical opening/retry authority')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    controller = module(bound(pins['controller_program']), '_queue_existing_controller')
    controller.seal(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['queue_manifest_sha256']))
    controller.seal(pins['controller_manifest']); controller.seal(pins['scientific_manifest'])
    bound(pins['existing_root_supervisor'])
    control_pins = read(bound(pins['controller_pins'])); controller.physical(control_pins)
    helper = controller.module(control_pins['ownership_helper'], '_queue_existing_ownership'); helper.GPU = pins['GPU0']
    qualification_release(pins, cfg, control_pins)
    index_guard(pins); family_receipt = family_live(pins, control_pins, helper)
    activation = inside(pins['qualification_activation']); output = inside(pins['qualification_output'])
    require(activation.is_dir() and not output.exists()
        and not any((activation / name).exists() for name in ('QUEUE_OWNER.json', 'OWNER.json', 'CHILD_OWNER.json', 'EXIT.json', 'QUEUE_RESULT.json')),
        'Fresh separate once-only GPU0 activation/output; no completed qualification retry')
    identity = helper.identity(os.getpid())
    require(identity is not None and identity['PID'] == identity['pgid'] == identity['sid'], 'Actual detached queue owner')
    fresh_json(activation / 'QUEUE_OWNER.json', dict(identity=identity, queue_release=binding(args.release), family_owner_receipt=family_receipt,
        exact_family_release=pins['family_release'], lane1_binding=pins['expected_lane1_binding'], automatic_retry=False, scores_read=False))
    # Exclusive once-only continuation reservation; root must not add lane0 while this queue owns it.
    fresh_json(inside(pins['full_family_activation']) / 'LANE_0_QUEUE_RESERVATION.json',
        dict(queue_identity=identity, queue_release=binding(args.release), family_release=pins['family_release'], automatic_retry=False))
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: STOP.__setitem__(0, True))
    deadline = began + pins['queue_hard_seconds']; qualification_attempts = 0; commit = None; failure = None
    qualification_binding = exit_binding = exit_record = None; wait_seconds = 0.
    try:
        wait_started = time.monotonic()
        while True:
            require(not STOP[0], 'Queue stopped before qualification; no next action')
            index_guard(pins); family_live(pins, control_pins, helper); controller.physical(control_pins)
            terminal = controller.dependency_terminal(control_pins, helper)
            rows = helper.query(['--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], 10)
            free = [row.split(',')[1].strip() for row in rows if row.split(',')[0].strip() == pins['GPU0']]
            require(len(free) == 1 and free[0].isdigit(), 'Exact GPU0 resource telemetry')
            available = int(free[0]) * 1024**2
            wait_seconds = time.monotonic() - wait_started
            remaining = min(pins['scheduling_seconds'] - wait_seconds, deadline - time.monotonic() - 600)
            controller.write(activation / 'QUEUE_WAIT.json', dict(UTC=datetime.now(timezone.utc).isoformat(), dependency_terminal=terminal,
                free_GPU0_bytes=available, required_bytes=36 * 1024**3, elapsed_wait_seconds=wait_seconds,
                remaining_seconds=max(0, remaining), qualification_attempts=0, scores_read=False))
            require(remaining > 0, 'Finite queue scheduling window exhausted; no qualification/retry')
            if terminal is not None and available >= 36 * 1024**3:
                fresh_json(activation / 'UNUSED2_TERMINAL_CUSTODY.json', terminal); break
            time.sleep(min(15, remaining))
        supervisor = module(HERE / 'qualification_supervisor.py', '_queue_reused_root_qualification_supervisor')
        supervisor.A = activation; supervisor.ACTIVATION_NAME = activation.name
        supervisor.RELEASE_SHA = cfg['qualification_release']['sha256']
        # Chain queue stop custody with the engine's original owned-child stop handler.
        def register(signum, handler):
            def chained(number, frame):
                STOP[0] = True; handler(number, frame)
            return signal.signal(signum, chained)
        supervisor.signal = SimpleNamespace(SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM, SIGKILL=signal.SIGKILL, signal=register)
        qualification_attempts = 1
        fresh_json(activation / 'QUALIFICATION_ATTEMPT.json', dict(attempts=1, release=cfg['qualification_release'],
            source=pins['scientific_manifest'], supervisor_original=pins['existing_root_supervisor'], automatic_retry=False))
        require(time.monotonic() + pins['qualification_hard_seconds'] < deadline and not STOP[0], 'Full qualification envelope must fit')
        code = supervisor.main()  # Original owned engine:300active +10cleanup +20terminal reserve.
        require(code == 0, 'Qualification failed; preserve original supervisor artifacts, no lane0 admission')
        qualification_binding, exit_binding, exit_record = validate_pass(pins, cfg, activation, output)
        commit = admit_lane0(pins, control_pins, controller, helper, qualification_binding, binding(args.release), deadline)
    except (Exception, KeyboardInterrupt) as error:
        failure = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if (activation / 'EXIT.json').is_file() and exit_record is None:
            exit_record = read(activation / 'EXIT.json'); exit_binding = binding(activation / 'EXIT.json')
        retained = {name: binding(activation / name) for name in ('OWNER.json', 'CHILD_OWNER.json', 'EXIT.json', 'QUALIFICATION_ATTEMPT.json', 'stdout.log', 'stderr.log')
                    if (activation / name).is_file()}
        worker_artifacts = {name: binding(output / name) for name in ('QUALIFIED.json', 'FAILURE.json', 'PROGRESS.json')
                            if (output / name).is_file()}
        result = dict(schema='graph-relation-GPU0-qualification-queue-result-v1', complete=commit is not None and failure is None,
            queue_identity=identity, queue_release=binding(args.release), exact_family_release=pins['family_release'],
            qualification_attempts=qualification_attempts, qualification=qualification_binding, qualification_exit=exit_binding,
            qualification_external_cost=exit_record, lane0_commit=commit, failure=failure, retained_artifacts=retained,
            retained_worker_artifacts=worker_artifacts,
            scheduling_wait_seconds=wait_seconds, inclusive_queue_seconds=time.monotonic() - began,
            queue_hard_seconds=pins['queue_hard_seconds'], CPU_user_seconds=usage.ru_utime - usage_start.ru_utime,
            CPU_system_seconds=usage.ru_stime - usage_start.ru_stime, peak_RSS_bytes=int(usage.ru_maxrss * 1024),
            scientific_fits_launched_by_queue=0, benchmark_or_validation_metrics_read=False, TEST_access=False,
            automatic_retry=False, anchor_reuse_authorized=False, comparative_opening_authorized=False,
            namespace_or_mount_changes=False)
        fresh_json(activation / 'QUEUE_RESULT.json', result)
    print(json.dumps(dict(queue_terminal=True, qualified_and_lane0_admitted=result['complete'], scores_read=False)), flush=True)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
