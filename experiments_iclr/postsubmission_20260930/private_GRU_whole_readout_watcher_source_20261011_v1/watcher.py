"""Finite private-GRU whole-family handoff; producer bindings, no scientific gate."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
HERE = Path(__file__).resolve().parent
BLOCKS = (10103, 10427, 10949)
ARMS = ('B4', 'G1', 'GF1', 'GM4', 'GN4', 'GI16', 'GFI16')
ENDPOINTS = ('P1_b0', 'P1_b1', 'P1_b2', 'P1_b3') + ARMS + ('GA4', 'inherited_WI4')
RAW_REFERENCES = ('P1_b0', 'B4', 'G1', 'GF1', 'GN4', 'GI16', 'GFI16', 'GA4', 'inherited_WI4')
ACQUISITION_MANIFEST_SHA = '6ae17dc319254a23910acef265c102ca83db471e069763add3652f27e32d1ab5'
READER_MANIFEST_SHA = 'd2382e9ca813a75529c7b9bd6cb8089a877dbca5a1b6e18893c468f32bcabd22'
SPEC_SHA = 'daa1bfc411f4f6a2a8e5511dad4f45d3ec47435dfd67ef2c42fc8dc7a998c7e2'
ACQUISITION_OWNER = dict(pid=640426, start_ticks=6055039154, pgid=640426)
ACQUISITION_CHILD = dict(pid=640428, start_ticks=6055039165, pgid=640428)
LIMITS = dict(wall_seconds=7200, cuda_bytes=34359738368, rss_bytes=17179869184)
WAIT_SECONDS = 604800
WHOLE_SECONDS = 619380
CLEANUP_RESERVE = 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def route():
    require(socket.gethostname() == 'anogena-2-0', 'Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=15).splitlines() == [UUID], 'Excluded wrong sole GPU')
    require(Path.cwd() == REPO, 'Authorized allocation repository required')


def receipt(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'Receipt outside authorized project')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'Exact receipt schema')
    require(receipt(item['path']) == item, 'Bound file differs: ' + item['path'])
    return Path(item['path'])


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def identity(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return None
    fields = path.read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgid=int(fields[2]))


def source_check(directory, item):
    directory = Path(directory).resolve()
    require(Path(item['path']).resolve() == directory / 'MANIFEST.json', 'Manifest/source directory linkage')
    manifest = json.loads(checked(item).read_text())
    for rel, fixed in manifest['files'].items():
        actual = receipt(directory / rel)
        require((actual['bytes'], actual['sha256']) == (fixed['bytes'], fixed['sha256']),
            'Sealed source differs: ' + rel)
    return manifest


def worker_command(source, config_path):
    return [str(REPO / '.venv/bin/python'), '-B', '-u', str(source / 'worker.py'), str(config_path)]


def start_ok(start, stage, source, config_path, config, expected_owner=None, expected_child=None):
    require(start['stage'] == stage and start['command'] == worker_command(source, config_path)
        and start['direct_child_wait'] is True and start['retry'] is False
        and start['child_wall_seconds'] == config['limits']['wall_seconds']
        and start['external_wall_seconds'] == config['limits']['wall_seconds'] + 60,
        'Exact producer stage/worker/CONFIG/finite-owner linkage')
    for key, expected in (('owner', expected_owner), ('child', expected_child)):
        value = start[key]
        require(set(value) == {'pid', 'start_ticks', 'pgid'} and all(type(v) is int and v > 0 for v in value.values())
            and (expected is None or value == expected), 'Exact producer birth identity: ' + key)
    require(start['child']['pgid'] == start['child']['pid'], 'Owned worker process group')
    return start


def owner_ok(path, stage, start, source, config_path, config):
    require(Path(path) == Path(config['owner_output']) / 'OWNER_END.json', 'Exact producer owner output')
    value = json.loads(Path(path).read_text())
    start_ok(value, stage, source, config_path, config, start['owner'], start['child'])
    require(value['status'] == 'COMPLETE_EXIT_0' and value['exit_code'] == 0
        and value['timed_out'] is False and value['child_pid_absent'] is True
        and value['owned_cuda_pid_absent'] is True, 'Successful producer owner exit/reap required')
    require(identity(value['child']['pid']) != value['child'], 'Owned worker remains live')
    return value


def reader_metadata_helpers(source):
    # Sealed modules have stdlib-only top-level imports. No numerical execution
    # occurs here; close_panel checks complete metadata and opaque byte receipts.
    for name in ('runtime', 'arithmetic', 'reader'):
        loaded_name = 'private_GRU_watched_reader' if name == 'reader' else name
        spec = importlib.util.spec_from_file_location(loaded_name, source / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[loaded_name] = module
        spec.loader.exec_module(module)
    return sys.modules['private_GRU_watched_reader']


def fill_reader_template(source, stage, base, output, root_decision):
    template = json.loads((source / (stage.upper() + '_CONFIG.disabled.json')).read_text())
    require(template['execution_enabled'] is False and template['stage'] == stage
        and template['blocks'] == list(BLOCKS) and template['endpoint_ids'] == list(ENDPOINTS)
        and template['custody'] == base['custody'] and template['inherited_sources'] == base['inherited_sources'],
        'Exact disabled sealed reader template')
    template.update(base, execution_enabled=True, stage=stage, root_decision=root_decision,
        output=str(output / (stage + '_v1')), owner_output=str(output / (stage + '_owner_v1')), limits=dict(LIMITS))
    return template


def endpoint_closure(path, action, reader, custodian, prior):
    require(Path(path) == Path(action['output']) / 'TEMPERATURES.json', 'Exact CAL producer output')
    value = json.loads(Path(path).read_text())
    rows = value['records']
    index = {(r['block'], r['endpoint_id']): r for r in rows}
    expected = {(b, e) for b in BLOCKS for e in ENDPOINTS}
    require(value['complete'] is True and value['retry'] is False and value['endpoints'] == 39
        and len(rows) == len(index) == 39 and set(index) == expected
        and value['actual_scalar_updates'] == 16500 and value['inherited_scalar_updates'] == 3000
        and value['total_bound_scalar_updates'] == 19500 and value['DEV_truth_read'] is False
        and value['panel'] == action['panel'] and value['role_custody'] == action['custody']
        and value['inherited_sources'] == action['inherited_sources']
        and value['custody']['stage'] == 'calibrate' and value['custody']['root_decision'] == action['root_decision']
        and value['custody']['source_manifest'] == action['source_manifest'],
        'Whole39 CAL endpoints bound to exact producer CONFIG and source')
    panel = json.loads(checked(action['panel']).read_text())
    banks = {(r['block'], r['endpoint_id']): r for r in panel['records']}
    for (b, e), ep in index.items():
        reader.endpoint_temperature(ep, banks[b, e]['exports']['cal'],
            custodian['files'][f'calibration_only/block_{b}_cal.npz'])
        inherited = e in ('P1_b0', 'inherited_WI4')
        require(ep['inherited'] is inherited, 'Exactly33new+6inherited temperatures')
        if inherited:
            family = 'P1' if e == 'P1_b0' else 'WI4'
            original = prior[family]['selected'][b]['endpoint']
            require(ep['original_endpoint'] == original
                and ep['original_temperature_source'] == prior[family]['receipts']['temperatures']
                and all(ep[k] == original[k] for k in ('updates', 'endpoint', 'selected_step', 'logT', 'temperature',
                    'CAL_minimum_NLL', 'cal_archive', 'cal_truth', 'objective_forward_evaluations', 'operator')),
                'Unchanged inherited CAL record; current origin/endpoint aliases stay explicit')
        else:
            require(ep['origin'] == 'new_private_GRU_CAL', 'Fresh CAL endpoint origin')
    require(sum(r['inherited'] for r in rows) == 6
        and sum(r['updates'] for r in rows if not r['inherited']) == 16500,
        'All33 fresh scalar fits precede DEV')
    return value


def cal_confidence_guard(temperatures, clarification_receipt):
    """Literal fitted-CAL verdict after whole closure; never a handoff gate."""
    rows = temperatures['records']
    endpoints = {(r['block'], r['endpoint_id']): r for r in rows}
    require(temperatures['complete'] is True and len(rows) == len(endpoints) == 39
        and set(endpoints) == {(b, e) for b in BLOCKS for e in ENDPOINTS}, 'Whole endpoints before literal CAL guard')
    contrasts = []
    for reference in RAW_REFERENCES:
        differences = [endpoints[b, 'GM4']['CAL_minimum_NLL'] - endpoints[b, reference]['CAL_minimum_NLL'] for b in BLOCKS]
        require(all(math.isfinite(v) for v in differences), 'Finite complete fitted-CAL differences')
        contrasts.append(dict(candidate='GM4', reference=reference, blocks=list(BLOCKS),
            per_block_delta_nat=differences, maximum_each_nat=.005, passed=all(v <= .005 for v in differences)))
    means = {e: statistics.mean(endpoints[b, e]['CAL_minimum_NLL'] for b in BLOCKS) for e in RAW_REFERENCES}
    candidate_mean = statistics.mean(endpoints[b, 'GM4']['CAL_minimum_NLL'] for b in BLOCKS)
    mean_pass = candidate_mean <= min(means.values())
    return dict(complete=True, prospective_clarification=clarification_receipt, evaluation_population='CAL',
        metric='CAL_minimum_NLL at selected scalar temperature', temperature_fitted_on_this_population=True,
        independent_confidence_evidence=False, endpoints=39, new_endpoints=33, inherited_endpoints=6,
        inherited_endpoints_unchanged=True, contrasts=contrasts, candidate_mean_NLL=candidate_mean,
        reference_mean_NLL=means, best_reference_mean_NLL=min(means.values()), mean_rule_pass=mean_pass,
        original_literal_CAL_confidence_guard_pass=all(r['passed'] for r in contrasts) and mean_pass,
        score_handoff_affected=False, raw_member_continuation_criteria_changed=False,
        separate_added_DEVELOPMENT_protection='Unchanged sealed reader calibrated_NLL confidence verdict; reported separately')


def terminate_owned(proc, birth, owner_output):
    """Adapted finite cleanup, only for this watcher's own readout processes."""
    cleanup = dict(owner=birth, worker=None, TERM_sent=False, KILL_sent=False)
    start_path = owner_output / 'OWNER_START.json'
    if start_path.exists():
        started = json.loads(start_path.read_text())
        if started.get('owner') == birth:
            cleanup['worker'] = started.get('child')
    worker = cleanup['worker']
    if worker and identity(worker['pid']) == worker:
        os.killpg(worker['pgid'], signal.SIGTERM)
        cleanup['TERM_sent'] = True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and ((worker and identity(worker['pid']) == worker) or proc.poll() is None):
        time.sleep(.25)
    if worker and identity(worker['pid']) == worker:
        os.killpg(worker['pgid'], signal.SIGKILL)
        cleanup['KILL_sent'] = True
    if proc.poll() is None and identity(proc.pid) == birth:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if identity(proc.pid) == birth:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    cleanup.update(owner_waited=True, owner_pid_absent=identity(proc.pid) != birth,
        worker_pid_absent=None if worker is None else identity(worker['pid']) != worker)
    return cleanup


def run_owner(source, config, path, output):
    write(path, config)
    config_receipt = receipt(path)
    stage, cap = config['stage'], config['limits']['wall_seconds']
    owner_output = Path(config['owner_output'])
    require(not Path(config['output']).exists() and not owner_output.exists(), 'Fresh reader/owner outputs')
    argv = [str(REPO / '.venv/bin/python'), '-B', '-u', str(source / 'owner.py'), str(path)]
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    transport = dict(stage=stage, config=config_receipt, command=argv, child_wall_seconds=cap,
        owner_wait_seconds=cap + 65, producer_output=config['output'], producer_owner_output=config['owner_output'], no_retry=True)
    with (output / (stage + '_owner_transport.log')).open('xb') as log:
        proc = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True)
        birth = identity(proc.pid)
        require(birth and birth['pgid'] == proc.pid, 'Owned direct readout owner identity')
        transport['owner'] = birth
        try:
            code = proc.wait(timeout=cap + 65)
            require(code == 0 and identity(proc.pid) != birth, 'Reader owner failed; no retry')
            require(receipt(path) == config_receipt, 'Producer CONFIG changed during reader stage')
            started = json.loads((owner_output / 'OWNER_START.json').read_text())
            start_ok(started, stage, source, path, config, birth)
            owner_ok(owner_output / 'OWNER_END.json', stage, started, source, path, config)
            transport.update(complete=True, exit_code=code, owner_waited=True, owner_pid_absent=True,
                owner_start=receipt(owner_output / 'OWNER_START.json'), owner_end=receipt(owner_output / 'OWNER_END.json'))
        except BaseException as error:
            transport.update(complete=False, failure=dict(type=type(error).__name__, message=str(error)))
            transport['cleanup'] = terminate_owned(proc, birth, owner_output)
            raise
        finally:
            write(output / (stage + '_OWNER_TRANSPORT.json'), transport)
    return transport


def main():
    route()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    require(config.get('execution_enabled') is True, 'Root must enable exact whole-family watcher')
    require(Path(config['output_directory']).is_absolute()
        and Path(config['output_directory']).resolve().is_relative_to(REPO), 'Project-local watcher output')
    output = Path(config['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    stage, started = 'admission', time.monotonic()
    result = dict(complete=False, partial_metrics_opened=False, partial_owner_logs_opened=False,
        scientific_gate_used_for_handoff=False, training_or_qualification_launched=False,
        model_retry=False, readout_retry=False, stage_owners=[])
    try:
        source_check(HERE, config['watcher_manifest'])
        binding = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
        for key in ('acquisition_config', 'acquisition_owner_start', 'acquisition_launch', 'reader_source_manifest', 'confidence_clarification'):
            require(config[key] == binding[key], 'Fixed producer/source binding: ' + key)
        require(config['wait_seconds'] == WAIT_SECONDS and config['whole_watch_wall_seconds'] == WHOLE_SECONDS
            and type(config['poll_seconds']) is int and 1 <= config['poll_seconds'] <= 60
            and config['calibrate_limits'] == config['score_limits'] == LIMITS, 'Exact finite watcher/reader envelopes')
        def timeout(signum, frame):
            raise TimeoutError('Whole-family watch ended with bounded cleanup reserve')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(WHOLE_SECONDS - CLEANUP_RESERVE)
        root_decision = receipt(checked(config['root_decision']))
        decision = json.loads(checked(root_decision).read_text())
        require(set(('watch', 'calibrate', 'score')).issubset(decision.get('admitted_stages', []))
            and decision.get('v2_spec_sha256') == SPEC_SHA, 'Explicit root whole-family readout admission')
        for key in ('watcher_manifest', 'acquisition_config', 'acquisition_owner_start', 'acquisition_launch',
                'reader_source_manifest', 'output_directory', 'wait_seconds', 'whole_watch_wall_seconds',
                'poll_seconds', 'calibrate_limits', 'score_limits', 'confidence_clarification'):
            require(decision.get(key) == config[key], 'Root decision must bind exact handoff: ' + key)
        clarification = json.loads(checked(config['confidence_clarification']).read_text())
        require(clarification['v2_spec_sha256'] == SPEC_SHA and clarification['reader_manifest_sha256'] == READER_MANIFEST_SHA
            and clarification['frozen_literal_population'] == 'CAL'
            and clarification['frozen_literal_references'] == list(RAW_REFERENCES)
            and clarification['frozen_literal_per_block_maximum_delta_nat'] == .005
            and clarification['original_CAL_verdict_preserved'] is True
            and clarification['prepared_reader_population'] == 'DEVELOPMENT'
            and clarification['raw_and_member_criteria_unchanged'] is True
            and clarification['continuation_still_raw_and_member_only'] is True
            and clarification['score_handoff_unconditional_after_whole_acquisition_CAL_owner_closure'] is True,
            'Exact prospective CAL/DEVELOPMENT confidence clarification')
        acq_path = checked(config['acquisition_config'])
        acq = json.loads(acq_path.read_text())
        require(acq['execution_enabled'] is True and acq['stage'] == 'acquire'
            and acq['blocks'] == list(BLOCKS) and acq['arms'] == list(ARMS)
            and acq['endpoint_ids'] == list(ENDPOINTS) and acq['source_manifest']['sha256'] == ACQUISITION_MANIFEST_SHA,
            'Exact admitted private-GRU acquisition')
        acquisition_source = Path(acq['source_manifest']['path']).parent
        source_check(acquisition_source, acq['source_manifest'])
        reader_source = Path(config['reader_source_manifest']['path']).parent
        require(config['reader_source_manifest']['sha256'] == READER_MANIFEST_SHA
            and reader_source != acquisition_source, 'Exact separate sealed private-GRU reader')
        source_check(reader_source, config['reader_source_manifest'])
        owner_start = json.loads(checked(config['acquisition_owner_start']).read_text())
        require(Path(config['acquisition_owner_start']['path']) == Path(acq['owner_output']) / 'OWNER_START.json',
            'Actual acquisition owner-output start receipt')
        start_ok(owner_start, 'acquire', acquisition_source, acq_path, acq, ACQUISITION_OWNER, ACQUISITION_CHILD)
        launch = json.loads(checked(config['acquisition_launch']).read_text())
        require(launch['owner_spawn_pid'] == ACQUISITION_OWNER['pid'] and launch['owner_start'] == owner_start,
            'Saved actual launch and live-owner start agree')
        for key, expected in (('owner_observed', ACQUISITION_OWNER), ('scientific_child_observed', ACQUISITION_CHILD)):
            require(all(launch[key][k] == v for k, v in expected.items()), 'Actual launch birth identity: ' + key)
        acquisition_owner = Path(acq['owner_output']) / 'OWNER_END.json'
        panel_path = Path(acq['output']) / 'COMPLETE_PANEL.json'
        result.update(config=receipt(config_path), root_decision=root_decision, watcher_manifest=config['watcher_manifest'],
            reader_source_manifest=config['reader_source_manifest'], acquisition_source_manifest=acq['source_manifest'],
            acquisition_config=config['acquisition_config'], acquisition_owner_start=config['acquisition_owner_start'],
            acquisition_launch=config['acquisition_launch'], confidence_clarification=config['confidence_clarification'],
            finite_wall_seconds=WHOLE_SECONDS, cleanup_reserve_seconds=CLEANUP_RESERVE)
        stage = 'waiting_for_whole_acquisition'
        write(output / 'WATCH_START.json', dict(result, pid=os.getpid(), stage=stage))
        deadline = time.monotonic() + WAIT_SECONDS
        while not acquisition_owner.exists():
            require(time.monotonic() < deadline, 'Finite acquisition closure wait exceeded')
            require(identity(ACQUISITION_OWNER['pid']) == ACQUISITION_OWNER,
                'Exact acquisition owner ended without closure receipt; no partial reads')
            time.sleep(min(config['poll_seconds'], max(.01, deadline - time.monotonic())))
        checked(config['acquisition_config'])
        checked(config['acquisition_owner_start'])
        owner_ok(acquisition_owner, 'acquire', owner_start, acquisition_source, acq_path, acq)
        base = dict(blocks=list(BLOCKS), endpoint_ids=list(ENDPOINTS), finite_external_owner=True,
            source_manifest=config['reader_source_manifest'], expected_runtime_versions=acq['expected_runtime_versions'],
            custody=acq['custody'], inherited_sources=acq['inherited_sources'], panel=receipt(panel_path),
            acquisition_owner_end=receipt(acquisition_owner))
        reader = reader_metadata_helpers(reader_source)
        panel, index, custodian, prior = reader.close_panel(base)
        require(panel['custody']['stage'] == 'acquire'
            and panel['custody']['root_decision'] == acq['root_decision']
            and panel['custody']['source_manifest'] == acq['source_manifest']
            and panel['stage1_closed_before_stage2'] is True, 'Complete panel bound to actual acquisition CONFIG')
        write(output / 'WHOLE_FAMILY_CLOSURE.json', dict(complete=True, cohort_fit_banks=33,
            cohort_optimizer_trajectories=51, new_fit_banks=30, new_optimizer_trajectories=48,
            external_optimizer_trajectories=12, exports=39, panel=base['panel'], owner_end=base['acquisition_owner_end'],
            acquisition_config=config['acquisition_config'], acquisition_owner_start=config['acquisition_owner_start'],
            reader_source_manifest=base['source_manifest'], no_partial_metric_access=True))
        cal_transport = cal_action = None
        for stage in ('calibrate', 'score'):
            source_check(reader_source, config['reader_source_manifest'])
            require(time.monotonic() - started + LIMITS['wall_seconds'] + 90 < WHOLE_SECONDS - CLEANUP_RESERVE,
                'Insufficient remaining inclusive watch budget; no stage launch')
            action = fill_reader_template(reader_source, stage, base, output, root_decision)
            if stage == 'score':
                checked(cal_transport['config'])
                cal_start = json.loads(checked(cal_transport['owner_start']).read_text())
                owner_ok(checked(cal_transport['owner_end']), 'calibrate', cal_start, reader_source,
                    Path(cal_transport['config']['path']), cal_action)
                temperatures_path = Path(cal_action['output']) / 'TEMPERATURES.json'
                complete_temperatures = endpoint_closure(temperatures_path, cal_action, reader, custodian, prior)
                action.update(temperatures=receipt(temperatures_path), calibration_owner_end=cal_transport['owner_end'])
                write(output / 'ALL_CAL_CLOSURE.json', dict(complete=True, endpoints=39, new_endpoints=33,
                    inherited_endpoints=6, new_scalar_updates=16500, inherited_scalar_updates=3000,
                    total_bound_scalar_updates=19500, CAL_config=cal_transport['config'],
                    CAL_owner_start=cal_transport['owner_start'], CAL_owner_end=cal_transport['owner_end'],
                    CAL_owner_waited_and_reaped=True, temperatures=action['temperatures'], no_DEV_truth_access_by_watcher=True))
                guard_path = output / 'CAL_CONFIDENCE_GUARD.json'
                write(guard_path, dict(cal_confidence_guard(complete_temperatures, config['confidence_clarification']),
                    temperatures=action['temperatures'], CAL_producer_config=cal_transport['config'],
                    CAL_owner_end=cal_transport['owner_end']))
                result['original_CAL_confidence_guard'] = receipt(guard_path)
            write(output / 'WATCH_PROGRESS.json', dict(stage=stage, whole_family_closed=True, partial_metrics_opened=False))
            transport = run_owner(reader_source, action, output / (stage.upper() + '_CONFIG.json'), output)
            result['stage_owners'].append(transport['owner_end'])
            if stage == 'calibrate':
                cal_transport, cal_action = transport, action
        stage = 'complete_report_receipt'
        report = Path(action['output']) / 'DEVELOPMENT_REPORT.json'
        closed = json.loads(report.read_text())
        require(closed['complete'] is True and len(closed['records']) == 39 and len(closed['comparisons']) == 156
            and closed['new_calibration_scalar_updates'] == 16500 and closed['inherited_calibration_scalar_updates'] == 3000
            and closed['total_bound_calibration_scalar_updates'] == 19500
            and closed['panel'] == base['panel'] and closed['temperatures'] == action['temperatures']
            and closed['release']['one_whole_release'] is True and closed['release']['blocks'] == list(BLOCKS)
            and closed['release']['custody']['stage'] == 'score'
            and closed['release']['custody']['root_decision'] == root_decision
            and closed['release']['custody']['source_manifest'] == config['reader_source_manifest'],
            'Complete exact whole DEV report from owned score producer')
        result.update(complete=True, report=receipt(report), all_CAL_owners_closed_before_DEV=True,
            scientific_gate_results_not_used_for_success=True,
            confidence_population_labels=dict(original_frozen_CAL= result['original_CAL_confidence_guard'],
                separately_added_DEVELOPMENT=dict(report=receipt(report), field='gates.calibrated_confidence_protection_pass',
                    population='DEVELOPMENT', metric='calibrated_NLL', does_not_replace_CAL=True)))
    except BaseException as error:
        result['failure'] = dict(type=type(error).__name__, message=str(error))
    finally:
        signal.alarm(0)
        result.update(last_stage=stage, seconds=time.monotonic() - started)
        write(output / 'WATCH_END.json', result)
    print(json.dumps(dict(complete=result['complete'], last_stage=stage, watch_end=str(output / 'WATCH_END.json'),
        report=result.get('report'), partial_metrics_opened=False)), flush=True)
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
