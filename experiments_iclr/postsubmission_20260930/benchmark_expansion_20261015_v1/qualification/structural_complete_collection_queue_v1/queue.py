"""Wait for six fits and seven checks; collect six banks and analyze 39 once."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
E = ROOT/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def require(value, message):
    if not value: raise ValueError(message)


def safe(path):
    path = Path(path); path = (path if path.is_absolute() else E/path).resolve()
    require(path.is_relative_to(ROOT), 'Path escapes allocation repository')
    return path


def read(path):
    path = safe(path); raw = path.read_bytes()
    return json.loads(raw), {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify(pin):
    path = safe(pin['path']); raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256']
            and ('bytes' not in pin or len(raw) == pin['bytes']), 'Pinned bytes changed: '+str(path))
    return path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def engineering_gate(queue, names):
    require(queue.get('status') == 'passed' and queue.get('TEST_accessed') is False
            and queue.get('automatic_retries') == 0, 'All seven PubMed invocations must pass')
    rows = queue.get('invocations', [])
    require([row.get('name') for row in rows] == names and len(rows) == 7,
            'Exact ordered seven PubMed invocations required')
    require(all(row.get('status') == 'passed' and row.get('exit_code') == 0
                and not row.get('timed_out') and not row.get('resource_limit')
                and row.get('result', {}).get('status') == 'passed'
                and row['result'].get('TEST_accessed') is False for row in rows),
            'Failed/incomplete PubMed prerequisite; no collection')


def main():
    require(socket.gethostname() == 'anogena-2-0' and subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() == [GPU],
        'Dedicated allocation required before project reads')
    parser = argparse.ArgumentParser(); parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True); args = parser.parse_args()
    request, request_pin = read(args.admission)
    require(request_pin['sha256'] == args.admission_sha256, 'Root admission bytes differ')
    source, source_pin = read(HERE/'MANIFEST.json'); inputs, inputs_pin = read(HERE/'INPUTS.json')
    require(request.get('schema') == 'root-structural-complete-collection-queue-admission-v1'
            and request.get('approved') is True and request.get('harness_manifest_sha256') == source_pin['sha256']
            and request.get('INPUTS_sha256') == inputs_pin['sha256']
            and request.get('delegates_terminal_hash_binding_after_all6_and_all7_success') is True
            and request.get('resources_and_counts_exactly_INPUTS') is True
            and request.get('automatic_retries') == request.get('TEST_calls') == 0, 'Exact conditional root admission required')
    require(sys.executable == request['python_executable'] == str(ROOT/'.venv/bin/python')
            and Path(sys.prefix).resolve() == ROOT/'.venv', 'Original root-qualified repository interpreter required')
    for name in ('published_source_review', 'qualified_python_review', 'peer_resource_review'): verify(request[name])
    for pin in source['files']: verify({**pin, 'path': str(HERE/pin['path'])})
    sources = {pin['path']: verify(pin) for pin in inputs['source_pins']}
    pilot = E/'pilot_20261015_structural_private_risk_v1'
    pilot_manifest, pilot_pin = read(pilot/'MANIFEST.json')
    for pin in pilot_manifest['files']: verify({**pin, 'path': str(pilot/pin['path'])})
    plan, plan_pin = read(pilot/'COLLECTION_PLAN.json')
    for pin in plan['source_pins']: verify(pin)
    require(plan['new_banks'] == 6 and plan['reused_banks'] == 33 and plan['result_banks'] == 39
            and plan['unique_historical_plus_new_training_fits'] == 45
            and plan['timeout_seconds_per_collection'] == inputs['collection_watchdog_seconds'] == 600,
            'Frozen complete collection scope differs')
    require('torch' not in sys.modules, 'Owner must remain free of model/tensor imports')
    sys.path.insert(0, str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
    from supervise import preflight, process, gpu_state
    launch = load(E/plan['launch_helpers'], '_structural_queue_launch_helpers')
    helper = load(E/plan['analysis_helpers'], '_structural_queue_analysis_helpers')
    selected = load(pilot/'selected_VALID.py', '_structural_queue_selected_VALID')
    study_launch, _ = read(sources['STRUCTURAL_LAUNCH_22.json'])
    engineering_launch, _ = read(sources['PUBMED_GPU_QUEUE_LAUNCH_24.json'])
    require(engineering_launch['source_sha256'] == next(pin['sha256'] for pin in inputs['source_pins']
            if pin['path'] == 'pubmed_gpu_engineering_queue_24.py'), 'PubMed launched source differs')
    prior = safe(inputs['prior_study']); engineering = safe(inputs['prior_engineering'])
    q = safe(inputs['queue_output']); collected = safe(inputs['collection_output']); analyzed = safe(inputs['analysis_output'])
    require(not collected.exists() and not analyzed.exists(), 'Existing collection/analysis output forbids duplicate work')
    q.mkdir(parents=True, exist_ok=False)  # Durable fixed one-attempt claim, also while waiting.
    started = time.monotonic(); child = None; lock = None; collection_started = False; collection_done = False
    children = [{'seed': job['seed'], 'id': job['id'], 'status': 'not_started'} for job in plan['jobs']]
    report = {'schema': 'structural-complete-collection-queue-v1', 'status': 'waiting', 'owner': process(os.getpid()),
              'source': source_pin, 'admission': request_pin, 'scientific_fits': 0, 'TRAIN_updates': 0,
              'TEST_accessed': False, 'automatic_retries': 0, 'maximum_concurrent_collectors': 1, 'invocations': []}
    def save():
        report['wall_seconds'] = time.monotonic()-started
        launch.write(q/'QUEUE.json', report, replace=True)
    def stop(*_): raise InterruptedError('Collection queue interrupted; stop without retry')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def alive(pin, label):
        owner = process(pin['owner_pid'])
        require(owner is not None and owner['start_ticks'] == str(pin['owner_start_ticks']), label+' owner disappeared without successful terminal files')
    def resources():
        peer = inputs['older_peer']; old = process(peer['pid']); peers = []
        if old is not None:
            require(old['start_ticks'] == peer['start_ticks'], 'Older peer PID reused')
            peers = [peer]
        host = preflight({'device': 'cpu', 'peers': peers}, {
            'minimum_host_headroom_after_peer_growth_bytes': inputs['host_RSS_reservation_bytes'],
            'host_RSS_reservation_bytes': inputs['host_RSS_reservation_bytes'],
            'disk_reservation_bytes': inputs['disk_reservation_bytes']})
        gpu = gpu_state(); require(gpu['uuid'] == GPU and len(gpu['processes']) <= 1, 'Dedicated sole peer/one collector slot required')
        growth = 0
        for row in gpu['processes']:
            require(row['pid'] == peer['pid'] and row['start_ticks'] == peer['start_ticks']
                    and row['gpu_memory_bytes'] <= peer['peak_GPU_bytes'], 'Unadmitted GPU peer or peer envelope exceeded')
            growth += peer['peak_GPU_bytes']-row['gpu_memory_bytes']
        require(gpu['free_bytes']-growth >= inputs['GPU_reservation_bytes'], 'GPU headroom after peer growth insufficient')
        return {'host': host, 'GPU': gpu, 'GPU_peer_growth_bytes': growth}
    def invoke(name, command, gpu_enabled, watchdog):
        nonlocal child
        row = {'name': name, 'command': command, 'status': 'starting', 'watchdog_seconds': watchdog,
               'resources': resources(), 'peak_RSS_sampled_bytes': 0, 'peak_GPU_sampled_bytes': 0}
        report['invocations'].append(row); save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU if gpu_enabled else '', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        log_path = q/(name+'.log')
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        row.update(status='running', child=process(child.pid)); begin = time.monotonic(); save()
        try:
            while child.poll() is None:
                if time.monotonic()-begin >= watchdog:
                    row['timed_out'] = True
                    raise TimeoutError('Invocation watchdog expired: '+name)
                current = process(child.pid); own = process(os.getpid())
                rss = (current['rss_bytes'] if current else 0)+(own['rss_bytes'] if own else 0)
                row['peak_RSS_sampled_bytes'] = max(row['peak_RSS_sampled_bytes'], rss)
                require(rss <= inputs['host_RSS_reservation_bytes'], 'Own host RSS reservation exceeded')
                peer = inputs['older_peer']; old = process(peer['pid'])
                if old is not None:
                    require(old['start_ticks'] == peer['start_ticks'] and old['rss_bytes'] <= peer['peak_host_RSS_bytes'],
                            'Older peer identity or host envelope changed')
                try: gpu = gpu_state()
                except ValueError as error:
                    if str(error) == 'GPU process could not be authenticated' and child.poll() is not None: break
                    raise
                require(gpu['uuid'] == GPU and len(gpu['processes']) <= (2 if gpu_enabled else 1), 'Third model worker appeared')
                for worker in gpu['processes']:
                    if worker['pid'] == child.pid:
                        require(gpu_enabled and row['child'] is not None and worker['start_ticks'] == row['child']['start_ticks'],
                                'CPU phase initialized CUDA or own child identity differs')
                        row['peak_GPU_sampled_bytes'] = max(row['peak_GPU_sampled_bytes'], worker['gpu_memory_bytes'])
                        require(worker['gpu_memory_bytes'] <= inputs['GPU_reservation_bytes'], 'Own GPU reservation exceeded')
                    else:
                        peer = inputs['older_peer']
                        require(worker['pid'] == peer['pid'] and worker['start_ticks'] == peer['start_ticks']
                                and worker['gpu_memory_bytes'] <= peer['peak_GPU_bytes'], 'Unadmitted GPU process or peer envelope exceeded')
                disk = sum(p.stat().st_size for folder in (q, collected, analyzed) if folder.exists()
                           for p in folder.rglob('*') if p.is_file())
                require(disk <= inputs['disk_reservation_bytes'], 'Own output disk reservation exceeded')
                save()
                try: child.wait(timeout=2)
                except subprocess.TimeoutExpired: pass
            row.update(exit_code=child.returncode, status='passed' if child.returncode == 0 else 'failed')
        except BaseException as error:
            row.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)})
            raise
        finally:
            launch.terminate(child); row.update(exit_code=child.returncode, wall_seconds=time.monotonic()-begin); save(); child = None
        require(row['status'] == 'passed', 'Invocation failed; no retry or later work: '+name)
        return log_path, row
    save()
    try:
        deadline = time.monotonic()+inputs['prerequisite_wait_seconds']
        while True:
            require(time.monotonic() < deadline, 'Finite prerequisite wait expired')
            study_ready = (prior/'END.json').exists()
            if study_ready:
                end, end_pin = read(prior/'END.json'); outcomes, outcomes_pin = read(prior/'OUTCOMES.json')
                selected.terminal_gate(end, outcomes, plan['jobs'])
            else: alive(study_launch, 'Structural')
            engineering_ready = False
            if (engineering/'QUEUE.json').exists():
                eng, eng_pin = read(engineering/'QUEUE.json')
                require(eng.get('status') != 'failed', 'PubMed prerequisite failed; no collection')
                require(eng.get('owner_pid') == engineering_launch['owner_pid']
                        and str(eng.get('owner_start_ticks')) == engineering_launch['owner_start_ticks']
                        and eng.get('publication_commit') == engineering_launch['publication_commit'], 'PubMed terminal owner/source differs')
                if eng.get('status') == 'passed':
                    engineering_gate(eng, inputs['engineering_invocations']); engineering_ready = True
            if not engineering_ready: alive(engineering_launch, 'PubMed engineering')
            report['prerequisite_progress'] = {'all6_complete': study_ready, 'all7_passed': engineering_ready}; save()
            if study_ready and engineering_ready: break
            time.sleep(inputs['poll_seconds'])
        engineering_pins = []
        for row in eng['invocations']:
            value, pin = read(engineering/row['name']/'RESULT.json')
            require(value == row['result'], 'PubMed terminal result changed after queue completion')
            engineering_pins.append(pin)
        prior_analysis, prior_pin = read(verify(plan['completed_33_bank_analysis']))
        require(prior_analysis['schema'] == 'residual-only-selected-VALID-33-bank-addition-v1'
                and prior_analysis['status'] in ('complete', 'complete_with_warnings')
                and prior_analysis['TEST_accessed'] is False and len(prior_analysis['rows']) == 33
                and len(prior_analysis['training_fit_outcomes']) == 39, 'Exact completed 33-bank analysis required')
        terminal_pins = [end_pin, outcomes_pin, eng_pin, prior_pin, *engineering_pins]
        report['prerequisites'] = {'structural_END': end_pin, 'structural_OUTCOMES': outcomes_pin,
                                   'PubMed_QUEUE': eng_pin, 'PubMed_results': engineering_pins,
                                   'reused_33_bank_analysis': prior_pin}; save()
        lock_path = safe(inputs['slot_lock']); lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a'); deadline = time.monotonic()+inputs['slot_wait_seconds']
        while True:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'Finite shared slot wait expired'); time.sleep(2)
        for pin in terminal_pins: verify(pin)
        require(not collected.exists() and not analyzed.exists(), 'Duplicate collector/analysis output appeared')
        admission, _ = read(pilot/'COLLECTION_ADMISSION_TEMPLATE.json')
        admission.update(approved=True, resource_admission_approved=True, pilot_manifest_sha256=pilot_pin['sha256'],
            allowed_other_GPU_pids=[inputs['older_peer']['pid']], slot_lock_path=str(lock_path),
            collection_output=str(collected), analysis_output=str(analyzed), terminal_END_sha256=end_pin['sha256'],
            terminal_OUTCOMES_sha256=outcomes_pin['sha256'], queue_root_admission=request_pin,
            prerequisite_PubMed_QUEUE=eng_pin, review_note='Exact terminal hashes bound under explicit root queue admission after all6/all7 gates')
        launch.write(q/'COLLECTION_ADMISSION.json', admission); _, admission_pin = read(q/'COLLECTION_ADMISSION.json')
        base = [sys.executable, '-B', '-u', str(pilot/'selected_VALID.py'), '--project-root', str(ROOT),
                '--admission', admission_pin['path'], '--admission-sha256', admission_pin['sha256']]
        path, _ = invoke('commands', base+['--mode', 'commands'], False, inputs['commands_watchdog_seconds'])
        commands = json.loads(path.read_text())
        expected = [[sys.executable, '-B', '-u', str(E/plan['collector']), '--project-root', str(ROOT),
                     '--fit', str(E/job['fit']), '--output', str(collected/job['collection'])] for job in plan['jobs']]
        require(commands['collectors'] == expected and commands['terminal_END'] == end_pin
                and commands['terminal_OUTCOMES'] == outcomes_pin, 'Frozen six collector commands/terminal pins differ')
        fit_values, fit_pins = [], []
        for job, outcome in zip(plan['jobs'], outcomes):
            fit, pin = read(E/job['fit']/'RESULT.json')
            require(pin['sha256'] == outcome['RESULT']['sha256'], 'Terminal fit receipt differs')
            fit_values.append(fit); fit_pins.append(pin)
        collected.mkdir(parents=True, exist_ok=False); collection_started = True
        launch.write(collected/'START.json', {'PLAN': plan_pin, 'admission': admission_pin, 'terminal_END': end_pin,
            'terminal_OUTCOMES': outcomes_pin, 'fit_RESULT': fit_pins, 'maximum_concurrent_collectors': 1, 'automatic_retries': 0})
        collect_start = time.monotonic()
        for index, command in enumerate(commands['collectors']):
            for pin in terminal_pins: verify(pin)
            children[index]['status'] = 'started'; launch.write(collected/'OUTCOMES.json', children, True)
            _, invocation = invoke('collect_'+str(index), command, True, inputs['collection_watchdog_seconds'])
            value, pin = read(collected/plan['jobs'][index]['collection']/'RESULT.json')
            children[index].update(helper.collection_outcome(plan['jobs'][index]['id'], value, pin, 'new'))
            children[index].update(exit_code=invocation['exit_code'], supervisor_wall_seconds=invocation['wall_seconds'])
            selected.collection_gate(value, fit_values[index], fit_pins[index])
            require(value['analysis_source']['sha256'] == next(p['sha256'] for p in plan['source_pins'] if p['path'] == plan['collector']),
                    'Collector source differs')
            launch.write(collected/'OUTCOMES.json', children, True)
        launch.write(collected/'END.json', {'status': 'complete', 'children': children, 'PLAN': plan_pin,
            'admission': admission_pin, 'terminal_END': end_pin, 'terminal_OUTCOMES': outcomes_pin,
            'wall_seconds': time.monotonic()-collect_start, 'failure': None, 'new_training_updates': 0,
            'TEST_accessed': False, 'automatic_retries': 0}); collection_done = True
        for pin in terminal_pins: verify(pin)
        invoke('analyze', base+['--mode', 'analyze'], False, inputs['analysis_watchdog_seconds'])
        result, result_pin = read(analyzed/'RESULT.json')
        require(result['schema'] == 'structural-private-risk-selected-VALID-39-bank-addition-v1'
                and result['status'] in ('complete', 'complete_with_warnings') and result['TEST_accessed'] is False
                and len(result['rows']) == 39 and len(result['training_fit_outcomes']) == 45
                and result['new_models'] == result['new_native_forwards'] == result['new_optimizer_steps'] == 0,
                'Complete 39-bank/45-fit analysis required; no partial summary')
        report.update(status='complete', complete_analysis=result_pin, collection_END=read(collected/'END.json')[1],
                      new_collections=6, reused_banks=33, complete_banks=39, paid_fits=45); save()
    except BaseException as error:
        if child is not None: launch.terminate(child)
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(limit=12)},
                      unreceipted_work='unknown if a child lacks terminal RESULT; never assume zero')
        if collection_started and not collection_done:
            for index, outcome in enumerate(children):
                if outcome['status'] != 'started': continue
                path = collected/plan['jobs'][index]['collection']/'RESULT.json'
                if path.exists():
                    try:
                        value, pin = read(path)
                        outcome.update(helper.collection_outcome(plan['jobs'][index]['id'], value, pin, 'new'))
                        outcome['observed_RESULT_status'] = value.get('status')
                    except Exception as receipt_error: outcome['RESULT_read_error'] = str(receipt_error)
                outcome.update(status='failed', failure=report['failure'])
                if report['invocations'] and report['invocations'][-1]['name'] == 'collect_'+str(index):
                    invocation = report['invocations'][-1]
                    outcome.update(exit_code=invocation.get('exit_code'), timed_out=invocation.get('timed_out', False),
                                   supervisor_wall_seconds=invocation.get('wall_seconds'))
            launch.write(collected/'OUTCOMES.json', children, True)
            launch.write(collected/'END.json', {'status': 'stopped_no_retry', 'children': children,
                'failure': report['failure'], 'new_training_updates': 0, 'TEST_accessed': False, 'automatic_retries': 0})
        save()
    finally:
        if lock is not None: lock.close()
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__': raise SystemExit(main())
