"""One durable tmux queue owner per GPU, all-fit smoke then global barrier then full fits."""
import argparse
import subprocess
import traceback
from common import *
from capacity import probe, capacity_decision


class Worker:
    def __init__(self, args):
        self.args = args
        self.config, config_pin = environment(args.worker)
        require(os.environ.get('TMUX'), 'Queue owner must run inside tmux')
        require(pin(args.plan)['sha256'] == args.plan_sha256, 'Plan hash differs')
        self.plan = read(args.plan)
        require(self.plan['worker'] == args.worker and self.plan['host_config'] == config_pin, 'Wrong worker plan')
        require(source_manifest() == self.plan['source_manifest'], 'Execution package changed')
        for field in ('inventory', 'authorization', 'resource_source', 'resource_catalogue', 'reporting_entrypoint', 'smoke_source_manifest'):
            verify(self.plan[field], self.plan['project_root'])
        self.inventory = read(self.plan['inventory']['path'])
        self.smokes = {row['id']: row for row in self.inventory['fits'] if row['worker'] == args.worker}
        self.resources = module(verify(self.plan['resource_source']), '_allfit_resources_worker')
        self.tasks = self.plan['tasks']; self.index = {row['id']: row for row in self.tasks}
        self.root = Path(self.plan['state_root']); self.root.mkdir(parents=True, exist_ok=True)
        self.supervisor_lease = lock(self.root / 'SUPERVISOR.lock')
        require(self.supervisor_lease is not None, 'A supervisor already owns this queue')
        self.blocked_profiles = {}
        self.ledger = Path(self.plan['host_ledger']); self.ledger.mkdir(parents=True, exist_ok=True)
        self.leases = []; self.processes = {}; self.states = {}; self.phase = 'initializing'
        self.retry = set(); self.last_report = 0
        self.started = time.time()
        write(self.root / 'OWNER.json', {'identity': identity(), 'plan': pin(args.plan), 'source_manifest': self.plan['source_manifest'],
            'inventory': self.plan['inventory'], 'release': self.inventory['release'], 'tmux': os.environ['TMUX'], 'started': self.started})

    def report(self, **extra):
        if time.time() - self.last_report < 10 and not extra:
            return
        counts = {}
        for value in self.states.values():
            counts[value['status']] = counts.get(value['status'], 0) + 1
        value = {'schema': 'all-fit-worker-status-v1', 'worker': self.args.worker, 'phase': self.phase,
            'updated': time.time(), 'owner': identity(), 'plan': pin(self.args.plan), 'source_manifest': self.plan['source_manifest'],
            'inventory': self.plan['inventory'], 'release': self.inventory['release'], 'counts': counts,
            'active_fit_ids': sorted(self.processes), 'expected_fit_count': len(self.smokes), 'TEST_accessed': False, **extra}
        write(self.root / 'STATUS.json', value)
        print(json.dumps({key: value[key] for key in ('worker', 'phase', 'counts', 'active_fit_ids', 'updated')}, sort_keys=True), flush=True)
        self.last_report = time.time()

    def acquire(self):
        self.phase = 'waiting_for_existing_owners'
        while True:
            handles = []
            for name in ('slot_lock', 'physical_lock', 'worker_lock'):
                handle = lock(self.plan[name])
                if handle is None:
                    break
                handles.append(handle)
            if len(handles) == 3:
                self.leases = handles; return
            for handle in handles:
                handle.close()
            self.report()
            if (self.root / 'STOP').exists():
                raise InterruptedError('Graceful stop requested while waiting for lease')
            time.sleep(self.args.poll_seconds)

    def prerequisites(self):
        path = self.plan['prerequisites']
        while path and (not Path(path).exists() or read(path).get('status') != 'PASS'):
            self.phase = 'waiting_for_allocation_prerequisites'; self.report()
            if (self.root / 'STOP').exists():
                raise InterruptedError('Graceful stop requested before prerequisite PASS')
            time.sleep(self.args.poll_seconds)

    def folder(self, task, phase):
        return self.root / ('smokes' if phase == 'smoke' else phase) / token(task['id'])

    def save_state(self, task, phase, state):
        self.states[task['id']] = state
        write(self.folder(task, phase) / 'STATE.json', state)

    def validate_result(self, task, phase, value, output):
        require(value.get('TEST_accessed') is False, 'Result TEST access changed')
        if phase == 'smoke':
            require(value.get('schema') == 'all-fit-two-update-smoke-result-v1' and value.get('id') == task['id'], 'Wrong smoke identity')
            for key in ('worker', 'runtime_job_hash', 'canonical_runtime_job_hash', 'science_contract_hash'):
                require(value.get(key) == self.smokes[task['id']][key], 'Smoke identity changed: ' + key)
            require(value.get('inventory') == self.plan['inventory'], 'Smoke inventory differs')
            require(value.get('descriptor') == task['descriptor'], 'Smoke descriptor differs')
            require(value.get('status') == 'PASS', 'Smoke failed')
            require(value.get('TRAIN_updates') == 2 and value.get('TRAIN_finite') is True and value.get('VALID_finite') is True,
                    'Smoke requires two finite TRAIN updates and finite VALID')
            require(value.get('optimizer_calls') == {'attempts': [2], 'completed': [2]}, 'Smoke optimizer calls differ')
            require(value.get('checkpoint_reload', {}).get('status') == 'PASS', 'Smoke checkpoint reload failed')
            return 'PASS', None
        if task['kind'] == 'fit':
            require(value.get('schema') == 'native-full-epoch-fit-result-v1' and value.get('job_id') == task['id'] and
                    value.get('job_hash') == task['runtime_job_hash'], 'Full fit identity differs')
        else:
            require(value.get('schema') == 'independent-own-selected-VALID-bank-result-v1', 'Wrong I4 result')
            require(sorted(row['job_id'] for row in value['members']) == sorted(task['dependencies']), 'I4 members differ')
            require(all(row['job_hash'] == self.index[row['job_id']]['runtime_job_hash'] for row in value['members']), 'I4 job hashes differ')
        status = value.get('status')
        require(status in ('complete', 'no_selected_state'), 'Full task failed')
        selected = value.get('selected_VALID_export') if task['kind'] == 'fit' and status == 'complete' else None
        if task['kind'] == 'fit' and status == 'complete':
            require(selected is not None and Path(selected['path']).resolve() == output / 'selected_VALID.pt', 'Selected export escapes own fit')
            verify(selected, self.plan['project_root'])
        return status, selected

    def collect(self, task, phase, directory):
        request = read(directory / 'INVOCATION.json')
        output = Path(request['output'])
        terminal_path = directory / 'TERMINAL.json'
        if not terminal_path.exists():
            # Both native and guardian inherited the locks; the caller only reaches
            # here after the attempt lease is free, including after supervisor loss.
            write(terminal_path, {'schema': 'all-fit-child-terminal-v1', 'exit_code': None,
                'unknown_partial_work': True, 'prior_partial_work_preserved': True,
                'failure': 'Guardian terminal missing after all attempt owners exited', 'finished': time.time()})
        terminal = read(terminal_path)
        native = output / 'RESULT.json'
        raw = None; status = 'failed'; selected = None; failure = None
        try:
            require(native.exists(), 'Missing native RESULT.json')
            raw = read(native)
            status, selected = self.validate_result(task, phase, raw, output)
            # Valid complete native results are recoverable after supervisor loss;
            # absence of exit status remains explicitly charged as uncertainty.
            require(terminal['exit_code'] == 0 if phase == 'smoke' else terminal['exit_code'] in (0, None),
                    'Smoke requires observed exit zero; full result permits documented supervisor-loss uncertainty')
            require(not terminal.get('forced_kill') and not terminal.get('signals_forwarded'), 'Interrupted native attempt')
        except Exception as exc:
            status = 'failed'; selected = None; failure = type(exc).__name__ + ': ' + str(exc)
        native_receipt = None
        if raw is not None:
            write(directory / 'NATIVE_RESULT.json', raw)
            native_receipt = pin(directory / 'NATIVE_RESULT.json')
        result = {'schema': 'all-fit-attempt-result-v1', 'id': task['id'], 'worker': self.args.worker, 'phase': phase,
            'attempt': int(directory.name), 'status': status, 'output': str(output), 'failure': failure,
            'native_result': native_receipt, 'terminal': pin(terminal_path), 'selected_VALID_export': selected,
            'exit_code': terminal['exit_code'], 'unknown_partial_work': terminal.get('unknown_partial_work', False),
            'prior_partial_work_preserved': True, 'finished': time.time()}
        write(directory / 'ATTEMPT_RESULT.json', result)
        result['attempt_result'] = pin(directory / 'ATTEMPT_RESULT.json')
        self.save_state(task, phase, result)

    def load_states(self, tasks, phase):
        self.states = {}
        for task in tasks:
            folder = self.folder(task, phase)
            attempts = sorted((folder / 'attempts').glob('[0-9][0-9][0-9][0-9]'))
            if attempts:
                latest = attempts[-1]
                require(not locked(latest / 'LEASE.lock'), 'Unexpected live attempt after physical lock acquisition')
                if (latest / 'ATTEMPT_RESULT.json').exists():
                    state = read(latest / 'ATTEMPT_RESULT.json')
                    state['attempt_result'] = pin(latest / 'ATTEMPT_RESULT.json')
                    if state['native_result']:
                        raw = read(verify(state['native_result']))
                        if state['status'] in ('PASS', 'complete', 'no_selected_state'):
                            self.validate_result(task, phase, raw, Path(state['output']))
                    self.save_state(task, phase, state)
                elif (latest / 'INVOCATION.json').exists():
                    self.collect(task, phase, latest)
                else:
                    require(False, 'Incomplete attempt custody; inspect ' + str(latest))
            elif (folder / 'STATE.json').exists():
                state = read(folder / 'STATE.json')
                require(state['status'].startswith('blocked_dependency_'), 'State exists without attempt custody')
                self.states[task['id']] = state
            else:
                self.states[task['id']] = {'status': 'pending', 'attempt': 0}
        self.retry = {task_id for task_id, state in self.states.items() if state['status'] == 'failed'} if self.args.retry_failed else set()

    def active_ledger(self):
        result = []
        for path in (self.ledger / 'reservations').glob('*.json'):
            record = read(path)
            if locked(Path(record['directory']) / 'LEASE.lock'):
                result.append(record)
            else:
                # Retain closed reservations outside the hot admission directory;
                # otherwise 13k attempts make every later start scan all history.
                retired = self.ledger / 'retired_reservations'
                retired.mkdir(parents=True, exist_ok=True)
                path.rename(retired / path.name)
        # Preserve the already-running allocation GRU, accounting for its
        # remaining growth. This identity is observation-only, never signal authority.
        if self.args.worker == 'allocation/gpu0':
            try:
                peer = identity(640428)
                if peer['start_ticks'] == 6055039165:
                    result.append({'directory': 'external-allocation-gru-640428', 'external_identity': peer,
                        'gpu_uuid': self.config['physical_gpu_uuid'], 'reservation': {
                            'vram_bytes': 32 * GiB, 'ram_bytes': 16 * GiB, 'disk_bytes': 0,
                            'cpu_slots': 2, 'thread_count': 2, 'profile': 'preserved_external_gru'}})
            except (FileNotFoundError, ProcessLookupError):
                pass
        return result

    def invocation(self, task, phase, directory):
        previous = self.states[task['id']]
        output = directory / 'output'
        python = str(Path(self.plan['project_root']) / '.venv/bin/python')
        if phase == 'smoke':
            argv = [python, '-B', '-u', str(SMOKE / 'run_one.py'), '--project-root', self.plan['project_root'],
                '--inventory', self.plan['inventory']['path'], '--inventory-sha256', self.plan['inventory']['sha256'],
                '--fit-id', task['id'], '--output', str(output), '--updates', '2', '--wall-seconds', str(self.args.smoke_wall_seconds)]
        else:
            verify(self.plan['reporting_entrypoint'], self.plan['project_root'])
            verify(task['runtime_entrypoint'], self.plan['project_root'])
            if task['kind'] == 'fit':
                verify(task['descriptor'], self.plan['project_root'])
            resume = None
            if task['kind'] == 'fit' and previous.get('output'):
                previous_output = Path(previous['output'])
                require(previous_output.is_relative_to(self.folder(task, 'full')), 'Resume output escapes own full task')
                if (previous_output / 'recovery.pt').exists():
                    output = previous_output; resume = output / 'recovery.pt'
            verify(task['preset'], self.plan['project_root'])
            argv = [python, '-B', '-u', self.plan['reporting_entrypoint']['path'], '--project-root', self.plan['project_root'],
                '--preset-file', task['preset']['path'], '--preset-sha256', task['preset']['sha256'],
                '--task-id', task['id'], '--output', str(output)]
            if resume:
                argv += ['--resume', str(resume)]
            if task['kind'] == 'aggregate_I4_exports':
                members = [self.states[dep]['selected_VALID_export'] for dep in task['dependencies']]
                for receipt in members:
                    verify(receipt, self.plan['project_root'])
                argv += ['--members', *[receipt['path'] for receipt in members]]
        return {'schema': 'all-fit-native-invocation-v1', 'id': task['id'], 'phase': phase, 'argv': argv,
            'project_root': self.plan['project_root'], 'output': str(output), 'environment': {key: os.environ[key] for key in ('BENCHMARK_HOST_CONFIG', 'BENCHMARK_HOST_CONFIG_SHA256', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')},
            'plan': pin(self.args.plan), 'resume_uses_smoke': False, 'created': time.time()}

    def start(self, task, phase):
        with lock(self.ledger / 'LEDGER.lock', blocking=True):
            active = self.active_ledger()
            candidate = {'gpu_uuid': self.config['physical_gpu_uuid'], 'reservation': task['reservation']}
            cache_key = (task['reservation']['profile'], tuple(sorted(row['directory'] for row in active)))
            if time.monotonic() - self.blocked_profiles.get(cache_key, -1e9) < self.args.poll_seconds:
                return False
            same_gpu = [row for row in active if row['gpu_uuid'] == candidate['gpu_uuid']]
            if len(same_gpu) >= 2:
                self.blocked_profiles[cache_key] = time.monotonic(); return False
            for row in same_gpu:
                if row.get('external_identity'):
                    compatible = (task['reservation']['vram_bytes'] + row['reservation']['vram_bytes'] <= 64 * GiB and
                        task['reservation']['ram_bytes'] + row['reservation']['ram_bytes'] <= 48 * GiB)
                else:
                    compatible = self.resources.can_pair(task['reservation'], row['reservation'])
                if not compatible:
                    self.blocked_profiles[cache_key] = time.monotonic(); return False
            live = probe(self.config, active)
            decision = capacity_decision(candidate, active, live, self.resources.can_pair)
            if not decision['admitted']:
                self.blocked_profiles[cache_key] = time.monotonic()
                write(self.root / 'CAPACITY_WAIT.json', {'id': task['id'], 'decision': decision, 'live': live})
                return False
            attempt = self.states[task['id']]['attempt'] + 1
            directory = self.folder(task, phase) / 'attempts' / f'{attempt:04d}'
            require(not directory.exists(), 'Attempt directory collision')
            request = self.invocation(task, phase, directory)
            directory.mkdir(parents=True)
            handle = lock(directory / 'LEASE.lock'); require(handle is not None, 'New attempt lease unavailable')
            record = {'directory': str(directory), 'worker': self.args.worker, 'id': task['id'], 'phase': phase,
                'gpu_uuid': self.config['physical_gpu_uuid'], 'reservation': task['reservation'], 'created': time.time()}
            try:
                write(directory / 'INVOCATION.json', request)
                write(directory / 'ADMISSION.json', {'decision': decision, 'live': live, 'active': active})
                write(self.ledger / 'reservations' / (token(str(directory)) + '.json'), record)
                fds = tuple([lease.fileno() for lease in self.leases] + [handle.fileno()])
                argv = [str(Path(self.plan['project_root']) / '.venv/bin/python'), '-B', '-u', str(HERE / 'guardian.py'),
                    '--invocation', str(directory / 'INVOCATION.json'), '--invocation-sha256', pin(directory / 'INVOCATION.json')['sha256'],
                    '--lease-fds', ','.join(map(str, fds))]
                with (directory / 'guardian.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen(argv, cwd=self.plan['project_root'], stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=fds, start_new_session=True)
                self.processes[task['id']] = (child, directory)
                self.retry.discard(task['id'])
                self.save_state(task, phase, {'status': 'running', 'attempt': attempt, 'directory': str(directory), 'output': request['output']})
            except BaseException as exc:
                write(directory / 'LAUNCH_ERROR.json', {'error': type(exc).__name__ + ': ' + str(exc), 'time': time.time()})
                raise
            finally:
                handle.close()
            return True

    def reap(self, phase):
        for task_id, (child, directory) in list(self.processes.items()):
            if child.poll() is None or locked(directory / 'LEASE.lock'):
                continue
            self.collect(self.index[task_id], phase, directory)
            del self.processes[task_id]

    def run_phase(self, phase):
        tasks = [task for task in self.tasks if phase == 'full' or task['kind'] == 'fit']
        if phase == 'smoke':
            tasks.sort(key=lambda task: task['smoke_order'])
        self.phase = phase; self.load_states(tasks, phase)
        success = {'PASS'} if phase == 'smoke' else {'complete', 'no_selected_state', 'blocked_dependency_no_selected_state'}
        while True:
            self.reap(phase)
            stopping = (self.root / 'STOP').exists()
            failed = [key for key, state in self.states.items() if state['status'] == 'failed' and key not in self.retry]
            if not stopping and not failed:
                for task in tasks:
                    state = self.states[task['id']]
                    if state['status'] in success or task['id'] in self.processes:
                        continue
                    if phase == 'full' and task['kind'] == 'aggregate_I4_exports':
                        statuses = [self.states[dep]['status'] for dep in task['dependencies']]
                        if any(status == 'no_selected_state' for status in statuses):
                            self.save_state(task, phase, {'status': 'blocked_dependency_no_selected_state', 'attempt': 0,
                                'dependencies': task['dependencies'], 'reason': 'Independent member ended without selected VALID state'})
                            continue
                        if any(status != 'complete' for status in statuses):
                            continue
                    if len(self.processes) >= 2:
                        break
                    if self.start(task, phase):
                        self.report(event='started', task_id=task['id'])
                    else:
                        # A preserved external job can block a heavy fit for days.
                        # Keep its fixed ownership, but try the next feasible fit.
                        continue
            self.report()
            if not self.processes:
                if stopping:
                    self.phase = 'stopped'; self.report(reason='Graceful STOP; no children interrupted'); return False
                if failed:
                    self.phase = phase + '_failed'; self.report(failed_fit_ids=failed); return False
                if all(state['status'] in success for state in self.states.values()):
                    return True
            time.sleep(self.args.poll_seconds)

    def smoke_summary(self):
        rows = []
        for task in self.tasks:
            if task['kind'] != 'fit':
                continue
            state = self.states[task['id']]
            require(state['status'] == 'PASS', 'Summary requires every assigned smoke PASS')
            result = read(verify(state['native_result']))
            self.validate_result(task, 'smoke', result, Path(state['output']))
            rows.append({'id': task['id'], 'status': 'PASS', 'runtime_job_hash': task['runtime_job_hash'],
                'descriptor': task['descriptor'], 'attempt': state['attempt'], 'result_receipt': state['native_result'],
                'result': result, 'attempt_result': state['attempt_result']})
        value = {'schema': 'allfit-worker-smoke-summary-v1', 'status': 'PASS', 'worker': self.args.worker,
            'inventory': self.plan['inventory'], 'expected_fit_ids': sorted(row['id'] for row in rows),
            'results': rows, 'fit_count': len(rows), 'TEST_accessed': False}
        path = self.root / 'SMOKE_SUMMARY.json'
        if path.exists():
            require(read(path) == value, 'Existing smoke summary bytes/identities must remain stable')
        else:
            write(path, value)
        return pin(path)

    def run(self):
        self.prerequisites(); self.acquire()
        if not self.run_phase('smoke'):
            return 2
        summary_pin = self.smoke_summary()
        self.phase = 'waiting_for_global_all_fit_PASS'; self.report(summary=summary_pin)
        while not global_barrier(self.plan['global_smoke_pass'], self.plan, summary_pin):
            if (self.root / 'STOP').exists():
                return 77
            self.report(); time.sleep(self.args.poll_seconds)
        write(self.root / 'TRAINING_BARRIER_ACCEPTED.json', {'global_pass': pin(self.plan['global_smoke_pass']),
            'summary': summary_pin, 'inventory': self.plan['inventory'], 'accepted': time.time()})
        if not self.run_phase('full'):
            return 3
        self.phase = 'complete'; self.report(full_task_count=len(self.tasks), summary=summary_pin)
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=WORKERS, required=True)
    p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--smoke-wall-seconds', type=int, default=1800)
    args = p.parse_args(); require(args.poll_seconds >= 1, 'Poll interval must be at least one second')
    worker = None
    try:
        worker = Worker(args)
        return worker.run()
    except BaseException as exc:
        if worker is not None:
            worker.phase = 'supervisor_error'
            worker.report(error=type(exc).__name__ + ': ' + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
