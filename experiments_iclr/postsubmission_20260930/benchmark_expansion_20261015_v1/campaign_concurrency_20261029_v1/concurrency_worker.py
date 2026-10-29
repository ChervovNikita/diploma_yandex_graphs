"""Four resource-compatible tasks per GPU with the frozen scientific queue intact."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent / "campaign_reporting_repair_20261024_v2"
REPAIR_SHA = "235cda9eed0838cb4596d9a2f30837ffae3ee18f98c6f956bb00c58f3ccda7fe"
sys.path.insert(0, str(REPAIR))
import recovery_worker as recovery
from common import *

HERE = Path(__file__).resolve().parent
MAX_CONCURRENT = 4
safeguards = recovery.named.safeguards
probe = safeguards.memory.probe


def own_source(expected):
    manifest = pin(HERE / "MANIFEST.json")
    require(manifest["sha256"] == expected, "Concurrency source manifest changed")
    for row in read(HERE / "MANIFEST.json")["files"]:
        path = (HERE / row["path"]).resolve()
        require(path.is_relative_to(HERE), "Concurrency source escapes package")
        actual = pin(path)
        require(all(actual[key] == row[key] for key in ("bytes", "sha256")), "Concurrency source changed")
    recovery.source(REPAIR_SHA)
    recovery.named.own_source(recovery.NAMED_SHA)
    return manifest


def capacity_decision(candidate, active, live, can_pair):
    """Replace only the old slot count; keep all resource/growth restrictions.

    The inherited decision sums remaining VRAM across this GPU and remaining
    RAM, CPU and disk across the entire host. Pair compatibility also preserves
    heavy-task exclusions. A four-task ceiling does not guarantee four tasks fit.
    """
    decision = safeguards.capacity_decision(candidate, active, live, can_pair)
    reasons = [reason for reason in decision["reasons"] if reason != "two GPU slots occupied"]
    if sum(row["gpu_uuid"] == candidate["gpu_uuid"] for row in active) >= MAX_CONCURRENT:
        reasons.append("four GPU slots occupied")
    return {**decision, "admitted": not reasons, "reasons": reasons,
            "maximum_concurrent_tasks_per_gpu": MAX_CONCURRENT,
            "capacity_policy": "four-slots-with-unchanged-aggregate-growth-checks-v1"}


class FourWorker(recovery.RecoveryWorker):
    def __init__(self, args, concurrency_source):
        self.concurrency_source = concurrency_source
        # The repair source remains the source passed to repair_native.py.
        super().__init__(args, recovery.source(REPAIR_SHA))

    @property
    def scheduler_source(self):
        return self._scheduler_source

    @scheduler_source.setter
    def scheduler_source(self, value):
        recovery.RecoveryWorker.scheduler_source.fset(self, value)
        self._scheduler_source.update(
            schema="campaign-four-slot-scheduler-producer-v1",
            actual_outer_source_manifest=self.concurrency_source,
            unchanged_reporting_repair_source_manifest=pin(REPAIR / "MANIFEST.json"),
            maximum_concurrent_tasks_per_gpu=MAX_CONCURRENT,
            changes=[*self._scheduler_source["changes"],
                     "User requested four concurrent resource-compatible tasks per GPU"],
            scientific_task_assignment_changed=False,
        )

    # These two methods are the reviewed execution7 implementations with only
    # their slot ceiling changed from 2 to MAX_CONCURRENT. Their other operations
    # (host lock, attempt custody, child launch and failure handling) are retained.
    def start(self, task, phase):
        with lock(self.ledger / 'LEDGER.lock', blocking=True):
            active = self.active_ledger()
            candidate = {'gpu_uuid': self.config['physical_gpu_uuid'], 'reservation': task['reservation']}
            cache_key = (task['reservation']['profile'], tuple(sorted(row['directory'] for row in active)))
            if time.monotonic() - self.blocked_profiles.get(cache_key, -1e9) < self.args.poll_seconds:
                return False
            same_gpu = [row for row in active if row['gpu_uuid'] == candidate['gpu_uuid']]
            if len(same_gpu) >= MAX_CONCURRENT:
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
                argv = [str(Path(self.plan['project_root']) / '.venv/bin/python'), '-B', '-u', str(safeguards.parent.HERE / 'guardian.py'),
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

    def run_phase(self, phase):
        tasks = [task for task in self.tasks if phase == 'full' or task['kind'] == 'fit']
        if phase == 'smoke':
            tasks.sort(key=lambda task: task['smoke_order'])
        self.phase = phase; self.load_states(tasks, phase)
        if phase == 'smoke':
            # Only unattempted IDs may start. Preserve failed evidence even if
            # --retry-failed was supplied for the later full-training phase.
            self.retry.clear()
        success = {'PASS'} if phase == 'smoke' else {'complete', 'no_selected_state', 'blocked_dependency_no_selected_state'}
        while True:
            self.reap(phase)
            stopping = (self.root / 'STOP').exists()
            failed = [key for key, state in self.states.items() if state['status'] == 'failed' and key not in self.retry]
            if phase == 'smoke':
                require(all(state['status'] in ('pending', 'running', 'PASS', 'failed') for state in self.states.values()),
                        'Unexpected smoke state; inspect retained evidence')
            if not stopping and (phase == 'smoke' or not failed):
                for task in tasks:
                    state = self.states[task['id']]
                    if phase == 'smoke' and state['status'] != 'pending':
                        continue
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
                    if len(self.processes) >= MAX_CONCURRENT:
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
                if failed and (phase != 'smoke' or not any(state['status'] == 'pending' for state in self.states.values())):
                    self.phase = phase + '_failed'; self.report(failed_fit_ids=failed); return False
                if all(state['status'] in success for state in self.states.values()):
                    return True
            time.sleep(self.args.poll_seconds)


def main():
    recovery.guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=WORKERS, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--process-title-dependency-dir", required=True)
    parser.add_argument("--recovery-config", required=True)
    parser.add_argument("--recovery-config-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--smoke-wall-seconds", type=int, default=1800)
    args = parser.parse_args()
    args.retry_failed = False
    require(args.poll_seconds >= 1 and args.smoke_wall_seconds == 1800, "Frozen polling policy required")
    source = own_source(args.source_sha256)
    recovery.named.set_process_title(args.process_title_dependency_dir)
    instance = None
    try:
        instance = FourWorker(args, source)
        return instance.run()
    except BaseException as exc:
        if instance is not None:
            instance.phase = "supervisor_error"
            instance.report(error=type(exc).__name__ + ": " + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
