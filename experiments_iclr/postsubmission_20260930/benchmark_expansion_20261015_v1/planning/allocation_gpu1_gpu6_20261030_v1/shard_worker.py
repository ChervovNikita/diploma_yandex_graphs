"""Continue the frozen allocation task set on three disjoint physical workers."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
sys.path.insert(0, str(E / 'campaign_concurrency_20261029_v1'))
import concurrency_worker as four
from host_extension import SHARDS, bind, config, pin, ROOT
from partition import original_plan, PLAN, PLAN_SHA, source, freeze, validate_partition
c = four.recovery.c
original_probe = four.probe
FOUR_SHA = '82b056580848c7074d6d9e2a3aa9a13f298c2123256d0d9f07c9139ba6bbe912'


class ShardWorker(four.FourWorker):
    def __init__(self, args):
        # Initialization is explicit because the old initializer owns the whole
        # allocation queue. All per-task methods below retain original custody.
        self.args = args
        self.config = bind(args.worker)
        c.require(os.environ.get('TMUX'), 'Persistent queue must run in tmux')
        self.additive_source = source(args.source_sha256)
        four.own_source(FOUR_SHA)
        self.plan = original_plan()
        self.partition_pin = pin(HERE / 'PARTITION.json')
        partition = c.read(HERE / 'PARTITION.json')
        validate_partition(partition, self.plan)
        c.require(partition['source'] == self.additive_source and partition['original_plan'] == pin(PLAN),
                  'Immutable partition source or original plan differs')
        assigned = partition['assignment'][args.worker]
        c.require(len(assigned) == len(set(assigned)), 'Duplicate assigned ID')
        index = {row['id']: row for row in self.plan['tasks']}
        self.tasks = [index[key] for key in assigned]
        self.index = {row['id']: row for row in self.tasks}
        c.require(all(set(row['dependencies']) <= set(self.index) for row in self.tasks), 'Foreign dependency')
        for field in ('inventory', 'authorization', 'resource_source', 'resource_catalogue', 'reporting_entrypoint', 'smoke_source_manifest'):
            c.verify(self.plan[field], ROOT)
        self.inventory = c.read(self.plan['inventory']['path'])
        self.smokes = {row['id']: row for row in self.inventory['fits'] if row['id'] in self.index}
        self.resources = c.module(c.verify(self.plan['resource_source']), '_allocation_shard_resources')
        self.canonical_root = Path(self.plan['state_root'])
        self.root = HERE / 'runs' / args.worker.replace('/', '_')
        self.root.mkdir(parents=True, exist_ok=True)
        self.supervisor_lease = c.lock(self.root / 'SUPERVISOR.lock')
        c.require(self.supervisor_lease is not None, 'Shard already has a supervisor')
        self.ledger = HERE / 'shared_disk_and_host_ledgers'
        self.ledger.mkdir(parents=True, exist_ok=True)
        self.leases = []; self.processes = {}; self.states = {}; self.phase = 'initializing'
        self.retry = set(); self.last_report = 0; self.blocked_profiles = {}; self.started = time.time()
        self.remote_disk_reserved = 0
        self.named_source = four.recovery.source(four.REPAIR_SHA)
        self.recovery_config = {'repairs': {}}
        self.process_title_dependency_dir = args.process_title_dependency_dir
        self.process_title_dependency = four.recovery.named.verify_dependency(args.process_title_dependency_dir)
        self._scheduler_source = {'schema': 'allocation-three-device-scheduler-v1',
            'actual_outer_source_manifest': self.additive_source, 'partition': self.partition_pin,
            'original_plan': pin(PLAN), 'physical_host': self.config,
            'unchanged_four_slot_source': pin(E / 'campaign_concurrency_20261029_v1/MANIFEST.json'),
            'native_scientific_descriptors_changed': False, 'automatic_retry': False,
            'maximum_concurrent_tasks_per_gpu': 4, 'original_smoke_receipts_reused': True,
            'server51_changed': False, 'shared_disk_reservations_count_all_three_workers': True}
        four.safeguards.preserve_owner(self.root / 'OWNER.json')
        c.write(self.root / 'OWNER.json', {'identity': c.identity(), 'started': self.started,
            'worker': args.worker, 'tmux': os.environ['TMUX'], 'producer': self._scheduler_source})
        # Resource measurements are host-local. Only disk reservations are shared
        # across the two hosts; never interpret a foreign PID through local /proc.
        four.probe = self.resource_probe

    def folder(self, task, phase):
        c.require(phase == 'full', 'Already-passed original smoke tests must not rerun')
        return self.canonical_root / 'full' / c.token(task['id'])

    def acquire(self):
        if self.args.worker == 'allocation/gpu0':
            paths = [self.canonical_root / 'SUPERVISOR.lock', *[self.plan[key] for key in ('slot_lock', 'physical_lock', 'worker_lock')]]
        else:
            paths = [ROOT / 'experiments_iclr/postsubmission_20260930/.expanded_dispatcher_leases' /
                     (self.config['physical_gpu_uuid'] + '.lock'), self.root / 'PHYSICAL_SLOT.lock']
        while True:
            handles = []
            for path in paths:
                handle = c.lock(path)
                if handle is None:
                    break
                handles.append(handle)
            if len(handles) == len(paths):
                self.leases = handles; return
            for handle in handles:
                handle.close()
            self.phase = 'waiting_for_physical_custody'; self.report()
            c.require(not (self.root / 'STOP').exists(), 'Shard stop requested')
            time.sleep(self.args.poll_seconds)

    def active_ledger(self):
        local = []; remote_disk = 0
        for path in (self.ledger / 'reservations').glob('*.json'):
            row = c.read(path)
            c.require(row['worker'] in SHARDS, 'Unexpected worker in allocation-only ledger')
            if c.locked(Path(row['directory']) / 'LEASE.lock'):
                if config(row['worker'])['expected_hostname'] == self.config['expected_hostname']:
                    local.append(row)
                else:
                    remote_disk += row['reservation']['disk_bytes']
            else:
                retired = self.ledger / 'retired_reservations'
                retired.mkdir(exist_ok=True)
                path.rename(retired / path.name)
        # The old owner may still be draining four preserved fits. Read its
        # reservations without retiring/mutating them; they are foreign-host
        # disk reservations on anogena-0, and host-local capacity on anogena-2-0.
        for path in (Path(self.plan['host_ledger']) / 'reservations').glob('*.json'):
            row = c.read(path)
            if row['worker'] != 'allocation/gpu0' or not c.locked(Path(row['directory']) / 'LEASE.lock'):
                continue
            if self.config['expected_hostname'] == 'anogena-2-0':
                local.append(row)
            else:
                remote_disk += row['reservation']['disk_bytes']
        if self.args.worker == 'allocation/gpu0':
            try:
                peer = c.identity(640428)
                if peer['start_ticks'] == 6055039165:
                    local.append({'directory': 'external-allocation-gru-640428', 'external_identity': peer,
                        'gpu_uuid': self.config['physical_gpu_uuid'], 'reservation': {
                            'vram_bytes': 32 * c.GiB, 'ram_bytes': 16 * c.GiB, 'disk_bytes': 0,
                            'cpu_slots': 2, 'thread_count': 2, 'profile': 'preserved_external_gru'}})
            except (FileNotFoundError, ProcessLookupError):
                pass
        self.remote_disk_reserved = remote_disk
        return local

    def resource_probe(self, config_value, active):
        live = original_probe(config_value, active)
        return {**live, 'free_disk': max(0, live['free_disk'] - self.remote_disk_reserved),
                'actual_shared_disk_free': live['free_disk'],
                'other_host_disk_reserved': self.remote_disk_reserved,
                'host_local_process_accounting': True}

    def start(self, task, phase):
        c.require(pin(HERE / 'PARTITION.json') == self.partition_pin, 'Task partition changed')
        c.require(pin(self.canonical_root / 'STOP') == c.read(HERE / 'PARTITION.json')['original_stop'],
                  'Original admission stop marker must remain unchanged')
        return super().start(task, phase)

    def invocation(self, task, phase, directory):
        request = super().invocation(task, phase, directory)
        argv = list(request['argv'])
        c.require(Path(argv[3]).resolve() == E / 'campaign_reporting_repair_20261024_v2/repair_native.py',
                  'Unexpected inherited native launcher')
        argv[3] = str(HERE / 'dispatch_native.py')
        return {**request, 'argv': argv, 'physical_shard': self.args.worker,
                'canonical_original_worker': 'allocation/gpu0', 'partition': self.partition_pin,
                'native_scientific_descriptor_changed': False}

    def collect(self, task, phase, directory):
        """Frozen collection semantics, canonical worker plus actual execution.

        The frozen scalar audit binds worker to the original task assignment.
        Keep that logical assignment and record the real physical worker beside
        it, never overwrite the actual host/config in invocation provenance.
        """
        request = c.read(directory / 'INVOCATION.json')
        output = Path(request['output'])
        terminal_path = directory / 'TERMINAL.json'
        if not terminal_path.exists():
            c.write(terminal_path, {'schema': 'all-fit-child-terminal-v1', 'exit_code': None,
                'unknown_partial_work': True, 'prior_partial_work_preserved': True,
                'failure': 'Guardian terminal missing after all attempt owners exited', 'finished': time.time()})
        terminal = c.read(terminal_path)
        native = output / 'RESULT.json'
        raw = None; status = 'failed'; selected = None; failure = None
        try:
            c.require(native.exists(), 'Missing native RESULT.json')
            raw = c.read(native)
            status, selected = self.validate_result(task, phase, raw, output)
            c.require(terminal['exit_code'] in (0, None), 'Full task terminal failure')
            c.require(not terminal.get('forced_kill') and not terminal.get('signals_forwarded'), 'Interrupted native attempt')
        except Exception as exc:
            status = 'failed'; selected = None; failure = type(exc).__name__ + ': ' + str(exc)
        native_receipt = None
        if raw is not None:
            c.write(directory / 'NATIVE_RESULT.json', raw)
            native_receipt = pin(directory / 'NATIVE_RESULT.json')
        result = {'schema': 'all-fit-attempt-result-v1', 'id': task['id'], 'worker': 'allocation/gpu0',
            'physical_shard': self.args.worker, 'actual_host_config': self.config,
            'actual_scheduler_producer': self._scheduler_source, 'phase': phase,
            'attempt': int(directory.name), 'status': status, 'output': str(output), 'failure': failure,
            'native_result': native_receipt, 'terminal': pin(terminal_path), 'selected_VALID_export': selected,
            'exit_code': terminal['exit_code'], 'unknown_partial_work': terminal.get('unknown_partial_work', False),
            'prior_partial_work_preserved': True, 'finished': time.time()}
        c.write(directory / 'ATTEMPT_RESULT.json', result)
        result['attempt_result'] = pin(directory / 'ATTEMPT_RESULT.json')
        self.save_state(task, phase, result)

    def run(self):
        self.acquire()
        summary_path = self.canonical_root / 'SMOKE_SUMMARY.json'
        summary = c.read(summary_path)
        c.require(c.global_barrier(self.plan['global_smoke_pass'], self.plan, pin(summary_path)),
                  'All 6460 original fits must have passed the exact smoke barrier')
        covered = {row['id']: row for row in summary['results']}
        for task in self.tasks:
            if task['kind'] == 'fit':
                row = covered[task['id']]
                c.require(row['status'] == 'PASS' and row['descriptor'] == task['descriptor']
                          and row['runtime_job_hash'] == task['runtime_job_hash'], 'Task smoke identity differs')
                c.verify(row['result_receipt'], ROOT)
        c.write(self.root / 'TRAINING_BARRIER_ACCEPTED.json', {
            'global_pass': pin(self.plan['global_smoke_pass']), 'original_summary': pin(summary_path),
            'partition': self.partition_pin, 'physical_shard': self.args.worker,
            'verified_original_smoke_fit_count': len(self.smokes), 'accepted': time.time()})
        if not self.run_phase('full'):
            return 3
        self.phase = 'complete'; self.report(full_task_count=len(self.tasks))
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=SHARDS, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--process-title-dependency-dir', required=True)
    args = parser.parse_args()
    args.plan = str(PLAN); args.plan_sha256 = PLAN_SHA
    args.retry_failed = False; args.poll_seconds = 10; args.smoke_wall_seconds = 1800
    bind(args.worker); source(args.source_sha256)
    four.recovery.named.set_process_title(args.process_title_dependency_dir)
    while not (HERE / 'PARTITION.json').exists():
        c.require(not (HERE / 'FREEZE_ERROR.json').exists(), 'Partition preparation failed; inspect explicit failure receipt')
        if args.worker == 'allocation/gpu0':
            try:
                freeze()
            except BaseException as exc:
                c.write(HERE / 'FREEZE_ERROR.json', {'error': type(exc).__name__ + ': ' + str(exc),
                    'updated': time.time(), 'identity': c.identity(), 'traceback': traceback.format_exc(),
                    'scientific_children_interrupted': False})
                raise
        c.write(HERE / ('WAIT-' + args.worker.replace('/', '_') + '.json'),
                {'phase': 'waiting_for_original_admission_stop_acknowledgement', 'updated': time.time(),
                 'identity': c.identity(), 'worker': args.worker})
        if not (HERE / 'PARTITION.json').exists():
            time.sleep(10)
    instance = None
    try:
        instance = ShardWorker(args)
        return instance.run()
    except BaseException as exc:
        if instance is not None:
            instance.phase = 'supervisor_error'
            instance.report(error=type(exc).__name__ + ': ' + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
