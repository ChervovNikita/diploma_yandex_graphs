"""Continue pending smokes on the exact execution5 plan; never run full training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'campaign_execution_20261016_v5'
PARENT_SHA = '1960d3a07693e9813e00789a4c8eb54393f197639cd89ee66cbe91076b99184c'
sys.path.insert(0, str(PARENT))
for name in ('common', 'capacity', 'worker'):
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve().parent == PARENT, 'Fresh interpreter required'
import common as c
from worker import Worker


def own_source(expected):
    source = c.pin(HERE / 'MANIFEST.json')
    c.require(source['sha256'] == expected, 'Exact continuation source required')
    for row in c.read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve()
        c.require(path.is_relative_to(HERE), 'Continuation source escapes package')
        actual = c.pin(path)
        c.require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Continuation source changed')
    c.require(c.source_manifest()['sha256'] == PARENT_SHA, 'Frozen execution5 helper changed')
    return source


class ContinuationWorker(Worker):
    def __init__(self, args, source):
        self.scheduler_source = {'schema': 'smoke-only-continuation-producer-v1',
            'actual_outer_source_manifest': source, 'unchanged_parent_source_manifest': c.source_manifest(),
            'plan': c.pin(args.plan), 'changes': 'continue pending independent smokes after failed fits',
            'automatic_retry': False, 'full_training_supported': False}
        old_owner = Path(c.read(args.plan)['state_root']) / 'OWNER.json'
        previous = old_owner.read_bytes() if old_owner.exists() else None
        # Original constructor authenticates the unchanged plan/inventory, owns
        # the supervisor lease, and validates its exact original helper source.
        super().__init__(args)
        self.scheduler_source['inventory'] = self.plan['inventory']
        if previous is not None:
            history = self.root / ('OWNER-before-continuation-' + hashlib.sha256(previous).hexdigest() + '.json')
            if history.exists():
                c.require(history.read_bytes() == previous, 'Prior owner history differs')
            else:
                with history.open('xb') as stream:
                    stream.write(previous)
        value = c.read(old_owner)
        value['actual_scheduler_producer'] = self.scheduler_source
        c.write(old_owner, value)
        c.write(self.root / ('CONTINUATION_OWNER-' + str(time.time_ns()) + '.json'),
                {'identity': c.identity(), 'producer': self.scheduler_source, 'TEST_accessed': False})

    def report(self, **extra):
        if time.time() - self.last_report < 10 and not extra:
            return
        return super().report(actual_scheduler_producer=self.scheduler_source, **extra)

    def invocation(self, task, phase, directory):
        c.require(phase == 'smoke', 'Continuation cannot invoke full training or aggregation')
        return {**super().invocation(task, phase, directory), 'actual_scheduler_producer': self.scheduler_source}

    def continue_pending(self):
        tasks = sorted((t for t in self.tasks if t['kind'] == 'fit'), key=lambda t: t['smoke_order'])
        self.phase = 'smoke_continuation'
        self.load_states(tasks, 'smoke')
        c.require(not self.retry, 'Failed unchanged fits may not be retried')
        while True:
            self.reap('smoke')
            stopping = (self.root / 'STOP').exists()
            c.require(all(s['status'] in ('pending', 'running', 'PASS', 'failed') for s in self.states.values()),
                      'Unexpected smoke state; do not skip unknown work')
            if not stopping:
                for task in tasks:
                    # PASS and failed receipts are retained. Only a genuinely
                    # unattempted pending ID may start under this wrapper.
                    if self.states[task['id']]['status'] != 'pending' or task['id'] in self.processes:
                        continue
                    if len(self.processes) >= 2:
                        break
                    if self.start(task, 'smoke'):
                        self.report(event='started', task_id=task['id'])
            self.report()
            if not self.processes:
                if stopping:
                    self.phase = 'stopped'; self.report(reason='Graceful STOP; no children interrupted'); return False
                pending = [key for key, state in self.states.items() if state['status'] == 'pending']
                if not pending:
                    failed = sorted(key for key, state in self.states.items() if state['status'] == 'failed')
                    if not failed and all(state['status'] == 'PASS' for state in self.states.values()):
                        return True
                    self.phase = 'smoke_incomplete'
                    self.report(failed_fit_ids=failed, all_fit_PASS=False, full_training_supported=False)
                    return False
            time.sleep(self.args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=c.WORKERS, required=True)
    parser.add_argument('--plan', required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--smoke-wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    c.require(args.poll_seconds >= 1 and args.smoke_wall_seconds == 1800, 'Keep original polling/wall policy')
    args.retry_failed = False
    worker = None
    try:
        source = own_source(args.source_sha256)
        worker = ContinuationWorker(args, source)
        worker.prerequisites(); worker.acquire()
        if not worker.continue_pending():
            return 2
        # Existing unchanged summary code still refuses any failed/missing ID.
        # This entrypoint has no global marker or full-training phase.
        summary = worker.smoke_summary()
        worker.phase = 'all_smokes_PASS_continuation_finished'
        worker.report(summary=summary, full_training_supported=False)
        return 0
    except BaseException as exc:
        if worker is not None:
            worker.phase = 'supervisor_error'
            worker.report(error=type(exc).__name__ + ': ' + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
