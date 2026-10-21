"""Frozen launch safeguards with an explicit campaign process title launcher."""
import argparse
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
SAFEGUARDS = HERE.parent / 'campaign_launch_safeguards_20261021_v1'
SAFEGUARDS_SHA = 'adf6710ccf68808d21d8f4aeb6a67b665f17b9ee06de18051886caf11fd7bfdc'
sys.path.insert(0, str(SAFEGUARDS))
if 'queue_worker' in sys.modules:
    assert Path(sys.modules['queue_worker'].__file__).resolve().parent == SAFEGUARDS, 'Fresh interpreter required'
import queue_worker as safeguards
from launch_named import PROCESS_TITLE, set_process_title, verify_dependency

c = safeguards.c


def own_source(expected):
    source = c.pin(HERE / 'MANIFEST.json')
    c.require(source['sha256'] == expected, 'Exact named process source required')
    for row in c.read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve()
        c.require(path.is_relative_to(HERE), 'Named process source escapes package')
        actual = c.pin(path)
        c.require(all(actual[key] == row[key] for key in ('bytes', 'sha256')), 'Named process source changed')
    safeguards.own_source(SAFEGUARDS_SHA)
    return source


class NamedWorker(safeguards.LaunchSafeguardsWorker):
    def __init__(self, args, source):
        self.named_source = source
        self.process_title_dependency_dir = args.process_title_dependency_dir
        self.process_title_dependency = verify_dependency(args.process_title_dependency_dir)
        super().__init__(args, source)

    @property
    def scheduler_source(self):
        return self._scheduler_source

    @scheduler_source.setter
    def scheduler_source(self, value):
        # The frozen parent assigns this before its first OWNER/STATUS write.
        self._scheduler_source = {**value, 'schema': 'campaign-named-processes-producer-v1',
            'actual_outer_source_manifest': self.named_source,
            'unchanged_launch_safeguards_source_manifest': c.pin(SAFEGUARDS / 'MANIFEST.json'),
            'changes': [*value['changes'], 'Exact campaign process title using a plaintext native script launcher'],
            'process_title': PROCESS_TITLE, 'process_title_dependency_dir': self.process_title_dependency_dir,
            'process_title_dependency': self.process_title_dependency,
            'original_native_argv_preserved_in_provenance': True,
            'native_scientific_descriptor_changed': False}

    def invocation(self, task, phase, directory):
        invocation = super().invocation(task, phase, directory)
        original = list(invocation['argv'])
        c.require(len(original) >= 4 and original[1:3] == ['-B', '-u'], 'Unexpected native Python command shape')
        launcher = HERE / 'launch_named.py'
        argv = [*original[:3], str(launcher)]
        if self.process_title_dependency_dir is not None:
            argv += ['--dependency-dir', self.process_title_dependency_dir]
        argv += ['--', *original[3:]]
        return {**invocation, 'argv': argv, 'native_argv': original,
                'process_title': PROCESS_TITLE, 'process_title_launcher': c.pin(launcher),
                'process_title_dependency_dir': self.process_title_dependency_dir}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=c.WORKERS, required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--process-title-dependency-dir')
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--smoke-wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    c.require(args.poll_seconds >= 1 and args.smoke_wall_seconds == 1800, 'Keep original polling/wall policy')
    args.retry_failed = False
    instance = None
    try:
        source = own_source(args.source_sha256)
        if args.process_title_dependency_dir is not None:
            root = Path(c.read(args.plan)['project_root']).resolve()
            dependency = Path(args.process_title_dependency_dir).resolve()
            c.require(dependency.is_relative_to(root), 'Process title dependency must stay inside the project')
            args.process_title_dependency_dir = str(dependency)
        # Name the idle capacity waiter/supervisor too. A missing dependency fails
        # before acquiring custody or replacing OWNER.
        set_process_title(args.process_title_dependency_dir)
        instance = NamedWorker(args, source)
        return instance.run()
    except BaseException as exc:
        if instance is not None:
            instance.phase = 'supervisor_error'
            instance.report(error=type(exc).__name__ + ': ' + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
