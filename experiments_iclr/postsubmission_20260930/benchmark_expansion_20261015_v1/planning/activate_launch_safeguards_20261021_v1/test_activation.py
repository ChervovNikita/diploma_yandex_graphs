"""Small local handoff guard fixtures; no subprocess or remote execution."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import activate as a


class ActivationTests(unittest.TestCase):
    def test_wrong_host_stops_before_gpu_or_project_access(self):
        with patch.object(a.socket, 'gethostname', return_value='wrong'), \
             patch.object(a.subprocess, 'check_output') as gpu:
            with self.assertRaisesRegex(ValueError, 'no project metadata'):
                a.guard('allocation/gpu0')
            gpu.assert_not_called()

    def test_full_gpu_table_required(self):
        with patch.object(a.socket, 'gethostname', return_value='olig'), \
             patch.object(a.subprocess, 'check_output', return_value='GPU-9c6c20fa-a620-479d-f72f-163ac74deef2\n'):
            with self.assertRaisesRegex(ValueError, 'full GPU identity'):
                a.guard('server51/gpu0')

    def test_only_waiting_exact_pass_owner_can_be_stopped(self):
        receipt = {'sha256': 'plan'}
        owner = {'identity': {'pid': 12}, 'plan': receipt}
        status = {'worker': 'allocation/gpu0', 'plan': receipt, 'owner': owner['identity'],
                  'phase': 'waiting_for_global_all_fit_PASS', 'active_fit_ids': [], 'counts': {'PASS': 2154}}
        a.require_waiting(status, owner, 'allocation/gpu0', 2154, receipt)
        for key, value in (('phase', 'full'), ('active_fit_ids', ['busy']), ('counts', {'PASS': 2153, 'failed': 1})):
            changed = dict(status, **{key: value})
            with self.subTest(field=key), self.assertRaises(ValueError):
                a.require_waiting(changed, owner, 'allocation/gpu0', 2154, receipt)

    def test_reconcile_requires_both_current_producers_and_waiting_status(self):
        producer = {'actual_outer_source_manifest': {'sha256': 'named'}}
        owner = {'identity': {'pid': 12}, 'actual_scheduler_producer': producer}
        status = {'owner': owner['identity'], 'actual_scheduler_producer': producer,
                  'phase': 'waiting_for_global_all_fit_PASS', 'active_fit_ids': []}
        self.assertTrue(a.ready(owner, status, 'named'))
        for key, value in (('owner', {'pid': 11}), ('phase', 'initializing'), ('active_fit_ids', ['busy']), ('actual_scheduler_producer', {})):
            self.assertFalse(a.ready(owner, dict(status, **{key: value}), 'named'))

    def test_native_venv_and_explicit_local_title_dependency(self):
        for worker, root in (('allocation/gpu0', a.ALLOCATION), ('server51/gpu0', a.SERVER51), ('server51/gpu1', a.SERVER51)):
            project = Path(root)
            expansion = project / a.RELATIVE
            plan = expansion / 'campaign_execution_20261016_v7' / ('PLAN-' + worker.replace('/', '_') + '.json')
            argv = a.command(project, expansion, worker, plan)
            self.assertEqual(argv[0], root + '/.venv/bin/python')
            self.assertEqual(argv[argv.index('--plan-sha256') + 1], a.PLAN_SHAS[worker])
            self.assertEqual(argv[-2:], ['--process-title-dependency-dir', str(expansion / 'campaign_process_title_dependency_20261021_v1')])
            self.assertNotIn('--retry-failed', argv)


if __name__ == '__main__':
    unittest.main(verbosity=2)
