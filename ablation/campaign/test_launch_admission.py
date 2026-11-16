"""Check that the documented worker budgets admit every planned task."""
from types import SimpleNamespace
import unittest

from .launch import parser
from .plan import build_plan
from .worker import resource_shortfall


class LaunchAdmissionTests(unittest.TestCase):
    def test_default_budgets_cover_all_required_resource_classes(self):
        args = parser().parse_args(['--plan', 'unused.json', '--root', 'unused'])
        budgets = {
            'gpu': SimpleNamespace(memory_gb=args.memory_gb_per_gpu, cpus=args.cpus_per_gpu),
            'cpu': SimpleNamespace(memory_gb=args.cpu_memory_gb, cpus=args.cpu_cpus),
        }
        failures = [task['id'] for task in build_plan()['tasks']
                    if resource_shortfall(task, budgets[task['kind']])]
        self.assertEqual(failures, [])
        self.assertGreaterEqual(args.cpu_memory_gb, 40)


if __name__ == '__main__':
    unittest.main()
