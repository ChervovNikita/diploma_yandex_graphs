"""Operational regression checks; no scientific training or heldout scoring."""
import ast
import copy
from pathlib import Path
import unittest
from unittest import mock

from host_extension import config, check_identity, UUIDS, SHARDS, validate_config
from partition import form_shards, live
from handover import admission_boundary


class Checks(unittest.TestCase):
    def test_owner_disappearance_is_not_a_live_owner(self):
        with mock.patch('partition.c.same_process', return_value=True), mock.patch('pathlib.Path.read_text', side_effect=FileNotFoundError):
            self.assertFalse(live({'pid': 5, 'start_ticks': 8}))

    def test_only_authorized_devices(self):
        for index in (0, 5, 7):
            with self.assertRaises(ValueError):
                config('anogena/gpu' + str(index))
        for index in (1, 2, 3, 4, 6):
            self.assertEqual(config('anogena/gpu' + str(index))['physical_gpu_uuid'], UUIDS[index])

    def test_host_and_inventory_are_real(self):
        value = config('anogena/gpu1')
        check_identity(value, 'anogena-0', UUIDS)
        with self.assertRaises(ValueError):
            check_identity(value, 'anogena-2-0', UUIDS)
        with self.assertRaises(ValueError):
            check_identity(value, 'anogena-0', list(reversed(UUIDS)))
        altered = copy.deepcopy(value); altered['physical_gpu_index'] = 2
        with self.assertRaises(ValueError):
            validate_config(altered)

    def test_disjoint_dependency_groups_and_preserved_attempts(self):
        tasks = [{'id': f'f{i}', 'kind': 'fit', 'cell_id': 'cell', 'dependencies': []} for i in range(24)]
        tasks += [{'id': 'bank', 'kind': 'aggregate_I4_exports', 'cell_id': 'cell',
                   'dependencies': ['f0', 'f1', 'f2', 'f3']}]
        states = {row['id']: {'status': 'pending'} for row in tasks}
        states['f2']['status'] = 'complete'; states['f5']['status'] = 'running'
        prior = {row['id']: 'allocation/gpu0' for row in tasks}
        for key in ('bank', 'f0', 'f1', 'f2', 'f3'):
            prior[key] = 'anogena/gpu1'
        prior['f5'] = 'anogena/gpu6'
        assigned, cost = form_shards(tasks, states, {'cell': 10}, prior)
        all_ids = [key for ids in assigned.values() for key in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(states))
        self.assertTrue({'bank', 'f0', 'f1', 'f2', 'f3'} <= set(assigned['anogena/gpu1']))
        self.assertIn('f5', assigned['anogena/gpu6'])
        self.assertEqual(set(assigned), set(SHARDS))
        self.assertTrue(all(assigned[key] for key in SHARDS))
        self.assertLessEqual(max(cost.values()) - min(cost.values()), 10)
        self.assertEqual(form_shards(tasks, states, {'cell': 10}, prior), (assigned, cost))

    def test_duplicate_dependency_rejected(self):
        tasks = [{'id': 'f', 'kind': 'fit', 'cell_id': 'c', 'dependencies': []},
                 {'id': 'b', 'kind': 'aggregate_I4_exports', 'cell_id': 'c', 'dependencies': ['f'] * 4}]
        with self.assertRaises(ValueError):
            form_shards(tasks, {row['id']: {'status': 'pending'} for row in tasks}, {}, {})

    def test_handover_requires_idle_boundary_or_dead_owner(self):
        owner = {'pid': 5, 'start_ticks': 8}
        with mock.patch('handover.live', return_value=False):
            self.assertEqual(admission_boundary(owner, 'stopped'), 'prior_owner_exited_no_new_admission')
        with mock.patch('handover.live', return_value=True):
            with mock.patch('pathlib.Path.read_text', return_value='hrtimer_nanosleep'):
                self.assertEqual(admission_boundary(owner, 'full'), 'prior_owner_asleep_outside_admission_ledger')
            with mock.patch('pathlib.Path.read_text', return_value='do_wait'):
                self.assertIsNone(admission_boundary(owner, 'full'))
            with mock.patch('pathlib.Path.read_text', side_effect=FileNotFoundError):
                self.assertIsNone(admission_boundary(owner, 'full'))

    def test_prior_supervisor_fences_and_all_ledger_generations_present(self):
        source = (Path(__file__).parent / 'shard_worker.py').read_text()
        self.assertIn("paths.insert(0, PREVIOUS / 'runs' / self.args.worker.replace('/', '_') / 'SUPERVISOR.lock')", source)
        self.assertIn("historical_ledgers = [Path(self.plan['host_ledger']), PREVIOUS / 'shared_disk_and_host_ledgers']", source)

    def test_all_source_is_plain_parseable_python(self):
        for path in Path(__file__).parent.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_reporting_entry_only_changes_host_authority(self):
        here = Path(__file__).parent
        old = (here.parent.parent / 'server51_campaign_port_v6/reporting/runtime_entry.py').read_text()
        old_guard = "    host_path=Path(__file__).resolve().parent.parent/'host_binding.py'\n    host_spec=importlib.util.spec_from_file_location('_server51_reporting_host_guard',host_path)\n    host=importlib.util.module_from_spec(host_spec);host_spec.loader.exec_module(host)\n    config=host.guard()  # Exact hostname, full device table and UUID mask before runtime/data reads."
        new_guard = "    from host_extension import guard, install_native_authority\n    config = guard()  # Genuine new identity; frozen scientific runtime remains unchanged."
        marker = '    sys.path.insert(0, str(Path(entry["path"]).parent))'
        addition = "    authority = install_native_authority('/composed/' in entry['path'])\n    output.parent.mkdir(parents=True, exist_ok=True)\n    atomic_json(output.parent / 'ACTUAL_HOST_AUTHORITY.json', authority)\n"
        self.assertEqual((here / 'reporting_entry.py').read_text(), old.replace(old_guard, new_guard).replace(marker, addition + marker))


if __name__ == '__main__':
    unittest.main()
