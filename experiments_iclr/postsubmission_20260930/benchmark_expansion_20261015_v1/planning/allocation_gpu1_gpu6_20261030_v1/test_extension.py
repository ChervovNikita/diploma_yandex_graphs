"""Operational regression checks; no scientific training or heldout scoring."""
import ast
import copy
from pathlib import Path
import unittest
from unittest import mock

from host_extension import config, check_identity, UUIDS, SHARDS, validate_config
from partition import form_shards, live


class Checks(unittest.TestCase):
    def test_owner_disappearance_is_not_a_live_owner(self):
        with mock.patch('partition.c.same_process', return_value=True), mock.patch('pathlib.Path.read_text', side_effect=FileNotFoundError):
            self.assertFalse(live({'pid': 5, 'start_ticks': 8}))

    def test_only_authorized_devices(self):
        for index in (0, 2, 3, 4, 5, 7):
            with self.assertRaises(ValueError):
                config('anogena/gpu' + str(index))
        self.assertEqual(config('anogena/gpu1')['physical_gpu_uuid'], UUIDS[1])
        self.assertEqual(config('anogena/gpu6')['physical_gpu_uuid'], UUIDS[6])

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
        assigned, cost = form_shards(tasks, states, {'cell': 10})
        all_ids = [key for ids in assigned.values() for key in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(states))
        self.assertTrue({'bank', 'f0', 'f1', 'f2', 'f3', 'f5'} <= set(assigned[SHARDS[0]]))
        self.assertLessEqual(max(cost.values()) - min(cost.values()), 10)
        self.assertEqual(form_shards(tasks, states, {'cell': 10}), (assigned, cost))

    def test_duplicate_dependency_rejected(self):
        tasks = [{'id': 'f', 'kind': 'fit', 'cell_id': 'c', 'dependencies': []},
                 {'id': 'b', 'kind': 'aggregate_I4_exports', 'cell_id': 'c', 'dependencies': ['f'] * 4}]
        with self.assertRaises(ValueError):
            form_shards(tasks, {row['id']: {'status': 'pending'} for row in tasks}, {})

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
