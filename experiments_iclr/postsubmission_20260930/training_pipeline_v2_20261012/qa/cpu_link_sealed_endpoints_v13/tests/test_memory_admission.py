import unittest

from launcher import admission_capacities, schedule_ready


class MemoryAdmission(unittest.TestCase):
    def test_owned_live_memory_is_recovered_once_and_reservation_still_counts(self):
        row = {'index': 0, 'free_mib': 12000}
        capacities = admission_capacities('allocation', [row], {}, {}, {'allocation/gpu0': 10000})
        runs = {'candidate': {'run': {'kind': 'model_fit', 'config': {'reservation_mib': 5900}}}}
        placement = {'candidate': {'worker': 'allocation/gpu0'}}
        active = [{'key': 'running', 'worker': 'allocation/gpu0', 'reserve_mib': 15000}]
        ready = schedule_ready(runs, {'candidate': 'registered'}, placement, {'candidate': []},
                               active, capacities, 'allocation')
        self.assertEqual(ready, ['candidate'])
        runs['candidate']['run']['config']['reservation_mib'] = 6500
        self.assertEqual(schedule_ready(runs, {'candidate': 'registered'}, placement,
                         {'candidate': []}, active, capacities, 'allocation'), [])

    def test_unknown_cuda_usage_and_declared_capacity_remain_conservative(self):
        row = {'index': 0, 'free_mib': 30000}
        self.assertEqual(admission_capacities('allocation', [row], {}, {}, {}),
                         {'allocation/gpu0': 28976})
        plan = {'worker_capacities_cuda_bytes': {'allocation/gpu0': 20000 * 1024**2}}
        self.assertEqual(admission_capacities('allocation', [row], {}, plan,
                         {'allocation/gpu0': 5000}), {'allocation/gpu0': 20000})

    def test_external_workload_can_reduce_capacity_to_zero(self):
        self.assertEqual(admission_capacities('server77', [{'index': 1, 'free_mib': 500}],
                         {}, {}, {}), {'server77/gpu1': 0})

if __name__ == '__main__':
    unittest.main()
