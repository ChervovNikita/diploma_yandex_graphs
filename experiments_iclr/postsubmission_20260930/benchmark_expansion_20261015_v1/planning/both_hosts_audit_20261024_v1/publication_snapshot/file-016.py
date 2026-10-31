import unittest
from wait_recovery import eligible


class EligibilityTests(unittest.TestCase):
    def status(self, phase='full', active=None, failed=None):
        return {'worker': 'server51/gpu1', 'phase': phase, 'active_fit_ids': ['b'] if active is None else active,
                'failed_fit_ids': ['a', 'b'] if failed is None else failed}

    def test_training_waits(self):
        self.assertFalse(eligible(self.status(), 'server51/gpu1', ['a', 'b']))

    def test_fully_failed_known_set_eligible(self):
        self.assertTrue(eligible(self.status('full_failed', []), 'server51/gpu1', ['b', 'a']))

    def test_unexpected_failure_fails(self):
        with self.assertRaises(ValueError):
            eligible(self.status('full_failed', [], ['a', 'b', 'c']), 'server51/gpu1', ['a', 'b'])

    def test_still_active_failure_fails(self):
        with self.assertRaises(ValueError):
            eligible(self.status('full_failed'), 'server51/gpu1', ['a', 'b'])

    def test_supervisor_error_fails(self):
        with self.assertRaises(ValueError):
            eligible(self.status('supervisor_error'), 'server51/gpu1', ['a', 'b'])

    def test_wrong_worker_fails(self):
        with self.assertRaises(ValueError):
            eligible(self.status(), 'allocation/gpu0', ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
