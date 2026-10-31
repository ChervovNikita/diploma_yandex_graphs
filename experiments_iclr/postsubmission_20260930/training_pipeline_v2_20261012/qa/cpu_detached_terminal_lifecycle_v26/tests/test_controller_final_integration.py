"""One release-focused suite for the frozen distributed master controller."""
import unittest


def load_tests(loader, existing, pattern):
    modules = ['test_launcher', 'test_detached_startup', 'test_memory_admission',
               'test_phase_closure', 'test_phase_rendezvous', 'test_master_plan',
               'test_master_polyformer_ownership', 'test_dependency_grouping',
               'test_author_master_integration', 'test_execution_receipts']
    return loader.loadTestsFromNames(modules)


if __name__ == '__main__':
    unittest.main()
