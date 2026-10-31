"""Exact PolyFormer ownership and source-specific independent RNG accounting."""
import copy
import unittest

from compile_master_plan import bank_roster, fit_semantics
from test_master_plan import study, run


class MasterPolyFormerOwnershipTests(unittest.TestCase):
    def fixture(self):
        value = study('polyfixture')
        value['runs'] = []
        for body in range(4):
            config = {'stage': 'comparative', 'polyformer_mode': 'polyformer_factorized_single',
                      'members': 1, 'initialization_seed': 70101 + 1000003 * body,
                      'dropout_seeds': [51 + body * 1009], 'factor_initialization': 'first_rademacher',
                      'factor_seed': 80101 + 1000003 * body, 'loss': {'variant': 'own'},
                      'learning_rate': .01}
            value['runs'].append(run('body'+str(body), 'model_fit', 'equipped_M1', config, body=body))
        value['runs'].append(run('bank', 'ensemble_assembly', 'equipped_I4',
            {'stage': 'bank_export', 'fit_parents': ['body'+str(i) for i in range(4)]},
            ['body'+str(i) for i in range(4)]))
        return value

    def test_grouped_optimizer_is_one_paid_owner_and_one_slow_body(self):
        row = self.fixture()['runs'][0]
        semantics = fit_semantics(row)
        self.assertEqual(semantics['optimizer_objects'], 1)
        self.assertEqual(semantics['physical_slow_bodies'], 1)
        self.assertEqual(semantics['known_optimizer_step_calls'], row['budget']['maximum'])
        for mode in ('polyformer_internal_be', 'polyformer_original_boundary_own'):
            shared = copy.deepcopy(row)
            shared['config'].update(polyformer_mode=mode, members=4)
            self.assertEqual(fit_semantics(shared)['physical_slow_bodies'], 1)
        row['budget']['unit'] = 'epochs'
        with self.assertRaises(ValueError):
            fit_semantics(row)

    def test_only_source_bound_independent_streams_are_allowed_to_differ(self):
        value = self.fixture()
        self.assertTrue(bank_roster(value)[0]['genuine_I4'])
        for key in ('learning_rate', 'loss'):
            invalid = copy.deepcopy(value)
            invalid['runs'][1]['config'][key] = 'different'
            with self.assertRaisesRegex(ValueError, 'learning/data/selection'):
                bank_roster(invalid)
        for key in ('budget', 'selection', 'adapter'):
            invalid = copy.deepcopy(value)
            if key == 'budget':
                invalid['runs'][1][key]['maximum'] += 1
            elif key == 'selection':
                invalid['runs'][1][key]['direction'] = 'min'
            else:
                invalid['runs'][1][key] = 'different'
            with self.assertRaisesRegex(ValueError, 'method/source/budget/selector'):
                bank_roster(invalid)
        invalid = copy.deepcopy(value)
        invalid['runs'][1]['config']['dropout_seeds'] = invalid['runs'][0]['config']['dropout_seeds']
        with self.assertRaisesRegex(ValueError, 'dropout seeds'):
            bank_roster(invalid)
        invalid = copy.deepcopy(value)
        invalid['runs'][1]['config']['factor_seed'] = invalid['runs'][0]['config']['factor_seed']
        with self.assertRaisesRegex(ValueError, 'factor seeds'):
            bank_roster(invalid)

    def test_unknown_families_cannot_acquire_generic_dropout_seed_exemption(self):
        value = self.fixture()
        for row in value['runs'][:4]:
            row['config'].pop('polyformer_mode')
            row['config']['architecture'] = 'factorized_single'
        with self.assertRaisesRegex(ValueError, 'learning/data/selection'):
            bank_roster(value)

    def test_link_minibatch_calls_do_not_become_coordinate_cycles(self):
        row = self.fixture()['runs'][0]
        row['budget'].update(unit='epochs', maximum=100)
        row['config'].update(link_inputs={'train': 'unopened'}, source_arm='original_boundary',
                             source_optimizer_count=1, train_minibatches_per_epoch=17)
        semantics = fit_semantics(row)
        self.assertEqual(semantics['known_optimizer_step_calls'], 1700)
        self.assertEqual(semantics['optimizer_objects'], 1)
        self.assertEqual(semantics['physical_slow_bodies'], 1)
        row['config'].pop('train_minibatches_per_epoch')
        self.assertIsNone(fit_semantics(row)['known_optimizer_step_calls'])


if __name__ == '__main__':
    unittest.main()
