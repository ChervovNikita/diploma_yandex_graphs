"""Pure source gate regression: legal native AMP skip stays narrow."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
from engine import _validate_spec, _training_counts


def spec():
    return {'fit_id': 'synthetic-gate', 'budget_steps': 1000, 'budget_unit': 'epochs',
        'validation_interval': 1, 'validate_initial': False, 'prefixes': [], 'inherited_fit_ids': [],
        'adapter_spec': {'run': {'adapter': 'owned/runtime/adapters/sehgnn.py:make_adapter',
            'config': {'sehgnn_arm': 'native_single', 'sehgnn_epochs': 1000,
                'source_optimizer_count': 1, 'native_AMP_overflow_skip_allowed': True,
                'historical_state_inherited': False, 'backbone': 'native_SeHGNN',
                'width': 512, 'classes': 5, 'feature_channels': 25, 'label_channels': 12}}},
        'native_amp_cycle_policy': {'schema': 'native-SeHGNN-AMP-complete-cycle-v1', 'optimizer_owner_attempts': 1}}


def counts(skipped):
    return {'optimizer_updates': 1 - skipped, 'source_optimizer_attempts': 1,
        'source_AMP_skipped_optimizer_maps': skipped, 'source_native_complete_cycles': 1,
        'source_native_AMP_enabled': 1, 'model_forwards': 1, 'backward_calls': 1,
        'supervised_examples': 4, 'accepted_private_corrections': 0}


class Gates(unittest.TestCase):
    def test_legal_source_skipped_and_finite_cycles(self):
        s = _validate_spec(spec())
        for skipped in (0, 1):
            self.assertEqual(_training_counts(counts(skipped), s), counts(skipped))

    def test_ordinary_success_still_requires_positive_maps(self):
        s = spec(); s.pop('native_amp_cycle_policy')
        self.assertEqual(_training_counts(counts(0), s)['optimizer_updates'], 1)
        with self.assertRaises(ValueError): _training_counts(counts(1), s)

    def test_incomplete_or_contradictory_cycles_rejected(self):
        for field, value in [('source_optimizer_attempts', 0), ('source_AMP_skipped_optimizer_maps', 0),
                ('source_native_complete_cycles', 0), ('source_native_AMP_enabled', 0),
                ('model_forwards', 0), ('backward_calls', 0), ('supervised_examples', 0),
                ('accepted_private_corrections', 1), ('optimizer_updates', -1)]:
            c = counts(1); c[field] = value
            with self.assertRaises(ValueError, msg=field): _training_counts(c, spec())
        c = counts(1); c.pop('source_AMP_skipped_optimizer_maps')
        with self.assertRaises(ValueError): _training_counts(c, spec())

    def test_source_geometry_and_FP32_arms_cannot_opt_in(self):
        for field, value in [('sehgnn_arm', 'ADD_PS'), ('source_optimizer_count', 4),
                ('native_AMP_overflow_skip_allowed', False), ('historical_state_inherited', True),
                ('backbone', 'other'), ('width', 256), ('classes', 3),
                ('feature_channels', 24), ('label_channels', 11), ('sehgnn_epochs', 200)]:
            s = spec(); s['adapter_spec']['run']['config'][field] = value
            with self.assertRaises(ValueError, msg=field): _validate_spec(s)

    def test_factory_budget_and_policy_cannot_opt_in(self):
        for key, value in [('budget_unit', 'optimizer_updates'), ('budget_steps', 200),
                ('native_amp_cycle_policy', {'schema': 'generic-AMP', 'optimizer_owner_attempts': 1})]:
            s = spec(); s[key] = value
            with self.assertRaises(ValueError, msg=key): _validate_spec(s)
        s = spec(); s['adapter_spec']['run']['adapter'] = 'owned/adapters/node.py:make_adapter'
        with self.assertRaises(ValueError): _validate_spec(s)


if __name__ == '__main__': unittest.main()
