"""Paired activation-matching checks with shared weights held fixed."""

import copy
import unittest

import torch

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.p01_placement.test_p01 import toy_bundle
from ablation.p01_placement.models import build_model as original_model
from ablation.p03_rms_init.activation import build_model, calibrate_model
from ablation.p05_hparam_grid.selected import build_selected
from run_common import set_seed


class ActivationMatchingTests(unittest.TestCase):
    def setUp(self):
        self.data, train, _, _, _ = toy_bundle()
        self.data.train_mask = train[:, 0]
        self.device = torch.device('cpu')

    def make(self, variant, model='GAT', depth=2):
        set_seed(17)
        return build_model(variant, model, depth, 12, 64, 3, 4, self.device)

    def test_xavier_pair_preserves_original_shared_weights_and_rng(self):
        set_seed(17)
        original = original_model('gnnm', 'GAT', 2, 12, 64, 3, 4, self.device)
        reference_rng = torch.get_rng_state().clone()
        raw = self.make('xavier_same_weights')
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        for name, parameter in original.state_dict().items():
            if name not in {'input_be_block.R', 'output_be_block.R'}:
                self.assertTrue(torch.equal(parameter, raw.state_dict()[name]), name)

    def test_matching_preserves_directions_and_shared_weights_for_all_backbones(self):
        for backbone in ['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep']:
            for depth in [1, 3, 5]:
                with self.subTest(backbone=backbone, depth=depth):
                    raw = self.make('xavier_same_weights', backbone, depth)
                    calibrated = copy.deepcopy(raw)
                    before_rng = torch.get_rng_state().clone()
                    calibrate_model(calibrated, self.data, 4)
                    self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                    for name, parameter in raw.state_dict().items():
                        actual = calibrated.state_dict()[name]
                        if name in {'input_be_block.R', 'output_be_block.R'}:
                            self.assertTrue(torch.allclose(
                                torch.nn.functional.cosine_similarity(parameter, actual, dim=1),
                                torch.ones(4), atol=1e-6))
                        else:
                            self.assertTrue(torch.equal(parameter, actual), name)
                    for row in calibrated.initialization_calibration['members']:
                        for target, actual in [('target_input_rms', 'matched_input_rms'),
                                               ('target_logit_rms', 'matched_logit_rms')]:
                            self.assertAlmostEqual(row[target], row[actual], delta=1e-5)

    def test_calibration_uses_features_and_train_mask_without_labels(self):
        first = self.make('xavier_activation_rms')
        second = copy.deepcopy(first)
        altered = copy.deepcopy(self.data)
        altered.y = torch.full_like(self.data.y.float(), float('nan'))
        calibrate_model(first, self.data, 4)
        calibrate_model(second, altered, 4)
        for name, parameter in first.state_dict().items():
            self.assertTrue(torch.equal(parameter, second.state_dict()[name]), name)

    def test_checkpoint_builder_needs_explicit_calibration(self):
        config = dict(model_module='ablation.p03_rms_init.models',
                      variant='xavier_activation_rms', model='GAT', num_layers=2,
                      hidden_dim=64, members=4)
        uncalibrated, _ = build_selected(config, self.data, 3, self.device)
        self.assertFalse(hasattr(uncalibrated, 'initialization_calibration'))
        calibrated, _ = build_selected(config, self.data, 3, self.device, calibrate=True)
        self.assertTrue(hasattr(calibrated, 'initialization_calibration'))


if __name__ == '__main__':
    unittest.main()
