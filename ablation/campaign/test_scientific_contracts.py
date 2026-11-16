"""Scientific-contract tests and CPU shape/backward checks for all node families.

These tiny graphs verify software contracts. They are excluded from scientific
results and cannot establish optimization quality or dataset-level accuracy.
"""

from itertools import product
import unittest

import torch

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.tabm_trainer import member_logits, train_step
from ablation.p01_placement.test_p01 import toy_bundle
from ablation.p05_hparam_grid.grid import configs_for
from ablation.p05_hparam_grid.selected import build_selected
from ablation.p03_rms_init.models import normalize_to_unit_rms
from ablation.campaign.plan import MODELS, VARIANTS, node_spec


class ScientificContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data, train, val, _, _ = toy_bundle(80)
        self.data.train_mask = train[:, 0]
        self.data.val_mask = val[:, 0]
        self.device = torch.device('cpu')

    def make(self, method, model='SAGE', depth=2, hp=None):
        spec = node_spec('roman-empire', model, depth, 0, method, hp)
        spec['hidden_dim'] = 64
        return build_selected(spec, self.data, 3, self.device)

    def test_every_variant_backbone_and_depth_has_finite_backward(self):
        for method, model, depth in product(['native0', 'gnnm', *VARIANTS],
                                             MODELS, range(1, 6)):
            with self.subTest(method=method, model=model, depth=depth):
                network, members = self.make(method, model, depth)
                optimizer = torch.optim.AdamW(network.parameters(), lr=3e-5)
                loss = train_step(network, self.data, optimizer, False, members)
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                logits = member_logits(network, self.data, members)
                self.assertEqual(tuple(logits.shape), (members, 80, 3))
                self.assertTrue(torch.isfinite(logits).all())

    def test_all_24_grid_configurations_build_for_each_backbone(self):
        for model, (name, hp) in product(MODELS, configs_for('combined').items()):
            with self.subTest(model=model, config=name):
                for method in ['native0', 'gnnm']:
                    network, members = self.make(method, model, 2, hp)
                    logits = member_logits(network, self.data, members)
                    self.assertEqual(tuple(logits.shape), (members, 80, 3))
                    self.assertTrue(torch.isfinite(logits).all())

    def test_output_identity_has_identical_initial_paths(self):
        for model, depth in product(MODELS, range(1, 6)):
            with self.subTest(model=model, depth=depth):
                network, members = self.make('output_only_identity', model, depth)
                logits = member_logits(network, self.data, members)
                self.assertTrue(torch.equal(logits[0], logits[1]))

    def test_rms_normalization_supports_different_feature_widths(self):
        for width in [8, 32, 64, 128, 512]:
            factor = torch.empty(4, width)
            torch.nn.init.xavier_uniform_(factor)
            normalize_to_unit_rms(factor, per_member=True)
            self.assertTrue(torch.allclose(factor.square().mean(dim=1),
                                           torch.ones(4), atol=1e-6))
            network, _ = self.make('xavier_row_rms')
            for block in [network.input_be_block, network.output_be_block]:
                self.assertTrue(torch.allclose(block.R.square().mean(dim=1),
                                               torch.ones(4), atol=1e-6))

    def test_member_zero_reuse_has_same_seed_and_architecture(self):
        for split in range(10):
            native = node_spec('roman-empire', 'GAT', 2, split, 'native0')
            self.assertEqual(native['members'], 1)
            self.assertEqual(native['method'], 'base')
            self.assertEqual(native['seed'], 1000 * split)
            seeds = [node_spec('roman-empire', 'GAT', 2, split, f'native{i}')['seed']
                     for i in range(4)]
            self.assertEqual(len(set(seeds)), 4)


if __name__ == '__main__':
    unittest.main()
