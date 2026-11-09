"""CPU checks for the ablation studies.

    python -m unittest ablation.p03_rms_init.test_p02_p03 -v
"""

import unittest

import torch
from torch_geometric.data import Data

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.study import PAPER_MODELS  # noqa: E402
from ablation.p02_full_ablations import models as ablations  # noqa: E402
from ablation.p03_rms_init import models as rms  # noqa: E402
from models import TABMModel  # noqa: E402
from run_common import set_seed  # noqa: E402


DEVICE = torch.device('cpu')
HIDDEN = 64


def graph():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(40, 12, generator=generator)
    edge_index = torch.randint(0, 40, (2, 200), generator=generator)
    return Data(x=x, edge_index=edge_index)


def member_outputs(model, members):
    data = graph()
    model.eval()
    with torch.no_grad():
        return [model(data, data.x, tabm_seed=i) for i in range(members)]


def main_table_model(model_name, members):
    return TABMModel(
        model_name=model_name, num_layers=2, input_dim=12, hidden_dim=HIDDEN, output_dim=3,
        hidden_dim_multiplier=1, num_heads=8, normalization='LayerNorm', dropout=0.2,
        tabm_inits=members, device=DEVICE,
    )


def build(module, variant, model_name='SAGE'):
    return module.build_model(variant, model_name, 2, 12, HIDDEN, 3, module.VARIANTS[variant], DEVICE)


class AblationModelTests(unittest.TestCase):
    def test_m4_is_exactly_the_main_table_model(self):
        for name in PAPER_MODELS:
            with self.subTest(model=name):
                set_seed(0)
                first = main_table_model(name, 4)
                set_seed(0)
                second = build(ablations, 'm4', name)
                for ours, main in zip(member_outputs(second, 4), member_outputs(first, 4)):
                    self.assertTrue(torch.equal(ours, main))

    def test_member_counts(self):
        for variant, members in ablations.VARIANTS.items():
            with self.subTest(variant=variant):
                model = build(ablations, variant)
                self.assertEqual(model.input_be_block.R.shape[0], members)
                self.assertEqual(model.output_be_block.R.shape[0], members)

    def test_m4_stays_out_of_the_default_matrix(self):
        self.assertNotIn('m4', ablations.DEFAULT_VARIANTS)


class RmsInitializationTests(unittest.TestCase):
    def test_both_factors_have_unit_rms_and_both_signs(self):
        model = build(rms, 'xavier_rms')
        for factor in (model.input_be_block.R, model.output_be_block.R):
            self.assertAlmostEqual(factor.pow(2).mean().sqrt().item(), 1.0, places=5)
            self.assertLess(factor.min().item(), 0.0)
            self.assertGreater(factor.max().item(), 0.0)

    def test_rescaling_only_changes_the_scale_of_xavier_all(self):
        set_seed(0)
        raw = build(ablations, 'xavier_all')
        set_seed(0)
        normalized = build(rms, 'xavier_rms')
        for raw_block, new_block in ((raw.input_be_block, normalized.input_be_block),
                                     (raw.output_be_block, normalized.output_be_block)):
            scale = (new_block.R / raw_block.R)
            self.assertLess((scale.max() - scale.min()).item() / scale.mean().item(), 1e-4)
            self.assertGreater(scale.mean().item(), 1.0)
            self.assertTrue(torch.equal(new_block.W.weight, raw_block.W.weight))
            self.assertTrue((new_block.S == 1).all() and (new_block.B == 0).all())


if __name__ == '__main__':
    unittest.main()
