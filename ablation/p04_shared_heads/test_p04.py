"""CPU checks for the shared-backbone controls.

    python -m unittest ablation.p04_shared_heads.test_p04 -v
"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.resumable import JobStore  # noqa: E402
from ablation.common.study import PAPER_MODELS  # noqa: E402
from ablation.common.tabm_trainer import (  # noqa: E402
    TrainSettings, member_logits, train_resumable, train_step)
from ablation.p04_shared_heads.models import VARIANTS, build_model  # noqa: E402


DEVICE = torch.device('cpu')
HIDDEN = 64
CLASSES = 3
NODES = 60


def toy_bundle():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(NODES, 12, generator=generator)
    y = x[:, :3].argmax(dim=1)
    edge_index = torch.randint(0, NODES, (2, 400), generator=generator)
    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = torch.ones(NODES, dtype=torch.bool)
    train = torch.zeros(NODES, 10, dtype=torch.bool)
    val = train.clone()
    test = train.clone()
    for split in range(10):
        order = torch.randperm(NODES, generator=generator)
        train[order[:30], split] = True
        val[order[30:45], split] = True
        test[order[45:], split] = True
    return data, train, val, test, False


def make(variant, model='SAGE', output_dim=CLASSES):
    return build_model(variant, model, 2, 12, HIDDEN, output_dim, VARIANTS[variant], DEVICE)


class ModelTests(unittest.TestCase):
    def test_independent_projectors_have_one_matrix_per_member(self):
        model = make('independent_projectors')
        self.assertEqual(len(model.input_be_block.layers), 4)
        self.assertEqual(len(model.output_be_block.layers), 4)
        self.assertFalse(torch.equal(model.input_be_block.layers[0].weight,
                                     model.input_be_block.layers[1].weight))

    def test_heads_have_one_shared_input_and_one_head_per_member(self):
        model = make('heads_single_pass')
        names = dict(model.named_parameters())
        self.assertIn('base.input_be_block.weight', names)
        self.assertNotIn('base.input_be_block.R', names)
        self.assertEqual(len(model.base.output_be_block.layers), 4)

    def test_every_step_passes_through_the_backbone_only_once(self):
        model = make('heads_single_pass')
        calls = []
        handle = model.base.residual_modules[0].register_forward_hook(
            lambda *args: calls.append(1))
        data = toy_bundle()[0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_step(model, data, optimizer, False, 4)
        handle.remove()
        self.assertEqual(len(calls), 1)

    def test_all_heads_learn(self):
        model = make('heads_single_pass')
        data = toy_bundle()[0]
        before = [layer.weight.detach().clone() for layer in model.base.output_be_block.layers]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_step(model, data, optimizer, False, 4)
        for start, layer in zip(before, model.base.output_be_block.layers):
            self.assertFalse(torch.equal(start, layer.weight))

    def test_logit_shapes_for_multiclass_and_binary(self):
        data = toy_bundle()[0]
        self.assertEqual(member_logits(make('heads_single_pass'), data, 4).shape,
                         (4, NODES, CLASSES))
        self.assertEqual(member_logits(make('heads_single_pass', output_dim=1), data, 4).shape,
                         (4, NODES))

    def test_every_backbone_trains(self):
        data = toy_bundle()[0]
        for variant in VARIANTS:
            for name in PAPER_MODELS:
                with self.subTest(variant=variant, model=name):
                    model = make(variant, name)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    train_step(model, data, optimizer, False, 4)


class Interrupt(Exception):
    pass


class ResumeTests(unittest.TestCase):
    def test_resume_is_identical_for_both_variants(self):
        def run(store, variant, save_every_steps=None, on_step=None):
            settings = TrainSettings(dataset='roman-empire', split=0, seed=0, members=4, lr=1e-3,
                                     num_steps=30, patience=300, save_every_sec=1e9,
                                     save_every_steps=save_every_steps)
            return train_resumable(settings, store, toy_bundle(), DEVICE,
                                   lambda: make(variant), on_step=on_step)

        def interrupt(step):
            if step == 20:
                raise Interrupt()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    continuous = run(JobStore(root / f'{variant}a'), variant)
                    store = JobStore(root / f'{variant}b')
                    with self.assertRaises(Interrupt):
                        run(store, variant, 7, interrupt)
                    resumed = run(store, variant, 7)
                    for key in ('best_step', 'val_metric', 'test_acc', 'test_loss'):
                        self.assertEqual(continuous[key], resumed[key], key)


if __name__ == '__main__':
    unittest.main()
