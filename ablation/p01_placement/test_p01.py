"""CPU checks for the placement study.

    python -m unittest ablation.p01_placement.test_p01 -v
"""

import os
import tempfile
import time
import types
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from ablation.common.test_support import import_datasets

import_datasets()

import run_tabm  # noqa: E402
from ablation.common.resumable import JobStore  # noqa: E402
from ablation.common.study import PAPER_MODELS  # noqa: E402
from ablation.common.tabm_trainer import TrainSettings, train_resumable  # noqa: E402
from ablation.common.worker import Job, run_worker  # noqa: E402
from ablation.p01_placement.models import LINEAR_TYPES, VARIANTS, build_model  # noqa: E402
from run_common import set_seed  # noqa: E402


DEVICE = torch.device('cpu')
HIDDEN = 64
CLASSES = 3


def toy_bundle(nodes=80):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(nodes, 12, generator=generator)
    y = (x[:, :3].argmax(dim=1)).long()
    source = torch.randint(0, nodes, (400,), generator=generator)
    target = torch.randint(0, nodes, (400,), generator=generator)
    edge_index = torch.stack([torch.cat([source, target]), torch.cat([target, source])])
    data = Data(x=x, y=y, edge_index=edge_index)
    train = torch.zeros(nodes, 10, dtype=torch.bool)
    val = train.clone()
    test = train.clone()
    for split in range(10):
        order = torch.randperm(nodes, generator=generator)
        train[order[:40], split] = True
        val[order[40:60], split] = True
        test[order[60:], split] = True
    return data, train, val, test, False


def make(variant, model='SAGE', layers=2):
    return build_model(variant, model, layers, 12, HIDDEN, CLASSES, VARIANTS[variant], DEVICE)


def settings(num_steps=30, save_every_steps=None, members=4):
    return TrainSettings(dataset='roman-empire', split=0, seed=0, members=members, lr=1e-3,
                         num_steps=num_steps, patience=300,
                         save_every_sec=1e9, save_every_steps=save_every_steps)


def train(store, variant='gnnm', model='SAGE', save_every_steps=None, on_step=None, num_steps=30):
    return train_resumable(
        settings(num_steps, save_every_steps, VARIANTS[variant]), store, toy_bundle(),
        DEVICE, lambda: make(variant, model), on_step=on_step,
    )


class Interrupt(Exception):
    pass


class TemporaryStoreTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)


class VariantTests(unittest.TestCase):
    def member_outputs(self, model, members=4):
        data = toy_bundle()[0]
        model.eval()
        with torch.no_grad():
            return [model(data, data.x, tabm_seed=i) for i in range(members)]

    def test_all_layer_starts_exactly_as_gnnm_for_every_backbone(self):
        for name in PAPER_MODELS:
            with self.subTest(model=name):
                set_seed(0)
                gnnm = make('gnnm', name)
                set_seed(0)
                all_layer = make('all_layer_be', name)
                for first, second in zip(self.member_outputs(gnnm), self.member_outputs(all_layer)):
                    self.assertTrue(torch.equal(first, second))

    def test_no_bare_linear_layer_remains_inside_residual_blocks(self):
        for name in PAPER_MODELS:
            with self.subTest(model=name):
                model = make('all_layer_be', name)
                self.assertGreater(len(model.factor_layers), 0)
                wrapped = {id(layer.shared) for layer in model.factor_layers}
                for module in model.base.residual_modules.modules():
                    if isinstance(module, LINEAR_TYPES):
                        self.assertIn(id(module), wrapped)

    def test_members_differ_at_start_where_factors_have_signs(self):
        for variant in ('gnnm', 'input_only', 'output_only_signs', 'all_layer_be'):
            with self.subTest(variant=variant):
                outputs = self.member_outputs(make(variant))
                self.assertFalse(torch.equal(outputs[0], outputs[1]))

    def test_variants_share_only_the_intended_projectors(self):
        input_only = make('input_only')
        self.assertEqual(input_only.input_be_block.R.shape[0], 4)
        self.assertNotIn('output_be_block.R', dict(input_only.named_parameters()))
        output_only = make('output_only_signs')
        self.assertNotIn('input_be_block.R', dict(output_only.named_parameters()))
        self.assertEqual(set(output_only.output_be_block.R.unique().tolist()), {-1.0, 1.0})
        self.assertEqual(make('gnnm_m1').input_be_block.R.shape[0], 1)

    def test_every_variant_and_backbone_trains(self):
        for name in PAPER_MODELS:
            for variant in ('input_only', 'output_only_signs', 'all_layer_be', 'gnnm_m1'):
                with self.subTest(model=name, variant=variant):
                    model = make(variant, name)
                    data = toy_bundle()[0]
                    model(data, data.x, tabm_seed=0).sum().backward()
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in model.parameters()))


class TrainerTests(TemporaryStoreTest):
    def test_matches_original_run_tabm(self):
        data, train_masks, val_masks, test_masks, _ = toy_bundle()
        original = run_tabm.train_single(
            'SAGE', 2, 0, data, train_masks, val_masks, test_masks, CLASSES, False, DEVICE,
            str(self.root), 'roman-empire', HIDDEN, 1e-3, 40,
        )
        ours = train_resumable(
            TrainSettings(dataset='roman-empire', split=0, seed=0, members=4, lr=1e-3,
                          num_steps=40, patience=300),
            JobStore(self.root / 'job'), toy_bundle(), DEVICE, lambda: make('gnnm'),
        )
        self.assertEqual(original['best_step'], ours['best_step'])
        self.assertEqual(original['val_metric'], ours['val_metric'])
        self.assertEqual(original['test_acc'], ours['test_acc'])
        self.assertEqual(original['test_loss'], ours['test_loss'])

    def resume_equals_continuous(self, variant, model):
        continuous = train(JobStore(self.root / f'{variant}{model}a'), variant, model)
        store = JobStore(self.root / f'{variant}{model}b')

        def interrupt(step):
            if step == 20:
                raise Interrupt()

        with self.assertRaises(Interrupt):
            train(store, variant, model, save_every_steps=7, on_step=interrupt)
        self.assertEqual(store.load_state()['progress']['step'], 14)
        resumed = train(store, variant, model, save_every_steps=7)
        self.assertEqual(resumed['resumes'], 1)
        for key in ('best_step', 'val_metric', 'test_acc', 'test_loss',
                    'test_pair_disagreement', 'stopped_step'):
            self.assertEqual(continuous[key], resumed[key], key)

    def test_resume_is_identical_after_a_crash(self):
        for variant, model in (('gnnm', 'SAGE'), ('all_layer_be', 'GAT'),
                               ('input_only', 'GT-sep')):
            with self.subTest(variant=variant, model=model):
                self.resume_equals_continuous(variant, model)

    def test_corrupted_state_falls_back_to_previous(self):
        store = JobStore(self.root / 'job')

        def interrupt(step):
            if step == 20:
                raise Interrupt()

        with self.assertRaises(Interrupt):
            train(store, save_every_steps=7, on_step=interrupt)
        store.state_path.write_bytes(b'truncated')
        self.assertEqual(store.load_state()['progress']['step'], 7)
        self.assertEqual(train(store, save_every_steps=7)['resumes'], 1)


class QueueTests(TemporaryStoreTest):
    def test_claim_is_exclusive_until_stale(self):
        first, second = JobStore(self.root / 'job'), JobStore(self.root / 'job')
        self.assertTrue(first.claim(600))
        self.assertFalse(second.claim(600))
        old = time.time() - 1000
        os.utime(first.claim_path, (old, old))
        self.assertTrue(second.claim(600))

    def test_restart_skips_done_and_retries_failed_up_to_the_limit(self):
        jobs = [Job('a', {'x': 1}), Job('b', {'x': 2})]
        calls = []

        def execute(job, store, stop):
            calls.append(job.key)
            if job.key == 'b':
                raise RuntimeError('boom')
            return {'ok': True}

        stop = types.SimpleNamespace(requested=False)
        self.assertEqual(run_worker(self.root, jobs, execute, max_failures=2, stop=stop), 1)
        self.assertEqual(calls, ['a', 'b', 'b'])
        self.assertEqual(run_worker(self.root, jobs, execute, max_failures=2, stop=stop), 1)
        self.assertEqual(calls, ['a', 'b', 'b'])

    def test_changed_protocol_is_refused(self):
        store = JobStore(self.root / 'job')
        store.ensure_config({'lr': 1})
        with self.assertRaises(ValueError):
            store.ensure_config({'lr': 2})


if __name__ == '__main__':
    unittest.main()
