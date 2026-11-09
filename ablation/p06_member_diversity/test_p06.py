"""CPU checks for the member diversity study.

    python -m unittest ablation.p06_member_diversity.test_p06 -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.resumable import JobStore  # noqa: E402
from ablation.common.tabm_trainer import (  # noqa: E402
    TrainSettings, evaluate, member_logits, train_resumable,
)
from ablation.p05_hparam_grid import grid  # noqa: E402
from ablation.p06_member_diversity import analyze  # noqa: E402
from ablation.p06_member_diversity.models import build_model  # noqa: E402
from ablation.p06_member_diversity.select_depths import select_cell  # noqa: E402


DEVICE = torch.device('cpu')
INPUTS = 12
CLASSES = 3
NODES = 80
STEPS = 30
LR = 1e-3


def toy_bundle():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(NODES, INPUTS, generator=generator)
    y = x[:, :CLASSES].argmax(dim=1)
    edge_index = torch.randint(0, NODES, (2, 500), generator=generator)
    data = Data(x=x, y=y, edge_index=edge_index)
    train = torch.zeros(NODES, 10, dtype=torch.bool)
    val = train.clone()
    test = train.clone()
    for split in range(10):
        order = torch.randperm(NODES, generator=generator)
        train[order[:40], split] = True
        val[order[40:60], split] = True
        test[order[60:], split] = True
    return data, train, val, test, CLASSES, CLASSES, False


def train_job(root, method, num_layers=2):
    bundle = toy_bundle()
    store = JobStore(Path(root) / 'roman-empire' / method / 'SAGE' / f'L{num_layers}' / 'split0')
    members = grid.members_for(method)
    settings = TrainSettings(dataset='roman-empire', split=0, seed=grid.seed_for(method, 0),
                             members=members, lr=LR, num_steps=STEPS, patience=300)
    result = train_resumable(
        settings, store, (bundle[0], bundle[1], bundle[2], bundle[3], False), DEVICE,
        lambda: build_model(method, 'SAGE', num_layers, INPUTS, 512, CLASSES, members, DEVICE),
        model_path=store.dir / 'model.pt',
    )
    result.update(dataset='roman-empire', method=method, model='SAGE', num_layers=num_layers,
                  split=0, members=members)
    store.write_done(result)
    return store, result


class SelectionTests(unittest.TestCase):
    def frame(self, rows):
        return pd.DataFrame(rows, columns=['split', 'num_layers', 'val_metric', 'test_metric'])

    def test_uses_validation_of_split_zero_and_the_lowest_depth_on_ties(self):
        frame = self.frame([(0, 1, 0.5, 0.9), (0, 2, 0.7, 0.1), (0, 3, 0.7, 0.2), (1, 1, 0.99, 0.9)])
        cell = select_cell(frame)
        self.assertEqual(cell['num_layers'], 2)
        self.assertEqual(cell['source'], 'split0_validation')
        self.assertEqual(cell['archive_test_metric'], 0.1)

    def test_falls_back_to_other_splits_and_marks_the_row(self):
        frame = self.frame([(1, 1, 0.5, 0.0), (2, 1, 0.5, 0.0), (1, 2, 0.8, 0.0), (2, 2, 0.8, 0.0)])
        cell = select_cell(frame)
        self.assertEqual(cell['num_layers'], 2)
        self.assertEqual(cell['source'], 'mean_validation_other_splits')
        self.assertNotEqual(cell['archive_test_metric'], cell['archive_test_metric'])

    def test_frozen_file_covers_every_cell(self):
        selected = pd.read_csv(Path(analyze.SELECTED_PATH))
        self.assertEqual(len(selected), 5 * 7 * 2)
        self.assertFalse(selected.duplicated(['dataset', 'model', 'method']).any())
        self.assertTrue(selected.num_layers.between(1, 5).all())


class MetricTests(unittest.TestCase):
    def test_hand_computed_multiclass_example(self):
        y = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor([
            [[2., 0.], [0., 2.], [0., 2.], [0., 2.]],
            [[2., 0.], [2., 0.], [2., 0.], [0., 2.]],
        ])
        metrics = analyze.prediction_metrics(logits, y, False, 'roman-empire')
        self.assertAlmostEqual(metrics['pair_disagreement'], 0.5)
        self.assertAlmostEqual(metrics['error_jaccard'], 0.0)
        self.assertAlmostEqual(metrics['double_fault'], 0.0)
        self.assertAlmostEqual(metrics['at_least_one_correct'], 1.0)
        self.assertAlmostEqual(metrics['mean_member_metric'], 0.75)

    def test_identical_members_have_no_diversity(self):
        y = torch.tensor([0, 1, 2, 1, 0, 2])
        logits = torch.randn(6, 3, generator=torch.Generator().manual_seed(1)).repeat(4, 1, 1)
        metrics = analyze.prediction_metrics(logits, y, False, 'roman-empire')
        self.assertAlmostEqual(metrics['pair_disagreement'], 0.0)
        self.assertAlmostEqual(metrics['js_divergence'], 0.0, places=6)
        self.assertAlmostEqual(metrics['probability_correlation'], 1.0, places=6)
        self.assertAlmostEqual(metrics['ensemble_gain'], 0.0, places=6)

    def test_binary_logits_without_a_class_dimension(self):
        y = torch.tensor([0., 1., 0., 1.])
        logits = torch.tensor([[-1., 1., 1., 1.], [-1., -1., -1., 2.]])
        metrics = analyze.prediction_metrics(logits, y, True, 'tolokers')
        self.assertAlmostEqual(metrics['pair_disagreement'], 0.5)
        self.assertAlmostEqual(metrics['double_fault'], 0.0)
        self.assertGreater(metrics['js_divergence'], 0.0)

    def test_cka_is_invariant_to_rotation_and_scale(self):
        generator = torch.Generator().manual_seed(2)
        first = torch.randn(200, 8, generator=generator)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
        self.assertAlmostEqual(analyze.linear_cka(first, first @ rotation * 3.0), 1.0, places=6)
        other = torch.randn(200, 8, generator=generator)
        self.assertLess(analyze.linear_cka(first, other), 0.2)

    def test_parameter_similarity_of_identical_factors(self):
        state = {}
        for prefix in ('input_be_block', 'output_be_block'):
            state[f'{prefix}.R'] = torch.ones(3, 4) * 2
            state[f'{prefix}.S'] = torch.ones(3, 5)
            state[f'{prefix}.B'] = torch.ones(3, 5) * 0.5
            state[f'{prefix}.W.weight'] = torch.randn(5, 4)
        rows = pd.DataFrame(analyze.parameter_rows(state))
        self.assertEqual(len(rows), 8)
        self.assertTrue(((rows.mean_pair_cosine - 1).abs() < 1e-6).all())
        self.assertTrue((rows.relative_spread.abs() < 1e-6).all())

    def test_effective_matrix_uses_member_factors(self):
        state = {}
        for prefix in ('input_be_block', 'output_be_block'):
            state[f'{prefix}.R'] = torch.tensor([[1., 1.], [1., -1.]])
            state[f'{prefix}.S'] = torch.ones(2, 3)
            state[f'{prefix}.B'] = torch.zeros(2, 3)
            state[f'{prefix}.W.weight'] = torch.ones(3, 2)
        rows = pd.DataFrame(analyze.parameter_rows(state)).set_index(['block', 'factor'])
        self.assertAlmostEqual(rows.loc[('input', 'effective'), 'mean_pair_cosine'], 0.0, places=6)


class CaptureTests(unittest.TestCase):
    def test_stages_match_the_network_and_leave_it_unchanged(self):
        data = toy_bundle()[0]
        mask = torch.ones(NODES, dtype=torch.bool)
        for method, members in (('gnnm', 4), ('ens_m0', 1)):
            for name, blocks in (('SAGE', 3), ('GT', 4)):
                layers = 3 if name == 'SAGE' else 2
                model = build_model(method, name, layers, INPUTS, 64, CLASSES, members, DEVICE)
                logits, stages = analyze.capture_member(model, data, 0, mask)
                self.assertEqual(list(stages)[0], 'input')
                self.assertEqual(list(stages)[-1], 'final')
                self.assertEqual(len(stages), blocks + 2)
                model.eval()
                self.assertTrue(torch.equal(logits, model(data, data.x, tabm_seed=0)))
                inner = getattr(model, 'model', model)
                self.assertFalse(inner.act._forward_hooks)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_saved_model_reproduces_the_reported_test_score(self):
        store, result = train_job(self.root, 'gnnm')
        bundle = toy_bundle()
        data = bundle[0]
        data.test_mask = bundle[3][:, 0]
        model = build_model('gnnm', 'SAGE', 2, INPUTS, 512, CLASSES, 4, DEVICE)
        model.load_state_dict(torch.load(store.dir / 'model.pt', weights_only=True))
        reloaded = evaluate(member_logits(model, data, 4), data, data.test_mask, False, 'roman-empire')
        self.assertEqual(reloaded['metric'], result['test_metric'])

    def test_analysis_of_trained_gnnm_and_ens(self):
        train_job(self.root, 'gnnm')
        for k in range(grid.ENSEMBLE_MEMBERS):
            train_job(self.root, f'ens_m{k}')
        runs = pd.DataFrame([
            {**json.loads(path.read_text()), 'job_dir': str(path.parent)}
            for path in sorted(self.root.rglob('done.json'))
        ])
        frame, stages, parameters = analyze.analyse_runs(runs, lambda dataset: toy_bundle(), DEVICE)
        self.assertEqual(sorted(frame.method), ['ens', 'gnnm'])
        self.assertTrue((frame.recheck_abs_delta < 1e-9).all())
        self.assertEqual(set(stages.stage), {'input', 'block_1', 'block_2', 'final'})
        self.assertEqual(len(parameters), 8)
        self.assertEqual(set(parameters.method), {'gnnm'})
        merged = analyze.compare(frame)
        self.assertEqual(len(merged), 1)
        summary = analyze.summarise(merged)
        self.assertEqual(summary['cells'], 1)
        self.assertAlmostEqual(
            merged.pair_disagreement_gap[0],
            merged.pair_disagreement_gnnm[0] - merged.pair_disagreement_ens[0])

    def test_incomplete_ensembles_are_reported_and_skipped(self):
        train_job(self.root, 'gnnm')
        train_job(self.root, 'ens_m0')
        runs = pd.DataFrame([
            {**json.loads(path.read_text()), 'job_dir': str(path.parent)}
            for path in sorted(self.root.rglob('done.json'))
        ])
        complete, incomplete = analyze.group_runs(runs)
        self.assertEqual([method for method, _ in complete], ['gnnm'])
        self.assertEqual(len(incomplete), 1)


if __name__ == '__main__':
    unittest.main()
