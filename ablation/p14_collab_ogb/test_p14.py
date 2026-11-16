"""CPU checks of the Collab GCN/GraphSAGE runner on a small synthetic graph.

    python -m unittest ablation.p14_collab_ogb.test_p14 -v
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ablation.common.resumable import JobStore
from ablation.common.worker import Job, Preempted
from ablation.p14_collab_ogb import run
from ablation.p14_collab_ogb.data import build_bundle
from ablation.p14_collab_ogb.models import build_model
from ablation.p14_collab_ogb.protocol import RECIPE

REPO = Path(__file__).resolve().parents[2]


def synthetic_split(nodes=60, seed=0):
    generator = torch.Generator().manual_seed(seed)
    edges = torch.randint(0, nodes, (300, 2), generator=generator)
    edges = edges[edges[:, 0] != edges[:, 1]]
    part = lambda a, b: edges[a:b]
    return torch.randn(nodes, 8, generator=generator), {
        'train': {'edge': part(0, 200), 'weight': torch.ones(200, 1), 'year': torch.full((200, 1), 2010)},
        'valid': {'edge': part(200, 240), 'weight': torch.ones(40, 1),
                  'edge_neg': torch.randint(0, nodes, (80, 2), generator=generator)},
        'test': {'edge': part(240, 280), 'weight': torch.ones(40, 1),
                 'edge_neg': torch.randint(0, nodes, (80, 2), generator=generator)},
    }


class Stop:
    def __init__(self, after=None):
        self.after, self.calls = after, 0

    @property
    def requested(self):
        self.calls += 1
        return self.after is not None and self.calls > self.after


class Store(JobStore):
    def heartbeat(self):
        pass


class CollabTests(unittest.TestCase):
    def setUp(self):
        self.saved_recipe = copy.deepcopy(RECIPE)
        RECIPE.update(batch_size=64, cap_schedule=[3, 6], extend_if_best_within=1)
        x, split = synthetic_split()
        self.bundle = build_bundle(x, split, with_test=True)
        self.args = SimpleNamespace(device='cpu', eval_batch_size=50)

    def tearDown(self):
        RECIPE.clear()
        RECIPE.update(self.saved_recipe)

    def config(self, backbone, variant, member=0):
        config = dict(study='test', backbone=backbone, variant=variant, cohort=0, seed=member,
                      smoke=0, recipe=dict(RECIPE), defer_test=True)
        if variant != 'native':
            config['members'] = 4
        return config

    def test_routes_with_identity_factors_equal_the_native_model(self):
        for backbone in ('gcn', 'sage'):
            torch.manual_seed(0)
            native = build_model(backbone, 'native', 8)
            gnnm = build_model(backbone, 'gnnm', 8)
            gnnm.load_state_dict({**gnnm.state_dict(), **{k: v for k, v in native.state_dict().items()
                                                          if not k.startswith('final')}})
            with torch.no_grad():
                gnnm.R.fill_(1)
                gnnm.final.W.weight.copy_(native.final.weight)
                gnnm.final.B.copy_(native.final.bias.expand(4, -1))
            links = self.bundle['valid']['links']
            graph = self.bundle['train_graph']
            expected = native.score(native.embed(self.bundle['x'], graph), links)
            for route in range(4):
                actual = gnnm.score(gnnm.embed(self.bundle['x'], graph, route), links, route)
                self.assertTrue(torch.allclose(expected, actual, atol=1e-5))

    def train(self, root, config, stop=None):
        store = Store(root)
        store.ensure_config(config)
        return run.train_job(Job('cohort0/x', config), store, stop or Stop(), self.args, self.bundle), store

    def test_every_method_trains_and_scores(self):
        for backbone in ('gcn', 'sage'):
            for variant in ('native', 'gnnm', 'gnnm_contrast'):
                with self.subTest(backbone=backbone, variant=variant), tempfile.TemporaryDirectory() as tmp:
                    config = self.config(backbone, variant)
                    result, store = self.train(Path(tmp) / 'fit', config)
                    self.assertGreaterEqual(result['stopped_epoch'], 3)
                    self.assertIn(result['stop_reason'], ('fixed_epochs', 'epoch_cap'))
                    store.write_done(result)
                    scorer = Store(Path(tmp) / 'scoring')
                    scored = run.score_selected(Job('cohort0/x', config), store.dir, scorer, self.args,
                                                self.bundle)
                    for key in ('val_metric', 'test_metric', 'test_hits@20', 'test_hits@100'):
                        self.assertIn(key, scored)
                    saved = torch.load(scorer.dir / 'predictions.pt', weights_only=False)
                    self.assertEqual(saved['valid']['member_scores'].shape[0], 1 if variant == 'native' else 4)

    def test_interrupted_training_resumes_to_the_same_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config('gcn', 'gnnm')
            straight, _ = self.train(Path(tmp) / 'a', config)
            with self.assertRaises(Preempted):
                self.train(Path(tmp) / 'b', config, Stop(after=5))
            resumed, _ = self.train(Path(tmp) / 'b', config)
            self.assertEqual(straight['best_epoch'], resumed['best_epoch'])
            self.assertAlmostEqual(straight['val_metric'], resumed['val_metric'], places=6)
            self.assertGreaterEqual(resumed['resumes'], 1)

    def test_limit_extends_only_when_the_best_epoch_is_at_the_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self.train(Path(tmp) / 'fit', self.config('sage', 'native'))
            if result['extensions']:
                self.assertEqual(result['stopped_epoch'], 6)
            else:
                self.assertEqual(result['stopped_epoch'], 3)

    def test_rescore_rebuilds_base_ensemble_and_shared_settings(self):
        from ablation.p14_collab_ogb import rescore
        links = {'edge_order_sha256': 'x', 'num_positive': 2}
        def artifact(variant, member, scores):
            config = dict(variant=variant, member=member, cohort=0)
            part = dict(links, scores=torch.tensor(scores))
            return {'config': config, 'valid': part, 'test': part}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'tasks/transfer/collab_buddy/cohort0'
            items = [('native', k, [1.0, 0.0, 0.5, 0.2]) for k in range(4)]
            items.append(('gnnm', None, [1.0, 0.9, 0.5, 0.2]))
            for variant, member, scores in items:
                name = f'native{member}' if variant == 'native' else variant
                path = base / name / 'output/training/scoring/cohort0' / name
                path.mkdir(parents=True)
                torch.save(artifact(variant, member, scores), path / 'predictions.pt')
            found = rescore.find_predictions(tmp, 'collab_buddy')
            rows = {(row['setting'], row['role']): row for row in rescore.cohort_rows(0, found[0])}
            self.assertEqual(set(setting for setting, _ in rows), {'BASE', 'ENS', 'gnnm'})
            self.assertAlmostEqual(rows[('gnnm', 'test')]['hits@20'], 100.0)
            self.assertAlmostEqual(rows[('BASE', 'test')]['hits@20'], 100.0)

    def test_hits_matches_a_direct_count(self):
        scores = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.5, 0.3])
        # Two positives (0.9, 0.2); negatives 0.8, 0.1, 0.5, 0.3. Hits@1: 0.9 > 0.8 only.
        self.assertAlmostEqual(run.hits(scores, 2, 1), 0.5)


if __name__ == '__main__':
    unittest.main()
