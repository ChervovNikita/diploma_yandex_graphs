"""Focused regressions for native parity, exact resume and ensemble scoring."""

import copy
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.utils import to_undirected

from ablation.common.resumable import JobStore, atomic_torch_save
from ablation.common.worker import Job, Preempted
from .collect import collect_cohort, collect_groups, main as collect_main
from .data import bounded_hash_tables, tensor_identity
from .models import BUDDY, GNNMBuddy
from .protocol import REPOSITORY, namespace
from .run import evaluate_scores, make_jobs, parse_args, train_job
from src.hashing import ElphHashes


class Stop:
    requested = False


class InterruptStore(JobStore):
    def __init__(self, path, stop):
        super().__init__(path)
        self.stop = stop

    def heartbeat(self):
        self.stop.requested = True
        super().heartbeat()


class BuddyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        cls.root = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_original_boundary_identity_matches_native(self):
        torch.manual_seed(19)
        native = BUDDY(namespace(), 7).eval()
        torch.manual_seed(19)
        routed = GNNMBuddy(7).eval()
        with torch.no_grad():
            for block in (routed.label_lin_layer, routed.lin_feat, routed.lin):
                block.R.fill_(1)
                block.S.fill_(1)
        inputs = (torch.rand(11, 8), torch.rand(11, 2, 7), torch.arange(11).float(), torch.ones(11))
        expected = native(*inputs)
        for route in range(4):
            torch.testing.assert_close(routed(*inputs, route=route), expected, rtol=0, atol=0)
        torch.testing.assert_close(native.lin_out.weight, routed.lin_out.weight, rtol=0, atol=0)

    def test_bounded_hashing_is_exact_native_hashing(self):
        edges = to_undirected(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 1], [1, 2, 3, 4, 5, 6, 7, 6]]))
        hashes = ElphHashes(namespace())
        expected, expected_cards = hashes.build_hash_tables(9, edges)
        actual, actual_cards = bounded_hash_tables(hashes, 9, edges, 3, lambda: None)
        torch.testing.assert_close(actual_cards, expected_cards, rtol=0, atol=0)
        for hop in expected:
            for name in expected[hop]:
                torch.testing.assert_close(actual[hop][name], expected[hop][name], rtol=0, atol=0)
        links = torch.tensor([[0, 1], [1, 7], [0, 8], [8, 8]])
        torch.testing.assert_close(hashes.get_subgraph_features(links, actual, actual_cards, 2),
                                   hashes.get_subgraph_features(links, expected, expected_cards, 4),
                                   rtol=0, atol=0)

    def fixture(self):
        torch.manual_seed(31)
        x = torch.randn(20, 3)
        degree = torch.ones(20)
        result = {}
        for split, count, positive in (('train', 128, 64), ('valid', 80, 20)):
            links = torch.randint(0, 20, (count, 2))
            result[split] = {'x': x, 'degrees': degree, 'links': links,
                             'subgraph_features': torch.rand(count, 8), 'num_positive': positive,
                             'edge_order_sha256': tensor_identity(links)}
        return result

    def test_resume_matches_uninterrupted_model_and_predictions(self):
        args = parse_args(['--root', str(self.root / 'resume'), '--smoke', '--smoke_epochs', '2',
                           '--cohorts', '0', '--device', 'cpu'])
        config = make_jobs(args, 'test_source', 'test_data')[4].config
        job = Job('gnnm', config)
        bundle = self.fixture()
        direct = JobStore(args.root / 'direct')
        direct.ensure_config(config)
        expected = train_job(job, direct, Stop(), args, bundle, self.root)
        stopped = Stop()
        interrupted = InterruptStore(args.root / 'interrupted', stopped)
        interrupted.ensure_config(config)
        with self.assertRaises(Preempted):
            train_job(job, interrupted, stopped, args, bundle, self.root)
        continued = JobStore(interrupted.dir)
        actual = train_job(job, continued, Stop(), args, bundle, self.root)
        self.assertEqual(actual['best_epoch'], expected['best_epoch'])
        self.assertEqual(actual['val_metric'], expected['val_metric'])
        self.assertEqual(actual['global_step'], expected['global_step'])
        self.assertEqual(actual['resumes'], 1)
        expected_model = torch.load(direct.dir / 'best_model.pt', weights_only=False)
        actual_model = torch.load(continued.dir / 'best_model.pt', weights_only=False)
        for name in expected_model:
            torch.testing.assert_close(actual_model[name], expected_model[name], rtol=0, atol=0)
        expected_predictions = torch.load(direct.dir / 'predictions.pt', weights_only=False)
        actual_predictions = torch.load(continued.dir / 'predictions.pt', weights_only=False)
        torch.testing.assert_close(actual_predictions['valid']['scores'], expected_predictions['valid']['scores'], rtol=0, atol=0)
        self.assertNotIn('test', actual_predictions)

    def write_cohort(self, root, cohort=0, changes=None, include_gnnm=True):
        args = parse_args(['--root', str(root), '--smoke', '--cohorts', str(cohort)])
        jobs = make_jobs(args, 'source', 'data')
        selected = jobs if include_gnnm else jobs[:4]
        scores = []
        for position, job in enumerate(selected):
            config = copy.deepcopy(job.config)
            config.update(changes or {})
            store = JobStore(root / job.key)
            store.ensure_config(config)
            prediction = torch.arange(80).float().roll(position * 13)
            scores.append(prediction)
            atomic_torch_save({'config': config, 'valid': {
                'scores': prediction, 'num_positive': 20, 'edge_order_sha256': 'same'}}, store.dir / 'predictions.pt')
            store.write_done(dict(config, best_epoch=position + 1, train_seconds=1.0, num_params=1,
                                  val_metric=evaluate_scores(prediction, 20)))
        return scores

    def test_ensemble_averages_selected_scores_before_official_metric(self):
        root = self.root / 'ensemble'
        self.assertIsNone(collect_cohort(root, 0))
        scores = self.write_cohort(root, include_gnnm=False)
        self.assertIsNone(collect_cohort(root, 0))
        result = collect_cohort(root, 0, include_smoke=True)['ENS4']
        self.assertEqual(result['member_best_epochs'], [1, 2, 3, 4])
        self.assertEqual(result['val_metric'], evaluate_scores(torch.stack(scores).mean(0), 20))
        self.assertFalse(result['test_evaluated'])
        complete, incomplete = collect_groups(root, [0, 1], include_smoke=True)
        self.assertEqual(complete, [])
        self.assertEqual(incomplete[0]['missing'], ['gnnm'])
        self.assertEqual(incomplete[1]['missing'], ['native0', 'native1', 'native2', 'native3', 'gnnm'])

    def test_copied_native_zero_artifacts_are_rejected(self):
        for member in (1, 2, 3):
            for copy_config in (False, True):
                with self.subTest(member=member, copy_config=copy_config):
                    root = self.root / f'copied_{member}_{copy_config}'
                    self.write_cohort(root)
                    first = root / 'cohort0/native0'
                    target = root / f'cohort0/native{member}'
                    files = ['done.json', 'predictions.pt']
                    if copy_config:
                        files.append('config.json')
                    for name in files:
                        shutil.copy2(first / name, target / name)
                    with self.assertRaisesRegex(RuntimeError, 'mismatch|Invalid'):
                        collect_cohort(root, 0, include_smoke=True)

    def test_collector_out_cli_keeps_fit_artifacts_at_input_root(self):
        root = self.root / 'cli_fits'
        output = self.root / 'cli_collection'
        self.write_cohort(root)
        before = {str(path.relative_to(root)): path.read_bytes()
                  for path in root.rglob('*') if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            status = collect_main(['--root', str(root), '--out', str(output),
                                   '--cohorts', '0', '--include-smoke'])
        self.assertEqual(status, 0)
        report = json.loads((output / 'results.json').read_text())
        self.assertEqual(report, json.loads(stream.getvalue()))
        self.assertEqual(report['complete_cohorts'], 1)
        self.assertEqual(report['incomplete_cohorts'], [])
        self.assertTrue((output / 'cohort0/ensemble/predictions.pt').exists())
        after = {str(path.relative_to(root)): path.read_bytes()
                 for path in root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        with contextlib.redirect_stdout(io.StringIO()):
            status = collect_main(['--root', str(root), '--cohorts', '0', '--include-smoke'])
        self.assertEqual(status, 0)
        self.assertTrue((root / 'cohort0/ensemble/predictions.pt').exists())

    def test_cross_cohort_bindings_cannot_be_mixed(self):
        changes = {
            'source': {'source_sha256': 'different_source'},
            'data': {'dataset_sha256': 'different_data'},
            'protocol': {'recipe': {'changed': True}},
            'runtime': {'dependencies': {'torch': 'different_version'}},
        }
        for name, change in changes.items():
            with self.subTest(binding=name):
                root = self.root / f'mixed_{name}'
                self.write_cohort(root, cohort=0)
                self.write_cohort(root, cohort=1, changes=change)
                with self.assertRaisesRegex(RuntimeError, 'cross-cohort'):
                    collect_groups(root, [0, 1], include_smoke=True)
        root = self.root / 'matched_cohorts'
        self.write_cohort(root, cohort=0)
        self.write_cohort(root, cohort=1)
        complete, incomplete = collect_groups(root, [0, 1], include_smoke=True)
        self.assertEqual([row['cohort'] for row in complete], [0, 1])
        self.assertEqual(incomplete, [])


if __name__ == '__main__':
    unittest.main()
