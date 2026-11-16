"""CPU regressions for molecular boundary parity and minibatch recovery."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ogb.graphproppred import Evaluator
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.data import Batch, Data

from ablation.common.resumable import JobStore, atomic_torch_save, atomic_write_json
from ablation.common.worker import Job, Preempted
from .collect import collect, main as collect_main
from .models import build_model, route_logits
from .run import main as run_main
from .protocol import METHODS, REFERENCE, make_jobs
from .training import load_dataset, score, train


class ToyMolecules(list):
    num_tasks = 1


def toy_data():
    data = ToyMolecules()
    for index in range(20):
        atom = torch.tensor([[(index + node) % size for size in get_atom_feature_dims()]
                             for node in range(3)])
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        bonds = torch.tensor([[index % size for size in get_bond_feature_dims()]
                              for _ in range(4)])
        data.append(Data(x=atom, edge_index=edges, edge_attr=bonds,
                         y=torch.tensor([[float(index % 2)]])))
    splits = {'train': torch.arange(12), 'valid': torch.arange(12, 16), 'test': torch.arange(16, 20)}
    return data, splits


def config_for(method='ens_m0', **changes):
    config = dict(REFERENCE, dataset='ogbg-molbace', model='gin', method=method,
                  seed=int(method[-1]) if method.startswith('ens_m') else 0,
                  cohort=0, members=4 if method == 'gnnm' else 1,
                  hidden_dim=16, num_layers=2, dropout=0.2, batch_size=4,
                  smoke_epochs=2, smoke_updates=None, smoke=True,
                  source_sha256='synthetic-test-only')
    config.update(changes)
    return config


class MolecularTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        root = Path(__file__).parent / 'test_runs'
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temporary.name)
        self.dataset, self.splits = toy_data()
        self.evaluator = Evaluator(name='ogbg-molbace')

    def tearDown(self):
        self.temporary.cleanup()

    def run_fit(self, config, directory, stop, hook=None):
        store = JobStore(self.root / directory)
        store.ensure_config(config)
        result = train(config, store, stop, self.dataset, self.splits, self.evaluator,
                       {'synthetic': True}, torch.device('cpu'), on_update=hook)
        prediction = torch.load(store.dir / 'predictions.pt', weights_only=False)
        state = store.load_state()
        return result, prediction, state

    def test_ogb_split_metadata_interface(self):
        self.dataset.meta_info = {'split': 'scaffold'}
        self.dataset.name = 'ogbg-molbace'
        self.dataset.eval_metric = 'rocauc'
        self.dataset.task_type = 'binary classification'
        self.dataset.y = torch.cat([graph.y for graph in self.dataset])
        self.dataset.get_idx_split = lambda: self.splits
        with patch('ablation.p09_molecular.training.PygGraphPropPredDataset', return_value=self.dataset):
            dataset, splits, evaluator, binding = load_dataset('ogbg-molbace', self.root)
            self.assertIs(dataset, self.dataset)
            self.assertEqual(binding['dataset'], 'ogbg-molbace')
            self.assertEqual(binding['split_counts'], {'train': 12, 'valid': 4, 'test': 4})
            self.assertEqual(set(splits), {'train', 'valid', 'test'})
            self.assertEqual(evaluator.eval_metric, 'rocauc')
            self.dataset.meta_info['split'] = 'random'
            with self.assertRaisesRegex(ValueError, 'scaffold'):
                load_dataset('ogbg-molbace', self.root)

    def test_graph_input_binding_rejects_resume_and_completed_skip(self):
        self.dataset.meta_info = {'split': 'scaffold'}
        self.dataset.name = 'ogbg-molbace'
        self.dataset.eval_metric = 'rocauc'
        self.dataset.task_type = 'binary classification'
        self.dataset.y = torch.cat([graph.y for graph in self.dataset])
        self.dataset.get_idx_split = lambda: self.splits
        with patch('ablation.p09_molecular.training.PygGraphPropPredDataset', return_value=self.dataset):
            original = load_dataset('ogbg-molbace', self.root)
            protocol = config_for()
            resume_store = JobStore(self.root / 'resume')
            resume_store.ensure_config(protocol)
            with self.assertRaises(Preempted):
                train(protocol, resume_store, SimpleNamespace(requested=True), *original,
                      torch.device('cpu'))
            self.dataset[0].x[0, 0] += 1
            changed = load_dataset('ogbg-molbace', self.root)
        self.assertEqual(original[3]['splits_and_labels_sha256'], changed[3]['splits_and_labels_sha256'])
        self.assertNotEqual(original[3]['graph_inputs_sha256'], changed[3]['graph_inputs_sha256'])
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            train(protocol, resume_store, SimpleNamespace(requested=False), *changed,
                  torch.device('cpu'))
        completed_root = self.root / 'completed'
        completed = JobStore(completed_root / 'fit')
        completed.ensure_config(protocol)
        completed.write_done(dict(protocol, data_binding=original[3]))
        with patch('ablation.p09_molecular.run.make_jobs', return_value=[Job('fit', protocol)]), \
                patch('ablation.p09_molecular.run.load_dataset', return_value=changed), \
                patch('ablation.p09_molecular.run.run_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'Completed job data binding changed'):
                run_main(['--root', str(completed_root), '--smoke_epochs', '2', '--device', 'cpu'])
            worker.assert_not_called()

    def test_native_identity_boundary_parity(self):
        batch = Batch.from_data_list(self.dataset[:4])
        for architecture in ('gin', 'gcn'):
            torch.manual_seed(10)
            native = build_model(config_for(model=architecture), 1).eval()
            torch.manual_seed(10)
            gnnm = build_model(config_for('gnnm', model=architecture), 1).eval()
            self.assertTrue(set(gnnm.input_be_block.R.unique().tolist()) <= {-1.0, 1.0})
            with torch.no_grad():
                gnnm.input_be_block.R.fill_(1)
            for route in range(4):
                self.assertTrue(torch.allclose(native(batch), gnnm(batch, member=route),
                                               atol=1e-6, rtol=1e-6))
            self.assertEqual(len(gnnm.gnn_node.convs), 2)
            self.assertEqual(route_logits(gnnm, batch, 4).shape, (4, 4, 1))

    def test_exact_resume_and_closed_smoke_test(self):
        for method in ('ens_m0', 'gnnm'):
            for update_limit in (None, 2):
                config = config_for(method, smoke_updates=update_limit)
                name = f'{method}_{update_limit}'
                expected, expected_predictions, expected_state = self.run_fit(
                    config, name + '_full', SimpleNamespace(requested=False))
                stop = SimpleNamespace(requested=False)

                def interrupt(updates):
                    stop.requested = updates == (update_limit or 1)

                with self.assertRaises(Preempted):
                    self.run_fit(config, name + '_resume', stop, interrupt)
                actual, actual_predictions, actual_state = self.run_fit(
                    config, name + '_resume', SimpleNamespace(requested=False))
                self.assertEqual(actual['validation_curve'], expected['validation_curve'])
                self.assertEqual(actual['best_epoch'], expected['best_epoch'])
                self.assertEqual(actual['updates'], expected['updates'])
                self.assertFalse(actual['test_evaluated'])
                self.assertEqual(set(actual_predictions['splits']), {'valid'})
                self.assertEqual(actual['resumes'], 1)
                for key in expected_state['model']:
                    self.assertTrue(torch.equal(expected_state['model'][key], actual_state['model'][key]), key)
                for parameter, state in expected_state['optimizer']['state'].items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(value, actual_state['optimizer']['state'][parameter][key]))
                self.assertIsNone(actual_state['scheduler'])
                selected = torch.load(self.root / (name + '_resume') / 'model.pt', weights_only=False)
                for key, value in actual_state['progress']['best_model'].items():
                    self.assertTrue(torch.equal(value, selected['model'][key]))
                self.assertTrue(torch.equal(expected_state['rng']['torch'], actual_state['rng']['torch']))
                self.assertTrue(torch.equal(expected_predictions['splits']['valid']['member_logits'],
                                            actual_predictions['splits']['valid']['member_logits']))

    def test_queue_counts_seeds_and_source_binding(self):
        args = SimpleNamespace(datasets=['ogbg-molbace', 'ogbg-molhiv'], models=None,
                               cohorts=list(range(10)), methods=METHODS,
                               smoke_epochs=None, smoke_updates=None, data_dir=self.root)
        jobs = make_jobs(args)
        self.assertEqual(len(jobs), 150)
        self.assertEqual(len({job.key for job in jobs}), 150)
        self.assertTrue(all(job.config['epochs'] == 100 for job in jobs))
        for job in jobs:
            config = job.config
            member = int(config['method'][-1]) if config['method'].startswith('ens_m') else 0
            self.assertEqual(config['seed'], 1000 * config['cohort'] + member)
            self.assertIsNone(config['scheduler'])
            self.assertFalse(config['smoke'])
            self.assertEqual(len(config['source_sha256']), 64)

    def test_source_mismatch_rejected(self):
        config = config_for(smoke_updates=1)
        self.run_fit(config, 'binding', SimpleNamespace(requested=False))
        config['source_sha256'] = 'changed'
        store = JobStore(self.root / 'binding')
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            train(config, store, SimpleNamespace(requested=False), self.dataset, self.splits,
                  self.evaluator, {'synthetic': True}, torch.device('cpu'))

    def test_ensemble_uses_mean_logits_and_member_zero_base(self):
        labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        native_logits = [torch.tensor([[0.0], [float(member + 1)], [2.0], [4.0]])
                         for member in range(4)]
        for method in METHODS:
            config = config_for(method, smoke=False, smoke_epochs=None)
            folder = self.root / method
            store = JobStore(folder)
            store.ensure_config(config)
            logits = (torch.stack(native_logits) if method == 'gnnm' else
                      native_logits[int(method[-1])].unsqueeze(0))
            splits = {split: {'indices': torch.arange(4), 'y_true': labels, 'member_logits': logits}
                      for split in ('valid', 'test')}
            binding = {'synthetic': True}
            atomic_torch_save({'config': config, 'data_binding': binding, 'splits': splits},
                              folder / 'predictions.pt')
            atomic_write_json(dict(config, data_binding=binding), folder / 'done.json')
        rows, incomplete = collect(self.root, datasets=['ogbg-molbace'], models=['gin'], cohorts=[0])
        self.assertFalse(incomplete)
        by_method = {row['method']: row for row in rows}
        expected = score(self.evaluator, labels, torch.stack(native_logits).mean(dim=0))
        self.assertEqual(by_method['ensemble']['test_rocauc'], expected)
        self.assertEqual(by_method['base']['test_rocauc'], score(self.evaluator, labels, native_logits[0]))


    def test_partial_duplicate_members_and_explicit_smoke_collection(self):
        labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        selection = {'datasets': ['ogbg-molbace'], 'models': ['gin'], 'cohorts': [0]}
        role = {'study': 'p09_molecular', 'smoke': True, 'smoke_epochs': 2, 'smoke_updates': None}
        atomic_write_json(role, self.root / 'protocol_role.json')
        for method in METHODS:
            if method == 'gnnm':
                rows, missing = collect(self.root, include_smoke=True, **selection)
                self.assertFalse(rows)
                self.assertEqual(missing[0]['missing'], ['gnnm'])
            config = config_for(method)
            store = JobStore(self.root / method)
            store.ensure_config(config)
            logits = torch.tensor([[0.0], [1.0], [2.0], [4.0]]).repeat(config['members'], 1, 1)
            prediction = {'indices': torch.arange(4), 'y_true': labels, 'member_logits': logits}
            binding = {'synthetic': True}
            atomic_torch_save({'config': config, 'data_binding': binding,
                               'splits': {'valid': prediction}}, store.dir / 'predictions.pt')
            atomic_write_json(dict(config, data_binding=binding), store.dir / 'done.json')
        rows, missing = collect(self.root, **selection)
        self.assertFalse(rows)
        self.assertEqual(set(missing[0]['missing']), set(METHODS))
        rows, missing = collect(self.root, include_smoke=True, **selection)
        self.assertEqual(len(rows), 3)
        self.assertFalse(missing)
        self.assertTrue(all(row['role'] == 'engineering_validation' for row in rows))
        self.assertTrue(all('test_rocauc' not in row for row in rows))
        output = self.root / 'summary'
        status = collect_main(['--root', str(self.root), '--out', str(output),
                               '--datasets', 'ogbg-molbace', '--models', 'gin',
                               '--cohorts', '0', '--include-smoke'])
        self.assertEqual(status, 0)
        summary = json.loads((output / 'results.json').read_text())
        self.assertEqual(summary['root_role'], role)
        self.assertTrue(all('test_mean' not in item for item in summary['summary']))
        shutil.copytree(self.root / 'ens_m0', self.root / 'duplicate_ens_m0')
        with self.assertRaisesRegex(ValueError, 'Duplicate cohort member'):
            collect(self.root, include_smoke=True, **selection)


if __name__ == '__main__':
    unittest.main()
