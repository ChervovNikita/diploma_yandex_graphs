"""Synthetic artifact tests for cross-directory transfer aggregation."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ablation.common.resumable import atomic_torch_save, atomic_write_json
from ablation.campaign.analyses import score_transfer


class TransferAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, module):
        methods = [f'native{i}' for i in range(4)] + ['gnnm']
        if module in ('p09_molecular', 'p11_collab_buddy'):
            methods.append('private_bn')
        if module == 'p11_collab_buddy':
            methods.append('link_same_query_global')
        meta = dict(module=module, pair='fixture', cohort=0, fits=[])
        self.payloads = {}
        for index, method in enumerate(methods):
            task_id = 'fit_' + method
            meta['fits'].append(task_id)
            base = self.root / 'tasks' / task_id / 'output'
            base.mkdir(parents=True, exist_ok=True)
            train_root = base / 'training'
            directory = train_root / 'cohort0' / method
            score_dir = train_root / 'scoring' / 'cohort0' / method
            config = dict(cohort=0, seed=index if method.startswith('native') else 0,
                          method=method, members=1 if method.startswith('native') else 4,
                          source='fixture', dataset='ogbg-molbace', protocol='fixture')
            labels = torch.tensor([[0.], [1.], [0.], [1.]])
            logits = torch.tensor([[-1.], [2.], [0.], [3.]]) + index / 10
            if module == 'p09_molecular':
                payload = dict(config=config, data_binding={'fixture': True}, splits={
                    role: dict(indices=torch.arange(4), y_true=labels,
                               member_logits=logits.repeat(config['members'], 1, 1))
                    for role in ('valid', 'test')})
            elif module == 'p11_collab_buddy':
                payload = dict(config=config, **{role: dict(
                    scores=torch.arange(80).float().roll(index), num_positive=20,
                    member_scores=torch.arange(80).float().roll(index).repeat(config['members'], 1),
                    edge_order_sha256=role + '_order') for role in ('valid', 'test')})
            else:
                payload = dict(config=config, identity={'data_sha256': 'valid_order',
                               'config_sha256': method, 'versions': {}},
                               test_edge_order_sha256='test_order',
                               predictions={name: torch.arange(80).float().roll(index)
                                            for name in ('valid_pos', 'valid_neg', 'test_pos', 'test_neg')})
            self.payloads[method] = payload
            completed = dict(config, val_metric=0.5, test_metric=0.5, stop_reason='patience')
            atomic_write_json(dict(fit_dir=str(directory), method=method,
                                   argv=['python', '-m', 'fixture', '--root', str(train_root)]),
                              base / 'done.json')
            score_dir.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(payload, score_dir / 'predictions.pt')
            atomic_torch_save(payload, score_dir / 'reference100_predictions.pt')
            atomic_write_json(completed, score_dir / 'done.json')
        return meta

    def run_case(self, meta):
        out = self.root / 'score'
        def ncn_score(directory, target, device):
            method = directory.name
            target.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(self.payloads[method], target / 'predictions.pt')
            atomic_torch_save(self.payloads[method], target / 'reference100_predictions.pt')
            return dict(val_metric=0.5, test_metric=0.5, stop_reason='patience')
        with patch('ablation.campaign.analyses.subprocess.run') as invoked,                 patch('ablation.p12_collab_ncn.score.score_fit', side_effect=ncn_score):
            result = score_transfer(meta, self.root, out, 'cpu')
        if meta['module'] != 'p12_collab_ncn':
            self.assertEqual(invoked.call_count, len(meta['fits']))
        return result

    def test_three_runners_and_reference100_ensembles(self):
        for module in ('p09_molecular', 'p11_collab_buddy', 'p12_collab_ncn'):
            with self.subTest(module=module):
                meta = self.fixture(module)
                result = self.run_case(meta)
                self.assertIn('BASE', result['settings'])
                self.assertIn('ENS', result['settings'])
                self.assertEqual(result['settings'], result['reference100_settings'])
                self.assertFalse(result['unresolved_convergence'])

    def test_ncn_valid_identity_drift_rejected(self):
        meta = self.fixture('p12_collab_ncn')
        self.payloads['native3']['identity']['data_sha256'] = 'reordered_valid'
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            self.run_case(meta)

    def test_molecular_extension_label_drift_rejected(self):
        meta = self.fixture('p09_molecular')
        task_root = self.root / 'tasks' / 'fit_private_bn' / 'output' / 'training'
        path = task_root / 'scoring/cohort0/private_bn/predictions.pt'
        payload = copy.deepcopy(self.payloads['private_bn'])
        payload['splits']['test']['y_true'] = payload['splits']['test']['y_true'].flip(0)
        atomic_torch_save(payload, path)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.run_case(meta)


if __name__ == '__main__':
    unittest.main()
