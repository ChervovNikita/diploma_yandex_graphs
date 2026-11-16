"""CPU-only regression tests for adapter provenance, failures and recovery."""

from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from . import tasks
from .analyses import final_report
from .decisions import select
from .schema import load_plan
from .store import CampaignStore, atomic_json
from ablation.common.resumable import JobStore

REPOSITORY = Path(__file__).resolve().parents[2]


class AdapterRecovery(unittest.TestCase):
    def setUp(self):
        parent = REPOSITORY / '.campaign_test_tmp' / 'adapter_tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / 'run'
        self.out = self.root / 'tasks' / 'transfer' / 'output'
        self.out.mkdir(parents=True)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temporary.cleanup()

    def transfer_fixture(self):
        trained = self.out / 'training/ogbg-molbace/gcn/cohort0/ens_m0'
        trained.mkdir(parents=True)
        atomic_json(trained / 'done.json', {'metric': 0.5, 'source_sha256': 'fixture-source'})
        claim = trained / 'claim'
        claim.write_text('retained legacy owner\n')
        lock = self.out.parent / '.ownership.lock'
        lock.touch()
        meta = dict(module='p09_molecular', pair='molbace_gcn', method='native0', cohort=0)
        return claim, lock, meta

    def test_legacy_claim_survives_without_campaign_ownership(self):
        claim, _, meta = self.transfer_fixture()
        with patch.dict(os.environ, {'GNNM_CAMPAIGN_LOCK_FDS': ''}), patch.object(tasks.subprocess, 'run') as run:
            tasks.transfer_fit(meta, self.out, 'cpu')
        self.assertTrue(claim.exists())
        self.assertEqual(run.call_args.kwargs['pass_fds'], ())

    def test_legacy_claim_clears_only_with_matching_owned_task_descriptor(self):
        claim, lock, meta = self.transfer_fixture()
        descriptor = os.open(lock, os.O_RDWR)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch.dict(os.environ, {'GNNM_CAMPAIGN_LOCK_FDS': str(descriptor)}), patch.object(tasks.subprocess, 'run') as run:
                receipt = tasks.transfer_fit(meta, self.out, 'cpu')
            self.assertFalse(claim.exists())
            self.assertEqual(run.call_args.kwargs['pass_fds'], (descriptor,))
            self.assertEqual(receipt['fit']['source_sha256'], 'fixture-source')
            self.assertTrue((self.out / 'command.json').is_file())
        finally:
            os.close(descriptor)

    def test_wrong_closed_and_externally_owned_descriptors_preserve_claim(self):
        claim, lock, meta = self.transfer_fixture()
        unrelated = self.directory / 'unrelated-file'
        unrelated.touch()
        wrong = os.open(unrelated, os.O_RDWR)
        try:
            with patch.dict(os.environ, {'GNNM_CAMPAIGN_LOCK_FDS': str(wrong)}):
                with self.assertRaisesRegex(ValueError, 'do not own'):
                    tasks.transfer_fit(meta, self.out, 'cpu')
        finally:
            os.close(wrong)
        with patch.dict(os.environ, {'GNNM_CAMPAIGN_LOCK_FDS': str(wrong)}):
            with self.assertRaises(OSError):
                tasks.transfer_fit(meta, self.out, 'cpu')
        owner = os.open(lock, os.O_RDWR)
        competing = os.open(lock, os.O_RDWR)
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch.dict(os.environ, {'GNNM_CAMPAIGN_LOCK_FDS': str(competing)}):
                with self.assertRaises(BlockingIOError):
                    tasks.transfer_fit(meta, self.out, 'cpu')
        finally:
            os.close(competing)
            os.close(owner)
        self.assertTrue(claim.exists())

    def node_meta(self):
        fake_repo = self.directory / 'fixture_repo'
        (fake_repo / 'data').mkdir(parents=True)
        (fake_repo / 'data/roman_empire.npz').write_bytes(b'fixture-data-identity')
        spec = dict(dataset='roman-empire', model='GCN', num_layers=1, split=0, seed=0,
                    members=1, lr=0.001, num_steps=2, continuation_steps=4, min_steps=1,
                    patience=1, defer_test=True, save_selected=True)
        return fake_repo, {'spec': spec}

    def test_node_fit_records_config_checkpoint_and_continuation_receipts(self):
        fake_repo, meta = self.node_meta()
        calls = []
        def fake_train(settings, store, bundle, device, build_model, stop):
            calls.append(settings.num_steps)
            torch.save({'weight': torch.ones(2)}, store.dir / 'model.pt')
            torch.save({'roles': {'val': {}}}, store.dir / 'predictions.pt')
            store.save_state({'progress': {'step': settings.num_steps}})
            return dict(stopped_step=settings.num_steps, best_step=settings.num_steps,
                        stop_reason='step_cap' if settings.num_steps == 2 else 'patience',
                        needs_convergence_extension=settings.num_steps == 2)
        with patch.object(tasks, 'REPO', fake_repo), patch.object(tasks, 'source_binding', return_value='source-v1'), \
             patch('datasets.load_dataset', return_value=(object(), None, None, None, 2, 2, False)), \
             patch('ablation.common.tabm_trainer.train_resumable', side_effect=fake_train):
            receipt = tasks.node_fit(meta, self.out, self.root, 'cpu')
        self.assertEqual(calls, [2, 4])
        config = json.loads((self.out / 'config.json').read_text())
        self.assertEqual(config['source_sha256'], 'source-v1')
        self.assertEqual(config['dataset_sha256'], hashlib.sha256(b'fixture-data-identity').hexdigest())
        self.assertEqual(receipt['selected_checkpoint_sha256'], hashlib.sha256((self.out / 'model.pt').read_bytes()).hexdigest())
        self.assertEqual(len(receipt['continuations']), 2)
        self.assertEqual(receipt['effective_cap'], 4)
        self.assertFalse(JobStore(self.out).has_state())
        self.assertEqual(json.loads((self.out / 'done.json').read_text()), receipt)
        with self.assertRaises(ValueError):
            JobStore(self.out).ensure_config(dict(config, lr=0.02))

    def test_deterministic_numerical_failure_is_evidence_and_io_failure_propagates(self):
        fake_repo, meta = self.node_meta()
        with patch.object(tasks, 'REPO', fake_repo), patch.object(tasks, 'source_binding', return_value='source-v1'), \
             patch('datasets.load_dataset', return_value=(object(), None, None, None, 2, 2, False)), \
             patch('ablation.common.tabm_trainer.train_resumable', side_effect=FloatingPointError('nonfinite gradient')):
            receipt = tasks.node_fit(meta, self.out, self.root, 'cpu')
        self.assertEqual(receipt['scientific_status'], 'numerical_failure')
        self.assertTrue((self.out / 'numerical_failure.json').is_file())
        other = self.directory / 'io_failure'
        with patch.object(tasks, 'REPO', fake_repo), patch.object(tasks, 'source_binding', return_value='source-v1'), \
             patch('datasets.load_dataset', return_value=(object(), None, None, None, 2, 2, False)), \
             patch('ablation.common.tabm_trainer.train_resumable', side_effect=OSError('storage unavailable')):
            with self.assertRaises(OSError):
                tasks.node_fit(meta, other, self.root, 'cpu')
        self.assertFalse((other / 'numerical_failure.json').exists())

    def cell_fixture(self, failed_index=2):
        fit_ids = [f'fit/{index}' for index in range(5)]
        for index, task_id in enumerate(fit_ids):
            directory = tasks.output(self.root, task_id)
            directory.mkdir(parents=True)
            receipt = dict(fit_seconds=1, needs_convergence_extension=False, num_layers=1)
            if index == failed_index:
                receipt.update(scientific_status='numerical_failure', error='nonfinite gradient')
            else:
                logits = torch.tensor([[[2., 0.], [0., 2.], [2., 0.], [0., 2.]]])
                artifact = dict(is_binary=False, roles={'val': {'indices': torch.arange(4),
                                'labels': torch.tensor([0, 1, 0, 1]), 'logits': logits}})
                torch.save(artifact, directory / 'predictions.pt')
            atomic_json(directory / 'done.json', receipt)
        return dict(fits=fit_ids, dataset='roman-empire', model='GCN', config_name='bad', depth=1, split=0)

    def test_numerical_candidate_keeps_surviving_methods_and_feasible_alternative(self):
        meta = self.cell_fixture()
        cell = tasks.node_cell(meta, self.root, 'cell/bad0')
        self.assertEqual(cell['scores'], {'BASE': 1.0, 'ENS': None, 'GNNM': 1.0})
        cells = []
        for split in range(3):
            cells.append(dict(cell, split=split, task_id=f'cell/bad{split}'))
            cells.append(dict(cell, split=split, task_id=f'cell/good{split}', config_name='good',
                              scores={'BASE': 0.9, 'ENS': 0.95, 'GNNM': 0.96}, scientific_failures={}))
        decision = select(cells)
        self.assertTrue(decision['ready'])
        self.assertEqual(decision['selections']['ENS']['config_name'], 'good')
        self.assertEqual(len(decision['infeasible_candidates']), 1)

    def test_surviving_predictions_still_require_validation_identity(self):
        meta = self.cell_fixture()
        path = tasks.output(self.root, 'fit/4') / 'predictions.pt'
        artifact = torch.load(path, weights_only=False)
        artifact['roles']['test'] = {}
        torch.save(artifact, path)
        with self.assertRaisesRegex(ValueError, 'held-out'):
            tasks.node_cell(meta, self.root, 'cell/bad')
        del artifact['roles']['test']
        artifact['roles']['val']['indices'] = torch.arange(4).flip(0)
        torch.save(artifact, path)
        with self.assertRaisesRegex(ValueError, 'ordering'):
            tasks.node_cell(meta, self.root, 'cell/bad')

    def test_missing_feasible_selection_is_explicit(self):
        directory = tasks.output(self.root, 'decision/a')
        directory.mkdir(parents=True)
        atomic_json(directory / 'done.json', {'ready': False, 'selections': {}})
        with self.assertRaisesRegex(ValueError, 'feasible, ready'):
            tasks.resolve_spec({'spec': {}, 'resolve': {'selection': 'decision/a', 'setting': 'GNNM'}}, self.root)

    def report_plan(self):
        plan = dict(schema='gnnm-campaign-v1', campaign_id='final-report-test', tasks=[
            dict(id='gate', kind='cpu', argv=['true'], metadata={'action': 'expand'}),
            dict(id='branch', kind='cpu', argv=['true'], dependencies=['gate'],
                 when=dict(task='gate', path='output/done.json', field='expand', equals=True)),
            dict(id='report/main', kind='cpu', argv=['true']),
            dict(id='report/final', kind='cpu', argv=['true'], optional_dependencies=['branch', 'report/main']),
        ])
        store = CampaignStore(self.root, plan)
        store.prepare()
        store.write_state('gate', status='complete')
        store.write_state('branch', status='skipped', reason='validation_gate_closed')
        store.write_state('report/main', status='complete')
        store.write_state('report/final', status='running')
        atomic_json(tasks.output(self.root, 'gate') / 'done.json', dict(expand=False, convergence_ready=True))
        atomic_json(tasks.output(self.root, 'report/main') / 'done.json', dict(complete=True))
        return plan, store

    def test_final_report_separates_execution_and_scientific_completeness(self):
        plan, store = self.report_plan()
        meta = {'reports': ['branch', 'report/main']}
        out = tasks.output(self.root, 'report/final')
        report = final_report(meta, self.root, out, plan)
        self.assertTrue(report['queue_terminal'])
        self.assertTrue(report['complete'])
        self.assertEqual(report['declared_skips'], ['branch'])
        atomic_json(tasks.output(self.root, 'report/main') / 'done.json', dict(complete=False, unresolved_convergence=['fit/a']))
        report = final_report(meta, self.root, out, plan)
        self.assertTrue(report['execution_complete'])
        self.assertFalse(report['scientific_complete'])
        self.assertFalse(report['complete'])
        store.write_state('report/main', status='failed')
        report = final_report(meta, self.root, out, plan)
        self.assertTrue(report['queue_terminal'])
        self.assertFalse(report['execution_complete'])
        self.assertIn('report/main', report['incomplete_tasks'])
        store.write_state('report/main', status='pending')
        self.assertFalse(final_report(meta, self.root, out, plan)['queue_terminal'])

    def test_final_report_rejects_undeclared_skip_and_missing_report_artifact(self):
        plan, store = self.report_plan()
        store.write_state('report/main', status='skipped', reason='manual_skip')
        (tasks.output(self.root, 'report/main') / 'done.json').unlink()
        report = final_report({'reports': ['report/main']}, self.root, tasks.output(self.root, 'report/final'), plan)
        self.assertFalse(report['complete'])
        self.assertEqual(report['missing_reports'], ['report/main'])
        self.assertIn('report/main', report['incomplete_tasks'])

    def test_legacy_worker_retains_capped_states_and_clears_converged_defaults(self):
        from ablation.common.worker import Job, run_worker
        for index, (result, retained) in enumerate([
                ({'stop_reason': 'step_cap'}, True),
                ({'stop_reason': 'epoch_cap'}, True),
                ({'needs_convergence_extension': True}, True),
                ({'stop_reason': 'patience'}, False),
                ({'metric': 0.5}, False)]):
            with self.subTest(result=result):
                root = self.directory / f'worker-case-{index}'
                job = Job('fit', {'fixture': index})
                def execute(job, store, stop):
                    store.save_state({'progress': {'epoch': 2000}, 'optimizer': {'fixture': True}})
                    return result
                with patch('setproctitle.setproctitle'):
                    code = run_worker(root, [job], execute, wait_for_others=False,
                                      stop=SimpleNamespace(requested=False))
                self.assertEqual(code, 0)
                store = JobStore(root / 'fit')
                self.assertEqual(store.has_state(), retained)
                self.assertEqual(store.read_done(), result)

    def test_recorded_real_dag_smoke_has_32_successes_and_closed_test_roles(self):
        directory = REPOSITORY / '.campaign_test_tmp/dag_smoke_run'
        plan_path = REPOSITORY / '.campaign_test_tmp/dag_smoke_plan.json'
        if not plan_path.exists():
            self.skipTest('Preparation-host engineering DAG receipt is absent')
        plan = load_plan(plan_path)
        states = CampaignStore(directory, plan).snapshot()
        self.assertEqual(Counter(row['status'] for row in states.values()), {'complete': 32})
        actions = Counter(task['metadata']['action'] for task in plan['tasks'])
        self.assertEqual(actions, dict(node_fit=21, node_cell=3, select=1, node_score=3, diversity=3, final_report=1))
        for task in plan['tasks']:
            if task['metadata']['action'] != 'node_fit':
                continue
            out = tasks.output(directory, task['id'])
            config = json.loads((out / 'config.json').read_text())
            receipt = json.loads((out / 'done.json').read_text())
            self.assertEqual(config['source_sha256'], receipt['source_sha256'])
            self.assertEqual(config['dataset_sha256'], receipt['dataset_sha256'])
            artifact = torch.load(out / 'predictions.pt', map_location='cpu', weights_only=False)
            self.assertNotIn('test', artifact['roles'])
            self.assertEqual(receipt['selected_checkpoint_sha256'], hashlib.sha256((out / 'model.pt').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
