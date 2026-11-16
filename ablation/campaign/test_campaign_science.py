"""Synthetic checks for validation decisions and paired scientific summaries."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.resumable import atomic_torch_save, atomic_write_json
from ablation.campaign.analyses import comparison, holm, paired_summary, transfer_report, budget_report
from ablation.campaign.decisions import expansion, select
from ablation.campaign.tasks import node_cell, output
from ablation.p05_hparam_grid.grid import configs_for


class CampaignScienceTests(unittest.TestCase):
    def cells(self, dataset='roman-empire', gnnm_gain=.01):
        rows = []
        for index, name in enumerate(configs_for('combined')):
            for depth in [1, 2]:
                for split in [0, 1, 2]:
                    value = .5 + index * .001 + depth * .0001
                    rows.append(dict(
                        task_id=f'{dataset}/{name}/{depth}/{split}',
                        dataset=dataset, model='GAT', role='val',
                        config_name=name, depth=depth, split=split,
                        scores={'BASE': value - .01, 'ENS': value, 'GNNM': value + gnnm_gain},
                        specs={key: {'num_layers': depth} for key in ('BASE', 'ENS', 'GNNM')},
                        fit_ids=[f'{dataset}/{name}/{depth}/{split}/fit{member}' for member in range(5)],
                        unresolved_convergence=[]))
        return rows

    def test_selection_is_validation_only_and_requires_all_pilot_splits(self):
        cells = self.cells()
        for row in cells:
            row['test_metric'] = 1000 if row['depth'] == 1 else -1000
        chosen = select(cells)
        self.assertEqual(chosen['selections']['GNNM']['depth'], 2)
        self.assertTrue(chosen['no_test_read'])
        with self.assertRaises(ValueError):
            select(cells[1:])
        changed = copy.deepcopy(cells)
        changed[0]['role'] = 'test'
        with self.assertRaises(ValueError):
            select(changed)

    def test_expansion_is_method_balanced_and_blocks_unconverged_selection(self):
        decisions = [select(self.cells(str(dataset), .01 if dataset < 3 else -.01))
                     for dataset in range(5)]
        expanded = expansion(decisions)
        self.assertTrue(expanded['expand'])
        self.assertEqual(len(expanded['chosen_names']), 3)
        self.assertEqual(len(expanded['common_ranks']), 9)
        decisions[0]['selected_unresolved'] = ['capped_fit']
        self.assertFalse(expansion(decisions)['expand'])

    def test_paired_summaries_holm_and_nonfinite_guard(self):
        summary = paired_summary([.01, .02, .03])
        self.assertAlmostEqual(summary['mean'], .02)
        self.assertEqual(summary['n'], 3)
        self.assertIsNotNone(summary['descriptive_paired_t_interval'])
        rows = {'a': {'exact_sign_flip_p': .01}, 'b': {'exact_sign_flip_p': .04}}
        holm(rows)
        self.assertEqual(rows['a']['holm_adjusted_p'], .02)
        self.assertEqual(rows['b']['holm_adjusted_p'], .04)
        with self.assertRaises(ValueError):
            paired_summary([.1, float('nan')])

    def test_comparison_retains_losses_and_rejects_duplicate_cells(self):
        rows = [dict(dataset='a', model='GAT', stage='gat_confirmation', split=split,
                     setting=setting, val_metric=value, test_metric=value,
                     unresolved_convergence=[])
                for split in [3, 4, 5]
                for setting, value in [('BASE', .7), ('ENS', .8), ('GNNM', .75)]]
        report = comparison(rows)
        contrast = report['paired_comparisons']['a/GAT/selected/GNNM-minus-ENS/test']
        self.assertAlmostEqual(contrast['mean'], -.05)
        self.assertEqual(contrast['paired_splits'], [3, 4, 5])
        with self.assertRaises(ValueError):
            comparison([*rows, rows[0]])

    def test_transfer_cohort_and_method_integrity(self):
        rows = [dict(pair='molbace_gcn', cohort=cohort, unresolved_convergence=[],
                     settings={method: dict(val_metric=value, test_metric=value)
                               for method, value in [('BASE', .5), ('ENS', .6), ('gnnm', .7)]})
                for cohort in [0, 1, 2]]
        for row in rows:
            row["reference100_settings"] = copy.deepcopy(row["settings"])
        report = transfer_report(rows)
        self.assertTrue(report['complete'])
        with self.assertRaises(ValueError):
            transfer_report([*rows, rows[0]])
        incomplete = copy.deepcopy(rows)
        del incomplete[1]['settings']['gnnm']
        with self.assertRaises(ValueError):
            transfer_report(incomplete)

    def test_budget_replay_uses_one_order_and_actual_member_updates(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        cell_ids, profile_ids = [], []
        for candidate in range(8):
            name = f'c{candidate}'
            profile_id = f'profile/{name}'
            folder = output(root, profile_id)
            folder.mkdir(parents=True)
            atomic_write_json({'config_name': name, 'depth': 1,
                               'profiles': {method: {'full_step_ms': value}
                                            for method, value in [('BASE', 100), ('ENS', 400), ('GNNM', 200)]}},
                              folder / 'done.json')
            profile_ids.append(profile_id)
            for split in range(3):
                fits = []
                for member in range(5):
                    fit_id = f'fit/{name}/{split}/{member}'
                    folder = output(root, fit_id)
                    folder.mkdir(parents=True)
                    atomic_write_json({'stopped_step': 10}, folder / 'done.json')
                    fits.append(fit_id)
                cell_id = f'cell/{name}/{split}'
                folder = output(root, cell_id)
                folder.mkdir(parents=True)
                atomic_write_json({'config_name': name, 'depth': 1, 'split': split,
                                   'specs': {method: {'lr': .001, 'num_layers': 1} for method in ['BASE', 'ENS', 'GNNM']},
                                   'fit_ids': fits, 'scores': {method: .5 + candidate * .01
                                                             for method in ['BASE', 'ENS', 'GNNM']}},
                                  folder / 'done.json')
                cell_ids.append(cell_id)
        report = budget_report({'cells': cell_ids, 'profiles': profile_ids}, root)
        self.assertEqual(report['common_gpu_seconds'], 24)
        self.assertEqual(report['matched']['BASE']['completed_candidates'], 8)
        self.assertEqual(report['matched']['ENS']['completed_candidates'], 2)
        self.assertEqual(report['matched']['GNNM']['completed_candidates'], 4)
        orders = [[row['config_name'] for row in report['full_trace'][method]]
                  for method in ['BASE', 'ENS', 'GNNM']]
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0], orders[2])
        self.assertTrue(report["ready_BASE"])
        self.assertTrue(report["ready_ENS"])
        self.assertTrue(report["ready_GNNM"])
        self.assertEqual(report["selections"]["ENS"]["spec"]["num_layers"], 1)

    def test_validation_cell_uses_native_members_and_blocks_test_artifacts(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        identifiers = [f'fit/{index}' for index in range(5)]
        for index, identifier in enumerate(identifiers):
            out = output(root, identifier)
            out.mkdir(parents=True)
            atomic_write_json(dict(fit_seconds=2, needs_convergence_extension=False), out / 'done.json')
            role = {'indices': torch.arange(4), 'labels': torch.tensor([0, 1, 0, 1]),
                    'logits': torch.tensor([[[2., 0.], [0., 2.], [2., 0.], [0., 2.]]])}
            if index == 4:
                role['logits'] = role['logits'].expand(4, -1, -1)
            atomic_torch_save({'roles': {'val': role}, 'is_binary': False}, out / 'predictions.pt')
        meta = dict(fits=identifiers, dataset='roman-empire', model='GAT',
                    config_name='reference', depth=1, split=0)
        result = node_cell(meta, root, 'cell/check')
        self.assertEqual(result['scores']['ENS'], 1)
        self.assertEqual(result['elapsed_seconds']['ENS'], 8)
        payload_path = output(root, identifiers[0]) / 'predictions.pt'
        payload = torch.load(payload_path, weights_only=True)
        payload['roles']['test'] = payload['roles']['val']
        atomic_torch_save(payload, payload_path)
        with self.assertRaises(ValueError):
            node_cell(meta, root, 'cell/check')


if __name__ == '__main__':
    unittest.main()
