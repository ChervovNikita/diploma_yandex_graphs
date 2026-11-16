"""Matched architecture-search scheduling and validation-selection regression tests."""

import copy
from itertools import product
import tempfile
import unittest
from pathlib import Path

from ablation.common.resumable import atomic_write_json
from ablation.campaign.architecture_search import (
    ALTERNATIVES, BACKBONES, DEPTHS, PILOT_SPLITS, STAGE,
    add_architecture_search, candidate_configs, execute_comparison, execute_selection,
)
from ablation.campaign.plan import Builder, node_spec
from ablation.campaign.schema import SCHEMA, validate_plan
from ablation.campaign.tasks import output
from ablation.p05_hparam_grid.grid import REFERENCE, REFERENCE_NAME


class ArchitectureSearchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def save(self, identifier, value):
        directory = output(self.root, identifier)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(value, directory / 'done.json')

    def pilot(self):
        grid = candidate_configs()
        candidates = []
        for index, ((name, hp), depth, split) in enumerate(product(grid.items(), DEPTHS, PILOT_SPLITS)):
            identifier = f'fit/architecture/{index}'
            spec = node_spec('roman-empire', 'GAT', depth, split, 'all_layer_be', hp)
            spec.update(val_metric=.5 + hp['lr'] * 100 + depth * .001,
                        test_metric=-depth * 1000, needs_convergence_extension=False)
            self.save(identifier, spec)
            candidates.append(dict(config_name=name, depth=depth, split=split, fit=identifier))
        return dict(candidates=candidates, dataset='roman-empire', model='GAT',
                    variant='all_layer_be', candidate_configs=grid,
                    depths=DEPTHS, pilot_splits=PILOT_SPLITS)

    def test_all_arms_have_identical_candidate_opportunities(self):
        builder = Builder()
        reports = add_architecture_search(builder)
        self.assertEqual(len(reports), 2)
        validate_plan(dict(schema=SCHEMA, campaign_id='architecture-test',
                           tasks=list(builder.tasks.values())))
        for model in BACKBONES:
            control = builder.tasks[f'decision/architecture/{model}/controls']['metadata']
            self.assertEqual(len(control['cells']), 45)
            control_configs = set()
            for identifier in control['cells']:
                cell = builder.tasks[identifier]['metadata']
                control_configs.add((cell['config_name'], cell['depth'], cell['split']))
            for variant in ALTERNATIVES:
                alternative = builder.tasks[f'decision/architecture/{model}/{variant}']['metadata']
                candidate_ids = {(item['config_name'], item['depth'], item['split'])
                                 for item in alternative['candidates']}
                self.assertEqual(control_configs, candidate_ids)
                self.assertEqual(len(candidate_ids), 45)
        for report_id in reports:
            self.assertEqual(len(builder.tasks[report_id]['metadata']['scores']), 56)

    def test_existing_reference_and_gat_fits_reduce_increment_to_604(self):
        builder = Builder()
        grid = candidate_configs()
        for name, hp in grid.items():
            for depth, split in product(DEPTHS, PILOT_SPLITS):
                builder.cell('roman-empire', 'GAT', depth, split, hp, name, 'gat_pilot', 2)
        for depth, split in product(DEPTHS, PILOT_SPLITS):
            builder.cell('roman-empire', 'SAGE', depth, split, REFERENCE,
                         REFERENCE_NAME, 'roman_controls', 10)
        for model, variant, depth, split in product(BACKBONES, ALTERNATIVES, DEPTHS, PILOT_SPLITS):
            builder.fit(node_spec('roman-empire', model, depth, split, variant), 'roman_ablation', 11)
        before = sum(task['metadata']['action'] == 'node_fit' for task in builder.tasks.values())
        add_architecture_search(builder)
        after = sum(task['metadata']['action'] == 'node_fit' for task in builder.tasks.values())
        self.assertEqual(after - before, 604)

    def test_validation_selection_ignores_test_values_and_keeps_caps_visible(self):
        metadata = self.pilot()
        selected = execute_selection(metadata, self.root)
        chosen = selected['selections']['all_layer_be']
        self.assertEqual(chosen['depth'], 5)
        self.assertEqual(chosen['spec']['lr'], 3e-4)
        self.assertTrue(selected['ready'])
        self.assertTrue(selected['no_test_read'])
        self.assertEqual(selected['candidate_count'], 45)
        path = output(self.root, chosen['fits'][0]) / 'done.json'
        import json
        fit = json.loads(path.read_text())
        fit['needs_convergence_extension'] = True
        atomic_write_json(fit, path)
        selected = execute_selection(metadata, self.root)
        self.assertEqual(selected['selected_unresolved'], [chosen['fits'][0]])

    def test_incomplete_or_changed_candidates_are_rejected(self):
        metadata = self.pilot()
        truncated = copy.deepcopy(metadata)
        truncated['candidates'].pop()
        with self.assertRaises(ValueError):
            execute_selection(truncated, self.root)
        path = output(self.root, metadata['candidates'][0]['fit']) / 'done.json'
        import json
        fit = json.loads(path.read_text())
        fit['lr'] = .5
        atomic_write_json(fit, path)
        with self.assertRaises(ValueError):
            execute_selection(metadata, self.root)

    def test_numerical_failures_preserved_without_selecting_failed_candidate(self):
        metadata = self.pilot()
        for item in metadata['candidates']:
            path = output(self.root, item['fit']) / 'done.json'
            import json
            fit = json.loads(path.read_text())
            fit['scientific_status'] = 'numerical_failure'
            fit.pop('val_metric')
            atomic_write_json(fit, path)
        selected = execute_selection(metadata, self.root)
        self.assertFalse(selected['ready'])
        self.assertEqual(len(selected['infeasible_candidates']), 15)
        self.assertEqual(selected['selections'], {})

    def test_joint_report_keeps_missing_arms_and_available_comparisons(self):
        scores = []
        for split, setting in product(range(3, 10), ['BASE', 'ENS', 'GNNM', 'all_layer_be']):
            identifier = f'score/{split}/{setting}'
            scores.append(identifier)
            self.save(identifier, dict(dataset='roman-empire', model='GAT', split=split,
                                       setting=setting, stage=STAGE, val_metric=.5,
                                       test_metric=.6, unresolved_convergence=[]))
        missing = 'score/3/heads_single_pass'
        scores.append(missing)
        directory = self.root / 'tasks' / missing
        directory.mkdir(parents=True)
        atomic_write_json({'status': 'skipped'}, directory / 'state.json')
        decision = 'decision/architecture'
        self.save(decision, {'ready': True, 'selected_unresolved': []})
        report = execute_comparison(dict(scores=scores, decisions=[decision],
                                         dataset='roman-empire', model='GAT',
                                         learning_rates=[3e-5, 1e-4, 3e-4]), self.root)
        self.assertFalse(report['complete'])
        self.assertTrue(report['evaluation_complete'])
        self.assertEqual(report['skipped_score_tasks'], [missing])
        self.assertTrue(report['paired_comparisons'])


if __name__ == '__main__':
    unittest.main()
