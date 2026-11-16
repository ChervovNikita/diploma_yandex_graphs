"""Checks that the final campaign cannot drift from the frozen protocol.

    python -m unittest ablation.fixed_protocol.test_fixed_protocol -v
"""

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ablation.campaign.plan_fixed_protocol import ROMAN_VARIANTS, build_plan, fixed_spec
from ablation.fixed_protocol import protocol

REPO = Path(__file__).resolve().parents[2]


class ProtocolTests(unittest.TestCase):
    def spec(self, **changes):
        spec = fixed_spec('roman-empire', 'SAGE', 3, 4, 'gnnm')
        spec.update(changes)
        return spec

    def test_frozen_values_are_the_paper_protocol_with_a_longer_limit(self):
        self.assertEqual(protocol.FROZEN['patience'], 300)
        self.assertEqual(protocol.FROZEN['min_steps'], 0)
        self.assertEqual(protocol.FROZEN['selection_interval'], 10)
        self.assertEqual(protocol.FROZEN['num_steps'], 40000)
        protocol.check_spec(self.spec())

    def test_any_protocol_change_is_refused(self):
        for key, value in (('patience', 1000), ('min_steps', 5000), ('selection_interval', 1),
                           ('num_steps', 5000), ('continuation_steps', 20000), ('hidden_dim', 256),
                           ('protocol_id', 'other')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.check_spec(self.spec(**{key: value}))

    def test_earlier_result_folders_are_refused(self):
        for name in ('results_merged', 'results_merged/fixed_protocol_x',
                     'ablation/campaign/runs/production_owner_20261112_giga'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                protocol.check_output_root(REPO / name, REPO)
        with self.assertRaises(ValueError):
            protocol.check_output_root(REPO / 'ablation/campaign/runs/new_root', REPO)
        protocol.check_output_root(REPO / 'ablation/campaign/runs/fixed_protocol_20261116', REPO)

    def test_a_folder_of_another_protocol_is_refused(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory) / 'fixed_protocol_root'
            root.mkdir()
            (root / 'PROTOCOL_ID').write_text('something-else\n')
            with self.assertRaises(ValueError):
                protocol.check_output_root(root, REPO)

    def test_collector_folders_are_never_reused(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            out = Path(directory) / 'fixed_protocol_out'
            protocol.fresh_output_dir(out, REPO)
            (out / 'table.csv').write_text('x\n')
            with self.assertRaises(FileExistsError):
                protocol.fresh_output_dir(out, REPO)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan()
        cls.fits = [task for task in cls.plan['tasks'] if task['metadata']['action'] == 'node_fit']

    def test_every_node_fit_uses_the_frozen_protocol(self):
        for task in self.fits:
            protocol.check_spec(task['metadata']['spec'])
        self.assertEqual(self.plan['campaign_id'], protocol.CAMPAIGN_ID)

    def test_main_rerun_covers_exactly_the_rerun_list(self):
        with protocol_path('rerun_list.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        expected = sum(4 if row['method'] == 'ens' else 1 for row in rows)
        rerun = {task['id'] for task in self.fits if task['metadata']['stage'] == 'main_rerun'}
        # GAT pilot cells reuse identical fits, so the reruns are counted through the score tasks.
        scores = [task for task in self.plan['tasks'] if task['id'].startswith('main_rerun/')]
        self.assertEqual(len(scores), len(rows))
        self.assertEqual(sum(len(task['metadata']['fits']) for task in scores), expected)
        self.assertTrue(rerun)

    def test_seeds_follow_the_paper(self):
        for task in self.fits:
            spec = task['metadata']['spec']
            if spec['method'] == 'base' and spec['seed'] >= 1000:
                self.assertEqual(spec['seed'] // 1000, spec['split'])
            elif spec['method'] != 'base':
                self.assertEqual(spec['seed'], spec['split'])

    def test_roman_ablation_is_complete(self):
        roman = Counter(task['metadata']['spec']['variant'] for task in self.fits
                        if task['metadata']['stage'] == 'roman_ablation')
        self.assertEqual(set(roman), set(ROMAN_VARIANTS))
        self.assertTrue(all(count == 7 * 5 * 10 for count in roman.values()))

    def test_first_half_of_ablation_splits_is_queued_first(self):
        for task in self.fits:
            spec = task['metadata']['spec']
            if task['metadata']['stage'] == 'roman_ablation':
                self.assertEqual(task['priority'], 4 if spec['split'] < 5 else 6)


class CollectTests(unittest.TestCase):
    """Uses the frozen archive table; earlier result folders are only read."""

    def fake_score(self, root, task_id, setting, val, test, depth=5, split=0, model='SAGE',
                   dataset='roman-empire', protocol_id=protocol.PROTOCOL_ID, role='scientific'):
        out = Path(root) / 'tasks' / task_id / 'output'
        out.mkdir(parents=True)
        (out / 'done.json').write_text(__import__('json').dumps(dict(
            dataset=dataset, model=model, num_layers=depth, split=split, setting=setting,
            val_metric=val, test_metric=test,
            config=dict(fixed_spec(dataset, model, depth, split, 'gnnm'), best_step=6000, stopped_step=6300,
                        stop_reason='patience', protocol_id=protocol_id, role=role))))

    def test_reruns_replace_only_their_own_archived_runs(self):
        from ablation.fixed_protocol import collect
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=REPO) as tmp:
            root = Path(tmp) / 'fixed_protocol_root'
            self.fake_score(root, 'main_rerun/roman-empire/SAGE/L5/split0/GNNM', 'GNNM', 0.99, 0.98)
            self.fake_score(root, 'score/roman/SAGE/L5/split0/input_only', 'input_only', 0.5, 0.4)
            (root / 'tasks/main_rerun/roman-empire/GCN/L1/split0/BASE/output').mkdir(parents=True)
            out = protocol.fresh_output_dir(Path(tmp) / 'fixed_protocol_out', REPO)
            status = collect.collect(root, collect.load_archive(), out)
            runs = pd.read_csv(out / 'main_runs.csv')
            archive = collect.load_archive()
            self.assertEqual(len(runs), len(archive))
            if (REPO / 'results_merged').exists():
                self.assertEqual(len(collect.archived_main(REPO / 'results_merged')), len(archive))
            cell = runs[(runs.model == 'SAGE') & (runs.num_layers == 5) & (runs.split == 0)
                        & (runs.method == 'gnnm') & (runs.dataset == 'roman-empire')]
            self.assertEqual(cell.source.tolist(), ['rerun'])
            self.assertAlmostEqual(cell.test_metric.item(), 0.98)
            self.assertEqual(status['main_rerun_unfinished'], ['main_rerun/roman-empire/GCN/L1/split0/BASE'])
            ablation = pd.read_csv(out / 'roman_ablation_summary.csv')
            self.assertIn('input_only', set(ablation.setting))

    def test_smoke_results_are_refused(self):
        from ablation.fixed_protocol import collect
        with tempfile.TemporaryDirectory(dir=REPO) as tmp:
            root = Path(tmp) / 'fixed_protocol_root'
            self.fake_score(root, 'main_rerun/x', 'GNNM', 0.9, 0.9, role='engineering_smoke')
            with self.assertRaises(ValueError):
                collect.score_results(root, 'main_rerun')

    def test_results_of_another_protocol_are_refused(self):
        from ablation.fixed_protocol import collect
        with tempfile.TemporaryDirectory(dir=REPO) as tmp:
            root = Path(tmp) / 'fixed_protocol_root'
            self.fake_score(root, 'main_rerun/x', 'GNNM', 0.9, 0.9, protocol_id='old')
            with self.assertRaises(ValueError):
                collect.score_results(root, 'main_rerun')


def protocol_path(name):
    return Path(protocol.__file__).with_name(name)


if __name__ == '__main__':
    unittest.main()
