"""CPU checks for the cost measurements.

    python -m unittest ablation.p07_cost_small_setup.test_p07 -v
"""

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.resumable import JobStore  # noqa: E402
from ablation.common.worker import Preempted, Job, run_worker  # noqa: E402
from ablation.p07_cost_small_setup import collect, run  # noqa: E402
from ablation.p07_cost_small_setup.models import VARIANTS, build_units  # noqa: E402


DEVICE = torch.device('cpu')
INPUTS = 12
CLASSES = 3
NODES = 80


def toy_data():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(NODES, INPUTS, generator=generator)
    data = Data(x=x, y=x[:, :CLASSES].argmax(dim=1),
                edge_index=torch.randint(0, NODES, (2, 400), generator=generator))
    order = torch.randperm(NODES, generator=generator)
    for name, indices in (('train_mask', order[:40]), ('val_mask', order[40:60]),
                          ('test_mask', order[60:])):
        mask = torch.zeros(NODES, dtype=torch.bool)
        mask[indices] = True
        setattr(data, name, mask)
    return data


def units_of(variant, model='SAGE'):
    return build_units(variant, model, 2, INPUTS, 64, CLASSES, 0, DEVICE)


class VariantTests(unittest.TestCase):
    def test_every_variant_measures_on_every_kind_of_backbone(self):
        data = toy_data()
        for variant in VARIANTS:
            for model in ('SAGE', 'GT', 'TAG', 'ResNet'):
                result = run.measure(units_of(variant, model), data, False, 'roman-empire', DEVICE,
                                     1, 2, lambda step: None)
                self.assertGreater(result['full_step_ms'], 0.0, (variant, model))
                self.assertEqual(len(result['full_step_series_ms']), 2)

    def test_ensemble_has_four_differently_seeded_full_models(self):
        units = units_of('ens')
        self.assertEqual(len(units), 4)
        self.assertEqual([members for _, members, _ in units], [1, 1, 1, 1])
        first, second = (next(unit[0].parameters()) for unit in units[:2])
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(units_of('base')), 1)

    def test_parameter_count_of_ensemble_is_four_bases(self):
        data = toy_data()
        count = lambda variant: run.measure(
            units_of(variant), data, False, 'roman-empire', DEVICE, 0, 1,
            lambda step: None)['num_params']
        self.assertEqual(count('ens'), 4 * count('base'))
        self.assertLess(count('gnnm'), 2 * count('base'))

    def test_ensemble_inference_uses_every_member(self):
        data = toy_data()
        units = units_of('ens')
        pooled = run.pooled_logits(units, data)
        single = torch.stack([unit[0](data, data.x, tabm_seed=0).detach() for unit in units]).mean(0)
        self.assertTrue(torch.allclose(pooled, single))

    def test_stop_request_abandons_the_job(self):
        stop = SimpleNamespace(requested=True)

        def tick(step):
            if stop.requested:
                raise Preempted()

        with self.assertRaises(Preempted):
            run.measure(units_of('base'), toy_data(), False, 'roman-empire', DEVICE, 1, 1, tick)


class LinearityTests(unittest.TestCase):
    def test_constant_step_time_is_exactly_linear(self):
        fit, error = collect.step_linearity([5.0] * 200)
        self.assertAlmostEqual(fit, 1.0)
        self.assertAlmostEqual(error, 0.0)

    def test_slower_second_half_is_caught_by_the_extrapolation(self):
        series = [5.0] * 100 + [15.0] * 100
        _, error = collect.step_linearity(series)
        self.assertAlmostEqual(error, -50.0)

    def test_member_scaling_recovers_slope_and_intercept(self):
        import pandas as pd
        frame = pd.DataFrame([
            {'dataset': 'a', 'model': 'SAGE', 'num_layers': 2, 'split': 0, 'variant': name,
             'train_step_ms': 3.0 + 2.0 * members}
            for name, members in (('gnnm_m2', 2), ('gnnm', 4), ('gnnm_m8', 8))
        ])
        table = collect.member_scaling(frame)
        self.assertAlmostEqual(table.median_r2[0], 1.0)
        self.assertAlmostEqual(table.mean_ms_per_member[0], 2.0)
        self.assertAlmostEqual(table.mean_intercept_ms[0], 3.0)

    def test_ratios_are_paired_by_configuration(self):
        import pandas as pd
        rows = []
        for model, base_ms in (('SAGE', 10.0), ('GT', 100.0)):
            for variant, factor in (('base', 1.0), ('ens', 4.0), ('gnnm', 8.0)):
                rows.append({'dataset': 'a', 'model': model, 'num_layers': 1, 'split': 0,
                             'variant': variant,
                             **{column: base_ms * factor for column in collect.COST_COLUMNS}})
        table = collect.by_variant(pd.DataFrame(rows)).set_index('variant')
        self.assertAlmostEqual(table.loc['gnnm', 'full_step_ms_vs_base'], 8.0)
        self.assertAlmostEqual(table.loc['gnnm', 'full_step_ms_vs_ens'], 2.0)


class WorkerIntegrationTests(unittest.TestCase):
    def test_jobs_cover_the_full_design_and_are_restartable(self):
        arguments = argparse.Namespace(
            datasets=['a'], variants=['base', 'gnnm'], models=['SAGE'], layers=[1, 2], splits=[0])
        jobs = run.make_jobs(arguments)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0].key, 'a/base/SAGE/L1/split0')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def execute(job, store, stop):
                calls.append(job.key)
                return {'value': 1}

            self.assertEqual(run_worker(root, jobs, execute, wait_for_others=False), 0)
            self.assertEqual(run_worker(root, jobs, execute, wait_for_others=False), 0)
            self.assertEqual(len(calls), 4)
            self.assertTrue(JobStore(root / jobs[0].key).is_done())


if __name__ == '__main__':
    unittest.main()
