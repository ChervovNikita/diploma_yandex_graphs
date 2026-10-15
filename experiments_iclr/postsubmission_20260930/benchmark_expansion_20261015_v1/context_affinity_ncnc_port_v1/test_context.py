"""Small exact graph checks for the declared target law; no native model fit."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
import torch

HERE = Path(__file__).resolve().parent
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
context = load(HERE/'context.py', '_link_context_tests')
planning = load(HERE.parent/'link/planning.py', '_link_context_frozen_selection')
core = load(HERE.parent/'context_affinity_v1/core.py', '_link_context_core_test')


class ContextTests(unittest.TestCase):
    def graph(self, edges, n=7):
        return context.TrainGraph(torch, torch.tensor(edges, dtype=torch.long).t().contiguous(), n)

    def test_simple_graph_removes_loops_deduplicates_without_uint8_overflow(self):
        g = self.graph([(0, 1)]*300+[(1, 0)]*300+[(0, 0), (1, 2)])
        self.assertEqual(g.degree.tolist(), [1, 2, 1, 0, 0, 0, 0])
        self.assertEqual(g.receipt['simple_undirected_edges'], 2)
        self.assertEqual(g.receipt['source_loop_records_removed'], 1)

    def test_own_pair_exclusion_and_all_three_logarithms(self):
        # Triangle0/1/2 with witness2 also connected to3; pair0/1 is removed.
        g = self.graph([(0, 1), (0, 2), (1, 2), (2, 3)])
        a, b, r = g.contexts(torch, [(0, 1, 0)], torch.float64)
        torch.testing.assert_close(a, torch.tensor([[math.log(2), math.log(2)]], dtype=torch.float64))
        expected = torch.tensor([[math.log(2), math.log1p(1/math.log(3)), math.log1p(1/3)]], dtype=torch.float64)
        torch.testing.assert_close(b, expected)
        self.assertTrue(r['query_statistics'][0]['pair_present_in_full_TRAIN_graph'])
        self.assertEqual(r['query_statistics'][0]['degree_without_pair'], [1, 1])

    def test_absent_query_uses_same_exclusion_law(self):
        g = self.graph([(0, 2), (1, 2), (2, 3)])
        a, b, r = g.contexts(torch, [(0, 1, 0)], torch.float64)
        self.assertFalse(r['query_statistics'][0]['pair_present_in_full_TRAIN_graph'])
        self.assertEqual(r['query_statistics'][0]['CN'], 1)
        torch.testing.assert_close(b[0, 2], torch.tensor(math.log1p(1/3), dtype=torch.float64))
        self.assertEqual(r['query_statistics'][0]['degree_without_pair'], [1, 1])

    def test_empty_witness_is_zero_and_graph_immutable(self):
        g = self.graph([(0, 1), (1, 2)])
        before = (g.degree.copy(), g.indices.copy(), g.indptr.copy())
        a, b, receipt = g.contexts(torch, [(0, 1, 0), (0, 6, 0)], torch.float64)
        self.assertTrue(torch.equal(b, torch.zeros(2, 3, dtype=torch.float64)))
        self.assertEqual(receipt['zero_B'], 2)
        self.assertEqual(g.degree.tolist(), before[0].tolist())
        self.assertEqual(g.indices.tolist(), before[1].tolist())
        self.assertEqual(g.indptr.tolist(), before[2].tolist())

    def test_semantic_conflict_duplicate_and_self_gates(self):
        keys = planning.query_keys([(1, 0), (0, 1), (2, 3), (3, 2), (4, 4), (4, 5), (5, 6)])
        groups = [1, 1, 1, 0, 0, 0, 0]
        rows, ids, r = context.select(planning, keys, groups, seed=5, nodes=7)
        self.assertEqual(rows, [0, 5, 6]); self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(r['self_rows_omitted_from_AUX'], 1)
        self.assertEqual(r['conflicting_semantic_objects'], 1)
        self.assertEqual(r['conflicting_source_rows_omitted_from_AUX'], 2)
        self.assertEqual(r['same_group_duplicate_rows_omitted_from_AUX'], 1)
        self.assertEqual(r['native_F_rows_removed'], 0)

    def test_frozen_hash_limit_and_native_order(self):
        keys = planning.query_keys([(0, i) for i in range(1, 601)]+[(1, i) for i in range(2, 602)])
        groups = [1]*600+[0]*600
        rows, ids, r = context.select(planning, keys, groups, seed=19, nodes=603)
        expected = []
        for group in (0, 1):
            eligible = [i for i, g in enumerate(groups) if g == group]
            expected += sorted(eligible, key=lambda i: (planning._hash(19, 'anchor', keys[i]), i))[:512]
        self.assertEqual(rows, sorted(expected)); self.assertEqual(r['selected_by_group'], {'0': 512, '1': 512})

    def test_relation_and_zero_support_are_preserved(self):
        torch.manual_seed(12)
        z = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([1, 1, 0]); ids = torch.tensor([1, 2, 3]); relations = torch.tensor([0, 1, 0])
        result = core.context_auxiliary(torch, preheads=z, labels=labels, object_ids=ids,
            relation_ids=relations, context_a=torch.zeros(3, 2, dtype=torch.float64),
            context_b=torch.zeros(3, 3, dtype=torch.float64), mode='split', reduction='binary_separate')
        self.assertEqual(result['diagnostics']['eligible_anchors'], 0)
        self.assertEqual(result['diagnostics']['omitted_anchors'], 3)
        result['loss'].backward(); self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))

    def test_common_split_equal_when_context_targets_equal_and_binary_sum(self):
        torch.manual_seed(18)
        z = torch.randn(4, 8, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0]*4+[1]*4); ids = torch.arange(8)
        factual = torch.randn(8, 3, dtype=torch.float64)
        kw = dict(preheads=z, labels=labels, object_ids=ids, context_a=factual, context_b=factual,
            reduction='binary_separate')
        common = core.context_auxiliary(torch, mode='common', **kw)
        split = core.context_auxiliary(torch, mode='split', **kw)
        torch.testing.assert_close(common['loss'], split['loss'], atol=0, rtol=0)
        balanced = core.context_auxiliary(torch, mode='common', **{**kw, 'reduction': 'class_balanced'})
        torch.testing.assert_close(common['A'], 2*balanced['A'], atol=0, rtol=0)
        g1 = torch.autograd.grad(common['loss'], z, retain_graph=True)[0]
        g2 = torch.autograd.grad(split['loss'], z)[0]
        torch.testing.assert_close(g1, g2, atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(2); unittest.main()
