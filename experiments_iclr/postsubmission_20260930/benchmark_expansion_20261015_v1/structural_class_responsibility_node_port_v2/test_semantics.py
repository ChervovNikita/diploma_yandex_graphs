"""Finite exact-table and session-local unchanged-rule checks; no native fits."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import unittest

path = Path(__file__).resolve().with_name("extension.py")
spec = importlib.util.spec_from_file_location("c2_node_semantic_extension", path)
extension = importlib.util.module_from_spec(spec); spec.loader.exec_module(extension)
table = extension.table_module()


class NodeTableSemantics(unittest.TestCase):
    def test_actual_target_records_preserve_multiplicity(self):
        # Includes two identical 0->1 records, existing 1->1, and 3->0.
        self.assertEqual(table.native_incoming_counts(4, [0,0,1,3], [1,1,1,0], [0,1,2]), (1,3,0))

    def test_fractional_ties_and_whole_vector_control(self):
        ids = list(range(14)); labels = [0]*7+[1]*7
        metrics = [0,0,1,3,3,8,9]*2
        actual = table.make_table(ids, labels, metrics, role="TRAIN")
        control = table.make_table(ids, labels, metrics, role="TRAIN", permutation_seed=101)
        self.assertEqual([row[0] for row in actual.q], [row[1] for row in actual.q])
        for i, source in enumerate(control.permutation):
            self.assertEqual(labels[i], labels[source])
            self.assertEqual([row[i] for row in control.weights], [row[source] for row in actual.weights])
        for label in (0,1):
            positions = [i for i,v in enumerate(labels) if v == label]
            for member in range(4):
                self.assertEqual(sum(actual.q[member][i] for i in positions), Fraction(7,4))
                self.assertEqual(sum(control.weights[member][i] for i in positions), 7)

    def test_all_ties_do_not_invent_diversity(self):
        actual = table.make_table([1,3,5], [0,0,0], [2,2,2], role="TRAIN")
        self.assertTrue(all(w == 1 for row in actual.weights for w in row))

    def test_no_nontrain_assignment(self):
        with self.assertRaises(ValueError):
            table.make_table([1], [0], [0], role="VALID")

    def test_capture_delegates_exact_objects_once_without_global_mutation(self):
        class Scalar:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def __float__(self): return float(self.value)
        f, w, token = Scalar(2), Scalar(3), object()
        class Native:
            other = token
            def step_private_responsibility(self, torch, core, optimizer, **kw):
                self.received = (torch, core, optimizer, kw)
                return token
        native = Native(); proxy = extension._RulesCapture(native)
        result = proxy.step_private_responsibility("torch", "core", "optimizer", F=f, F_weighted=w)
        self.assertIs(result, token); self.assertIs(proxy.other, token)
        self.assertEqual(proxy.capture, [(2.,3.)])
        self.assertIs(native.received[3]["F"], f); self.assertIs(native.received[3]["F_weighted"], w)
        self.assertNotIn("step_private_responsibility", native.__dict__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
