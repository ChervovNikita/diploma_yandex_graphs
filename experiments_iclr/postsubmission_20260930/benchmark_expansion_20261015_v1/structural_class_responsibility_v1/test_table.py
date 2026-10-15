"""Exact rational/mass/tie/control checks; no tensors or task data."""
from fractions import Fraction
import random
import unittest

from table import make_table, native_incoming_counts


class StructuralTableChecks(unittest.TestCase):
    def test_fractional_ties_and_nondivisible_class_counts(self):
        table = make_table(range(8), [0]*5+[1]*3, [1, 1, 1, 2, 5, 7, 7, 9], role="TRAIN")
        for member in range(4):
            self.assertEqual(table.q[member][0], table.q[member][1])
            self.assertEqual(table.q[member][1], table.q[member][2])
            self.assertEqual(sum(table.q[member][:5]), Fraction(5, 4))
            self.assertEqual(sum(table.weights[member][:5]), 5)
            self.assertEqual(sum(table.q[member][5:]), Fraction(3, 4))
        self.assertEqual([table.q[m][0] for m in range(4)], [Fraction(5, 12), Fraction(5, 12), Fraction(1, 6), Fraction()])

    def test_all_tied_or_single_object_is_exact_uniform_no_redraw(self):
        for count in (1, 2, 3, 5, 17):
            table = make_table(range(count), [0]*count, [4]*count, role="TRAIN", permutation_seed=101)
            self.assertTrue(all(value == Fraction(1, 4) for row in table.q for value in row))
            self.assertTrue(all(value == 1 for row in table.weights for value in row))

    def test_control_moves_complete_vectors_within_classes(self):
        ids, labels, metrics = tuple(range(13)), [0]*5+[1]*8, [0, 0, 3, 5, 9, 1, 2, 2, 2, 5, 6, 8, 11]
        before = random.getstate()
        original = make_table(ids, labels, metrics, role="TRAIN")
        control = make_table(ids, labels, metrics, role="TRAIN", permutation_seed=101)
        self.assertEqual(before, random.getstate())
        self.assertGreater(control.summary()["moved_positions"], 0)
        for destination, source in enumerate(control.permutation):
            self.assertEqual(labels[destination], labels[source])
            self.assertEqual(tuple(control.weights[m][destination] for m in range(4)),
                             tuple(original.weights[m][source] for m in range(4)))
        for label in set(labels):
            positions = [i for i, value in enumerate(labels) if value == label]
            self.assertEqual(sorted(tuple(control.weights[m][i] for m in range(4)) for i in positions),
                             sorted(tuple(original.weights[m][i] for m in range(4)) for i in positions))
        self.assertEqual(control.digest(), make_table(ids, labels, metrics, role="TRAIN", permutation_seed=101).digest())

    def test_structured_ties_do_not_depend_on_object_order(self):
        first = make_table([5, 1, 3, 9, 2], [0]*5, [1, 1, 1, 2, 5], role="TRAIN")
        second = make_table([2, 9, 3, 1, 5], [0]*5, [5, 2, 1, 1, 1], role="TRAIN")
        for object_id in first.ids:
            a, b = first.ids.index(object_id), second.ids.index(object_id)
            self.assertEqual(tuple(first.q[m][a] for m in range(4)), tuple(second.q[m][b] for m in range(4)))

    def test_native_incoming_degree_counts_existing_multiset_only(self):
        self.assertEqual(native_incoming_counts(4, [0, 0, 1, 2, 3], [1, 1, 1, 0, 3], [1, 3, 0]), (3, 1, 1))

    def test_role_missing_class_and_duplicate_ids_are_rejected(self):
        for role, ids, labels in [("VALID", [1], [0]), ("TRAIN", [1], [None]), ("TRAIN", [1, 1], [0, 0])]:
            with self.assertRaises(ValueError):
                make_table(ids, labels, [2]*len(ids), role=role)


if __name__ == "__main__":
    unittest.main()
