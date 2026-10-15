"""Finite CPU DataLoader bookkeeping only; no model or task data."""
import unittest

import torch
from source_rng import OmittedTestIteratorRNG, check_source_loader_flags, loader_rng_semantics


class SourceIteratorChecks(unittest.TestCase):
    def test_serial_integer_and_tensor_dataset_match_exact_draw(self):
        before = torch.get_rng_state().clone()
        record = loader_rng_semantics(torch)
        self.assertEqual(len(record["cases"]), 24)
        self.assertTrue(all(row["exact_cpu_rng_state_match"] for row in record["cases"]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(record["model_calls"], 0)

    def test_changed_worker_or_shuffle_flags_fail_before_iteration(self):
        loader = torch.utils.data.DataLoader
        train = loader([0, 1], batch_size=32, shuffle=True, num_workers=0)
        valid = loader([0, 1], batch_size=32, shuffle=False, num_workers=0)
        check_source_loader_flags(torch, train, valid)
        for altered in (loader([0, 1], batch_size=32, shuffle=False, num_workers=1),
                        loader([0, 1], batch_size=32, shuffle=True, num_workers=0),
                        loader([0, 1], batch_size=32, generator=torch.Generator())):
            with self.assertRaises(ValueError):
                check_source_loader_flags(torch, train, altered)

    def test_emulation_is_disabled_without_explicit_declaration(self):
        self.assertIsNone(OmittedTestIteratorRNG.create(None, None, {}, None, None))
        with self.assertRaises(ValueError):
            OmittedTestIteratorRNG.create(None, None,
                {"source_rng_emulation": {}, "family": "node"}, None, None)


if __name__ == "__main__":
    unittest.main()
