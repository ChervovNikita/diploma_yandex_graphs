"""Finite tensor/ownership checks; no native model, task dataset or fit."""
import copy
import os
from pathlib import Path
import unittest

import torch

from core import (FRAME_POLICY, assign_private_parameters, build_residual,
                  remove_private_ownership)
from extension import PHASE, _source_bindings, source_mapping


class PrivateInputTensorChecks(unittest.TestCase):
    def test_high_width_initializer_matches_existing_molecular_candidate(self):
        fit_seed = 101
        residual = build_residual(torch, torch.zeros(1), width=300, seed=fit_seed+901003)
        generator = torch.Generator(device="cpu").manual_seed(fit_seed+901003)
        basis, _ = torch.linalg.qr(torch.randn(300, 32, generator=generator), mode="reduced")
        self.assertTrue(torch.equal(residual.down, basis.T.reshape(4, 8, 300)))
        self.assertEqual(residual.metadata()["actual_frame"], "one_disjoint_orthonormal_frame")

    def test_zero_up_and_different_task_gradient_paths_without_global_rng(self):
        before = torch.get_rng_state().clone()
        residual = build_residual(torch, torch.zeros(1), width=64, seed=901104)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        encoded = torch.linspace(-1.2, .8, 6*2*64).reshape(6, 2, 64)
        outputs = [residual(encoded, member) for member in range(4)]
        self.assertTrue(all(torch.equal(output, encoded) for output in outputs))
        coefficient = torch.linspace(.2, 1.4, encoded.numel()).reshape_as(encoded)
        sum((output*coefficient).sum() for output in outputs).backward()
        self.assertTrue(torch.equal(residual.down.grad, torch.zeros_like(residual.down)))
        self.assertTrue(bool((residual.up.grad.reshape(4, -1).norm(dim=1) > 0).all()))
        self.assertFalse(torch.allclose(residual.up.grad[0], residual.up.grad[1]))

    def test_dimension_policy_does_not_invent_disjoint_low_width_subspaces(self):
        high = build_residual(torch, torch.zeros(1), width=64, seed=901104)
        low = build_residual(torch, torch.zeros(1), width=16, seed=901104)
        tiny = build_residual(torch, torch.zeros(1), width=3, seed=901104)
        torch.testing.assert_close(high.down.reshape(32, 64)@high.down.reshape(32, 64).T,
                                   torch.eye(32), atol=1e-5, rtol=1e-5)
        for member in range(4):
            torch.testing.assert_close(low.down[member]@low.down[member].T,
                                       torch.eye(8), atol=1e-5, rtol=1e-5)
        self.assertGreater(float((low.down[0]@low.down[1].T).norm()), .1)
        self.assertFalse(low.metadata()["four_disjoint_subspaces_fit"])
        self.assertEqual(tiny.metadata()["rank"], 3)
        self.assertEqual(low.metadata()["frame_policy"], FRAME_POLICY)

    def test_residual_state_restore_keeps_parameter_objects(self):
        residual = build_residual(torch, torch.zeros(1), width=16, seed=101)
        state = copy.deepcopy(residual.state_dict())
        other = build_residual(torch, torch.zeros(1), width=16, seed=303)
        parameter_ids = [id(p) for p in other.parameters()]
        other.load_state_dict(state, strict=True)
        self.assertEqual(parameter_ids, [id(p) for p in other.parameters()])
        self.assertTrue(torch.equal(other.down, residual.down))
        self.assertTrue(torch.equal(other.up, residual.up))

    def test_untrained_private_bank_removal_preserves_source_groups(self):
        anchor = torch.nn.Parameter(torch.tensor([.2, .3]))
        bias = torch.nn.Parameter(torch.tensor([.1]))
        prior_be = torch.nn.Parameter(torch.ones(4, 2))
        optimizer = torch.optim.Adam([{"params": [anchor, bias, prior_be], "lr": .02, "weight_decay": .1}])
        group = optimizer.param_groups[0]
        options = {k: v for k, v in group.items() if k != "params"}
        residual = build_residual(torch, anchor, width=16, seed=101)
        remove_private_ownership(optimizer, (prior_be,))
        self.assertEqual(assign_private_parameters(optimizer, anchor, residual.parameters()), 0)
        self.assertIs(optimizer.param_groups[0], group)
        self.assertEqual({k: v for k, v in group.items() if k != "params"}, options)
        self.assertEqual({id(p) for p in group["params"]}, {id(anchor), id(bias), *(id(p) for p in residual.parameters())})
        with self.assertRaises(ValueError):
            assign_private_parameters(optimizer, anchor, residual.parameters())
        self.assertEqual(len(optimizer.state), 0)  # no optimizer transition occurred

    def test_source_receipts_and_unsupported_interfaces_fail_closed(self):
        for family in ("node", "ogb", "link"):
            self.assertGreater(len(_source_bindings(family, Path(os.environ["PORTABLE_TEST_REPOSITORY_ROOT"]))), 1)
        with self.assertRaises(ValueError):
            _source_bindings("node", Path(os.environ["PORTABLE_TEST_REPOSITORY_ROOT"]), "0"*64)
        with self.assertRaises(ValueError):
            source_mapping(None, "unsupported")


if __name__ == "__main__":
    unittest.main()
