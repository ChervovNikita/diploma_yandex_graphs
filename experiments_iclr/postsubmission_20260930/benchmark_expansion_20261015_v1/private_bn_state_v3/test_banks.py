"""CPU semantics of state ownership, not graph training or quality evidence."""
import copy
import unittest

import torch

from banks import RouteBatchNormState, attach_molecular


class OwnershipChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(313)
        torch.set_num_threads(1)

    def test_delayed_backward_preserves_train_outputs_and_gradients(self):
        for momentum in (.1, None):
            native = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.BatchNorm1d(4, momentum=momentum),
                                         torch.nn.GELU(), torch.nn.Linear(4, 2))
            ordinary = copy.deepcopy(native)
            manager = RouteBatchNormState(native)
            xs = [torch.randn(11, 6) + m for m in range(4)]
            out = []
            for m, x in enumerate(xs):
                with manager.route(m):
                    out.append(native(x))
            reference = [ordinary(x) for x in xs]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(out, reference)))
            sum(x.square().mean() for x in out).backward()
            sum(x.square().mean() for x in reference).backward()
            self.assertTrue(all(torch.equal(p.grad, q.grad) for p, q in
                                zip(native.parameters(), ordinary.parameters())))
            manager.assert_native_identity()
            self.assertEqual(len(list(manager.parameters())), 0)
            self.assertEqual(native[1].num_batches_tracked.item(), 0)
            self.assertEqual(ordinary[1].num_batches_tracked.item(), 4)
            self.assertTrue(all(b.num_batches_tracked.item() == 1 for b in manager.banks[0]))

    def test_eval_uses_matching_route_without_mutation(self):
        native = torch.nn.BatchNorm1d(3, affine=True, momentum=1.)
        manager = RouteBatchNormState(native)
        for m in range(4):
            with manager.route(m):
                native(torch.randn(10, 3) + 3*m)
        native.eval()
        before = copy.deepcopy(manager.state_dict())
        x = torch.randn(7, 3)
        for m in range(4):
            bank = manager.banks[0][m]
            expected = (x-bank.running_mean)/(bank.running_var+native.eps).sqrt()
            expected = expected*native.weight+native.bias
            with manager.route(m):
                self.assertTrue(torch.allclose(native(x), expected, atol=1e-6))
        self.assertTrue(all(torch.equal(v, manager.state_dict()[k]) for k, v in before.items()))

    def test_checkpoint_restores_all_route_state(self):
        native = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.BatchNorm1d(4))
        manager = RouteBatchNormState(native)
        for m in range(4):
            for _ in range(m+1):
                with manager.route(m):
                    native(torch.randn(8, 4))
        fresh_native = copy.deepcopy(native)
        fresh = RouteBatchNormState(fresh_native)
        fresh.load_state_dict(copy.deepcopy(manager.state_dict()))
        native.eval(); fresh_native.eval()
        x = torch.randn(9, 4)
        for m in range(4):
            with manager.route(m):
                a = native(x)
            with fresh.route(m):
                b = fresh_native(x)
            self.assertTrue(torch.equal(a, b))

    def test_exception_and_nesting_restore_original_objects(self):
        native = torch.nn.BatchNorm1d(3)
        manager = RouteBatchNormState(native)
        original = native.running_mean
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with manager.route(2):
                with self.assertRaises(ValueError):
                    with manager.route(1):
                        pass
                raise RuntimeError("deliberate")
        self.assertIs(native.running_mean, original)
        manager.assert_native_identity()
        with self.assertRaises(ValueError):
            with manager.route(-1):
                pass

    def test_unused_branch_and_untracked_state(self):
        native = torch.nn.ModuleDict({"used": torch.nn.BatchNorm1d(3),
                                      "unused": torch.nn.BatchNorm1d(3),
                                      "untracked": torch.nn.BatchNorm1d(3, track_running_stats=False)})
        manager = RouteBatchNormState(native)
        with manager.route(1):
            native["used"](torch.randn(9, 3))
            native["untracked"](torch.randn(9, 3))
        self.assertEqual(manager.banks[0][1].num_batches_tracked.item(), 1)
        self.assertEqual(manager.banks[1][1].num_batches_tracked.item(), 0)
        self.assertEqual(manager.ignored_untracked, ["untracked"])
        with self.assertRaises(ValueError):
            RouteBatchNormState(torch.nn.Linear(3, 2))

    def test_molecular_interface_registers_state_and_keeps_optimizer_parameters(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.native = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
                self._anchors = tuple(self.native.parameters())

            def run_member(self, member, batch):
                return self.native(batch)

            def assert_source_identity(self):
                assert all(p is q for p, q in zip(self.native.parameters(), self._anchors))

        class Session:
            pass

        session = Session(); session.model = Wrapper()
        parameters = tuple(session.model.parameters())
        attach_molecular(session)
        self.assertTrue(all(p is q for p, q in zip(parameters, session.model.parameters())))
        self.assertEqual(len(parameters), len(tuple(session.model.parameters())))
        out = [session.model.run_member(m, torch.randn(8, 3)) for m in range(4)]
        sum(x.square().mean() for x in out).backward()
        self.assertTrue(any(k.startswith("route_bn_state.banks.") for k in session.model.state_dict()))
        self.assertEqual(session.model.native[0].num_batches_tracked.item(), 0)

    def test_attached_candidate_rejects_bypass_and_active_checkpoint(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.native = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))

            def run_member(self, member, batch):
                return self.native(batch)

            def assert_source_identity(self):
                pass

        class Session:
            pass

        session = Session(); session.model = Wrapper(); attach_molecular(session)
        model, x = session.model, torch.randn(8, 3)
        idle_state = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "run_member"):
            model.native(x)
        with self.assertRaisesRegex(RuntimeError, "run_member"):
            model.native[0](x)
        with model.route_bn_state.route(0):
            for save in (model.state_dict, model.native.state_dict, model.route_bn_state.state_dict):
                with self.assertRaisesRegex(RuntimeError, "active"):
                    save()
            with self.assertRaisesRegex(RuntimeError, "active"):
                model.load_state_dict(idle_state)
        model.load_state_dict(idle_state)
        self.assertTrue(torch.isfinite(model.run_member(2, x)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
