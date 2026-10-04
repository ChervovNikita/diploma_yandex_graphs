"""Unrun synthetic objective/state checks; no data/predictive study."""
import argparse
from copy import deepcopy
import json
import unittest
import bank
import bank_driver
from native_driver import native_train_step
from runtime import load_runtime
from schedule import BankClock
from views import native_edges

RT = None
RECIPE = dict(in_channels=3, hidden_channels=4, out_channels=5, local_layers=2, global_layers=1,
              heads=2, beta=-1, pre_ln=False, in_dropout=0.2, dropout=0.3, global_dropout=0.3)


def exact(test, a, b):
    if isinstance(a, RT.torch.Tensor):
        test.assertEqual(a.dtype, b.dtype)
        test.assertEqual(tuple(a.shape), tuple(b.shape))
        test.assertTrue(RT.torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            exact(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(type(a), type(b))
        test.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            exact(test, left, right)
    else:
        test.assertEqual(a, b)


class NativeReferenceChecks(unittest.TestCase):
    def fixture(self):
        model, optimizer, _ = bank.build_bank(RT, "tied_persistent", 17, synthetic_recipe=RECIPE)
        x = RT.torch.arange(36, dtype=RT.torch.float32).reshape(12, 3) / 20
        edges = native_edges(12, ((a, b) for a in range(12) for b in range(a + 1, 12)))
        graph = RT.torch.tensor(edges, dtype=RT.torch.long).t().contiguous()
        return model, optimizer, x, graph, tuple(range(10)), (0,) * 5 + (1,) * 5

    def test_four_streamed_native_CEs_match_mean4_objective_and_single_Adam_step(self):
        for global_stage in (False, True):
            model, optimizer, x, graph, ids, labels = self.fixture()
            bank.set_stage(model, global_stage)
            reference = deepcopy(model)
            ref_optimizer = RT.torch.optim.Adam(reference.parameters(), lr=0.001, weight_decay=0.0)
            rng = bank.rng_state(RT, "cpu")
            buffers = bank.cpu_tree(RT, dict(model.named_buffers()))
            update = 201 if global_stage else 1
            actual = native_train_step(RT, model, optimizer, x, graph, ids, labels, update)
            post_rng = bank.rng_state(RT, "cpu")
            bank.restore_rng(RT, "cpu", rng)
            reference.train()
            logits = reference(x, graph)
            index, target = RT.torch.tensor(ids), RT.torch.tensor(labels)
            losses = [RT.torch.nn.functional.nll_loss(RT.torch.nn.functional.log_softmax(
                logits[m], dim=1).index_select(0, index), target) for m in range(4)]
            mean4 = RT.torch.stack(losses).mean()
            mean4.backward()
            self.assertAlmostEqual(actual, float(mean4.detach()), places=6)
            for a, b in zip(model.parameters(), reference.parameters()):
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    RT.torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
            ref_optimizer.step()
            for a, b in zip(model.parameters(), reference.parameters()):
                RT.torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
            exact(self, post_rng, bank.rng_state(RT, "cpu"))
            exact(self, buffers, bank.cpu_tree(RT, dict(model.named_buffers())))
            exact(self, buffers, bank.cpu_tree(RT, dict(reference.named_buffers())))

    def test_full_native_reference_state_replay_and_selected_local_transition(self):
        for global_stage in (False, True):
            model, optimizer, x, graph, ids, labels = self.fixture()
            bank.set_stage(model, global_stage)
            update = 201 if global_stage else 1
            native_train_step(RT, model, optimizer, x, graph, ids, labels, update)
            logits = bank_driver.eval_logits(RT, model, x, graph)
            binding = {"synthetic_native_reference": True}
            selection = {"global": global_stage, "actual_update": update}
            image = bank.snapshot(RT, model, optimizer, "cpu", selection, binding)
            other, other_optimizer, _, _, _, _ = self.fixture()
            bank.restore(RT, other, other_optimizer, "cpu", image, binding)
            exact(self, image, bank.snapshot(RT, other, other_optimizer, "cpu", selection, binding))
            self.assertTrue(RT.torch.equal(logits, bank_driver.eval_logits(RT, other, x, graph)))
            a = native_train_step(RT, model, optimizer, x, graph, ids, labels, update + 1)
            state = bank.snapshot(RT, model, optimizer, "cpu", selection, binding)
            bank.restore(RT, other, other_optimizer, "cpu", image, binding)
            b = native_train_step(RT, other, other_optimizer, x, graph, ids, labels, update + 1)
            self.assertEqual(a, b)
            exact(self, state, bank.snapshot(RT, other, other_optimizer, "cpu", selection, binding))
            if not global_stage:
                live = bank.rng_state(RT, "cpu")
                clock = BankClock(200)  # Clock fixture; no full scientific schedule.
                bank.transition(RT, model, optimizer, "cpu", image, clock)
                self.assertTrue(bank.stage(model))
                self.assertEqual(clock.actual_update, 200)
                exact(self, live, bank.rng_state(RT, "cpu"))
                exact(self, image["model"], bank.cpu_tree(RT, model.state_dict()))
                exact(self, image["optimizer"], bank.cpu_tree(RT, optimizer.state_dict()))
                self.assertEqual(clock.advance(), 201)
                bank.restore(RT, model, optimizer, "cpu", image, binding)
                self.assertFalse(bank.stage(model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "NATIVE_GNNM_OBJECTIVE_STATE_CHECKS_UNRUN_ROOT_ADMISSION_REQUIRED"}))
    else:
        RT = load_runtime(execute=True, manifest_sha256=args.manifest_sha256, protocol_sha256=args.protocol_sha256)
        RT.torch.set_num_threads(1)
        unittest.main(argv=[__file__])
