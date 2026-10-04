"""Unrun, admission-required synthetic CPU implementation tests; no prediction study."""
import argparse
from copy import deepcopy
import json
import unittest

from bank import (build_bank, cpu_tree, families, restore, restore_rng, rng_state,
                  set_stage, snapshot, stage, transition, untie)
from driver import correct_count, edge_tensors, eval_logits, train_step
from runtime import load_runtime
from schedule import BankClock, pass_plan
from views import TrainRole, construct_views, native_edges

RT = None
FIXTURE_RECIPE = dict(in_channels=3, hidden_channels=4, out_channels=5, local_layers=2,
                      global_layers=1, in_dropout=0.2, dropout=0.3, global_dropout=0.3,
                      heads=2, beta=-1, pre_ln=False)


def assert_tree_equal(test, left, right):
    if isinstance(left, RT.torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(tuple(left.shape), tuple(right.shape))
        test.assertTrue(RT.torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class NumericalChecks(unittest.TestCase):
    def test_probability_pool_is_class_softmax_then_member_mean(self):
        logits = RT.torch.tensor([[[100.0, 0.0, -10.0, -10.0, -10.0]],
                                  [[0.0, 5.0, -10.0, -10.0, -10.0]],
                                  [[0.0, 5.0, -10.0, -10.0, -10.0]],
                                  [[0.0, 5.0, -10.0, -10.0, -10.0]]])
        self.assertEqual(correct_count(RT, logits, (0,), (1,)), 1)
        self.assertEqual(int(logits.mean(0).argmax(-1)[0]), 0)

    def fixture(self, condition="tied_persistent"):
        model, optimizer, _ = build_bank(RT, condition, 17, synthetic_recipe=FIXTURE_RECIPE)
        role = TrainRole(12, 0, tuple(range(10)), (0,) * 5 + (1,) * 5, (10,), (11,))
        bundle = construct_views(role, native_edges(12, ((a, b) for a in range(12) for b in range(a + 1, 12))), 17)
        x = RT.torch.arange(36, dtype=RT.torch.float32).reshape(12, 3) / 20
        return model, optimizer, x, bundle, edge_tensors(RT, bundle, "cpu")

    def test_initial_member_functions_private_rows_and_storage(self):
        tied, _, x, bundle, graphs = self.fixture()
        before = rng_state(RT, "cpu")
        independent = untie(RT, tied)
        assert_tree_equal(self, before, rng_state(RT, "cpu"))
        for global_stage in (False, True):
            set_stage(tied, global_stage)
            set_stage(independent, global_stage)
            a, b = eval_logits(RT, tied, x, graphs["native"]), eval_logits(RT, independent, x, graphs["native"])
            self.assertTrue(RT.torch.equal(a, b))
            tied.train()
            independent.train()
            pre_rng = rng_state(RT, "cpu")
            a = RT.torch.stack([tied.forward_member(x, graphs[view], member)
                                for member, view in pass_plan("tied_persistent", 17, 201 if global_stage else 1)])
            post_rng = rng_state(RT, "cpu")
            restore_rng(RT, "cpu", pre_rng)
            b = RT.torch.stack([independent.forward_member(x, graphs[view], member)
                                for member, view in pass_plan("untied_persistent", 17, 201 if global_stage else 1)])
            self.assertTrue(RT.torch.equal(a, b))
            assert_tree_equal(self, post_rng, rng_state(RT, "cpu"))
        for member, family in enumerate(families(independent)):
            assert_tree_equal(self, tied.core.state_dict(), family.core.state_dict())
            for name in ("stem", "local_head", "global_head"):
                original, copy = getattr(tied, name), getattr(family, name)
                self.assertEqual(copy.members, 1)
                self.assertTrue(RT.torch.equal(original.weight, copy.weight))
                for factor in ("R", "S", "B"):
                    self.assertTrue(RT.torch.equal(getattr(original, factor)[member:member + 1], getattr(copy, factor)))
        pointer_sets = [{p.data_ptr() for p in f.parameters()} for f in families(independent)]
        self.assertTrue(all(not a.intersection(b) for i, a in enumerate(pointer_sets) for b in pointer_sets[i + 1:]))

    def test_accumulation_matches_summed_loss_and_forward_buffers(self):
        for condition in ("tied_persistent", "untied_persistent"):
            for global_stage in (False, True):
                model, optimizer, x, bundle, graphs = self.fixture(condition)
                set_stage(model, global_stage)
                reference = deepcopy(model)
                reference_optimizer = RT.torch.optim.Adam(reference.parameters(), lr=0.001, weight_decay=0.0)
                live_rng = rng_state(RT, "cpu")
                buffers = cpu_tree(RT, dict(model.named_buffers()))
                update = 201 if global_stage else 1
                actual_loss = train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                                         bundle["role"].train_labels, condition, 17, update)
                post_rng = rng_state(RT, "cpu")
                restore_rng(RT, "cpu", live_rng)
                reference.train()
                losses = []
                index = RT.torch.tensor(bundle["role"].train_ids)
                labels = RT.torch.tensor(bundle["role"].train_labels)
                for member, view in pass_plan(condition, 17, update):
                    logits = reference.forward_member(x, graphs[view], member)
                    losses.append(RT.torch.nn.functional.nll_loss(
                        RT.torch.nn.functional.log_softmax(logits, dim=1).index_select(0, index), labels))
                reference_loss = RT.torch.stack(losses).sum() / 4
                reference_loss.backward()
                self.assertAlmostEqual(actual_loss, float(reference_loss.detach()), places=6)
                for (_, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
                    if actual.grad is None:
                        self.assertIsNone(expected.grad)
                    else:
                        RT.torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7)
                reference_optimizer.step()
                for (_, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
                    RT.torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
                assert_tree_equal(self, post_rng, rng_state(RT, "cpu"))
                assert_tree_equal(self, buffers, cpu_tree(RT, dict(model.named_buffers())))
                assert_tree_equal(self, buffers, cpu_tree(RT, dict(reference.named_buffers())))

    def test_full_restore_and_next_update_replay_both_stages_and_banks(self):
        for condition in ("tied_persistent", "untied_persistent"):
            for global_stage in (False, True):
                model, optimizer, x, bundle, graphs = self.fixture(condition)
                set_stage(model, global_stage)
                update = 201 if global_stage else 1
                train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                           bundle["role"].train_labels, condition, 17, update)
                logits = eval_logits(RT, model, x, graphs["native"])
                binding = {"condition": condition, "fixture": True}
                selection = {"global": global_stage, "actual_update": update, "schedule_cursor": update}
                image = snapshot(RT, model, optimizer, "cpu", selection, binding)
                restored, restored_optimizer, _, _, _ = self.fixture(condition)
                restore(RT, restored, restored_optimizer, "cpu", image, binding)
                assert_tree_equal(self, image, snapshot(RT, restored, restored_optimizer, "cpu", selection, binding))
                self.assertTrue(RT.torch.equal(logits, eval_logits(RT, restored, x, graphs["native"])))
                loss_a = train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                                    bundle["role"].train_labels, condition, 17, update + 1)
                state_a = snapshot(RT, model, optimizer, "cpu", selection, binding)
                restore(RT, restored, restored_optimizer, "cpu", image, binding)
                loss_b = train_step(RT, restored, restored_optimizer, x, graphs, bundle["role"].train_ids,
                                    bundle["role"].train_labels, condition, 17, update + 1)
                self.assertEqual(loss_a, loss_b)
                assert_tree_equal(self, state_a, snapshot(RT, restored, restored_optimizer, "cpu", selection, binding))
                families(restored)[0].core.dropout = 0.0
                with self.assertRaises(ValueError):
                    restore(RT, restored, restored_optimizer, "cpu", image, binding)

    def test_selected_local_transition_preserves_live_rng_and_actual_cursor(self):
        for condition in ("tied_persistent", "untied_persistent"):
            model, optimizer, x, bundle, graphs = self.fixture(condition)
            train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                       bundle["role"].train_labels, condition, 17, 1)
            local = snapshot(RT, model, optimizer, "cpu", {"global": False, "actual_update": 1}, {"fixture": True})
            train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                       bundle["role"].train_labels, condition, 17, 2)
            live_rng = rng_state(RT, "cpu")
            clock = BankClock(200)  # Clock fixture, not 200 numerical training updates.
            transition(RT, model, optimizer, "cpu", local, clock)
            self.assertTrue(stage(model))
            self.assertEqual(clock.actual_update, 200)
            assert_tree_equal(self, live_rng, rng_state(RT, "cpu"))
            assert_tree_equal(self, local["model"], cpu_tree(RT, model.state_dict()))
            assert_tree_equal(self, local["optimizer"], cpu_tree(RT, optimizer.state_dict()))
            self.assertEqual(clock.advance(), 201)
            restored, restored_optimizer, _, _, _ = self.fixture(condition)
            restore(RT, restored, restored_optimizer, "cpu", local, {"fixture": True})
            self.assertFalse(stage(restored))  # A local checkpoint is a valid final selected image.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Only after explicit root admission")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "NUMERICAL_CHECKS_NOT_RUN_ROOT_ADMISSION_REQUIRED", "synthetic_only": True}))
    else:
        RT = load_runtime(execute=True, manifest_sha256=args.manifest_sha256, protocol_sha256=args.protocol_sha256)
        RT.torch.set_num_threads(1)
        unittest.main(argv=[__file__])
