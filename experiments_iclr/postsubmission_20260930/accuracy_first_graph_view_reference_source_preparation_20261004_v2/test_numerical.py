"""Unrun, root-admission-required synthetic CPU checks. No predictive experiment."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from contract import fit_spec, pass_plan
from driver import correct_count, pool_conventional
from native_reference import build, evaluate, restore, snapshot, train_step, transition
from runtime import ROOT, descriptor, load_runtime

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


class NumericalReferences(unittest.TestCase):
    def fixture(self, kind):
        member = 0 if kind == "native_member" else None
        model, optimizer, _ = build(RT, kind, 0, member, synthetic_recipe=RECIPE)
        role = RT.views.TrainRole(12, 0, tuple(range(10)), (0,) * 5 + (1,) * 5, (10,), (11,))
        bundle = RT.views.construct_views(role, RT.views.native_edges(12,
                    ((a, b) for a in range(12) for b in range(a + 1, 12))), 17)
        edges = dict(bundle["views"], native=bundle["native"])
        graphs = {name: RT.torch.tensor(value, dtype=RT.torch.long).t().contiguous() for name, value in edges.items()}
        x = RT.torch.arange(36, dtype=RT.torch.float32).reshape(12, 3) / 20
        return model, optimizer, x, bundle, graphs

    def test_coefficients_accumulation_heads_buffers_and_single_step(self):
        for kind in ("native_member", "view_augmented_single"):
            for global_stage in (False, True):
                model, optimizer, x, bundle, graphs = self.fixture(kind)
                model._global = global_stage
                reference = deepcopy(model)
                ref_optimizer = RT.torch.optim.Adam(reference.parameters(), lr=0.001, weight_decay=0.0)
                rng = RT.helpers.rng_state(RT, "cpu")
                buffers = RT.helpers.cpu_tree(RT, dict(model.named_buffers()))
                update = 201 if global_stage else 1
                loss = train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                                  bundle["role"].train_labels, kind, update)
                post_rng = RT.helpers.rng_state(RT, "cpu")
                RT.helpers.restore_rng(RT, "cpu", rng)
                reference.train()
                ids = RT.torch.tensor(bundle["role"].train_ids)
                labels = RT.torch.tensor(bundle["role"].train_labels)
                losses = [RT.torch.nn.functional.nll_loss(RT.torch.nn.functional.log_softmax(
                    reference(x, graphs[name]), dim=1).index_select(0, ids), labels) * coefficient
                    for name, coefficient in pass_plan(kind)]
                total = RT.torch.stack(losses).sum()
                total.backward()
                self.assertAlmostEqual(loss, float(total.detach()), places=6)
                for (_, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
                    if a.grad is None:
                        self.assertIsNone(b.grad)
                    else:
                        RT.torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
                ref_optimizer.step()
                for a, b in zip(model.parameters(), reference.parameters()):
                    RT.torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
                exact(self, post_rng, RT.helpers.rng_state(RT, "cpu"))
                exact(self, buffers, RT.helpers.cpu_tree(RT, dict(model.named_buffers())))
                exact(self, buffers, RT.helpers.cpu_tree(RT, dict(reference.named_buffers())))

    def test_complete_restore_and_next_update_replay_both_stages_and_kinds(self):
        for kind in ("native_member", "view_augmented_single"):
            for global_stage in (False, True):
                model, optimizer, x, bundle, graphs = self.fixture(kind)
                model._global = global_stage
                update = 201 if global_stage else 1
                train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                           bundle["role"].train_labels, kind, update)
                logits = evaluate(RT, model, x, graphs["native"])
                binding = {"synthetic": True, "kind": kind}
                selection = {"global": global_stage, "actual_update": update}
                image = snapshot(RT, model, optimizer, "cpu", selection, binding)
                other, other_optimizer, _, _, _ = self.fixture(kind)
                restore(RT, other, other_optimizer, "cpu", image, binding)
                exact(self, image, snapshot(RT, other, other_optimizer, "cpu", selection, binding))
                self.assertTrue(RT.torch.equal(logits, evaluate(RT, other, x, graphs["native"])))
                loss_a = train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                                    bundle["role"].train_labels, kind, update + 1)
                state_a = snapshot(RT, model, optimizer, "cpu", selection, binding)
                restore(RT, other, other_optimizer, "cpu", image, binding)
                loss_b = train_step(RT, other, other_optimizer, x, graphs, bundle["role"].train_ids,
                                    bundle["role"].train_labels, kind, update + 1)
                self.assertEqual(loss_a, loss_b)
                exact(self, state_a, snapshot(RT, other, other_optimizer, "cpu", selection, binding))

    def test_local_transition_preserves_live_rng_and_local_final_restore(self):
        for kind in ("native_member", "view_augmented_single"):
            model, optimizer, x, bundle, graphs = self.fixture(kind)
            binding = {"synthetic": True}
            train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                       bundle["role"].train_labels, kind, 1)
            local = snapshot(RT, model, optimizer, "cpu", {"global": False, "actual_update": 1}, binding)
            train_step(RT, model, optimizer, x, graphs, bundle["role"].train_ids,
                       bundle["role"].train_labels, kind, 2)
            rng = RT.helpers.rng_state(RT, "cpu")
            transition(RT, model, optimizer, "cpu", local, 200)  # Actual clock fixture, not 200 fits.
            self.assertTrue(model._global)
            exact(self, rng, RT.helpers.rng_state(RT, "cpu"))
            exact(self, local["model"], RT.helpers.cpu_tree(RT, model.state_dict()))
            exact(self, local["optimizer"], RT.helpers.cpu_tree(RT, optimizer.state_dict()))
            restore(RT, model, optimizer, "cpu", local, binding)
            self.assertFalse(model._global)

    def test_fixed_probability_pool_and_member0_physical_alias(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            results = []
            for member in range(4):
                value = RT.torch.full((24492, 5), -10.0)
                value[:, :2] = RT.torch.tensor([100.0, 0.0] if member == 0 else [0.0, 5.0])
                binding = {"fit": fit_spec("native_member", 0, member), "synthetic": True}
                selection = {"global": bool(member % 2), "actual_update": 1 if member % 2 == 0 else 201}
                path = Path(directory) / (str(member) + ".pt")
                RT.torch.save({"bindings": binding, "selection": selection, "selected_raw_logits": value}, path)
                results.append({"bindings": binding, "selection": selection,
                                "completed_actual_updates": 2700, "final_checkpoint": descriptor(path)})
            pooled = pool_conventional(RT, list(reversed(results)))
            self.assertEqual(pooled["native_single_alias"], results[0]["final_checkpoint"])
            self.assertEqual(correct_count(RT, pooled["native_single_raw_logits"], (0,), (0,)), 1)
            self.assertEqual(correct_count(RT, pooled["independent4_raw_logits"], (0,), (1,)), 1)
            self.assertEqual(int(pooled["independent4_raw_logits"].mean(0)[0].argmax()), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "REFERENCE_NUMERICAL_CHECKS_UNRUN_ROOT_ADMISSION_REQUIRED"}))
    else:
        RT = load_runtime(execute=True, manifest_sha256=args.manifest_sha256, protocol_sha256=args.protocol_sha256)
        RT.torch.set_num_threads(1)
        unittest.main(argv=[__file__])
