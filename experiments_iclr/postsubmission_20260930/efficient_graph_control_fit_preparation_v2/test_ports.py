"""Synthetic CPU qualification tests. Prepared only; NEVER RUN in this round."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ports import (FEATURES, FIXED_WIDTH, MEMBERS, IndependentMembers,
    PackedBackendUnavailable, PolyFormerMIMO, new_native, make_mimo_tuples,
    independent_tuple_positions, mimo_loss, pooled_log_probabilities)
from qualifications import (check_parameter_budget, check_independent_equivalence,
    check_member_independence, check_mimo_correspondence)


def synthetic_graph(backbone):
    # No dataset files, labels, checkpoint or graph preprocessing is read.
    size = 6 * FEATURES[backbone] * (13 if backbone == "polyformer_mono" else 1)
    values = torch.sin(torch.arange(size, dtype=torch.float64) / 97) * 0.1
    shape = (6, 13, FEATURES[backbone]) if backbone == "polyformer_mono" else (6, FEATURES[backbone])
    edges = torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,0,0,1,2,3,4,5],
                          [1,0,2,1,3,2,4,3,5,4,0,5,0,1,2,3,4,5]])
    graph = SimpleNamespace(teacher_backbone=backbone, teacher_input=values.reshape(shape),
        teacher_edge_index=edges)
    train = SimpleNamespace(nodes=torch.tensor([5,1,4,0]), labels=torch.tensor([3,0,4,1]))
    return graph, train


def matched_families(backbone, identical=False):
    first = new_native(backbone, FIXED_WIDTH[backbone], 17)
    models = [copy.deepcopy(first) if identical else
        new_native(backbone, FIXED_WIDTH[backbone], 17 + 100003 * m) for m in range(MEMBERS)]
    reference = IndependentMembers(backbone, copy.deepcopy(models), backend="sequential").double()
    packed = IndependentMembers(backbone, copy.deepcopy(models), backend="vmap").double()
    return reference, packed


class PortsQualification(unittest.TestCase):
    def test_complete_module_counts_and_fixed_widths(self):
        for backbone in FIXED_WIDTH:
            reference, packed = matched_families(backbone)
            self.assertEqual(check_parameter_budget(reference)["actual_parameters"],
                check_parameter_budget(packed)["actual_parameters"])
        check_parameter_budget(PolyFormerMIMO(17))

    def test_polyformer_outputs_gradients_and_adam(self):
        self._equivalence("polyformer_mono")

    def test_polynormer_both_stages_outputs_gradients_and_adam(self):
        self._equivalence("polynormer_r")

    def _equivalence(self, backbone):
        graph, train = synthetic_graph(backbone)
        reference, packed = matched_families(backbone)
        try:
            result = check_independent_equivalence(reference, packed, graph, train,
                rtol=1e-8, atol=1e-10)
            self.assertTrue(result["deterministic_equivalence"])
            for stage in ((False, True) if backbone == "polynormer_r" else (False,)):
                check_member_independence(packed, graph, train, global_stage=stage)
        except PackedBackendUnavailable as error:
            # A visible skip is an unsupported backend, NEVER a qualification pass.
            self.skipTest(str(error))

    def test_polyformer_training_member_randomness(self):
        self._randomness("polyformer_mono")

    def test_polynormer_training_member_randomness(self):
        self._randomness("polynormer_r")

    def _randomness(self, backbone):
        graph, _ = synthetic_graph(backbone)
        _, packed = matched_families(backbone, identical=True)
        packed.train()
        try:
            logits = packed(graph)
        except PackedBackendUnavailable as error:
            self.skipTest(str(error))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertFalse(torch.equal(logits[0], logits[1]),
            "Identical members received identical training randomness")

    def test_complete_tuple_label_correspondence_and_inference(self):
        graph, train = synthetic_graph("polyformer_mono")
        positions = torch.tensor([[0,2,1,3],[1,0,3,2],[2,3,0,1],[3,1,2,0]])
        tuples = make_mimo_tuples(graph, train, positions)
        check_mimo_correspondence(graph, train, positions, tuples)
        # A deliberate label-slot mistake must be detectable by the hook.
        broken = copy.copy(tuples)
        object.__setattr__(broken, "labels", tuples.labels.roll(1, dims=1))
        with self.assertRaises(AssertionError):
            check_mimo_correspondence(graph, train, positions, broken)
        model = PolyFormerMIMO(17).double().eval()
        repeated_positions = torch.arange(len(train.nodes))[:, None].expand(-1, MEMBERS)
        repeated = make_mimo_tuples(graph, train, repeated_positions)
        with torch.no_grad():
            native = model(graph)[:, train.nodes].permute(1, 0, 2)
            explicit = model.forward_tuples(repeated)
        torch.testing.assert_close(native, explicit, rtol=1e-8, atol=1e-10)
        # Head-summed loss and its gradient are checked independently of the model.
        logits = torch.arange(4 * 4 * 5, dtype=torch.float64).reshape(4,4,5) / 20
        logits.requires_grad_()
        candidate = mimo_loss(logits, tuples)
        oracle = torch.stack([F.cross_entropy(logits[:, m], tuples.labels[:, m])
            for m in range(MEMBERS)]).sum()
        torch.testing.assert_close(candidate, oracle, rtol=1e-12, atol=1e-12)
        a = torch.autograd.grad(candidate, logits, retain_graph=True)[0]
        b = torch.autograd.grad(oracle, logits)[0]
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_tuple_sampler_and_both_declared_reducers(self):
        positions = independent_tuple_positions(9, generator=torch.Generator().manual_seed(17))
        for m in range(MEMBERS):
            torch.testing.assert_close(positions[:, m].sort().values, torch.arange(9), rtol=0, atol=0)
        logits = torch.tensor([[[4.,0.]], [[0.,1.]], [[-1.,2.]], [[1.,0.]]])
        probability = pooled_log_probabilities(logits, "native_probability").exp()
        common = pooled_log_probabilities(logits, "common_logit").exp()
        torch.testing.assert_close(probability, logits.softmax(-1).mean(0))
        torch.testing.assert_close(common, logits.mean(0).softmax(-1))
        self.assertFalse(torch.equal(probability, common))


if __name__ == "__main__":
    unittest.main()
