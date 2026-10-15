"""Meaningful finite tensor/selection checks; no native model or scientific fit."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


objective = load(HERE/"objective.py", "c1_node_semantics")
selection = load(HERE/"selection.py", "c1_node_selection_semantics")
ancestor = HERE.parent/"molecular/objectives.py"
assert hashlib.sha256(ancestor.read_bytes()).hexdigest() == "75e7d59f8360aa2688b9cd974e9e28863e475c683f3dd073e2eacd6665aff71a"
molecular = load(ancestor, "c1_pinned_molecular_ancestor")


def reference(h, labels):
    z = h/np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
    m,n,d = z.shape; classes = sorted(set(labels.tolist())); per_class = {c:[] for c in classes}
    if len(classes) >= 2:
        for route in range(m):
            for i in range(n):
                positive, by_class = [], {c:[] for c in classes}
                for peer in range(m):
                    for j in range(n):
                        if (route,i) == (peer,j): continue
                        score = float(z[route,i]@z[peer,j]/.2)
                        by_class[labels[j]].append(math.exp(score))
                        if labels[i] == labels[j]: positive.append(score)
                if positive:
                    denominator = sum(sum(v)/len(v) for v in by_class.values() if v)
                    per_class[labels[i]].append(math.log(denominator)-np.mean(positive))
    alignment = np.mean([np.mean(v) for v in per_class.values() if v]) if any(per_class.values()) else 0.
    gamma = .5/math.sqrt(d); penalties = []
    for c in classes:
        rows = labels == c
        if rows.sum() >= 2:
            std = np.sqrt(z[:,rows].var(axis=1)+1e-8)
            penalties.append(np.maximum(gamma-std,0).mean()/gamma)
    return alignment, np.mean(penalties) if penalties else 0.


class GeometrySemantics(unittest.TestCase):
    def setUp(self): self.generator = torch.Generator().manual_seed(15102026)

    def run_objective(self, h, labels, mode="alignment_variance"):
        return objective.NodeGeometry(mode)(prehead=h, labels=torch.as_tensor(labels),
            node_ids=torch.arange(h.shape[1]), role="TRAIN")

    def test_exact_binary_ancestor_loss_and_gradient(self):
        h = torch.randn(4,7,5,generator=self.generator,dtype=torch.float64,requires_grad=True)
        y = torch.tensor([0.,0.,float("nan"),1.,1.,1.,0.],dtype=torch.float64)
        for mode in ("alignment", "alignment_variance"):
            a = self.run_objective(h,y,mode)
            b = molecular.MolecularObjective(molecular.ObjectiveConfig(mode=mode))(
                prehead=h,labels=y,graph_ids=torch.arange(7),role="TRAIN")
            torch.testing.assert_close(a["loss"],b.loss,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(a["loss"],h,retain_graph=True)[0],
                torch.autograd.grad(b.loss,h,retain_graph=True)[0],rtol=0,atol=0)

    def test_multiclass_scalar_reference_and_finite_difference(self):
        h = torch.randn(4,6,3,generator=self.generator,dtype=torch.float64,requires_grad=True)
        y = torch.tensor([0,0,3,3,8,8])
        a = self.run_objective(h,y); alignment,variance = reference(h.detach().numpy(),y.numpy())
        self.assertAlmostEqual(float(a["alignment"].detach()),alignment,places=12)
        self.assertAlmostEqual(float(a["variance"].detach()),variance,places=12)
        self.assertTrue(torch.autograd.gradcheck(lambda x:self.run_objective(x,y)["loss"],(h,),
            eps=1e-6,atol=5e-5,rtol=1e-4,fast_mode=True))

    def test_singleton_classes_keep_cross_route_positive_not_variance(self):
        h = torch.randn(4,3,5,generator=self.generator,requires_grad=True)
        a = self.run_objective(h,[0,2,5])
        self.assertEqual(a["diagnostics"]["alignment"]["eligible_anchors"],12)
        self.assertEqual(a["diagnostics"]["alignment"]["positive_count_min"],3)
        self.assertEqual(a["diagnostics"]["conditional_variance"]["eligible_classes"],0)
        a["loss"].backward();self.assertTrue(torch.isfinite(h.grad).all())

    def test_one_class_and_exact_collapse_limitations(self):
        offsets = torch.randn(4,1,5,generator=self.generator,dtype=torch.float64)
        h = offsets.expand(-1,6,-1).clone().requires_grad_()
        a = self.run_objective(h,[0]*6)
        self.assertEqual(float(a["alignment"].detach()),0)
        self.assertGreater(float(a["variance"].detach()),.99)
        grad = torch.autograd.grad(a["variance"],h)[0]
        self.assertLess(float(grad.abs().max()),1e-10)

    def test_missing_and_empty_support_zero_gradient(self):
        h = torch.randn(4,5,5,generator=self.generator,requires_grad=True)
        a = self.run_objective(h,[0.,0.,float("nan"),2.,2.])
        a["loss"].backward();self.assertEqual(float(h.grad[:,2].norm()),0)
        b = self.run_objective(h,[float("nan")]*5)
        self.assertEqual(float(b["loss"].detach()),0)
        self.assertEqual(b["diagnostics"]["missing_labels"],5)

    def test_gradient_support_excludes_downstream_head(self):
        x = torch.randn(6,4,generator=self.generator)
        shared = torch.randn(4,5,generator=self.generator,requires_grad=True)
        private = torch.randn(4,1,5,generator=self.generator,requires_grad=True)
        head = torch.randn(5,3,generator=self.generator,requires_grad=True)
        h = (x@shared)[None,:,:]*private
        a = self.run_objective(h,[0,0,1,1,2,2])
        grads = torch.autograd.grad(a["loss"],(shared,private,head),allow_unused=True)
        self.assertGreater(float(grads[0].norm()),0);self.assertGreater(float(grads[1].norm()),0)
        self.assertIsNone(grads[2])

    def test_fixed_class_preserving_hash_budget_and_source_order(self):
        ids,labels = list(range(900)),[i%3 for i in range(900)]
        positions,record,sha = selection.select(ids,labels,seed=101,role="TRAIN")
        self.assertEqual(len(positions),512);self.assertEqual(positions,sorted(positions))
        self.assertEqual([c["selected"] for c in record["class_counts"]],[171,171,170])
        self.assertEqual(selection.select(ids,labels,seed=101,role="TRAIN")[2],sha)
        other = selection.select(ids[::-1],labels[::-1],seed=101,role="TRAIN")[1]
        self.assertEqual(set(other["selected_ids"]),set(record["selected_ids"]))
        self.assertFalse(record["epoch_change"]);self.assertFalse(record["native_CE_population_changed"])
        with self.assertRaises(ValueError):selection.select(list(range(520)),[i//2 for i in range(520)],seed=1,role="TRAIN")

    def test_roles_duplicate_ids_bad_classes_and_configuration(self):
        h = torch.randn(4,3,5,generator=self.generator)
        for labels,ids,role in [([0,1,2],[0,1,2],"VALID"),([0,1,2],[0,0,2],"TRAIN"),
                ([0.,.5,2.],[0,1,2],"TRAIN")]:
            with self.assertRaises(ValueError):objective.NodeGeometry("alignment")(
                prehead=h,labels=torch.tensor(labels),node_ids=torch.tensor(ids),role=role)
        with self.assertRaises(ValueError):objective.Config(mode="alignment",temperature=.3)


if __name__ == "__main__":
    torch.set_num_threads(2);unittest.main(verbosity=2)
