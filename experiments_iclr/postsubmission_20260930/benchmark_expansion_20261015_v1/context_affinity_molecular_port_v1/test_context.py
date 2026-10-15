"""Native categorical histogram extraction semantics; no native model or fit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

spec=importlib.util.spec_from_file_location("molecular_context_test",Path(__file__).with_name("context.py"))
context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)


def batch():
    x=torch.tensor([[0]*9,[1]*9,[0]*9,[2]*9],dtype=torch.long)
    return SimpleNamespace(x=x,batch=torch.tensor([0,0,0,1]),num_graphs=2,
        smoke_graph_id=torch.tensor([10,20]),edge_index=torch.tensor([[0,1],[1,0]]),
        edge_attr=torch.tensor([[0,1,0],[0,1,0]],dtype=torch.long))


class MolecularContextSemantics(unittest.TestCase):
    def test_native_per_field_frequencies_and_directed_records(self):
        source=batch();before=source.x.clone(),source.edge_index.clone(),source.edge_attr.clone()
        a,b,r=context.build(torch,source,(3,)*9,(2,)*3)
        torch.testing.assert_close(a[0,:3],torch.tensor([2/3,1/3,0.]))
        torch.testing.assert_close(a[1,:3],torch.tensor([0.,0.,1.]))
        torch.testing.assert_close(b[0],torch.tensor([1.,0.,0.,1.,1.,0.]))
        self.assertEqual(r["directed_bond_record_counts"],[2,0]);self.assertEqual(r["zero_bond_contexts"],1)
        self.assertTrue(torch.equal(before[0],source.x));self.assertTrue(torch.equal(before[1],source.edge_index))
        self.assertTrue(torch.equal(before[2],source.edge_attr))

    def test_edgeless_zero_and_factual_signature(self):
        source=batch();source.edge_index=torch.empty(2,0,dtype=torch.long);source.edge_attr=torch.empty(0,3,dtype=torch.long)
        a,b,r=context.build(torch,source,(3,)*9,(2,)*3)
        self.assertEqual(float(b.norm()),0.);self.assertEqual(r["zero_bond_contexts"],2)
        self.assertEqual(context.build(torch,source,(3,)*9,(2,)*3)[2],r)
        source.x[0,0]=1
        self.assertNotEqual(context.build(torch,source,(3,)*9,(2,)*3)[2]["sha256"],r["sha256"])

    def test_same_class_donors_zero_context_uniform_and_distinct_channels(self):
        a=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        b=torch.zeros(4,3);labels=torch.zeros(4,dtype=torch.long);ids=torch.arange(4)
        r=context.target_diagnostics(torch,a,b,labels,ids)
        self.assertEqual(r["supported_anchors"],4);self.assertEqual(r["zero_B"],4)
        self.assertEqual(r["B_nonuniform_anchors_at_1e-8"],0)
        self.assertGreater(r["A_nonuniform_anchors_at_1e-8"],0)
        self.assertGreater(r["distinct_target_anchors_at_1e_8"],0)

    def test_singleton_support_is_omitted(self):
        a=torch.ones(2,3);b=torch.ones(2,2);labels=torch.tensor([0,1]);ids=torch.tensor([1,2])
        r=context.target_diagnostics(torch,a,b,labels,ids)
        self.assertEqual(r["supported_anchors"],0);self.assertEqual(r["omitted_anchors"],2)

    def test_invalid_native_category_and_cross_graph_edge_rejected(self):
        source=batch();source.x[0,0]=3
        with self.assertRaises(ValueError):context.build(torch,source,(3,)*9,(2,)*3)
        source=batch();source.edge_index[1,0]=3
        with self.assertRaises(ValueError):context.build(torch,source,(3,)*9,(2,)*3)


if __name__=="__main__":torch.set_num_threads(2);unittest.main(verbosity=2)
