"""Selected sparse incoming operator versus independent scalar reference."""
import importlib.util
import math
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location("node_context_semantics",Path(__file__).with_name("context.py"))
context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)


def scalar_reference(x,edges,selected):
    degree=[1]*len(x);records=[]
    for source,target in edges.T.tolist():
        if source!=target:degree[target]+=1;records.append((source,target))
    result=[]
    for target in selected.tolist():
        row=x[target]/degree[target]
        for source,destination in records:
            if destination==target:row=row+x[source]/math.sqrt(degree[target]*degree[source])
        result.append(row)
    return torch.stack(result)


class NodeContextSemantics(unittest.TestCase):
    def test_directed_duplicates_and_full_native_degrees(self):
        x=torch.tensor([[1.,2.],[3.,0.],[0.,4.],[2.,1.]],dtype=torch.float64)
        edges=torch.tensor([[0,0,1,2,2,3,1],[1,1,0,1,2,3,1]])
        selected=torch.tensor([1,3]);before=x.clone(),edges.clone()
        a,b,r=context.build(torch,x,edges,selected)
        torch.testing.assert_close(a,x[selected],rtol=0,atol=0)
        torch.testing.assert_close(b,scalar_reference(x,edges,selected),rtol=1e-14,atol=1e-14)
        self.assertEqual(r["target_row_degrees"],[4,1]);self.assertEqual(r["removed_native_self_records"],3)
        self.assertEqual(r["sparse_shape"],[2,4]);self.assertEqual(r["selected_nonself_records"],3)
        self.assertTrue(torch.equal(before[0],x));self.assertTrue(torch.equal(before[1],edges))

    def test_repeated_self_records_become_one_self_contribution(self):
        x=torch.tensor([[1.,0.],[0.,2.],[3.,1.]])
        edges=torch.tensor([[0,0,1,1,2,2],[0,0,1,1,2,2]])
        selected=torch.tensor([2,0]);a,b,r=context.build(torch,x,edges,selected)
        torch.testing.assert_close(b,x[selected],rtol=0,atol=0)
        self.assertEqual(r["removed_native_self_records"],6);self.assertEqual(r["sparse_records"],2)

    def test_no_edges_and_zero_features_are_reported(self):
        x=torch.tensor([[0.,0.],[1.,0.],[0.,2.]])
        a,b,r=context.build(torch,x,torch.empty(2,0,dtype=torch.long),torch.tensor([0,2]))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(r["zero_A"],1);self.assertEqual(r["zero_B"],1)
        self.assertFalse(r["full_N_by_D_propagated_cache"])

    def test_target_signatures_bind_unselected_neighbor_features(self):
        x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        edge=torch.tensor([[0,2],[1,1]]);selected=torch.tensor([1])
        a,b,r=context.build(torch,x,edge,selected);x[0,0]=2
        aa,bb,rr=context.build(torch,x,edge,selected)
        torch.testing.assert_close(a,aa,rtol=0,atol=0)
        self.assertNotEqual(r["B"]["bytes_sha256"],rr["B"]["bytes_sha256"])
        self.assertNotEqual(r["source_features"]["bytes_sha256"],rr["source_features"]["bytes_sha256"])

    def test_invalid_topology_and_duplicate_cohort_fail(self):
        x=torch.ones(3,2)
        with self.assertRaises(ValueError):context.build(torch,x,torch.tensor([[0],[3]]),torch.tensor([1]))
        with self.assertRaises(ValueError):context.build(torch,x,torch.empty(2,0,dtype=torch.long),torch.tensor([1,1]))


if __name__=="__main__":torch.set_num_threads(2);unittest.main(verbosity=2)
