"""Pinned original boundary source laws on tiny CPU graph fixtures only."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from adapters.molecular import validate_cost_contract
from adapters.molecular_boundary import ARM,make_adapter,validate_boundary_config
from adapters.molecular_endpoints import export_bank,receipt
from compile_molecular_boundary_matrix import cost_contract
from engine import run_trajectory,restore_checkpoint
from test_molecular import MOLECULAR_RUNTIME,run
import test_molecular as molecular_fixture
from test_molecular_endpoints import FakeRegistry


class BoundaryCostContractTests(unittest.TestCase):
    def test_bound_exact_molecular_tail_calls(self):
        row=run();config=row["config"]
        for graphs,batches,tail in [(32901,258,5),(1210,10,58)]:
            config["molecular_cost"]=cost_contract(config,row,graphs)
            config["train_minibatches_per_epoch"]=batches
            self.assertEqual(validate_cost_contract(config,row)["last_batch_graphs"],tail)
            self.assertEqual(config["molecular_cost"]["maximum_optimizer_step_calls"],100*batches)
            config["molecular_cost"]["drop_last"]=True
            with self.assertRaisesRegex(ValueError,"include-tail"):
                validate_cost_contract(config,row)


@unittest.skipUnless(MOLECULAR_RUNTIME,"Allocation CPU Torch/PyG/OGB fixtures required")
class BoundarySourceCPUFixtureTests(unittest.TestCase):
    def setup_fixture(self,root):
        import torch
        package=Path(__file__).resolve().parents[1]
        reference=Path(os.environ.get("GNNM_REFERENCE_PROJECT",str(package.parent)))
        source=root/"source";source.mkdir();sources={}
        for name in ["models","factors","objectives"]:
            path=source/(name+".py");shutil.copyfile(reference/"portable_internal_be_public_interface_20261007_v2/core"/path.name,path)
            sources[name]=receipt(path,root)
        boundary=source/"native_models.py";shutil.copyfile(reference/"common_wrapper_native_source_20261010_v1/models.py",boundary)
        self.assertEqual(hashlib.sha256(boundary.read_bytes()).hexdigest(),"07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303")
        row=run();row.update(run_id="boundary",seed=101,split=0,method="molecular_"+ARM,adapter="adapters/molecular_boundary.py:make_adapter",parents=[])
        row["budget"]["maximum"]=1;row["config"].update(arm=ARM,members=4,is_original_gnnm=True,molecular_source=sources,
            boundary_source=receipt(boundary,root),head_policy="source_boundary_scalar_graph_adaptation")
        row["config"]["model"].update(hidden=8,layers=2,dropout=.2)
        row["config"]["optimizer"].update(name="AdamW",weight_decay=0)
        row["config"]["training"].update(batch_size=2,eval_batch_size=2,views_per_update=1)
        row["config"]["contrastive"].update(alignment_weight=0,residual_weight=0,temperature=.2)
        fixture=molecular_fixture.MolecularCPUFixtureTests();roles={}
        for role,ids,molecules,labels in [("train",10,("methane","ethane","ethanol"),[0.,1.,0.]),("select",20,("ethanol","methanol"),[0.,1.])]:
            path=root/(role+".pt");torch.save(fixture.pack(ids,molecules,labels),path);roles[role]=receipt(path,root)
        row["config"]["molecular_inputs"]=roles
        row["config"]["molecular_cost"]=cost_contract(row["config"],row,3);row["config"]["train_minibatches_per_epoch"]=2
        endpoint={}
        for role,ids in [("cal",30),("evaluate",40)]:
            pack=fixture.pack(ids,("ethanol","methanol"),[0.,1.]);path=root/(role+"_graphs.pt")
            torch.save({k:v for k,v in pack.items() if k!="y"},path);endpoint[role+"_graphs"]=receipt(path,root)
            # Targets are purposefully never created in the exporter fixture.
            endpoint[role+"_targets"]={"path":role+"_absent_targets.pt","bytes":0,"sha256":"0"*64}
        study={"study_id":"boundary-engineering","dataset":"synthetic-molecule-fixture","runs":[row],
            "source":{"files":[*sources.values(),row["config"]["boundary_source"]]},
            "data":{"task":"graph_binary_classification","roles":roles,"target_policy":"complete_binary","evidence":{"endpoint_inputs":endpoint}}}
        return row,study,reference

    def test_source_init_and_original_sequential_gradient_update_match(self):
        import torch
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);row,study,reference=self.setup_fixture(root)
                actual=make_adapter(torch=torch,project_root=root,study=study,run=row,device="cpu")
                expected=make_adapter(torch=torch,project_root=root,study=study,run=row,device="cpu")
                first,last=actual.model.body.input.block,actual.model.body.head.block
                self.assertEqual(set(first.R.detach().unique().tolist()),{-1.,1.})
                for block in [first,last]:
                    self.assertTrue(torch.equal(block.S,torch.ones_like(block.S)))
                    self.assertTrue(torch.equal(block.B,torch.zeros_like(block.B)))
                    self.assertFalse(block.W.bias is not None)
                self.assertTrue(torch.equal(last.R,torch.ones_like(last.R)))
                self.assertIsNone(actual.model.body.message_factors)
                self.assertFalse(any(type(m).__name__=="FactorLinear" for m in actual.model.modules()))
                # Execute the exact pinned source train_step_tabm definition,
                # with a graph-task interface and matched private RNG streams.
                native_training=(reference/"source_snapshot/run_tabm.py").read_bytes()
                self.assertEqual(hashlib.sha256(native_training).hexdigest(),"b1d748e50c854802b93f4d4fa7485f096daa1cd36d3dd3bd1c89fbfa63cbd124")
                tree=ast.parse(native_training)
                definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train_step_tabm")
                context={"F":torch.nn.functional,"TABM_INITS":4}
                exec(compile(ast.Module(body=[definition],type_ignores=[]),"pinned_original_train_step_tabm","exec"),context)
                class SourceInterface(torch.nn.Module):
                    def __init__(self,adapter):super().__init__();self.adapter=adapter;self.body=adapter.model
                    def forward(self,data,x,tabm_seed):
                        stream=self.adapter.streams[tabm_seed]
                        with torch.random.fork_rng(devices=[]):
                            torch.set_rng_state(stream["cpu"]);self.body.member=tabm_seed
                            result=self.body({"graph":data})[0]
                            stream["cpu"]=torch.get_rng_state().clone()
                        return result
                source_model=SourceInterface(expected)
                generator=torch.Generator().manual_seed(row["config"]["shuffle_seed"]+19709+1000)
                order=torch.randperm(3,generator=generator);losses=[]
                for positions in order.split(2):
                    batch,labels=expected._batch(expected.train,positions);data=batch["graph"]
                    data.y=labels[:,0];data.train_mask=torch.ones(len(labels),dtype=torch.bool)
                    loss=context["train_step_tabm"](source_model,data,expected.optimizer,True);losses.append((loss,len(labels)))
                metrics=actual.train_step(1)
                self.assertAlmostEqual(metrics["objective"],sum(v*n for v,n in losses)/3,places=7)
                for name,value in actual.model.state_dict().items():
                    self.assertTrue(torch.equal(value,expected.model.state_dict()[name]),name)
                self.assertEqual(metrics["operation_counts"]["optimizer_updates"],2)
                self.assertEqual(metrics["operation_counts"]["model_forwards"],8)
                self.assertEqual(metrics["operation_counts"]["backward_calls"],8)
                self.assertEqual(metrics["operation_counts"]["training_label_opportunities"],12)
                for block in [first,last]:
                    for param in [block.W.weight,block.R,block.S,block.B]:
                        self.assertIsNotNone(param.grad);self.assertTrue(torch.isfinite(param.grad).all())
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in actual.model.body.convs.parameters()))
        finally:torch.set_num_threads(previous)

    def test_selected_state_restore_and_four_route_export_without_targets(self):
        import torch
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);row,study,_=self.setup_fixture(root)
                adapter=make_adapter(torch=torch,project_root=root,study=study,run=row,device="cpu")
                output=root/"runs";directory=output/study["study_id"]/row["run_id"]
                result=run_trajectory(torch,adapter,{"fit_id":row["run_id"],"budget_steps":1,"budget_unit":"epochs",
                    "validation_interval":1,"validate_initial":False,"prefixes":[],"inherited_fit_ids":[],"adapter_spec":{"synthetic_fixture":True}},directory)
                self.assertEqual(result["selected"]["step"],1)
                expected=adapter.predict_pack(adapter.select).clone();state=adapter.capture_state()
                with torch.no_grad():
                    for parameter in adapter.model.parameters():parameter.add_(17.)
                restore_checkpoint(torch,directory/"selected.pt",adapter,expected_fit_id=row["run_id"])
                self.assertTrue(torch.equal(adapter.predict_pack(adapter.select),expected))
                for a,b in zip(adapter.streams,state["streams"]):self.assertTrue(torch.equal(a["cpu"],b["cpu"]))
                expected_cal=adapter.predict_pack(torch.load(root/"cal_graphs.pt",weights_only=True)).clone()
                package=Path(__file__).resolve().parents[1];(root/"adapters").mkdir()
                for name in ["molecular.py","molecular_boundary.py","node.py","classifier.py"]:
                    shutil.copyfile(package/"adapters"/name,root/"adapters"/name)
                registry=FakeRegistry({row["run_id"]:{"state":"complete","contract":row,"result":result}})
                bank_dir=output/study["study_id"]/"bank";bank_dir.mkdir()
                endpoint={"endpoint_inputs":study["data"]["evidence"]["endpoint_inputs"],"fusion":"mean_logits","target_policy":"complete_binary","fit_parents":[row["run_id"]]}
                exported=export_bank(project_root=root,study=study,run={"parents":[row["run_id"]],"config":endpoint},output_root=output,output_dir=bank_dir,registry=registry,device="cpu")
                bank=torch.load(root/exported["files"]["cal"]["path"],weights_only=True)
                self.assertEqual(tuple(bank["logits"].shape),(4,2,1))
                self.assertFalse(exported["CAL_targets_read"]);self.assertFalse(exported["evaluation_targets_read"])
                self.assertFalse((root/"cal_absent_targets.pt").exists());self.assertFalse((root/"evaluate_absent_targets.pt").exists())
                self.assertTrue(torch.equal(bank["logits"],expected_cal))
        finally:torch.set_num_threads(previous)


if __name__=="__main__":unittest.main()
