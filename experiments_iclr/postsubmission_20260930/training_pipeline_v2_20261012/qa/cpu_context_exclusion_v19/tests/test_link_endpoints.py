"""Synthetic endpoint custody, selected ancestry, raw metric and failure costs."""
import copy
from contextlib import nullcontext
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adapters import link_endpoints as ep
from engine import AdapterStepError
from test_link import config

try:
    import torch
except ImportError:
    torch=None


class Registry:
    def __init__(self,rows):self.rows=rows
    def run(self,study_id,identifier):return self.rows[identifier]


class LinkEndpointClosureTests(unittest.TestCase):
    def test_whole_search_selection_and_comparative_roster_must_close(self):
        rows=[{"run_id":stage,"config":{"stage":stage}} for stage in ("search","search_selection","comparative")]
        study={"study_id":"synthetic","runs":rows,"release":{"global_plan_gates_required":True}}
        registry=Registry({r["run_id"]:{"state":"complete"} for r in rows})
        ep.whole_fit_roster_closed(study,registry)
        registry.rows["comparative"]["state"]="running"
        with self.assertRaisesRegex(ValueError,"has not closed"):ep.whole_fit_roster_closed(study,registry)
        study["release"]["global_plan_gates_required"]=False
        with self.assertRaisesRegex(ValueError,"mandatory"):ep.whole_fit_roster_closed(study,registry)


@unittest.skipIf(torch is None,"Torch unavailable locally; allocation CPU fixture required")
class LinkEndpointTensorTests(unittest.TestCase):
    def fixture(self,root):
        query={"ids":torch.arange(52),"pairs":torch.arange(104).reshape(52,2)%8}
        torch.save(query,root/"queries.pt")
        # Synthetic only: positive pool is deliberately interspersed/reversed.
        mask=torch.zeros(52,dtype=torch.bool);mask[[7,41]]=True
        original=torch.empty(52,dtype=torch.long);original[mask]=torch.tensor([1,0]);original[~mask]=torch.arange(50).flip(0)
        torch.save({"ids":query["ids"],"is_positive":mask,"original_index":original},root/"targets.pt")
        inputs={"evaluate_queries":ep.receipt(root,root/"queries.pt"),"evaluate_targets":ep.receipt(root,root/"targets.pt")}
        fits=[];registered={}
        for body in range(4):
            cfg=config("native_independent4_body","NCN");cfg.update(initialization_seed=9200+body,stage="comparative")
            row={"run_id":"fit-"+str(body),"body":body,"kind":"model_fit","method":"original_NCN_I4","split":0,
                 "config":cfg,"adapter":"synthetic:factory","budget":{"unit":"epochs","policy":"full_budget","maximum":100},
                 "selection":{"metric":"hits@50","direction":"max","ties":"first"}}
            fits.append(row)
            checkpoint=root/"outputs"/"synthetic"/row["run_id"]/"selected.pt";checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b"synthetic selected authority "+bytes([body]))
            registered[row["run_id"]]={"state":"complete","contract":row,"result":{"completed_steps":100,"selected":{"path":"selected.pt","step":body+1}}}
        study={"study_id":"synthetic","dataset":"ogbl-collab","data":{"task":"link_prediction","roles":{
            role:fits[0]["config"]["link_inputs"][role] for role in ("train","select")},"evidence":{"endpoint_inputs":inputs,"endpoint_nodes":8}},
            "release":{"global_plan_gates_required":True},"runs":fits}
        shared={"query_input":inputs["evaluate_queries"],"support_policy":"full_TRAIN_only","fusion":"mean_raw_scores"}
        export={"run_id":"export","kind":"ensemble_assembly","parents":[r["run_id"] for r in fits],"config":{**shared,"fit_parents":[r["run_id"] for r in fits]}}
        output=root/"export";output.mkdir()
        kwargs={"project_root":root,"study":study,"run":export,"output_root":root/"outputs","output_dir":output,"registry":Registry(registered),"device":"cpu"}
        return kwargs,query,mask,original

    def fake_export(self,kwargs):
        made=[];restored=[]
        def factory(**arguments):
            body=arguments["run"]["body"];made.append(arguments["run"]["run_id"])
            return SimpleNamespace(body=body,provenance={"adapter_setup_operations":{"full_support_graph_builds":1}})
        def restore(t,path,adapter,expected_fit_id):
            restored.append(expected_fit_id);return {"step":adapter.body+1}
        def predict(adapter,pairs):return torch.arange(len(pairs),dtype=torch.float32)+adapter.body,{"encoder_calls":1,"query_records":len(pairs)}
        with patch.object(ep,"load_factory",return_value=factory),patch.object(ep,"restore_checkpoint",side_effect=restore),patch.object(ep,"predict_unlabelled",side_effect=predict):
            result=ep.export_bank(**kwargs)
        return result,made,restored

    def test_queries_reject_targets_and_targets_require_complete_separate_index(self):
        query={"ids":torch.arange(3),"pairs":torch.tensor([[0,1],[1,2],[2,2]])}
        ep.query_pack(torch,query,3)
        with self.assertRaisesRegex(ValueError,"label-free"):ep.query_pack(torch,{**query,"is_positive":torch.ones(3,dtype=torch.bool)},3)
        target={"ids":query["ids"],"is_positive":torch.tensor([True,False,False]),"original_index":torch.tensor([0,1,0])}
        ep.targets(torch,target,query["ids"])
        target["original_index"][2]=1
        with self.assertRaisesRegex(ValueError,"coverage"):ep.targets(torch,target,query["ids"])

    def test_export_opens_only_query_pack_and_four_own_selected_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs,query,_,_=self.fixture(root)
            opened=[];original_load=ep.load_bound
            def load(t,base,row):opened.append(row["path"]);return original_load(t,base,row)
            with patch.object(ep,"load_bound",side_effect=load):result,made,restored=self.fake_export(kwargs)
            self.assertEqual(opened,["queries.pt"])
            self.assertEqual(made,restored);self.assertEqual(len(set(made)),4)
            self.assertEqual([r["selected_epoch"] for r in result["ancestry"]],[1,2,3,4])
            bank=ep.load_bound(torch,root,result["bank"])
            self.assertTrue(torch.equal(ep.pool_scores(torch,bank["scores"]),torch.arange(52)+1.5))
            self.assertEqual(result["operation_counts"]["model_constructions"],4)
            self.assertFalse(result["evaluation_targets_read"])

    def test_fixed_body0_reuses_exact_selected_authority_without_model_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs,_,_,_=self.fixture(root);source,_,_=self.fake_export(kwargs)
            registry=kwargs["registry"];registry.rows["export"]={"state":"complete","contract":kwargs["run"],"result":source}
            singleton={"run_id":"single","parents":["export","fit-0"],"config":{**kwargs["run"]["config"],"fit_parents":["fit-0"],"reuse_export_parent":"export","single_reference":"fixed_I4_body0"}}
            output=root/"singleton";output.mkdir();args={**kwargs,"run":singleton,"output_dir":output}
            with patch.object(ep,"load_factory",side_effect=AssertionError("Reuse constructed a model")),patch.object(ep,"predict_unlabelled",side_effect=AssertionError("Reuse predicted again")):
                result=ep.export_bank(**args)
            self.assertEqual(result["operation_counts"]["model_constructions"],0)
            self.assertEqual(result["operation_counts"]["prediction_pack_attempts"],0)
            self.assertTrue(torch.equal(ep.load_bound(torch,root,result["bank"])["scores"],ep.load_bound(torch,root,source["bank"])["scores"][:1]))
            source["ancestry"][0]["selected_epoch"]=99
            with self.assertRaisesRegex(ValueError,"exact current"):ep.export_bank(**args)
            source["ancestry"][0]["selected_epoch"]=1
            registry.rows["fit-0"]["result"]["completed_steps"]=99
            with self.assertRaisesRegex(ValueError,"Partial fit"):ep.export_bank(**args)

    def test_wrong_body_or_reused_checkpoint_cannot_supply_genuine_i4(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs,_,_,_=self.fixture(Path(directory))
            kwargs["registry"].rows["fit-2"]["contract"]["body"]=1
            with self.assertRaisesRegex(ValueError,"body"):self.fake_export(kwargs)

    def test_closure_precedes_query_or_target_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs,_,_,_=self.fixture(root)
            kwargs["registry"].rows["fit-3"]["state"]="running"
            with patch.object(ep,"load_bound",side_effect=AssertionError("Unclosed endpoint decoded authority")):
                with self.assertRaisesRegex(ValueError,"has not closed"):ep.export_bank(**kwargs)
                with self.assertRaisesRegex(ValueError,"has not closed"):ep.evaluate_bank(**kwargs)

    def test_raw_native_hits50_restores_pool_index_and_strict_ties(self):
        from ogb.linkproppred import Evaluator
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs,query,mask,_=self.fixture(root)
            score=torch.ones(52);score[7]=2.;score[41]=1.
            torch.save({"ids":query["ids"],"scores":torch.stack([score]*4)},root/"bank.pt")
            exported={"bank":ep.receipt(root,root/"bank.pt"),"query_input":kwargs["run"]["config"]["query_input"],"support_policy":"full_TRAIN_only"}
            kwargs["registry"].rows["export"]={"state":"complete","contract":{"kind":"ensemble_assembly"},"result":exported}
            row={"parents":["export"],"config":{**kwargs["run"]["config"],"target_input":kwargs["study"]["data"]["evidence"]["endpoint_inputs"]["evaluate_targets"],"export_parent":"export",
                "metric":{"hits_k":50,"evaluator_source_sha256":ep.file_sha(Path(inspect.getfile(Evaluator)))}}}
            result=ep.evaluate_bank(**{**kwargs,"run":row})
            self.assertEqual(result["hits@50"],.5);self.assertEqual(result["positive_records"],2)
            self.assertEqual(result["negative_records"],50);self.assertFalse(result["CAL_used"])
            self.assertFalse(result["confidence_claims_available"])

    def test_partial_inference_failure_is_retained_with_completed_lower_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs,_,_,_=self.fixture(root)
            fake=SimpleNamespace(provenance={"adapter_setup_operations":{"full_support_graph_builds":1}})
            error=AdapterStepError("synthetic second query pass failed",operation_counts={"encoder_calls":1,"query_records":3})
            with patch.object(ep,"load_factory",return_value=lambda **kw:fake),patch.object(ep,"restore_checkpoint",return_value={"step":1}),patch.object(ep,"predict_unlabelled",side_effect=error):
                with self.assertRaises(AdapterStepError):ep.export_bank(**kwargs)
            costs=json.loads((kwargs["output_dir"]/"failure_costs.json").read_text())
            self.assertEqual(costs["operation_counts"]["query_records"],3)
            self.assertEqual(costs["operation_counts"]["prediction_pack_attempts"],1)
            self.assertEqual(costs["operation_counts"]["prediction_packs_completed"],0)
            self.assertTrue(costs["partial_inference_counts_are_completed_lower_bounds"])

    def test_native_unlabelled_tail_and_failure_counts(self):
        class Model:
            def __init__(self):self.calls=0;self.fail=False
            def eval(self):pass
            def encoder(self,x,graph):return x
            def decoder(self,h,graph,pairs):
                self.calls+=1
                if self.fail and self.calls==2:raise RuntimeError("synthetic decoder failure")
                return pairs[0].float()
        def iterator(device,length,batch,shuffle):return (torch.arange(i,min(i+batch,length)) for i in range(0,length,batch))
        model=Model();adapter=SimpleNamespace(t=torch,model=model,_scope=lambda training:nullcontext(),graph={"x":torch.ones(8,2)},full_graph=None,
            device="cpu",utils=SimpleNamespace(PermIterator=iterator),config={"training":{"eval_batch_size":3}})
        pairs=torch.tensor([[0,1],[1,2],[2,3],[3,4],[4,5]])
        score,counts=ep.predict_unlabelled(adapter,pairs)
        self.assertEqual(len(score),5);self.assertEqual(counts["encoder_calls"],1);self.assertEqual(counts["decoder_query_passes"],2)
        model.fail=True;model.calls=0
        with self.assertRaises(AdapterStepError) as caught:ep.predict_unlabelled(adapter,pairs)
        self.assertEqual(caught.exception.operation_counts["query_records"],3);self.assertIsNone(adapter._active_counts)


if __name__=="__main__":unittest.main()
