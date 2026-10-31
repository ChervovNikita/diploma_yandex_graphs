"""Exact source arithmetic/ownership/state checks using synthetic CPU inputs."""
import ast
import copy
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adapters import private_cmcl as cmcl, masked_core as core, private_adam as adam, private_polyformer_source as p
from adapters import polyformer, private_polyformer_controls as controls

PROJECT = Path(os.environ.get("GNNM_REFERENCE_PROJECT", Path(__file__).resolve().parents[2]))
NUMERIC = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "torch_geometric"))


class PrivateSourceStaticTests(unittest.TestCase):
    def test_exact_source_bytes_and_math_AST(self):
        for row in [*cmcl.SOURCE.values(), *core.SOURCE.values(), adam.SOURCE, *polyformer.SOURCE_FILES.values()]:
            self.assertEqual(p.receipt(PROJECT, row["path"]), row)
        tree = ast.parse(p.bound(PROJECT, core.SOURCE["method"]).read_bytes())
        function = next(n for n in next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "Session").body if isinstance(n,ast.FunctionDef) and n.name == "train_step")
        transformed = p.ProviderConstants({(int,19717):"NODES",(int,500):"FEATURES",(int,3):"CLASSES",(float,.005):"BASE_LR",(float,.0005):"ATTN_LR"}).visit(copy.deepcopy(function))
        self.assertEqual(ast.dump(function), ast.dump(transformed))
        self.assertEqual(len(p.DOMAIN),9); self.assertIn(p.SOURCE_ANCHOR,p.DOMAIN)

    def test_OIPG_diagnostic_ancestry_projection_guard_and_no_benchmark_family(self):
        import sys
        path=PROJECT/"source_supply_private_gradient_helper_20261009_v2/source_supply.py"
        spec=importlib.util.spec_from_file_location("private_gradient_diagnostic",path)
        helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
        direction=helper.project_nonincrease_cone((2.,3.),((1.,0.),(0.,1.)))
        self.assertEqual(direction,(0.,0.))
        direction=helper.project_nonincrease_cone((2.,-3.),((1.,0.),(2.,0.),(0.,0.)))
        self.assertEqual(direction,(0.,-3.))
        reference={"J":2.,"risk":1.,"own":3.}
        passed=helper.check_guard_values(reference,{"J":1.9,"risk":1.,"own":2.9},("risk",),("own",),armijo_rhs=1.99)
        failed=helper.check_guard_values(reference,{"J":1.9,"risk":1.01,"own":2.9},("risk",),("own",),armijo_rhs=1.99)
        self.assertTrue(passed.accepted);self.assertEqual(failed.violations,("risk",))
        with self.assertRaises(helper.ContractError):helper.project_nonincrease_cone((float("nan"),),())
        with self.assertRaises(helper.ContractError):helper.Config().require_enabled()


@unittest.skipUnless(NUMERIC, "Allocation scientific interpreter required for synthetic CPU checks")
class PrivateSourceSyntheticTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch; torch.set_num_threads(1)
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        rows={r["path"]:r for r in [*cmcl.SOURCE.values(),*core.SOURCE.values(),adam.SOURCE,*polyformer.SOURCE_FILES.values(),controls.boundary.BOUNDARY_SOURCE]}
        for relative,row in rows.items():
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.bound(PROJECT,row).read_bytes())
        n=144;generator=torch.Generator().manual_seed(913)
        nodes=torch.arange(n)
        self.graph={"x":torch.randn(n,7,generator=generator),"edge_index":torch.stack([nodes.repeat(3),torch.cat([nodes,(nodes+1)%n,(nodes-1)%n])])}
        self.rewired=torch.stack([nodes.repeat(3),torch.cat([nodes,(nodes+5)%n,(nodes-5)%n])])
        self.train={"ids":nodes[:80],"y":nodes[:80]%5}; self.select={"ids":nodes[80:100],"y":nodes[80:100]%5}
        self.inputs={}
        for key,value in (("graph",self.graph),("train",self.train),("select",self.select)):
            torch.save(value,self.root/(key+".pt"));self.inputs[key]=p.receipt(self.root,key+".pt")
        torch.save({"edge_index":self.rewired},self.root/"rewired.pt")
        self.c={"classes":5,"learning_rate":.005,"dropout":.5,"native":p.NATIVE,"node_inputs":self.inputs,
            "identity":{"split":0,"input_custody_sha256":self.inputs["graph"]["sha256"],"TRAIN_custody_sha256":self.inputs["train"]["sha256"],"VALID_custody_sha256":self.inputs["select"]["sha256"]}}
        self.study={"study_id":"synthetic-private-source","dataset":"synthetic_fixture","data":{"task":"node_classification"},"source":{"synthetic":True}}
        self.context={"registry":object(),"output_root":self.root,"paid_parent_checkpoints":{}}

    def tearDown(self): self.temporary.cleanup()

    def row(self,family,name,body=0):
        c=copy.deepcopy(self.c)
        if family=="cmcl":
            c.update(architecture="private_cmcl",cmcl_condition=name,cmcl_source=cmcl.SOURCE,members=4,physical_slow_bodies=1,source_optimizer_count=1,source_optimizer_calls_per_cycle=1)
        elif family=="core":
            bodies,members,owners,_,_=core.LAW[name]
            c.update(architecture="masked_core",core_condition=name,core_source=core.SOURCE,members=members,physical_slow_bodies=bodies,source_optimizer_count=owners,optimizer_calls_per_cycle=owners,
                macro_cycle_law="one_complete_old_state_all_owner_cycle",genuine_own_selected_I4=False,auxiliary_graph=p.receipt(self.root,"rewired.pt") if name=="shared4_core_rewired" else None)
        elif family=="adam":
            c.update(architecture="private_adam",core_condition="shared4_own",core_source=core.SOURCE,private_adam_source=adam.SOURCE,members=4,physical_slow_bodies=1,source_optimizer_count=1,source_optimizer_calls_per_cycle=1,
                private_adam_normalization="private_R_S_epsilon_and_weight_decay_divided_by_M")
        else:
            c.update(control_condition=name,architecture={"ordinary_m1":"single","factorized_m1":"factorized_single","original_boundary":"original_gnnm","shared4_own":"internal_be"}[name],core_condition="shared4_own" if name=="shared4_own" else "single_native",
                core_source=core.SOURCE,members=4 if name in ("original_boundary","shared4_own") else 1,physical_slow_bodies=1,source_optimizer_count=1)
        c["fusion"]="mean_logits" if name=="original_boundary" else "mean_probabilities"
        return {"run_id":family+"-"+name+"-b"+str(body),"kind":"model_fit","method":family+"-"+name,"seed":9101,"body":body,"config":c,
            "budget":{"unit":"epochs" if family=="core" else "optimizer_updates","policy":"full_budget","maximum":5000},
            "selection":{"metric":"accuracy","direction":"max","ties":"first","evaluation_interval":1,"checkpoint0":False}}

    def make(self,family,name,body=0,context=True):
        module={"cmcl":cmcl,"core":core,"adam":adam,"control":controls}[family]
        return module.make_adapter(torch=self.t,project_root=self.root,study=self.study,run=self.row(family,name,body),device="cpu",execution_context=self.context if context else None)

    def test_CMCL_orientation_owners_source_replay_and_shared_gradient(self):
        t=self.t; adapters={name:self.make("cmcl",name) for name in cmcl.CONDITIONS}
        own=adapters["own_floor"]; session=own.source_session
        env={"math":math,"MEMBERS":4,"CLASSES":5,"OWNER_K":3,"BETA":.75,"LAMBDA":1.,"require":p.require,"CLOSED":session.caps}
        p.definitions(self.root,cmcl.SOURCE["objective"],("node_terms","assignment","member_terms"),env)
        z=t.tensor([[10.,-5.,-4.,-3.,-2.]],requires_grad=True);truth=t.tensor([2])
        ce,kl=env["node_terms"](t,z,truth)
        self.assertTrue(t.equal(kl,-math.log(5)-z.log_softmax(-1).mean(-1)))
        mask=env["assignment"](t,t.zeros(4,2),t.zeros(4,2));self.assertTrue(t.equal(mask,t.tensor([[1,1],[1,1],[1,1],[0,0]],dtype=t.bool)))
        before=t.get_rng_state().clone()
        for name,adapter in adapters.items():
            result=adapter.train_step(1);self.assertTrue(t.equal(before,t.get_rng_state()))
            for key,value in cmcl.COUNTS[name].items():self.assertEqual(result["operation_counts"][key],value)
            self.assertEqual(len(adapter.optimizer.optimizers),1)
        own_named=dict(own.model.named_parameters())
        for name in ("private_cmcl","private_uniform","private_constant_credit"):
            other=dict(adapters[name].model.named_parameters())
            for key,parameter in own_named.items():
                if not key.endswith((".r",".s")): self.assertTrue(t.equal(parameter.grad,other[key].grad),key)
            self.assertTrue(any(not t.equal(parameter.grad,other[key].grad) for key,parameter in own_named.items() if key.endswith((".r",".s"))))

    def test_CORE_exact_negative_paths_masks_audit_and_all_owner_counts(self):
        t=self.t;method=core.method_namespace(t,self.root,self.c,self.graph["x"],"synthetic_fixture")
        indices=method.negative_indices(40,core.random.Random(91))
        self.assertTrue(all(len(set(row.tolist()))==32 and i not in row for i,row in enumerate(indices)))
        z=t.randn(40,7,requires_grad=True);targets=t.randn(40,7)
        loss=method.core_loss(z,targets,indices,chunk=11)
        zn=t.nn.functional.normalize(z,dim=-1,eps=1e-8);xn=t.nn.functional.normalize(targets,dim=-1,eps=1e-8)
        scores=t.cat([(zn*xn).sum(-1,keepdim=True),(zn[:,None]*zn[indices]).sum(-1)],dim=-1)/.2
        reference=t.nn.functional.cross_entropy(scores,t.zeros(40,dtype=t.long))
        self.assertTrue(t.allclose(loss,reference,atol=1e-6))
        grad=t.autograd.grad(loss,z,retain_graph=True)[0]
        self.assertTrue(t.allclose(grad,t.autograd.grad(reference,z,retain_graph=True)[0],atol=1e-6))
        detached=t.cat([(zn*xn).sum(-1,keepdim=True),(zn[:,None]*zn.detach()[indices]).sum(-1)],dim=-1)/.2
        detached_grad=t.autograd.grad(t.nn.functional.cross_entropy(detached,t.zeros(40,dtype=t.long)),z)[0]
        self.assertGreater(float((grad-detached_grad).abs().sum()),.01)
        for name,law in core.LAW.items():
            adapter=self.make("core",name);session=adapter.source_session
            self.assertEqual(len(session.bodies),law[0]);self.assertEqual(len(session.optimizers),law[2])
            if session.spec["masked"]:
                ids=session.anchor_ids[0];self.assertEqual(int(session.masked[0].list_mat[0][ids].count_nonzero()),0)
                expected=p.mono_tokens(t,self.root,session.masked[0].list_mat[0],self.rewired if name=="shared4_core_rewired" else self.graph["edge_index"],K=2,preprocessing="pinned_mono_base")
                self.assertTrue(all(t.equal(a,b) for a,b in zip(expected,session.masked[0].list_mat)))
            before=t.get_rng_state().clone();result=session.train_step(audit=session.spec["core"])
            self.assertTrue(t.equal(before,t.get_rng_state()));self.assertEqual(session.counters["optimizer_steps"],law[2]);self.assertEqual(session.counters["backwards"],law[4])
            self.assertEqual(session.counters["factual_forwards"]+session.counters["masked_forwards"],law[3])
            self.assertEqual(result["decoder_data_gradient_divisor"],4 if name=="untied4_shared_decoder_core" else 1)
            if session.spec["core"]:self.assertEqual(session.audit["head"]["nonzero_gradients"],0)

    def test_private_Adam_changes_only46_private_group_fields_and_same_gradients(self):
        t=self.t;normalized=self.make("adam","normalized");native=self.make("core","shared4_own")
        inventory=normalized.source_session._M_normalization_inventory
        self.assertEqual(inventory["changed_private_groups"],46);self.assertEqual(inventory["unchanged_native_groups"],52)
        for row in inventory["groups"]:
            expected=dict(row["native"])
            if row["private_R_S"]:expected.update(eps=expected["eps"]/4,weight_decay=expected["weight_decay"]/4)
            self.assertEqual(expected,row["normalized"])
        normalized.train_step(1);native.train_step(1)
        for (name,parameter),(other_name,other) in zip(normalized.model.named_parameters(),native.model.named_parameters()):
            self.assertEqual(name,other_name);self.assertTrue(t.equal(parameter.grad,other.grad),name)
        adam_source=adam.normalization_namespace(t,self.root,self.c,self.graph["x"])
        self.assertEqual(adam_source["work"](normalized.source_session,1)["optimizer_steps"],1)

    def test_real_engine_selected_restore_all_streams_moments_and_strict_tie(self):
        from engine import run_trajectory,restore_checkpoint
        t=self.t
        for family,name in (("cmcl","private_cmcl"),("core","untied4_shared_decoder_core"),("adam","normalized")):
            adapter=self.make(family,name)
            contract={"fit_id":family,"budget_steps":2,"budget_unit":"epochs" if family=="core" else "optimizer_updates","validation_interval":1,"validate_initial":False,"prefixes":[],"inherited_fit_ids":[],"adapter_spec":{}}
            output=self.root/("trajectory-"+family);result=run_trajectory(t,adapter,contract,output)
            self.assertEqual(result["operation_counts"]["optimizer_updates"],2*(5 if family=="core" else 1))
            replacement=self.make(family,name)
            restore_checkpoint(t,output/"selected.pt",replacement,expected_fit_id=family)
            first=replacement.train_step(3)
            restore_checkpoint(t,output/"selected.pt",replacement,expected_fit_id=family)
            self.assertEqual(first,replacement.train_step(3))
            first_event=replacement.observe_validation(10,{"selection_metric":1.})
            tie_event=replacement.observe_validation(11,{"selection_metric":1.})
            self.assertTrue(first_event["save_selected"]);self.assertFalse(tie_event["save_selected"])
        readonly=self.make("cmcl","own_floor",context=False)
        with self.assertRaisesRegex(RuntimeError,"Restoration-only"):readonly.train_step(1)

    def test_genuine_separate_controls_and_actual_boundary(self):
        from engine import inspect_independence
        for name in ("ordinary_m1","factorized_m1"):
            adapters=[self.make("control",name,body) for body in range(4)];inspect_independence(adapters)
            for adapter in adapters:
                self.assertEqual(len(adapter.logits_fn(adapter.model,False,0)),1)
                self.assertEqual(len(adapter.optimizer.optimizers),1)
                self.assertEqual(sum(isinstance(m,adapter.source_session.factors.FactorLinear) for m in adapter.model.modules()),23 if name=="factorized_m1" else 0)
        boundary=self.make("control","original_boundary")
        self.assertEqual(boundary.model.boundary_sites,("lin1","lin3"));self.assertEqual(boundary.model.factor_sites,0)
        result=boundary.train_step(1);self.assertEqual(result["operation_counts"]["backward_calls"],4)
        self.assertEqual(boundary.fusion,"mean_logits")

    def test_actual_common_exporter_four_own_selected_source_M1_checkpoints(self):
        from engine import run_trajectory
        from adapters import node_endpoints
        t=self.t;output=self.root/"export-output";records={};parents=[]
        for body in range(4):
            row=self.row("control","factorized_m1",body);row["adapter"]="fixture:make_adapter"
            identifier=row["run_id"];parents.append(identifier)
            contract={"fit_id":identifier,"budget_steps":1,"budget_unit":"optimizer_updates","validation_interval":1,"validate_initial":False,"prefixes":[],"inherited_fit_ids":[],"adapter_spec":{}}
            result=run_trajectory(t,self.make("control","factorized_m1",body),contract,output/self.study["study_id"]/identifier)
            records[identifier]={"state":"complete","contract":row,"result":result}
        class Registry:
            def run(self,study,identifier):return records[identifier]
        paths={}
        for role,ids in (("cal",t.tensor([100,101])),("evaluate",t.tensor([102,103]))):
            t.save(ids,self.root/(role+"_ids.pt"));paths[role+"_ids"]=role+"_ids.pt"
        row={"parents":parents,"config":{"fit_parents":parents,"endpoint_inputs":paths,"fusion":"mean_probabilities"}}
        directory=self.root/"export";directory.mkdir()
        with patch.object(node_endpoints,"load_factory",return_value=controls.make_adapter):
            result=node_endpoints.export_bank(project_root=self.root,study=self.study,run=row,output_root=output,output_dir=directory,registry=Registry(),device="cpu")
        self.assertTrue(result["independent_complete_learners"]);self.assertFalse(result["CAL_targets_read"]);self.assertFalse(result["evaluation_targets_read"])
        self.assertEqual(len({a["actual_model_seed"] for a in result["ancestry"]}),4)
        self.assertEqual(tuple(t.load(directory/"cal_logits.pt",weights_only=True)["logits"].shape),(4,2,5))
