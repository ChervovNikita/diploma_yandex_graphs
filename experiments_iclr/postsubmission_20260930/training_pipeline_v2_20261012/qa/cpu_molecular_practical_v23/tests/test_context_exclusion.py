"""Exact context algebra, target custody, native ownership and restore checks."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adapters.context_exclusion import (SOURCE_ARMS, SOURCE_FILES, SOURCE_CONTRACT, make_adapter,
                                       objective_module, receipt, target_payload, validate_config)
from compile_context_exclusion_matrix import DOMAIN, I4, make_study


PROJECT = Path(os.environ.get("GNNM_REFERENCE_PROJECT", Path(__file__).resolve().parents[2]))
NUMERIC = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "torch_geometric"))


def source_bindings(root, *, copy_sources=False):
    result = {}
    for key, (relative, digest) in SOURCE_FILES.items():
        source = PROJECT / relative
        if not source.is_file():raise AssertionError("Missing pinned source: " + str(source))
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:raise AssertionError("Pinned source changed")
        target = root / relative
        if copy_sources:
            target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(source.read_bytes())
        result[key] = {"path": relative, "bytes":source.stat().st_size,"sha256":digest}
    return result


class ContextMetadataTests(unittest.TestCase):
    def test_exact_source_and_only_row_guard_delta(self):
        bindings = source_bindings(PROJECT)
        self.assertEqual(len(bindings), 7)
        helper = objective_module(PROJECT / SOURCE_FILES["objective"][0])
        self.assertEqual(helper.CONTEXT_NAMES, tuple(SOURCE_CONTRACT["contexts"]))
        self.assertEqual(helper.COEFFICIENT, .05)
        self.assertEqual(len(DOMAIN), 9)
        self.assertIn({"learning_rate":.001,"dropout":.2}, DOMAIN)

    def metadata_fixture(self, root):
        files=[];splits=[]
        def record(name):
            row={"path":name,"bytes":1,"sha256":hashlib.sha256(name.encode()).hexdigest()};files.append(row);return name
        graph=record("graph.pt")
        for split in range(20):
            node={"graph":graph,**{role:record("split_"+str(split)+"/"+role+".pt") for role in ("train","select")}}
            endpoint={k:record("split_"+str(split)+"/"+k+".pt") for k in ("cal_ids","cal_targets","evaluate_ids","evaluate_targets")}
            splits.append({"split_id":split,"node_inputs":node,"endpoint_inputs":endpoint,"evaluate_purpose":"development",
                           "role_class_counts":{"train":{"0":20,"1":20,"2":20}}})
        custody={"dataset_id":"wikics","status":"prepared","benchmark_fits":0,"predictive_scores":0,
                 "request_hash":"a"*64,"model_graph":{"classes":3},"files":files,"splits":splits,
                 "request":{"graph_policy":{"fixture":True},"evaluation_exposure":"development"}}
        bindings=source_bindings(PROJECT)
        targets={"source_contract":SOURCE_CONTRACT,"request_hash":custody["request_hash"],"source_bindings":bindings,
                 "SELECT_CAL_TEST_tensors_opened":False,"model_constructed":False,"optimizer_created":False,
                 "splits":[{"split_id":i,"target":{"path":"targets/"+str(i)+".pt","bytes":1,"sha256":"b"*64}} for i in range(20)]}
        sources={"git_commit":"fixture","files":[{"path":"fixture.py","sha256":"c"*64}]}
        profile={"reservation_mib":1024,"wall_seconds":1000,"estimated_seconds_per_update":.01,"evidence":"synthetic scheduling declaration"}
        return make_study(root,custody,root/"data",targets,sources,bindings,{"*":profile},"runtime")

    def test_exact_roster_counts_and_i4_search_body0_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            study,counts,banks=self.metadata_fixture(Path(directory))
        self.assertEqual(counts["optimizer_starts_by_stage"],{"search":162,"comparative":1080})
        self.assertEqual(counts["atomic_fit_groups"],1242)
        self.assertEqual(counts["full_budget_optimizer_updates"],6210000)
        self.assertEqual(counts["comparative_banks"],720)
        self.assertEqual(counts["genuine_I4_banks"],180)
        self.assertEqual(counts["all_run_records"],3411)
        runs={r["run_id"]:r for r in study["runs"]}
        for bank in banks:
            if bank["genuine_I4"]:
                parents=[runs[i] for i in bank["fit_parents"]]
                self.assertEqual([p["body"] for p in parents],[0,1,2,3])
                self.assertTrue(all(p["config"]==parents[0]["config"] for p in parents))
                self.assertEqual(len({p["seed"]+1000003*p["body"] for p in parents}),4)
            if bank["arm"]=="original_boundary":self.assertEqual(runs[bank["export"]]["config"]["fusion"],"mean_logits")
        for row in study["runs"]:
            if row["kind"]=="model_fit":validate_config(study,row)

    def test_allocation_guard_precedes_file_reads_in_both_modes(self):
        import compile_context_exclusion_matrix as compiler
        argv=["program","--project-root","missing","--custody","missing","--source-bindings","missing","--output","missing"]
        for flags in ([],["--prepare-targets"]):
            with patch("sys.argv",argv+flags),patch.object(compiler,"verify_allocation",side_effect=RuntimeError("guard")),patch.object(Path,"read_text") as opened:
                with self.assertRaisesRegex(RuntimeError,"guard"):compiler.main()
                opened.assert_not_called()


@unittest.skipUnless(NUMERIC,"Torch/NumPy/PyG runtime unavailable; allocation QA required")
class ContextNumericalTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        self.t,self.np=torch,np;torch.set_num_threads(1)
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.sources=source_bindings(self.root,copy_sources=True)
        helper=objective_module(self.root/SOURCE_FILES["objective"][0])
        generator=torch.Generator().manual_seed(913)
        n=66;features=5
        self.graph={"x":torch.randn(n,features,generator=generator),
                    "edge_index":torch.stack([torch.arange(n).repeat(3),torch.cat([torch.arange(n),(torch.arange(n)+1)%n,(torch.arange(n)+7)%n])])}
        ids=torch.randperm(60,generator=generator);y=(torch.arange(60)//20)[ids]
        self.train={"ids":ids,"y":y};self.select={"ids":torch.arange(60,66),"y":torch.arange(6)%3}
        self.inputs={}
        for name,value in [("graph",self.graph),("train",self.train),("select",self.select)]:
            path=self.root/(name+".pt");torch.save(value,path);self.inputs[name]=receipt(self.root,path.name)
        ancestry={"graph":self.inputs["graph"],"train":self.inputs["train"],"objective_sha256":SOURCE_FILES["objective"][1]}
        self.targets=target_payload(torch,np,helper,self.graph,self.train,ancestry)
        torch.save(self.targets,self.root/"targets.pt");self.target_binding=receipt(self.root,"targets.pt")

    def tearDown(self):self.temporary.cleanup()

    def make(self,arm="shared4_context_private_masked",body=0):
        architecture,members,mode=SOURCE_ARMS[arm]
        config={"context_arm":arm,"architecture":architecture,"members":members,"context_mode":mode,
                "backbone":"SAGE","width":128,"depth":2,"weight_decay":0.,"learning_rate":.001,"dropout":.2,
                "classes":3,"fusion":"mean_logits" if architecture=="original_gnnm" else "mean_probabilities",
                "node_inputs":self.inputs,"context_source":self.sources,
                "context_targets":self.target_binding if mode!="none" else None}
        run={"seed":72101,"body":body,"config":config,"budget":{"unit":"optimizer_updates","policy":"full_budget","maximum":5000},
             "selection":{"metric":"accuracy","direction":"max","ties":"first","evaluation_interval":1,"checkpoint0":False}}
        study={"dataset":"synthetic_fixture","data":{"task":"node_classification"}}
        return make_adapter(torch=self.t,project_root=self.root,study=study,run=run,device="cpu")

    def test_targets_preserve_nonself_class_degree_order_and_native_rng(self):
        t,np=self.t,self.np;helper=objective_module(self.root/SOURCE_FILES["objective"][0])
        before=np.random.get_state()
        arrays,meta=helper.prepare_targets(np,self.graph["x"].numpy(),self.graph["edge_index"].numpy(),self.train["ids"].numpy(),self.train["y"].numpy())
        after=np.random.get_state();self.assertTrue(np.array_equal(before[1],after[1]))
        self.assertTrue(np.array_equal(arrays["positive"].sum(-1),arrays["shuffled_positive"].sum(-1)))
        self.assertFalse(np.diagonal(arrays["positive"],axis1=1,axis2=2).any())
        self.assertTrue(all(c>0 for c in meta["changed_incidence_entries_per_context"]))
        order=self.targets["single_order"]
        bank=self.targets["banks"]["bank"]["weights"]
        self.assertTrue(t.equal(self.targets["banks"]["single"]["weights"],bank[:,order[:,None],order[None,:]]))

    def test_source_580row_algebra_and_gradients_equal_dynamic_function(self):
        t,np=self.t,self.np;path=self.root/SOURCE_FILES["objective"][0]
        spec=importlib.util.spec_from_file_location("untouched_context_fixture",path)
        source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
        adapted=objective_module(path)
        rows=580;generator=t.Generator().manual_seed(419)
        x=t.randn(rows,5,generator=generator);edges=t.stack([t.arange(rows), (t.arange(rows)+1)%rows])
        ids=t.arange(rows);y=t.arange(rows)%10
        arrays,_=source.prepare_targets(np,x.numpy(),edges.numpy(),ids.numpy(),y.numpy())
        targets=source.tensor_bank(t,arrays,np.arange(rows,dtype=np.int64),"cpu")
        for mode in ("private","mixture","private_full","shuffled_private"):
            h=t.randn(4,rows,6,generator=generator,requires_grad=True)
            original=source.context_auxiliary(t,t.nn.functional,h,targets,mode)
            gradient=t.autograd.grad(original[0],h)[0]
            changed=adapted.context_auxiliary(t,t.nn.functional,h,targets,mode)
            self.assertTrue(t.equal(original[0],changed[0]))
            self.assertTrue(t.equal(gradient,t.autograd.grad(changed[0],h)[0]))
            self.assertEqual(original[3],changed[3])

    def test_all_seven_source_modes_live_gradients_and_exact_factor_starts(self):
        t=self.t
        for arm in SOURCE_ARMS:
            adapter=self.make(arm)
            result=adapter.train_step(1)
            self.assertEqual(result["operation_counts"]["optimizer_updates"],1)
            self.assertEqual(result["operation_counts"]["model_forwards"],SOURCE_ARMS[arm][1])
            if SOURCE_ARMS[arm][2]!="none":
                self.assertEqual(result["operation_counts"]["context_extra_native_forwards"],0)
                upstream=adapter.model.source_model.body if SOURCE_ARMS[arm][1]==4 else adapter.model.source_model
                self.assertIsNotNone(upstream.input_linear.weight.grad)
                if SOURCE_ARMS[arm][0]!="single":self.assertIsNotNone(upstream.input_linear.r.grad)
        adapter=self.make("factorized_m1_context_mixture")
        body=adapter.model.source_model
        self.assertTrue(t.all(body.input_linear.r.abs()==1))
        self.assertTrue(t.all(body.output_linear.r==1))

    def test_actual_checkpoint_restore_rng_target_guard_and_genuine_i4(self):
        from engine import run_trajectory,restore_checkpoint,inspect_independence
        t=self.t;adapter=self.make()
        contract={"fit_id":"synthetic-context","budget_steps":2,"budget_unit":"optimizer_updates","validation_interval":1,
                  "validate_initial":False,"prefixes":[],"inherited_fit_ids":[],"adapter_spec":{}}
        result=run_trajectory(t,adapter,contract,self.root/"trajectory")
        self.assertEqual(result["operation_counts"]["optimizer_updates"],2)
        replacement=self.make()
        restore_checkpoint(t,self.root/"trajectory/selected.pt",replacement,expected_fit_id="synthetic-context")
        replacement.model.eval();first=replacement.logits_fn(replacement.model,False,1)
        restore_checkpoint(t,self.root/"trajectory/selected.pt",replacement,expected_fit_id="synthetic-context")
        self.assertTrue(t.equal(first,replacement.logits_fn(replacement.model,False,1)))
        state=copy.deepcopy(replacement.capture_state())
        a=replacement.train_step(3)
        restore_checkpoint(t,self.root/"trajectory/selected.pt",replacement,expected_fit_id="synthetic-context")
        self.assertEqual(a,replacement.train_step(3))
        adapters=[self.make("factorized_m1_context_mixture",body=i) for i in range(4)]
        inspect_independence(adapters)
        self.assertEqual(len({a.provenance["actual_model_seed"] for a in adapters}),4)
        with t.no_grad():replacement.context_targets["weights"].add_(.01)
        with self.assertRaises(ValueError):replacement.restore_state(state)


if __name__=="__main__":unittest.main()
