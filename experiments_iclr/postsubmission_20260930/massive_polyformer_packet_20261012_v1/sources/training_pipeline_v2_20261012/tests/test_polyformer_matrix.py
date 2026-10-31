"""Scoped domains/counts and selected PolyFormer independent bank engineering."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from compile_polyformer_matrix import (author_status, coefficient_domain, make_study,
                                      parameter_domain, templates)
from adapters.polyformer_endpoints import independent_contracts
import test_polyformer as fixture


PACKAGE = Path(__file__).resolve().parents[1]


def fixture_inputs(dataset="pubmed"):
    custody = {"dataset_id":dataset,"request_hash":"synthetic-fixture",
        "request":{"graph_policy":{"scope":"synthetic_graph_schema"},
                   "evaluation_exposure":{"scope":"engineering_only"}},
        "provider_files":[],"model_graph":{"nodes":8,"features":3,"classes":3},
        "splits":[{"split_id":0,"evaluate_purpose":"development","role_counts":{"train":4,"select":2,"cal":1,"evaluate":1},
            "node_inputs":{"graph":"graph.pt","train":"train.pt","select":"select.pt"},
            "endpoint_inputs":{"cal_ids":"cal_ids.pt","evaluate_ids":"evaluate_ids.pt",
                               "cal_targets":"sealed/cal.pt","evaluate_targets":"sealed/evaluate.pt"}}]}
    matrix = json.loads((PACKAGE/"docs/MATRIX_AXES_DRAFT.json").read_text())
    recipe = next(r for r in matrix["baseline_recipe_catalogue"]["rows"] if r["recipe_id"] == "polyformer_mono_"+dataset)
    source = {"git_commit": "engineering-fixture", "files": [{"path": "fixed.py", "bytes": 0, "sha256": "0"*64}]}
    profile = {"reservation_mib": 1024, "wall_seconds": 100, "estimated_seconds_per_update": 1., "evidence": "synthetic scheduling assumption"}
    return custody, recipe, source, profile


class PolyFormerMatrixTests(unittest.TestCase):
    def test_source_native_anchor_grouped_adam_and_effective_domains(self):
        _, recipe, _, _ = fixture_inputs()
        rows = templates()
        self.assertEqual(len(rows), 34)
        self.assertEqual(sum(r["members"] == 1 for r in rows), 18)
        self.assertEqual(sum(len(parameter_domain(recipe,r,"native_fixed"))*2 for r in rows), 5440)
        self.assertEqual(sum(len(parameter_domain(recipe,r,"neighbors"))*2 for r in rows), 41920)
        for row in rows:
            domain = parameter_domain(recipe,row,"native_fixed")
            anchor = recipe["native_point"]
            self.assertTrue(any(p["native"]["hidden"]==anchor["width"] and p["native"]["nlayer"]==anchor["depth"]
                and p["native"]["dropout"]==anchor["dropout"] and p["learning_rate"]==anchor["learning_rate_groups"]["other"]
                and p["attn_lr"]==anchor["learning_rate_groups"]["attnmodule"]
                and p["weight_decay"]==anchor["weight_decay_groups"]["other"]
                and p["attn_weight_decay"]==anchor["weight_decay_groups"]["attnmodule"] for p in domain))
            if row["members"] == 1:
                self.assertNotIn("pool_weight",row["loss"]["coefficients"])
                self.assertNotIn("residual_weight",row["loss"]["coefficients"])
            if row["matched_paired_own_control"]:
                self.assertEqual(len(coefficient_domain(row,"neighbors")),1)
                self.assertEqual(row["loss"]["coefficients"]["alignment_weight"],0.)

    def test_full_source_horizon_two_search_seeds_and_fixed_body0_reuse(self):
        custody,recipe,source,profile = fixture_inputs()
        root = Path.cwd()
        spec,counts,banks,domains = make_study(root,custody,root/"synthetic-data",recipe,source,profile,
                                              "native_fixed","frozen-polyformer/runtime")
        self.assertEqual(counts["optimizer_starts_by_stage"], {"search":5440,"comparative":264})
        self.assertEqual(counts["comparative_served_banks"],156)
        self.assertEqual(counts["genuine_I4_banks"],54)
        self.assertEqual(counts["reused_I4_body0_artifacts"],54)
        self.assertEqual(counts["comparative_served_prediction_views"],462)
        self.assertEqual(counts["all_run_records"],6206)
        self.assertEqual(counts["full_budget_optimizer_updates"],5704*2000)
        self.assertEqual(counts["author_replication_fits"],0)
        fits = {r["run_id"]:r for r in spec["runs"] if r["kind"]=="model_fit"}
        for row in spec["runs"]:
            self.assertTrue(row["adapter"].startswith("frozen-polyformer/runtime/"))
            if row["kind"]=="model_fit":
                self.assertEqual(row["budget"],{"unit":"optimizer_updates","policy":"full_budget","maximum":2000})
                self.assertEqual(set(row["config"]["node_inputs"]),{"graph","train","select"})
        for bank in banks:
            if bank["arm"].endswith("_I4"):
                parents=[fits[p] for p in bank["fit_parents"]]
                independent_contracts(spec,parents)
                singleton=next(b for b in banks if b["arm"]==bank["arm"][:-3] and b["seed"]==bank["seed"])
                self.assertEqual(singleton["fit_parents"],bank["fit_parents"][:1])
                self.assertTrue(singleton["single_is_fixed_body0"])
        cohort=[fits[p] for p in next(b for b in banks if b["arm"].endswith("_I4"))["fit_parents"]]
        cohort=copy.deepcopy(cohort)
        cohort[3]["config"]["dropout_seeds"][0]=cohort[0]["config"]["dropout_seeds"][0]
        with self.assertRaisesRegex(ValueError,"dropout"):
            independent_contracts(spec,cohort)

    def test_scoped_provider_and_author_reproduction_boundaries(self):
        custody,recipe,source,profile=fixture_inputs()
        self.assertEqual(author_status(recipe)["author_model_fit_count"],0)
        self.assertEqual(author_status(recipe)["status"],"BLOCKED_EXACT_AUTHOR_REPRODUCTION")
        root=Path.cwd()
        wrong=copy.deepcopy(recipe);wrong["dataset_scope"]=["cora"]
        with self.assertRaisesRegex(ValueError,"dataset-specific"):
            make_study(root,custody,root/"data",wrong,source,profile)
        wrong=copy.deepcopy(recipe);wrong["operator"]["base"]="cheb"
        with self.assertRaises(ValueError):
            make_study(root,custody,root/"data",wrong,source,profile)
        wrong=copy.deepcopy(custody);wrong["splits"][0]["role_counts"]["train"]=513
        with self.assertRaisesRegex(ValueError,"bounded panel"):
            make_study(root,wrong,root/"data",recipe,source,profile)


@unittest.skipIf(fixture.torch is None or not fixture.SOURCES_PRESENT or not fixture.PYG_PRESENT,
                 "Pinned PolyFormer Torch/PyG CPU sources required")
class PolyFormerEndpointFixtureTests(unittest.TestCase):
    def test_semantic_dispatch_four_own_selectors_private_streams_and_target_free_export(self):
        import torch
        from adapters import polyformer
        from adapters.polyformer_endpoints import export_bank
        from engine import run_trajectory
        previous = torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="polyformer-matrix-engineering-",dir=fixture.PROJECT) as name:
                root=fixture.PROJECT;directory=Path(name);output=directory/"runs"
                inputs={}
                for key,value in {"graph":{"x":torch.arange(24,dtype=torch.float32).reshape(8,3)/24,
                    "edge_index":torch.tensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]])},
                    "train":{"ids":torch.tensor([0,1,2,3]),"y":torch.tensor([0,0,1,1])},
                    "select":{"ids":torch.tensor([4,5]),"y":torch.tensor([0,1])}}.items():
                    path=directory/(key+".pt");torch.save(value,path);inputs[key]=str(path.relative_to(root))
                endpoint={}
                for role,identifier in (("cal",6),("evaluate",7)):
                    path=directory/(role+"_ids.pt");torch.save(torch.tensor([identifier]),path)
                    endpoint[role+"_ids"]=str(path.relative_to(root))
                    endpoint[role+"_targets"]=str((directory/(role+"_targets.pt")).relative_to(root))
                study={"study_id":"engineering-polyformer","dataset":"engineering-only","runs":[],
                    "data":{"prior_exposure":"synthetic_engineering_fixture"},"source":{}}
                for body in range(4):
                    cfg={"native":dict(fixture.SMALL_NATIVE),"classes":2,"node_inputs":inputs,
                        "polyformer_mode":"polyformer_factorized_single","initialization_seed":1009+body,
                        "dropout_seeds":[301010+body*10000,302019+body*10000],"preprocessing":"pinned_mono_base",
                        "train_views":2,"fusion":"mean_probabilities","learning_rate":.005,"attn_lr":.0005,
                        "weight_decay":.001,"attn_weight_decay":1e-8,"factor_initialization":"first_rademacher",
                        "factor_seed":900001+body,"independence_group":"fixture-i4",
                        "loss":{"variant":"two_view_class_full_cross_view","coefficients":{"alignment_weight":.05,"temperature":.2}}}
                    row={"run_id":"body"+str(body),"kind":"model_fit","method":"polyformer_equipped_first_paired_class_full",
                        "split":0,"seed":101,"body":body,"config":cfg,"parents":[],
                        "adapter":str((PACKAGE/"adapters/polyformer.py").relative_to(root))+":make_adapter",
                        "budget":{"maximum":1},"selection":{"metric":"accuracy","direction":"max","ties":"first"}}
                    study["runs"].append(row)
                rows={}
                for row in study["runs"]:
                    adapter=polyformer.make_adapter(torch=torch,project_root=root,study=study,run=row,device="cpu")
                    self.assertEqual(adapter.provenance["semantic_method_id"],row["method"])
                    self.assertEqual(adapter.provenance["actual_model_seed"],row["config"]["initialization_seed"])
                    result=run_trajectory(torch,adapter,{"fit_id":row["run_id"],"budget_steps":1,"budget_unit":"optimizer_updates",
                        "validation_interval":1,"validate_initial":False,"prefixes":[],"inherited_fit_ids":[],
                        "adapter_spec":{"fixture":True}},output/study["study_id"]/row["run_id"])
                    self.assertEqual(result["operation_counts"]["model_forwards"],2)
                    self.assertEqual(result["selected"]["step"],1)
                    rows[row["run_id"]]={"state":"complete","contract":row,"result":result}
                class Registry:
                    def run(self,study_id,identifier):return rows[identifier]
                registry=Registry();out=output/study["study_id"]/"export";out.mkdir()
                run={"parents":list(rows),"config":{"fit_parents":list(rows),"fusion":"mean_probabilities","endpoint_inputs":endpoint}}
                bank=export_bank(project_root=root,study=study,run=run,output_root=output,output_dir=out,registry=registry,device="cpu")
                self.assertEqual(len(bank["ancestry"]),4)
                self.assertTrue(bank["independent_complete_learners"])
                self.assertFalse((directory/"cal_targets.pt").exists())
                self.assertFalse((directory/"evaluate_targets.pt").exists())
                raw=torch.load(root/bank["files"]["cal"]["path"],map_location="cpu",weights_only=True)["logits"]
                out2=out.parent/"singleton";out2.mkdir()
                run2={"parents":["body0"],"config":{**run["config"],"fit_parents":["body0"],"singleton_reference":"fixed_body0"}}
                single=export_bank(project_root=root,study=study,run=run2,output_root=output,output_dir=out2,registry=registry,device="cpu")
                own=torch.load(root/single["files"]["cal"]["path"],map_location="cpu",weights_only=True)["logits"]
                self.assertTrue(torch.equal(own,raw[:1]))
                self.assertTrue(single["singleton_reference_is_fixed_body0"])
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
