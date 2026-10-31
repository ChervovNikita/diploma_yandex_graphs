"""One CPU fixture for selected I4 ancestry, label-free export and CAL placement."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


class FakeRegistry:
    def __init__(self, rows):
        self.rows = rows
    def run(self, study, identifier):
        return self.rows[identifier]


@unittest.skipUnless(importlib.util.find_spec("torch"), "Allocation Torch runtime required")
class EndpointTests(unittest.TestCase):
    def test_selected_independent_bank_roundtrip_without_export_target_access(self):
        import torch
        from adapters.node import make_adapter
        from adapters.node_endpoints import export_bank, scalar_calibration, evaluate_bank
        from engine import run_trajectory
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = Path(__file__).resolve().parents[1]
            source = package / "fixtures/native_models.py"
            if not source.exists():
                source = package.parent / "common_wrapper_native_source_20261010_v1/models.py"
            shutil.copyfile(source, root / "models.py")
            torch.save({"x": torch.arange(32,dtype=torch.float32).reshape(8,4)/32,
                        "edge_index": torch.tensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]])}, root/"graph.pt")
            torch.save({"ids":torch.arange(4),"y":torch.tensor([0,1,2,0])},root/"train.pt")
            torch.save({"ids":torch.tensor([4,5]),"y":torch.tensor([1,2])},root/"select.pt")
            torch.save(torch.tensor([6]), root/"cal_ids.pt")
            torch.save(torch.tensor([7]), root/"evaluate_ids.pt")
            inputs = {"cal_ids":"cal_ids.pt", "evaluate_ids":"evaluate_ids.pt",
                      "cal_targets":"cal_targets.pt", "evaluate_targets":"evaluate_targets.pt"}
            config={"node_inputs":{"graph":"graph.pt","train":"train.pt","select":"select.pt"},
                    "native_model_source":"models.py","backbone":"SAGE","depth":1,"width":8,
                    "classes":3,"ffn_multiplier":1,"heads":2,"normalization":"LayerNorm",
                    "dropout":.2,"learning_rate":.001,"weight_decay":0.,"fusion":"mean_probabilities",
                    "members":4}
            study={"study_id":"fixture","data":{"prior_exposure":"engineering_fixture"},"source":{}}
            rows={}
            output=root/"runs"
            # Factory must be a source-bound path within the project.
            for name in ("node.py","classifier.py","node_losses.py"):
                target=root/"adapters"/name;target.parent.mkdir(exist_ok=True)
                shutil.copyfile(package/"adapters"/name,target)
            for body in range(4):
                identifier="body"+str(body)
                run={"run_id":identifier,"kind":"model_fit","method":"single","seed":101,"body":body,
                     "adapter":"adapters/node.py:make_adapter","config":copy.deepcopy(config),
                     "selection":{"metric":"accuracy","direction":"max","ties":"first"},
                     "budget":{"maximum":2},"parents":[]}
                adapter=make_adapter(torch=torch,project_root=root,study=study,run=run,device="cpu")
                result=run_trajectory(torch,adapter,{"fit_id":identifier,"budget_steps":2,
                    "budget_unit":"optimizer_updates","validation_interval":1,"validate_initial":False,
                    "prefixes":[],"inherited_fit_ids":[],"adapter_spec":{"fixture":True}},output/"fixture"/identifier)
                rows[identifier]={"state":"complete","contract":run,"result":result}
            registry=FakeRegistry(rows)
            export_run={"parents":list(rows),"config":{"fit_parents":list(rows),
                        "endpoint_inputs":inputs,"fusion":"mean_probabilities"}}
            out=output/"fixture"/"export";out.mkdir()
            exported=export_bank(project_root=root,study=study,run=export_run,output_root=output,
                                 output_dir=out,registry=registry,device="cpu")
            self.assertTrue(exported["independent_complete_learners"])
            self.assertFalse(exported["evaluation_targets_read"])
            self.assertFalse((root/"cal_targets.pt").exists())
            rows["export"]={"state":"complete","result":exported}
            torch.save({"ids":torch.tensor([6]),"y":torch.tensor([0])},root/"cal_targets.pt")
            cal_run={"parents":["export"],"config":{"export_parent":"export","endpoint_inputs":inputs,
                       "fusion":"mean_probabilities","placement":"after_serving_pool","steps":500,"learning_rate":.01}}
            calibrated=scalar_calibration(project_root=root,study=study,run=cal_run,output_root=output,
                                         output_dir=out,registry=registry,device="cpu")
            self.assertFalse((root/"evaluate_targets.pt").exists())
            rows["cal"]={"state":"complete","result":calibrated}
            torch.save({"ids":torch.tensor([7]),"y":torch.tensor([1])},root/"evaluate_targets.pt")
            eval_run={"parents":["export","cal"],"config":{"export_parent":"export",
                      "calibration_parent":"cal","endpoint_inputs":inputs,"fusion":"mean_probabilities"}}
            scored=evaluate_bank(project_root=root,study=study,run=eval_run,output_root=output,
                                 output_dir=out,registry=registry,device="cpu")
            self.assertEqual(scored["metrics"]["raw"]["accuracy"],scored["metrics"]["calibrated"]["accuracy"])
            self.assertEqual(len(scored["member_diagnostics"]["accuracies"]),4)


if __name__ == "__main__":
    unittest.main()
