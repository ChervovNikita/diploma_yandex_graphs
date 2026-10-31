"""Frozen startup and paid endpoint-failure custody; no training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from registry import Registry
from run_endpoint import execute
from test_orchestration import study


class ExecutionReceiptTests(unittest.TestCase):
    def test_endpoint_failure_retains_paid_costs_and_runner_identity(self):
        class PaidFailure(RuntimeError):
            endpoint_costs = {"forward_graphs": 7, "status": "adapter_value"}
        def fail(**kwargs):
            raise PaidFailure("fixture")
        spec = study()
        spec["runs"][0].update(kind="calibration", budget={}, selection={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("run_endpoint.load", return_value=(spec, "digest")), \
                 patch("run_endpoint.require_approval", return_value={}), \
                 patch("run_endpoint.require_phase_approval"), \
                 patch("run_endpoint.owner_identity", return_value={"pid": 42}), \
                 patch("run_endpoint.load_factory", return_value=fail):
                with self.assertRaises(PaidFailure):
                    execute(root, root/"study.json", "fit", root/"registry.sqlite",
                            root/"out", root/"approval.json", "cpu")
            result = json.loads((root/"out/test/fit/failure.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["owner"], {"pid": 42})
            self.assertEqual(result["endpoint_costs"]["forward_graphs"], 7)
            registry = Registry(root/"registry.sqlite")
            self.assertEqual(registry.run("test", "fit")["result"], result)
            registry.close()

    def test_start_script_uses_plan_frozen_launcher(self):
        source = Path(__file__).resolve().parents[1]/"start_massive.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package=root/"phase/live"; package.mkdir(parents=True)
            (package/"start_massive.sh").write_bytes(source.read_bytes())
            frozen=root/"phase/frozen"; frozen.mkdir()
            (frozen/"launcher.py").write_text("import json,sys; print(json.dumps({'frozen':True,'args':sys.argv[1:]}))")
            (package/"launcher.py").write_text("raise RuntimeError('live launcher was used')")
            bindir=root/"bin";bindir.mkdir()
            git=bindir/"git";git.write_text("#!/bin/sh\nprintf '%s\\n' '"+str(root)+"'\n");git.chmod(0o755)
            plan=root/"phase/plan.json"
            plan.write_text(json.dumps({"runtime_launcher":"frozen/launcher.py","runtime_package":"frozen"}))
            env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ["PATH"],GNNM_PYTHON=sys.executable)
            result=subprocess.run(["bash",str(package/"start_massive.sh"),"allocation",str(plan),str(root/"phase/approval.json")],
                                  env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            value=json.loads(result.stdout)
            self.assertTrue(value["frozen"])
            self.assertIn("--detach",value["args"])
            plan.write_text(json.dumps({"runtime_launcher":"../outside/launcher.py","runtime_package":"frozen"}))
            invalid=subprocess.run(["bash",str(package/"start_massive.sh"),"allocation",str(plan),str(root/"phase/approval.json")],
                                   env=env,capture_output=True,text=True)
            self.assertNotEqual(invalid.returncode,0)

if __name__ == "__main__":
    unittest.main()
