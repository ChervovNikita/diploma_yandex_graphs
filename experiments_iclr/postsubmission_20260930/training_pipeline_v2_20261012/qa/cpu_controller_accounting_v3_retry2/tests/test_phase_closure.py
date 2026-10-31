"""Cross-host evidence gates must prevent partial or failed-cohort scoring."""
import json
from pathlib import Path
import tempfile
import unittest

from launcher import gate_ready
from phase_closure import collect, write_host_receipts


class RegistryFixture:
    def __init__(self, rows):
        self.rows = rows
    def states(self):
        return {k:r["state"] for k,r in self.rows.items()}
    def run(self, study, identifier):
        return self.rows[study+"/"+identifier]


class PhaseClosureTests(unittest.TestCase):
    def fixture(self, stage):
        runs = {"fixture/"+host: {"study":"fixture", "run":{"run_id":host,"kind":"model_fit", "config":{"stage":stage}}}
                for host in ("allocation", "server77")}
        placement = {k:{"worker":k.split("/")[1]+"/gpu0"} for k in runs}
        rows = {k:{"state":"complete", "contract":r["run"],"result":{"fixture":True}} for k,r in runs.items()}
        plan = {"global_gates":{"gate":{"path":"gates/global.json","source_stages":[stage],
                    "required_for_stages":["evaluation"],"required_hosts":["allocation","server77"]}}}
        return plan, runs, placement, rows

    def test_missing_host_never_closes_and_changed_proof_never_opens_gate(self):
        plan,runs,placement,rows=self.fixture("comparative")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);registry=RegistryFixture(rows)
            write_host_receipts(root,plan,"plan",runs,placement,registry,"allocation")
            self.assertFalse(gate_ready(root,plan,"plan","evaluation"))
            with self.assertRaises(FileNotFoundError):
                collect(root,plan,"plan",runs,placement,"gate")
            write_host_receipts(root,plan,"plan",runs,placement,registry,"server77")
            collect(root,plan,"plan",runs,placement,"gate")
            self.assertTrue(gate_ready(root,plan,"plan","evaluation"))
            proof=root/"gates/server77_gate.json"
            proof.write_text(proof.read_text()+" ")
            with self.assertRaises(ValueError):
                gate_ready(root,plan,"plan","evaluation")

    def test_failed_comparative_bank_blocks_host_proof(self):
        plan,runs,placement,rows=self.fixture("comparative")
        rows["fixture/allocation"]["state"]="failed"
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_host_receipts(root,plan,"plan",runs,placement,RegistryFixture(rows),"allocation")
            self.assertFalse((root/"gates/allocation_gate.json").exists())

    def test_failed_search_is_retained_but_not_disguised_as_success(self):
        plan,runs,placement,rows=self.fixture("search")
        rows["fixture/allocation"]["state"]="failed"
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_host_receipts(root,plan,"plan",runs,placement,RegistryFixture(rows),"allocation")
            value=json.loads((root/"gates/allocation_gate.json").read_text())
            self.assertEqual(value["results"]["fixture/allocation"]["state"],"failed")


if __name__ == "__main__":
    unittest.main()
