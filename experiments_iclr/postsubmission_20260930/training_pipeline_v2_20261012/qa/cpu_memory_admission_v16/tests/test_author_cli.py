"""Standard-library engineering fixtures; no author data/model fits or GO."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from author_cli_spec import (SPLIT_SEEDS,author_counts,receipt,register_author_spec,validate_author_spec)
from registry import Registry
from compile_polyformer_author_study import build_author_study,write_packet,RUNTIME_FILES
from run_author_cli import (Progress,execute_claimed_group,require_global_author_gate)
from spec import canonical,fingerprint


def fixture(root,source):
    work=root/"work";work.mkdir();(work/"training.py").write_text(source)
    (work/"raw.bin").write_bytes(b"synthetic-opaque-raw-provider")
    (work/"bases").mkdir();(work/"processed").mkdir()
    run={"run_id":"synthetic-cli","kind":"author_cli_group","method":"synthetic-author-cli-observation-fixture",
         "split":"native_source_ten_seed_sequence","seed":42,"body":0,"parents":[],
         "config":{"stage":"author_cli","reservation_mib":1,"wall_seconds":10,"estimated_seconds":1},
         "budget":{"unit":"optimizer_updates_per_repeat","policy":"author_native_validation_patience","maximum":2000,"patience":250,"repeats":10,"resume":False},
         "selection":{"metric":"accuracy","direction":"max","ties":"first_strict_above_initial_zero","evaluation_interval":1,"checkpoint0":False,"artifact":"source_in_memory_no_checkpoint"},
         "cli":{"cwd":"work","argv":["-u","training.py"],"python_executable":sys.executable,"execution_repository":str(root),
                "source_files":[receipt(root,"work/training.py")],"raw_files":[receipt(root,"work/raw.bin")],
                "environment":{},"runtime_versions":{},"quarantine":{"stdout":"quarantine/stdout.log","stderr":"quarantine/stderr.log"},
                "basis_dir":"work/bases","processed_dir":"work/processed",
                "protocol":{"split_seeds":SPLIT_SEEDS,"repeat_grouping":"ten_sequential_repeats_one_process"}}}
    value={"schema_version":1,"study_id":"synthetic-author-cli-engineering","stage":"engineering","hypothesis":"test TEST-free source-stream observation and real registry lifecycle",
           "dataset":"synthetic-opaque-provider","data":{"roles":{"train":"synthetic","select":"synthetic","evaluate":"synthetic"},
               "role_protocol":"author_native_train_select_test_no_cal","evidence":["synthetic fixture"],"prior_exposure":"engineering"},
           "source":{"git_commit":"synthetic-fixture-not-scientific-author-source","files":[receipt(root,"work/training.py")]},"runs":[run],
           "comparisons":[{"engineering_only":True}],"release":{"policy":"whole_roster","global_plan_gates_required":True,
             "required_before_launch":["whole_search"],"author_output_gate":"whole_author"},"resources":{"wall_seconds":10}}
    return validate_author_spec(value)


SUCCESS_SOURCE="""import sys
for repeat in range(10):
 print('RP',repeat,'Launched...',flush=True)
 for epoch in range(250):
  print(epoch,'epochs trained. Current acc: 0.1 0.2 TEST_SENTINEL epochs trained. Current loss: 1.1 1.2 TEST_LOSS_SENTINEL',flush=True)
 print('run_'+str(repeat+1)+' \\t test_acc: FINAL_TEST_SENTINEL',flush=True)
print('FINAL_TEST_AGGREGATE_SENTINEL')
sys.stderr.write('STDERR_TEST_SENTINEL\\n')
"""


class AuthorCLITests(unittest.TestCase):
    def test_compiler_eight_group_disabled_contract_and_argument_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"repo/experiments_iclr/phase";root.mkdir(parents=True)
            packet=root/"author";packet.mkdir();(packet/"source").mkdir();(packet/"dependencies/site").mkdir(parents=True)
            commands=[];jobs=[];raw_custody={};profiles={"jobs":{}}
            names={"citeseer":"citeseer","cs":"coauthor-cs","pubmed":"pubmed","physics":"coauthor-physics"}
            for base in ["mono","cheb"]:
                for dataset in ["citeseer","cs","pubmed","physics"]:
                    argv=["python","-u","training.py","--dataset",dataset,"--base",base]
                    commands.append(" ".join(argv));identifier=dataset+"-"+base
                    work=packet/"workdirs"/identifier;work.mkdir(parents=True)
                    (work/"bases").mkdir();(work/"datasets"/dataset/"raw").mkdir(parents=True)
                    (work/"datasets"/dataset/"processed").mkdir()
                    raw=work/"datasets"/dataset/"raw/raw.bin";raw.write_bytes(b"synthetic opaque provider fixture")
                    bound=receipt(root,raw.relative_to(root))
                    raw_custody[names[dataset]]={"raw_files":[dict(bound,path="/synthetic-provider/raw.bin")]}
                    jobs.append({"job_id":identifier,"cwd":"workdirs/"+identifier,"canonical_dataset":names[dataset],
                                 "provider_raw_dir":"datasets/"+dataset+"/raw","python_argv":argv[1:],"original_script_argv":argv,
                                 "base":base,"dataset":dataset,"source_order":len(jobs),
                                 "output":{"stdout":"quarantine/"+identifier+".stdout.log","stderr":"quarantine/"+identifier+".stderr.log"}})
                    profiles["jobs"][identifier]={"reservation_mib":1,"wall_seconds":100,"estimated_seconds":1,"evidence":"synthetic resource fixture"}
            (packet/"source/node_classification.sh").write_text("\n".join(commands)+"\n")
            inventory=[dict(receipt(root,"author/source/node_classification.sh"),path="node_classification.sh")]
            for job in jobs:
                (packet/job["cwd"]/"node_classification.sh").write_text((packet/"source/node_classification.sh").read_text())
            roster={"roster":jobs,"planned_model_fits":80,"raw_custody":raw_custody,
                    "source_commit":"synthetic-engineering-source","source_repository":"synthetic-engineering-fixture"}
            (packet/"AUTHOR_ROSTER.json").write_text(canonical(roster))
            qualified={"dependency_and_unchanged_cli_help_qualified":True,"scientific_training_processes_started":0,"GO_created":False,
                       "contract_sha256":receipt(root,"author/AUTHOR_ROSTER.json")["sha256"]}
            metadata={"PREPARATION_SUMMARY.json":qualified,"FINAL_VERIFICATION.json":{"scientific_fits":0,"scores":0,"quarantine_empty":True},
                      "SOURCE_INVENTORY.json":inventory,"PRIVATE_DEPENDENCY_INVENTORY.json":[],"RAW_CACHE_ANCESTRY.json":{},
                      "AUTHOR_IMPORT_AND_RUNTIME_CHECK.json":{"stdout":"{}"},"SEABORN_WHEEL_INVENTORY.json":[]}
            for name,content in metadata.items():(packet/name).write_text(canonical(content))
            runtime=root/"runtime";runtime.mkdir()
            for name in RUNTIME_FILES:(runtime/name).write_text("# synthetic controller binding\n")
            value=build_author_study(root,"author","runtime",profiles)
            value["stage"]="engineering"
            plan,summary=write_packet(root,"disabled",value)
            self.assertEqual(summary["counts"]["author_cli_groups"],8)
            self.assertEqual(summary["counts"]["author_sequential_repeat_trajectories"],80)
            self.assertEqual(summary["counts"]["author_registered_maximum_optimizer_updates"],160000)
            self.assertFalse(plan["scientific_launch_enabled"])
            self.assertTrue(summary["standalone_plan_cannot_satisfy_whole_master_HPO_gate"])
            self.assertFalse((root/"disabled/gate_receipts").exists())
            roster["roster"][0]["python_argv"].append("--epochs")
            (packet/"AUTHOR_ROSTER.json").write_text(canonical(roster))
            qualified["contract_sha256"]=receipt(root,"author/AUTHOR_ROSTER.json")["sha256"]
            (packet/"PREPARATION_SUMMARY.json").write_text(canonical(qualified))
            with self.assertRaisesRegex(ValueError,"arguments/order"):
                build_author_study(root,"author","runtime",profiles)

    def test_explicit_kind_native_counts_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=fixture(Path(tmp),SUCCESS_SOURCE)
            self.assertEqual(author_counts(value)["author_sequential_repeat_trajectories"],10)
            self.assertIsNone(author_counts(value)["author_native_actual_updates"])
            self.assertNotIn("adapter",value["runs"][0])
            wrong=copy.deepcopy(value);wrong["runs"][0]["budget"]["policy"]="full_budget"
            with self.assertRaisesRegex(ValueError,"native stopping"):
                validate_author_spec(wrong)
            wrong=copy.deepcopy(value);wrong["data"]["roles"]["cal"]="invented"
            with self.assertRaisesRegex(ValueError,"invented CAL"):
                validate_author_spec(wrong)

    def test_TEST_tokens_never_interpreted_and_unknown_partial_cost(self):
        observed=Progress();observed.feed(b"RP 0 Launched...")
        row=observed.feed(b"0 epochs trained. Current acc: 0.1 0.2 garbage_TEST epochs trained. Current loss: 1.1 1.2 garbage_TEST_loss")
        self.assertEqual(row["SELECT_accuracy"],.2)
        self.assertNotIn("garbage",canonical(row))
        cost=observed.costs(child_started=True,exit_code=-15)
        self.assertEqual(cost["known_completed_epoch_optimizer_updates"],1)
        self.assertIsNone(cost["actual_optimizer_updates"])
        self.assertTrue(cost["unobserved_update_count_is_unknown"])

    def test_real_subprocess_custody_complete_and_duplicate_claim_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=fixture(root,SUCCESS_SOURCE);run=value["runs"][0]
            registry=Registry(root/"registry.sqlite")
            try:
                register_author_spec(registry,value);register_author_spec(registry,value)
                owner={"pid":os.getpid(),"hostname":"synthetic-engineering-parent"}
                registry.claim(value["study_id"],run["run_id"],owner)
                result=execute_claimed_group(root,value,run,registry,root/"result",owner,
                    ([sys.executable,"-u","training.py"],root/"work",dict(os.environ)))
                self.assertEqual(result["status"],"complete")
                self.assertEqual(result["cost"]["actual_optimizer_updates"],2500)
                self.assertEqual(result["cost"]["repeat_returns_observed"],10)
                self.assertIn("FINAL_TEST_SENTINEL",(root/"quarantine/stdout.log").read_text())
                safe=canonical(registry.run(value["study_id"],run["run_id"])["result"])+(root/"result/TRAIN_SELECT_PROGRESS.jsonl").read_text()
                for sentinel in ["TEST_SENTINEL","FINAL_TEST_SENTINEL","STDERR_TEST_SENTINEL","FINAL_TEST_AGGREGATE_SENTINEL"]:
                    self.assertNotIn(sentinel,safe)
                self.assertTrue((root/"result/START.json").is_file())
                self.assertIn("deadline_UTC",result)
                with self.assertRaisesRegex(ValueError,"already claimed"):
                    registry.claim(value["study_id"],run["run_id"],owner)
            finally:
                registry.close()

    def test_wall_deadline_retains_failed_cost_and_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source="import time\nprint('RP 0 Launched...',flush=True)\nprint('0 epochs trained. Current acc: 0.1 0.2 TEST epochs trained. Current loss: 1.1 1.2 TEST',flush=True)\ntime.sleep(10)\n"
            value=fixture(root,source);run=value["runs"][0];run["config"]["wall_seconds"]=.25
            registry=Registry(root/"registry.sqlite")
            try:
                register_author_spec(registry,value);owner={"pid":os.getpid(),"hostname":"synthetic-engineering-parent"}
                registry.claim(value["study_id"],run["run_id"],owner)
                result=execute_claimed_group(root,value,run,registry,root/"result",owner,
                    ([sys.executable,"-u","training.py"],root/"work",dict(os.environ)))
                self.assertEqual(result["status"],"failed")
                self.assertEqual(result["failure_type"],"declared_wall_deadline")
                self.assertEqual(result["cost"]["known_completed_epoch_optimizer_updates"],1)
                self.assertIsNone(result["cost"]["actual_optimizer_updates"])
                self.assertEqual(registry.run(value["study_id"],run["run_id"])["state"],"failed")
                self.assertFalse(result["resume_supported"])
            finally:
                registry.close()

    def gate_fixture(self,root,rows,name):
        records={"synthetic/"+row["run_id"]:row for row in rows}
        assigned={key:"allocation/gpu0" for key in records};keys=sorted(records)
        gate={"path":"global.json","source_stages":["search","search_selection"] if name=="whole_search" else ["author_cli"],
              "required_for_stages":["author_cli"] if name=="whole_search" else [],"required_hosts":["allocation"],"all_packet_source_run_keys":keys}
        plan={"global_gates":{name:gate}};digest=fingerprint(plan)
        proof={"status":"complete","plan_hash":digest,"closed_phase":name,"host":"allocation","run_keys":keys,
               "results":{key:{"state":"failed" if records[key]["kind"] in {"model_fit","author_cli_group"} else "complete",
                                  "contract_hash":fingerprint(records[key]),"result_hash":"synthetic-cost-metadata-only"} for key in keys}}
        (root/"host.json").write_text(canonical(proof))
        (root/"global.json").write_text(canonical({"status":"complete","plan_hash":digest,"closed_phase":name,"host_receipts":{"allocation":receipt(root,"host.json")}}))
        return plan,digest,records,assigned

    def test_global_HPO_gate_rejects_subset_and_failed_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[{"run_id":"search","kind":"model_fit","config":{"stage":"search"}},
                  {"run_id":"select","kind":"selection","config":{"stage":"search_selection"}}]
            plan,digest,records,assigned=self.gate_fixture(root,rows,"whole_search")
            require_global_author_gate(root,plan,digest,"whole_search",records,assigned)
            plan["global_gates"]["whole_search"]["all_packet_source_run_keys"].pop()
            with self.assertRaisesRegex(ValueError,"every materialized"):
                require_global_author_gate(root,plan,digest,"whole_search",records,assigned)
            plan,digest,records,assigned=self.gate_fixture(root,rows,"whole_search")
            proof=json.loads((root/"host.json").read_text());proof["results"]["synthetic/select"]["state"]="failed"
            (root/"host.json").write_text(canonical(proof))
            closure=json.loads((root/"global.json").read_text());closure["host_receipts"]["allocation"]=receipt(root,"host.json")
            (root/"global.json").write_text(canonical(closure))
            with self.assertRaisesRegex(ValueError,"Incomplete required"):
                require_global_author_gate(root,plan,digest,"whole_search",records,assigned)

    def test_whole_author_closes_terminal_failures_without_reading_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[{"run_id":"author","kind":"author_cli_group","config":{"stage":"author_cli"}}]
            plan,digest,records,assigned=self.gate_fixture(root,rows,"whole_author")
            result=require_global_author_gate(root,plan,digest,"whole_author",records,assigned)
            self.assertEqual(result["source_run_keys"],["synthetic/author"])
            self.assertNotIn("selected_TEST",canonical(result))


if __name__=="__main__":
    unittest.main()
