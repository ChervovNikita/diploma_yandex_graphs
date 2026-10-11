"""Fetch complete existing Physics records without rerunning predictive scores."""
import base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r'''
import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=="anogena-2-0"
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True,timeout=15).splitlines()==["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
H=Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_SAGE_Physics_confirmation_root_20261011_v1")
def pinned(item):
 p=Path(item["path"]);assert p.resolve().is_relative_to(H)
 data=p.read_bytes();assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"];return data
end=json.loads((H/"after_fit_pipeline_v1/PIPELINE_END.json").read_text())
assert end==payload["pipeline_end"] and end["terminal"] and end["status"]=="COMPLETE_FIXED_PIPELINE" and end["one_TEST_release"]
cohort=json.loads((H/"cohort_owner_v1/OWNER_END.json").read_text());assert cohort["complete"] and cohort["banks"]==60 and cohort["fits"]==90
external=json.loads((H/"external_cohort_owner_v1/OWNER_END.json").read_text());assert external["exit_code"]==0 and external["child_pid_absent"]
owners={k:json.loads(pinned(item)) for k,item in end["owners"].items()}
for v in owners.values():
 assert v["exit_code"]==0
 assert v.get("child_pid_absent",True) and v.get("owned_cuda_pid_absent",True)
cal=json.loads(pinned(end["calibration"]));assert cal["terminal"] and cal["scalar_endpoints"]==60 and len(cal["records"])==60 and cal["all_finite"]
assert all(r["updates"]==500 and r["status"]=="finite_fixed_endpoint" for r in cal["records"])
test=json.loads(pinned(end["complete_TEST"]));assert test["terminal"] and test["one_complete_TEST_truth_release"] and len(test["rows"])==60 and test["all_raw_finite"] and test["all_calibrated_finite"]
parts={"PIPELINE_CLOSURE.json":{"pipeline_end":end,"cohort_end":cohort,"external_end":external,"owners":owners},"COMPLETE_CAL.json":cal,"ARM_AGGREGATES.json":test["arm_aggregates"],"RESULT_METADATA.json":{k:v for k,v in test.items() if k not in ("rows","arm_aggregates","all_ordered_pair_contrasts")}}
for block in test["blocks"]:parts[f"BANK_ROWS_{block}.json"]={"block":block,"rows":[r for r in test["rows"] if r["block"]==block]}
for kind,by_pair in test["all_ordered_pair_contrasts"].items():
 for candidate in test["arms"]:
  parts[f"CONTRASTS_{kind}_{candidate}.json"]={"kind":kind,"candidate":candidate,"contrasts":{k:v for k,v in by_pair.items() if v["candidate"]==candidate}}
files=[]
for name,value in parts.items():
 data=(json.dumps(value,separators=(",",":"))+"\n").encode();assert len(data)<2000000
 files.append({"name":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"data":base64.b64encode(data).decode()})
print(json.dumps({"complete":True,"original_TEST_receipt":end["complete_TEST"],"partitions":files,"predictive_scores_recomputed":False}))
'''
def main():
 observation=json.loads((H/"COHORT_OBSERVATION_V17.json").read_text());end=observation["after_fit_pipeline"]["end"];assert end
 out=H/"complete_closed_evidence_v1";assert not out.exists()
 spec=importlib.util.spec_from_file_location("Physics_complete_fetch_transport",P/"publication/publish_exact_inventory_v19.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 argv=["ssh","-p","2222","-i","/Users/alex/.ssh/mlspace__private_key_anogena.txt","-o","BatchMode=yes","-o","ConnectTimeout=15","-o","StrictHostKeyChecking=yes","-o","UpdateHostKeys=no","anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru",shlex.join(["/usr/bin/python3","-I","-S","-B","-c",REMOTE,"whole-Physics-result","literal-route","all-fixed-arms"]) ]
 r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps({"pipeline_end":end}).encode())).decode(),capture_output=True,text=True,timeout=60)
 if r.returncode:
  failure={"returncode":r.returncode,"stdout":r.stdout[-4000:],"stderr":r.stderr[-1500:],"scores_recomputed":False}
  (H/"FETCH_FAILURE_V2.json").write_text(json.dumps(failure,indent=2)+"\n")
  print(json.dumps(failure));return
 v=json.loads(r.stdout);assert v["complete"] and not v["predictive_scores_recomputed"];out.mkdir();receipts=[]
 for item in v["partitions"]:
  data=base64.b64decode(item.pop("data"));assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"]
  p=out/item["name"];assert p.parent==out and not p.exists();p.write_bytes(data);receipts.append(item)
 (out/"FETCH_RECEIPTS.json").write_text(json.dumps({"complete":True,"original_TEST_receipt":v["original_TEST_receipt"],"files":receipts,"native_arrays_or_checkpoints_copied":False,"predictive_scores_recomputed":False},indent=2)+"\n")
 print(json.dumps({"complete":True,"partitions":len(receipts),"bytes":sum(r["bytes"] for r in receipts),"raw_and_calibrated_banks":60,"calibration_endpoints":60}))
if __name__=="__main__":main()
