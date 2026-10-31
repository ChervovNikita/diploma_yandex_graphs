"""Guarded allocation-only staging and exact-master sync/audit orchestration.

stage prepares tooling and synthetic checks only. run requires an exact master
and raw summary binding. It creates no tar; bundle creation is a separate CLI.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zlib

OUT = Path(__file__).resolve().parents[1]
PHASE = OUT.parent
spec = importlib.util.spec_from_file_location("guarded_allocation_transport", PHASE / "training_pipeline_v2_20261012/preparation/prepare_molecular_packets_allocation.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

STAGE = helper.GUARD + r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
def atomic(path,value):
 with path.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 path.chmod(0o444)
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
rows=[]
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o444)
 rows.append(receipt(target))
atomic(stage/'SOURCE_INVENTORY.json',rows)
controller=phase/payload['qa_controller'];assert controller.is_dir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',FINAL_SYNC_TEST_CONTROLLER=str(controller))
command=[str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p','test_final_sync.py','-v']
started=time.monotonic();child=subprocess.run(command,cwd=stage,env=env,capture_output=True,text=True,timeout=120)
qa={'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-started,'stdout':child.stdout,'stderr':child.stderr,'actual_input_fits':0,'GO_created':False}
atomic(stage/'ENGINEERING_QA_RESULT.json',qa)
result={'schema':'generic-final-master-sync-tooling-allocation-stage-v1','destination':payload['destination'],'source_inventory':receipt(stage/'SOURCE_INVENTORY.json'),'engineering_QA':receipt(stage/'ENGINEERING_QA_RESULT.json'),'engineering_QA_passed':child.returncode==0,'exact_final_master_received':False,'final_master_audit_or_manifest_executed':False,'actual_input_models_or_scores':0,'data_copies':0,'final_tar_created':False,'server77_access':False,'GO_created':False}
atomic(stage/'TOOLING_READY.json',result)
print(json.dumps({'result':result,'qa':qa}),flush=True)
sys.exit(0 if child.returncode==0 else 1)
'''

RUN = helper.GUARD + r'''
request=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
stage=phase/request['tooling'];inventory=stage/'SOURCE_INVENTORY.json'
assert hashlib.sha256(inventory.read_bytes()).hexdigest()==request['tooling_inventory_sha256']
for row in json.loads(inventory.read_text()):
 path=phase/row['path'];assert path.is_file() and not path.is_symlink()
 raw=path.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
out=phase/request['output'];assert not out.exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
python=str(repo/'.venv/bin/python')
bound=['--project-root',str(phase),'--master',request['master'],'--master-hash',request['master_hash'],'--summary',request['summary'],'--summary-sha256',request['summary_sha256']]
records=[]
def run(command,label):
 started=time.monotonic();child=subprocess.run(command,cwd=stage/'runtime',env=env,capture_output=True,text=True,timeout=1800)
 result={'label':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-started,'stdout':child.stdout,'stderr':child.stderr,'scientific_fits':0,'GO_created':False}
 records.append(result);print(json.dumps(result),flush=True)
 if out.is_dir():
  target=out/(label+'_RESULT.json');assert not target.exists();target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');target.chmod(0o444)
 if child.returncode:sys.exit(1)
run([python,'-B',str(stage/'runtime/build_final_master_sync.py'),*bound,'--tooling-inventory',str(inventory.relative_to(phase)),'--tooling-inventory-sha256',request['tooling_inventory_sha256'],'--output',request['output']],'FINAL_MANIFEST_BUILD')
manifest=out/'MANIFEST.json';manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
run([python,'-B',str(stage/'runtime/audit_final_master_source_loading.py'),*bound,'--manifest',str(manifest.relative_to(phase)),'--manifest-sha256',manifest_sha,'--output',request['output']+'/SOURCE_LOADING_AUDIT.json'],'ACTUAL_FINAL_LAUNCHER_SOURCE_AUDIT')
summary=json.loads((out/'SUMMARY.json').read_text());audit=json.loads((out/'SOURCE_LOADING_AUDIT.json').read_text())
proof={'schema':'exact-disabled-final-master-sync-and-source-audit-v1','plan_hash':request['master_hash'],'manifest_sha256':manifest_sha,'statistics':summary['statistics'],'loaded_studies':audit['loaded_studies'],'registered_records':audit['registered_records'],'model_groups':audit['expected_counts']['model_groups'],'typed_counts_recomputed_from_frozen_controller':True,'all_phase_reads_within_manifest_union':audit['all_phase_reads_within_manifest_union'],'optimizer_total_is_registered_not_executed':True,'data_copies':0,'final_tar_created':False,'scientific_fits':0,'GO_created':False,'server77_access':False}
target=out/'SYNC_SOURCE_AUDIT_PROOF.json';assert not target.exists();target.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n');target.chmod(0o444)
print(json.dumps({'proof':proof}),flush=True)
'''


def relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or not value:
        raise ValueError("Exact project-relative path required")
    return value


def invoke(code, arguments, label, timeout):
    transport = OUT / "transport"
    transport.mkdir(exist_ok=True)
    if (transport / (label + ".json")).exists():
        raise FileExistsError("Preserve prior attempt; use a fresh label")
    result = helper.invoke(code, arguments, transport, label, timeout)
    print(json.dumps(result, indent=2))
    if result["exit_code"] != 0:
        raise RuntimeError("Guarded attempt failed; complete receipt retained: " + label)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage", "run"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--destination", default=OUT.name)
    parser.add_argument("--qa-controller", help="Exact frozen controller for synthetic metadata fixture")
    parser.add_argument("--tooling", default=OUT.name)
    parser.add_argument("--tooling-inventory-sha256")
    parser.add_argument("--master")
    parser.add_argument("--master-hash")
    parser.add_argument("--summary")
    parser.add_argument("--summary-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.label):
        raise ValueError("Single immutable attempt label required")
    if args.mode == "stage":
        if not args.qa_controller:
            parser.error("stage requires --qa-controller")
        files = []
        for folder in ("runtime", "tests", "preparation"):
            for path in sorted((OUT / folder).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                raw = path.read_bytes()
                files.append({"path": str(path.relative_to(OUT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
        payload = {"destination": relative(args.destination), "qa_controller": relative(args.qa_controller), "files": files}
        encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
        for offset in range(0, len(encoded), 16000):
            invoke(helper.TRANSFER, [args.label, str(offset), encoded[offset:offset+16000]], args.label + "_TRANSFER_" + str(offset), 60)
        invoke(STAGE, [args.label, hashlib.sha256(encoded.encode()).hexdigest()], args.label + "_STAGE", 180)
    else:
        for name in ("tooling_inventory_sha256", "master", "master_hash", "summary", "summary_sha256", "output"):
            if not getattr(args, name):
                parser.error("run requires --" + name.replace("_", "-"))
        value = {name: getattr(args, name) for name in ("tooling", "tooling_inventory_sha256", "master", "master_hash", "summary", "summary_sha256", "output")}
        for name in ("tooling", "master", "summary", "output"):
            relative(value[name])
        for name in ("tooling_inventory_sha256", "master_hash", "summary_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", value[name]):
                raise ValueError("Exact SHA256 required")
        encoded = base64.b64encode(zlib.compress(json.dumps(value).encode())).decode()
        invoke(RUN, [encoded], args.label + "_RUN", 4000)


if __name__ == "__main__":
    main()
