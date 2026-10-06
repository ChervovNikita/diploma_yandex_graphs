import base64,hashlib,json,pathlib,shlex,subprocess,zlib
from datetime import datetime,timezone
PHASE=pathlib.Path(__file__).resolve().parents[1]
HERE=pathlib.Path(__file__).resolve().parent
PREFIX="experiments_iclr/postsubmission_20260930/"
folders=["amazon_saved_utility_B_target_sign_reader_release_root_20261006_v1", "amazon_target_sign_cpu_reader_independent_source_review_20261006_v1", "amazon_target_sign_cpu_reader_custody_delta_independent_review_20261006_v2"]
files=[]
for folder in folders:
    for f in sorted((PHASE/folder).iterdir()):
        if f.is_file():
            b=f.read_bytes();files.append(dict(path=str(f.relative_to(PHASE)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
f=HERE/"ROOT_RELEASE.json";b=f.read_bytes();files.append(dict(path=str(f.relative_to(PHASE)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
remote=r'''
import base64,hashlib,json,os,pathlib,socket,subprocess,sys,time,zlib
repo=pathlib.Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
phase=repo/"experiments_iclr/postsubmission_20260930"
assert socket.gethostname()=="anogena-2-0" and repo.resolve()==repo and phase.resolve()==phase
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()==["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
records=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
for row in records:
    rel=pathlib.PurePosixPath(row["path"]);assert not rel.is_absolute() and ".." not in rel.parts
    f=phase.joinpath(*rel.parts);assert f.resolve().is_relative_to(phase) and not f.is_symlink()
    data=base64.b64decode(row["data"]);assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["sha256"]
    if f.exists():assert f.read_bytes()==data and f.stat().st_mode&0o222==0
    else:f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data);f.chmod(0o444)
execution=phase/"amazon_saved_utility_target_sign_cpu_execution_root_20261006_v1"
sentinel=execution/"REMOTE_LAUNCH.json";assert not sentinel.exists(),"Once-only CPU diagnostic already attempted"
release=execution/"ROOT_RELEASE.json";program=phase/"amazon_saved_utility_B_target_sign_reader_release_root_20261006_v1/diagnose_target_sign.py"
python=phase/"native_ncn_runtime_20261005_v1/.venv/bin/python"
argv=[str(python),"-B",str(program),"--execute-authorized","--release",str(release),"--release-sha256",hashlib.sha256(release.read_bytes()).hexdigest(),"--output",str(execution/"diagnostic")]
with sentinel.open("x") as f:json.dump(dict(argv=argv,whole_child_timeout_seconds=120,fit=False,GPU_execution=False,retry=False),f,indent=2)
sentinel.chmod(0o444)
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="";env["PYTHONPATH"]=str(repo/".venv/lib/python3.11/site-packages")
start=time.monotonic();result={"retry":False,"fit":False,"GPU_execution":False}
try:
    r=subprocess.run(argv,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
    result.update(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
except subprocess.TimeoutExpired as e:
    result.update(exit_code=None,timed_out=True,stdout=(e.stdout or b"").decode() if isinstance(e.stdout,bytes) else e.stdout,stderr=(e.stderr or b"").decode() if isinstance(e.stderr,bytes) else e.stderr)
result["whole_child_observed_seconds"]=time.monotonic()-start
with (execution/"TERMINAL.json").open("x") as f:json.dump(result,f,indent=2)
(execution/"TERMINAL.json").chmod(0o444)
output=execution/"diagnostic/TARGET_SIGN_DIAGNOSTIC.json"
if result.get("exit_code")==0:
    assert output.is_file() and output.stat().st_mode&0o222==0
    data=output.read_bytes();result["aggregate"]={"path":str(output.relative_to(phase)),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"data_base64":base64.b64encode(data).decode()}
print(json.dumps(result))
'''
ssh=["ssh","-p","2222","-i","/Users/alex/.ssh/mlspace__private_key_anogena.txt","-o","IdentitiesOnly=yes","-o","BatchMode=yes","-o","UpdateHostKeys=no","-o","StrictHostKeyChecking=yes","anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"]
receipt=HERE/"EXECUTE_RECEIPT.json";assert not receipt.exists()
start=datetime.now(timezone.utc).isoformat()
r=subprocess.run([*ssh,shlex.join(["/usr/bin/python3","-I","-S","-B","-c",remote])],input=base64.b64encode(zlib.compress(json.dumps(files).encode(),9)).decode(),capture_output=True,text=True,timeout=160)
value=dict(start_UTC=start,terminal_UTC=datetime.now(timezone.utc).isoformat(),ssh_exit_code=r.returncode,stderr=r.stderr,automatic_retry=False)
if r.returncode==0:
    result=json.loads(r.stdout);aggregate=result.get("aggregate")
    if aggregate:
        data=base64.b64decode(aggregate.pop("data_base64"));assert len(data)==aggregate["bytes"] and hashlib.sha256(data).hexdigest()==aggregate["sha256"]
        target=HERE/"TARGET_SIGN_DIAGNOSTIC.json";assert not target.exists();target.write_bytes(data);target.chmod(0o444)
    value["result"]=result
else:value["stdout"]=r.stdout
receipt.write_text(json.dumps(value,indent=2)+"\n");receipt.chmod(0o444)
print(json.dumps(dict(ssh_exit_code=r.returncode,result_exit_code=value.get("result",{}).get("exit_code"),aggregate=value.get("result",{}).get("aggregate"),failure_stderr=value.get("result",{}).get("stderr") if value.get("result",{}).get("exit_code")!=0 else None)))
