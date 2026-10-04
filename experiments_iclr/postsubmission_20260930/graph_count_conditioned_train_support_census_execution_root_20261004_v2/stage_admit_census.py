"""Stage reviewed TRAIN-only integer geometry and release one owned census."""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'graph_count_conditioned_train_support_census_preparation_20261004_v3'
REVIEW = PHASE / 'graph_count_conditioned_train_support_census_independent_source_review_20261004_v3'
PIN = 'e6c7f6bc64e9d3454f2a7881acb16f493ec00b2126de0397b62c7b64a8b064e2'
REVIEW_PIN = 'c4484d36b6bc0f519bf4a435c31ac6698ebb350ff971c34a4e702af43c153e0b'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
    with (HERE/name).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def transport(identity,code):
    file=PHASE/'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
    spec=importlib.util.spec_from_file_location('census_owned_project_transport',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.HERE=HERE
    return module.run('count_census_v2_'+identity+'_20261004',code)


def context():
    return 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\nrepo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(REMOTE))+')\nassert Path.cwd()==repo and os.uname().nodename=="peptide"\n'


def packet(folder,pin):
    assert sha(folder/'MANIFEST.json')==pin
    files=[]
    for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
        p=folder/row['path'];assert p.resolve().is_relative_to(folder) and not p.is_symlink()
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];files.append(p)
    return files+[folder/'MANIFEST.json',folder/'SEAL.json']


def stage():
    reviewed=json.loads((REVIEW/'REVIEW.json').read_text())
    assert sha(REVIEW/'REVIEW.json')==REVIEW_PIN and reviewed['status']=='PASS'
    assert reviewed['candidate_manifest_sha256']==PIN and reviewed['execution_authorized'] is False
    files=packet(SOURCE,PIN)+packet(REVIEW,'7871a403dfdc14bc684ad040be005458ac49a98d8c6eede815375db972325b0a')
    for row in json.loads((SOURCE/'INPUTS.json').read_text())['inputs']+json.loads((SOURCE/'PLAN.json').read_text())['source_pins']:
        p=PHASE/row['path'];assert p.resolve().is_relative_to(PHASE) and not p.is_symlink()
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];files.append(p)
    rows=[]
    for p in dict.fromkeys(files):
        assert p.stat().st_size<2_000_000 and p.suffix in ('.py','.json','.md','.diff','.patch')
        raw=p.read_bytes();rows.append(dict(path=str(p.relative_to(PHASE)),bytes=len(raw),sha256=sha(p),data=base64.b64encode(raw).decode()))
    save('SOURCE_STAGE_INVENTORY.json',[{k:v for k,v in row.items() if k!='data'} for row in rows])
    packed=zlib.compress(json.dumps(rows).encode(),9);encoded=base64.b64encode(packed).decode();chunks=[encoded[n:n+40000] for n in range(0,len(encoded),40000)]
    for number,chunk in enumerate(chunks,1):
        code=context()+'s=root/"source_chunks";s.mkdir(parents=True,exist_ok=True)\nraw='+repr(chunk)+'.encode()\nwith (s/'+repr('part%03d.b64'%number)+').open("xb") as f:f.write(raw)\nprint(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))))\n'
        v=transport('chunk%03d'%number,code);assert v['sha256']==hashlib.sha256(chunk.encode()).hexdigest()
    code=context()+'import zlib\npacked=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,'+str(len(chunks)+1)+')),validate=True)\nassert hashlib.sha256(packed).hexdigest()=='+repr(hashlib.sha256(packed).hexdigest())+'\nrows=json.loads(zlib.decompress(packed))\n'
    code+='for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if p.exists():assert not p.is_symlink() and p.read_bytes()==raw\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as f:f.write(raw)\nprint(json.dumps(dict(status="EXACT_REVIEWED_TRAIN_CENSUS_SOURCE_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),numerical_execution=False)))\n'
    save('SOURCE_STAGE_RECEIPT.json',transport('join',code));print('Exact reviewed census source staged; no array load or execution.')


def admit():
    assert (HERE/'SOURCE_STAGE_RECEIPT.json').exists() and sha(SOURCE/'MANIFEST.json')==PIN and sha(REVIEW/'REVIEW.json')==REVIEW_PIN
    plan=json.loads((SOURCE/'PLAN.json').read_text());assert plan['execution_directory']==HERE.name
    authority=json.loads((PHASE/plan['runtime_authority']).read_text())
    release=json.loads((SOURCE/'ROOT_RELEASE_TEMPLATE.json').read_text())
    release.update(status='APPROVED',authorized_stage='TRAIN_support_census',root_authorization_reference=str(REMOTE/'ROOT_ADMISSION.json'),source_manifest_sha256=PIN,plan_sha256=sha(SOURCE/'PLAN.json'),independent_source_review_path=str(REMOTE_PHASE/REVIEW.name/'REVIEW.json'),independent_source_review_sha256=REVIEW_PIN)
    save('ROOT_RELEASE.json',release)
    save('ROOT_ADMISSION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='APPROVED_ONE_FULL_TRAIN_SUPPORT_CENSUS',source_manifest_sha256=PIN,independent_source_review_sha256=REVIEW_PIN,seeds=plan['seeds'],full_batches_per_seed=17,populations=['positive','native_sampled_negative'],query_rows_per_population_per_batch=65536,model_or_feature_reads=False,VALID_TEST_reads=False,optimizer_updates=0,source_v1_v2_failures_preserved=True,cap_scope='Child observation/cleanup window; parent collection/hash/write and parent RSS are excluded. Sampled RSS, not an instantaneous OS bound.',scientific_fit_admitted=False,automatic_retry=False))
    rows=[]
    for name in ['ROOT_RELEASE.json','ROOT_ADMISSION.json']:
        f=HERE/name;rows.append(dict(path=str(REMOTE/name),sha256=sha(f),data=base64.b64encode(f.read_bytes()).decode()))
    code=context()+'runtime=phase/'+repr(plan['runtime_authority'])+'\na=json.loads(runtime.read_text());py=Path(a["interpreter_path"]);assert hashlib.sha256(py.read_bytes()).hexdigest()==a["interpreter_sha256"]\n'
    code+='q=subprocess.run(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)\nselected=[r.split(",") for r in q.stdout.splitlines() if r.split(",")[0].strip()=='+repr(plan['GPU_UUID'])+'];assert len(selected)==1\ngpu=[v.strip() for v in selected[0]];assert gpu[1]=="NVIDIA A100 80GB PCIe" and int(gpu[2])==81920 and int(gpu[3])>=73728\n'
    code+='available=next(int(r.split()[1])*1024 for r in Path("/proc/meminfo").read_text().splitlines() if r.startswith("MemAvailable:"));assert available>=24*1024**3\nassert not (root/"run01").exists() and not (root/"supervision/run01").exists() and not (root/"CENSUS_ATTEMPT_SPENT.json").exists()\nrows='+repr(rows)+'\nfor row in rows:\n p=Path(row["path"]);assert p.resolve().is_relative_to(root);b=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(b).hexdigest()==row["sha256"]\n with p.open("xb") as f:f.write(b)\nprint(json.dumps(dict(status="EXACT_CENSUS_RUNTIME_AND_RESOURCES_CHECKED",GPU=gpu,MemAvailable_bytes=available,numerical_execution=False)))\n'
    save('RESOURCE_AND_RELEASE_RECEIPT.json',transport('release',code))
    env=dict(CUDA_VISIBLE_DEVICES=plan['GPU_UUID'],PYTHONPATH='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/buddy_extra_v1/site',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',GNNM_SSH_DESTINATION='shmelev@192.168.18.77')
    remote_source=REMOTE_PHASE/SOURCE.name
    gate_code='import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(remote_source))+');from census import gate;gate(Path('+repr(str(REMOTE/'ROOT_RELEASE.json'))+'),'+repr(sha(HERE/'ROOT_RELEASE.json'))+');print("EXACT_STDLIB_CENSUS_GATE_PASS")'
    command=[authority['interpreter_path'],'-B','-c',gate_code]
    code=context()+'env='+repr(env)+'\nr=subprocess.run('+repr(command)+',cwd=repo,env=dict(os.environ,**env),capture_output=True,text=True,timeout=30)\nprint(json.dumps(dict(status="PASS" if r.returncode==0 else "FAILED",exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,numerical_execution=False)))\n'
    v=transport('gate',code);save('PRENUMERICAL_GATE.json',v);assert v['status']=='PASS',v
    command=[authority['interpreter_path'],'-B',str(remote_source/'supervise.py'),'--release',str(REMOTE/'ROOT_RELEASE.json'),'--release-sha256',sha(HERE/'ROOT_RELEASE.json')]
    code=context()+'env='+repr(env)+';command='+repr(command)+'\nassert not (root/"run01").exists() and not (root/"supervision/run01").exists() and not (root/"CENSUS_ATTEMPT_SPENT.json").exists()\nwith (root/"DETACHED_STDOUT.txt").open("xb") as out,(root/"DETACHED_STDERR.txt").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**env),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\nraw=(Path("/proc")/str(child.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();v=dict(UTC=datetime.now(timezone.utc).isoformat(),status="OWNED_FULL_TRAIN_CENSUS_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(f[19]),command=command,environment=env,model_fits=0,VALID_TEST_reads=False)\nwith (root/"DETACHED_LAUNCH.json").open("x") as s:json.dump(v,s,indent=2);s.write("\\n")\nprint(json.dumps(v))\n'
    v=transport('launch',code);save('DETACHED_LAUNCH.json',v);print(json.dumps({k:v[k] for k in ['UTC','status','supervisor_PID','supervisor_start_ticks']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','admit']);args=parser.parse_args()
    stage() if args.mode=='stage' else admit()
