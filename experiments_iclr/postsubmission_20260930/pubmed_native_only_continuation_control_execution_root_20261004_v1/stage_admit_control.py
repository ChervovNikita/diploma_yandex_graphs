"""Stage exact reviewed control source, then launch one fresh native diagnostic."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json
import zlib

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SOURCE=PHASE/'pubmed_native_only_continuation_control_source_20261004_v2'
PIN='456a22a28420268e9ae47fd49f6ac322e36856b40a313340da4b31bde954ce7e'
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE=REPO/'experiments_iclr/postsubmission_20260930'
REMOTE=REMOTE_PHASE/HERE.name


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
 with (HERE/name).open('x') as stream:
  json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def transport(identity,code):
 path=PHASE/'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
 spec=importlib.util.spec_from_file_location('native_control_project_transport',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.HERE=HERE
 return module.run('pubmed_native_control_v1_'+identity+'_20261004',code)


def context():
 return 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\nrepo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(REMOTE))+')\nassert Path.cwd()==repo and os.uname().nodename=="peptide"\n'


def packet(folder,pin):
 assert folder.resolve().is_relative_to(PHASE) and sha(folder/'MANIFEST.json')==pin
 files=[]
 for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
  p=folder/row['path'];assert p.resolve().is_relative_to(folder) and not p.is_symlink()
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];files.append(p)
 return files+[folder/'MANIFEST.json',folder/'SEAL.json']


def reviewed_folder(relative):
 review=PHASE/relative;assert review.resolve().is_relative_to(PHASE) and not review.is_symlink()
 d=json.loads((review/'REVIEW.json').read_text())
 assert d['candidate_manifest_sha256']==PIN and d['status']=='PASS' and d['execution_authorized'] is False
 assert not d.get('blocking_findings')
 return review


def stage(relative):
 review=reviewed_folder(relative)
 files=packet(SOURCE,PIN)+packet(review,sha(review/'MANIFEST.json'))
 for row in json.loads((SOURCE/'INPUTS.json').read_text())['files']:
  p=PHASE/row['path'];assert p.resolve().is_relative_to(PHASE) and not p.is_symlink()
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];files.append(p)
 rows=[]
 for p in dict.fromkeys(files):
  assert p.stat().st_size<2_000_000 and p.suffix in ('.py','.json','.md','.diff','.patch')
  raw=p.read_bytes();rows.append(dict(path=str(p.relative_to(PHASE)),bytes=len(raw),sha256=sha(p),data=base64.b64encode(raw).decode()))
 save('SOURCE_STAGE_INVENTORY.json',[{k:v for k,v in row.items() if k!='data'} for row in rows])
 packed=zlib.compress(json.dumps(rows).encode(),9);encoded=base64.b64encode(packed).decode()
 chunks=[encoded[n:n+40000] for n in range(0,len(encoded),40000)]
 for number,chunk in enumerate(chunks,1):
  code=context()+'s=root/"source_chunks";s.mkdir(parents=True,exist_ok=True)\nraw='+repr(chunk)+'.encode()\nwith (s/'+repr('part%03d.b64'%number)+').open("xb") as f:f.write(raw)\nprint(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))))\n'
  v=transport('chunk%03d'%number,code);assert v['sha256']==hashlib.sha256(chunk.encode()).hexdigest()
 code=context()+'import zlib\npacked=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,'+str(len(chunks)+1)+')),validate=True)\nassert hashlib.sha256(packed).hexdigest()=='+repr(hashlib.sha256(packed).hexdigest())+'\nrows=json.loads(zlib.decompress(packed))\n'
 code+='for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if p.exists():assert not p.is_symlink() and p.read_bytes()==raw\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as f:f.write(raw)\n'
 binding=json.loads((SOURCE/'SOURCE_BINDING.json').read_text())
 code+='prerequisites={}\nfor name,row in '+repr(binding['prerequisite_receipts'])+'.items():\n p=phase/row["path"];b=p.read_bytes();assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"];d=json.loads(b);assert d["status"]==row["required_status"];prerequisites[name]=dict(path=row["path"],sha256=row["sha256"],status=d["status"],verified_by_root=True)\n'
 code+='print(json.dumps(dict(status="EXACT_REVIEWED_NATIVE_CONTROL_SOURCE_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),prerequisites=prerequisites,numerical_execution=False)))\n'
 save('SOURCE_STAGE_RECEIPT.json',transport('join',code))
 save('REVIEW_BINDING.json',dict(path=str((review/'REVIEW.json').relative_to(PHASE)),sha256=sha(review/'REVIEW.json'),candidate_manifest_sha256=PIN))
 print('Exact independently reviewed native control staged; no numerical execution.')


def admit():
 staged=json.loads((HERE/'SOURCE_STAGE_RECEIPT.json').read_text())
 rb=json.loads((HERE/'REVIEW_BINDING.json').read_text());review=reviewed_folder(str(Path(rb['path']).parent))
 assert sha(review/'REVIEW.json')==rb['sha256'] and sha(SOURCE/'MANIFEST.json')==PIN
 plan=json.loads((SOURCE/'PLAN.json').read_text());binding=json.loads((SOURCE/'SOURCE_BINDING.json').read_text())
 assert plan['execution_directory']==HERE.name
 stage='native_continuation_control';release=json.loads((SOURCE/'ROOT_RELEASE_TEMPLATE.json').read_text())
 prerequisites={}
 for name,row in binding['prerequisite_receipts'].items():
  observed=staged['prerequisites'][name]
  assert observed['path']==row['path'] and observed['sha256']==row['sha256'] and observed['status']==row['required_status'] and observed['verified_by_root'] is True
  prerequisites[name]=dict(path=row['path'],sha256=row['sha256'],verified_by_root=True)
 release.update(status='APPROVED',authorized_stages=[stage],root_source_review_approved=True,
  source_manifest_sha256=PIN,plan_sha256=sha(SOURCE/'PLAN.json'),caps=plan['caps'],
  invocation=plan['stages'][stage]['invocation'],input_authority=plan['input_authority'],
  existing_qualification_reused=True,prerequisites=prerequisites,
  independent_source_review_path=str(REMOTE_PHASE/rb['path']),independent_source_review_sha256=rb['sha256'],
  root_authorization_reference=str(REMOTE/'ROOT_ADMISSION.json'))
 save('ROOT_RELEASE_native_continuation_control.json',release)
 save('ROOT_ADMISSION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='APPROVED_ONE_FRESH_NATIVE_REPEAT_CONTROL',
  source_manifest_sha256=PIN,independent_source_review_sha256=rb['sha256'],maximum_updates=252,
  scope='Five fresh native warm epochs plus two independent complete epoch6 continuations. TRAIN/features only; zero VALID/TEST/score/state-file reads.',
  old_shared4_fixed_rule_failure_preserved=True,v1_source_and_superseded_PASS_preserved=True,
  comparator_profile_caps_unchanged=True,scientific_fit_admitted=False,
  inference='Exact prestate/stream/copy predicates are prerequisites to interpreting numeric discrepancy. Fresh warm-state history differs from failed shared4; no automatic causal attribution or tolerance/gate amendment.',
  cap_scope='Owned child observation/cleanup; parent collection/hash/write and parent RSS excluded. Sampled RSS, no process escape sandbox.',
  automatic_retry=False))
 rows=[]
 for name in ['ROOT_RELEASE_native_continuation_control.json','ROOT_ADMISSION.json']:
  f=HERE/name;rows.append(dict(path=str(REMOTE/name),sha256=sha(f),data=base64.b64encode(f.read_bytes()).decode()))
 profile=plan['execution_profile'];environment=dict(profile['environment'],GNNM_SSH_DESTINATION='shmelev@192.168.18.77',PYTHONDONTWRITEBYTECODE='1')
 code=context()+'py=Path('+repr(profile['interpreter_path'])+');assert hashlib.sha256(py.read_bytes()).hexdigest()=='+repr(profile['interpreter_sha256'])+'\n'
 code+='r=subprocess.run(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)\ngpus=[s.split(",") for s in r.stdout.splitlines() if s.split(",")[0].strip()=='+repr(environment['CUDA_VISIBLE_DEVICES'])+'];assert len(gpus)==1;gpu=[s.strip() for s in gpus[0]];assert gpu[1]=="NVIDIA A100 80GB PCIe" and int(gpu[2])==81920 and int(gpu[3])>=16384\n'
 code+='available=next(int(s.split()[1])*1024 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"));assert available>=8*1024**3\nassert not (root/"native_continuation_control/run01").exists() and not (root/"supervision/native_continuation_control/run01").exists() and not (root/"NATIVE_CONTROL_ATTEMPT_SPENT.json").exists()\n'
 code+='rows='+repr(rows)+'\nfor row in rows:\n p=Path(row["path"]);assert p.resolve().is_relative_to(root);b=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(b).hexdigest()==row["sha256"]\n with p.open("xb") as f:f.write(b)\nprint(json.dumps(dict(status="EXACT_NATIVE_CONTROL_RUNTIME_RESOURCES_AND_RELEASE_CHECKED",GPU=gpu,MemAvailable_bytes=available,numerical_execution=False)))\n'
 save('RESOURCE_AND_RELEASE_RECEIPT.json',transport('release',code))
 remote_source=REMOTE_PHASE/SOURCE.name
 gate='import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(remote_source))+');from common import gate;gate(Path('+repr(str(REMOTE/'ROOT_RELEASE_native_continuation_control.json'))+'),'+repr(sha(HERE/'ROOT_RELEASE_native_continuation_control.json'))+','+repr(stage)+');print("EXACT_NATIVE_CONTROL_STDLIB_GATE_PASS")'
 command=[profile['interpreter_path'],'-B','-c',gate]
 code=context()+'environment='+repr(environment)+'\nr=subprocess.run('+repr(command)+',cwd=repo,env=dict(os.environ,**environment),capture_output=True,text=True,timeout=45)\nprint(json.dumps(dict(status="PASS" if r.returncode==0 else "FAILED",exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,numerical_execution=False)))\n'
 v=transport('gate',code);save('PRENUMERICAL_GATE.json',v);assert v['status']=='PASS',v
 command=[profile['interpreter_path'],'-B',str(remote_source/'supervise.py'),'--root-release',str(REMOTE/'ROOT_RELEASE_native_continuation_control.json'),'--release-sha256',sha(HERE/'ROOT_RELEASE_native_continuation_control.json')]
 code=context()+'environment='+repr(environment)+';command='+repr(command)+'\nassert not (root/"native_continuation_control/run01").exists() and not (root/"supervision/native_continuation_control/run01").exists() and not (root/"NATIVE_CONTROL_ATTEMPT_SPENT.json").exists()\nwith (root/"DETACHED_STDOUT.txt").open("xb") as out,(root/"DETACHED_STDERR.txt").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\nraw=(Path("/proc")/str(child.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();v=dict(UTC=datetime.now(timezone.utc).isoformat(),status="OWNED_FRESH_NATIVE_REPEAT_CONTROL_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(f[19]),command=command,environment=environment,engineering_updates_planned=252,scientific_fits=0,VALID_TEST_access=False)\nwith (root/"DETACHED_LAUNCH.json").open("x") as s:json.dump(v,s,indent=2);s.write("\\n")\nprint(json.dumps(v))\n'
 v=transport('launch',code);save('DETACHED_LAUNCH.json',v)
 print(json.dumps({k:v[k] for k in ['UTC','status','supervisor_PID','supervisor_start_ticks']}))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','admit']);parser.add_argument('--review')
 args=parser.parse_args()
 if args.mode=='stage':
  assert args.review;stage(args.review)
 else:admit()
