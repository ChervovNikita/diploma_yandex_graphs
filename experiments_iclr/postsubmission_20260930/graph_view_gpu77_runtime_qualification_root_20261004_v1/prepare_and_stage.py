from remote_transport import run,HERE,REPO,PHASE,PYTHON
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,zlib,difflib
P=HERE.parent
ORIGINAL=P/'graph_view_full_shape_qualification_root_20261004_v2'
SOURCE=P/'accuracy_first_graph_view_source_preparation_20261004_v2'
COVERAGE=P/'graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1/split0_COVERAGE.json'
UUID='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
UUIDS=[UUID,'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
REMOTE=PHASE/HERE.name
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(name,value):
 with (HERE/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
original=(ORIGINAL/'check_one.py').read_text()
successor=original
replacements={
 "UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'":"UUID = "+repr(UUID)+"\nPHYSICAL_UUIDS = "+repr(UUIDS),
 "capture_output=True, text=True, check=True).stdout.splitlines() == [UUID]":"capture_output=True, text=True, check=True).stdout.splitlines() == PHYSICAL_UUIDS",
 "    source = PHASE / ('accuracy_first_graph_view_source_preparation_20261004_v2' if args.family == 'bank'":"    assert args.family == 'bank' and args.condition == 'tied_persistent'\n    source = ROOT / 'staged_phase' / ('accuracy_first_graph_view_source_preparation_20261004_v2' if args.family == 'bank'",
 "    coverage_path = PHASE / 'graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1/split0_COVERAGE.json'":"    coverage_path = ROOT / 'staged_phase/graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1/split0_COVERAGE.json'\n    assert hashlib.sha256(coverage_path.read_bytes()).hexdigest() == "+repr(sha(COVERAGE)),
 "        component_stage_fresh_initialization=True, co_resident_with_original_Amazon_queue=True,":"        component_stage_fresh_initialization=True, co_resident_with_DDI_queue=True,",
 "        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),":"        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),\n        runtime=runtime_descriptor, source_provenance=list(rt.source_provenance),\n        input_bindings_sha256=hashlib.sha256((ROOT / 'INPUT_BINDINGS.json').read_bytes()).hexdigest(),\n        coverage_sha256=hashlib.sha256(coverage_path.read_bytes()).hexdigest(),\n        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"
}
for old,new in replacements.items():
 assert successor.count(old)==1,old
 successor=successor.replace(old,new)
needle='    torch, np = rt.torch, rt.numpy\n'
addition="""    assert Path(sys.executable).resolve() == Path(PYTHON_PATH).resolve()
    import os, torch_geometric
    assert os.environ['CUDA_VISIBLE_DEVICES'] == UUID
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    runtime_descriptor = dict(python_executable=sys.executable, python_version=sys.version,
        torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
        numpy_version=np.__version__, torch_geometric_version=torch_geometric.__version__,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], physical_gpu_uuids=PHYSICAL_UUIDS,
        CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False)
    runtime_descriptor['package_descriptors'] = [dict(name=name, path=module.__file__,
        bytes=Path(module.__file__).stat().st_size,
        sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        for name, module in [('torch',torch),('numpy',np),('torch_geometric',torch_geometric)]]
"""
assert successor.count(needle)==1
successor=successor.replace(needle,needle+addition)
successor=successor.replace("REFERENCE_PROTOCOL = '56c96715bc84eb23a9da8cb982f23a1147db18e16787df5690b7e7163b464708'", "REFERENCE_PROTOCOL = '56c96715bc84eb23a9da8cb982f23a1147db18e16787df5690b7e7163b464708'\nPYTHON_PATH = "+repr(PYTHON))
compile(successor,str(HERE/'check_one.py'),'exec')
(HERE/'check_one.py').write_text(successor)
(HERE/'HOST_PORTABILITY_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True),successor.splitlines(keepends=True),fromfile='one_gpu_v2/check_one.py',tofile='gpu77/check_one.py')))
(HERE/'INPUT_BINDINGS.json').write_bytes((ORIGINAL/'INPUT_BINDINGS.json').read_bytes())
(HERE/'DETERMINISTIC_RUNTIME_PROSPECTIVE_POLICY.json').write_bytes((ORIGINAL/'DETERMINISTIC_RUNTIME_PROSPECTIVE_POLICY.json').read_bytes())
supervisor=f'''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
REPO=Path({str(REPO)!r})
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
ROOT=PHASE/{HERE.name!r}
PYTHON={PYTHON!r}
UUID={UUID!r}
UUIDS={UUIDS!r}
assert Path.cwd()==REPO and os.uname().nodename=='peptide'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==UUIDS
env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT),CUBLAS_WORKSPACE_CONFIG=':4096:8')
def proc(pid):
 raw=(Path('/proc')/str(pid)/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 return dict(PID=pid,state=f[0],parent=int(f[1]),group=int(f[2]),session=int(f[3]),start_ticks=int(f[19]))
def save(name,value):
 with (ROOT/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\\n')
rows=[];start=time.perf_counter()
for stage in ('local','global'):
 free=int(subprocess.run(['nvidia-smi','--id='+UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip())
 if free<35000:
  rows.append(dict(stage=stage,status='NOT_STARTED_INSUFFICIENT_FREE_MEMORY',free_MiB=free));break
 command=[PYTHON,'-B',str(ROOT/'check_one.py'),'--family','bank','--condition','tied_persistent','--stage',stage]
 started=time.perf_counter()
 with (ROOT/(stage+'.stdout')).open('xb') as out,(ROOT/(stage+'.stderr')).open('xb') as err:
  child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  identity=proc(child.pid)
  save(stage+'_CHILD_STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),identity=identity,command=command,selected_GPU_UUID=UUID,free_MiB_at_start=free))
  exit_code=child.wait()
 row=dict(stage=stage,exit_code=exit_code,wall_seconds=time.perf_counter()-started,identity=identity,command=command,child_closed_and_reaped=True,stdout_sha256=hashlib.sha256((ROOT/(stage+'.stdout')).read_bytes()).hexdigest(),stderr_sha256=hashlib.sha256((ROOT/(stage+'.stderr')).read_bytes()).hexdigest())
 result=ROOT/('bank_tied_persistent_'+stage+'.json')
 if exit_code==0:
  row['result']=json.loads(result.read_text());row['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
 else:row['stderr']=(ROOT/(stage+'.stderr')).read_text()
 save(stage+'_PHYSICAL_TERMINAL.json',row);rows.append(row)
 if exit_code!=0:break
save('QUALIFICATION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),cases=rows,expected_cases=2,wall_seconds=time.perf_counter()-start,all_component_checks_complete=len(rows)==2 and all(r.get('exit_code')==0 for r in rows),route=dict(login='shmelev@192.168.18.77',repository=str(REPO),physical_UUIDs=UUIDS,selected_GPU_UUID=UUID),predictive_values_read=False,VALIDATION_or_TEST_targets_read=False,scientific_training_updates=0,checkpoint_or_logits_saved=False,eligible_as_donor=False,automatic_retry=False,job_signals_sent=False,co_resident_with_DDI_queue=True))
save('SUPERVISOR_TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),identity=proc(os.getpid()),completed=True,wall_seconds=time.perf_counter()-start,job_signals_sent=False))
'''
compile(supervisor,str(HERE/'supervisor.py'),'exec');(HERE/'supervisor.py').write_text(supervisor)
files=[]
manifest=json.loads((SOURCE/'MANIFEST.json').read_text())
assert sha(SOURCE/'MANIFEST.json')=='eb34fcb8bc406219007ee39ee9789f1de9bc56acf3cb16164aeab95b4493752f'
for row in manifest['payload']:
 f=SOURCE/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'];files.append((f,'staged_phase/'+str(f.relative_to(P))))
files.append((SOURCE/'MANIFEST.json','staged_phase/'+str((SOURCE/'MANIFEST.json').relative_to(P))))
if (SOURCE/'SEAL.json').is_file():files.append((SOURCE/'SEAL.json','staged_phase/'+str((SOURCE/'SEAL.json').relative_to(P))))
bind=json.loads((SOURCE/'SOURCE_BINDINGS.json').read_text())
for desc in [*bind['immutable_neural_files'],bind['v6_manifest'],bind['v6_seal'],bind['native_recipe_source'],bind['block_source'],*bind['context']]:
 f=P/desc['path'];assert f.stat().st_size==desc['bytes'] and sha(f)==desc['sha256'];files.append((f,'staged_phase/'+desc['path']))
files.append((COVERAGE,'staged_phase/'+str(COVERAGE.relative_to(P))))
for name in ['check_one.py','supervisor.py','INPUT_BINDINGS.json','DETERMINISTIC_RUNTIME_PROSPECTIVE_POLICY.json','HOST_PORTABILITY_DIFF.patch']:
 files.append((HERE/name,name))
rows=[]
for f,rel in files:
 raw=f.read_bytes();rows.append(dict(path=rel,bytes=len(raw),sha256=sha(f),data=base64.b64encode(raw).decode()))
assert len({r['path'] for r in rows})==len(rows)
save('STAGE_INVENTORY.json',[{k:v for k,v in r.items() if k!='data'} for r in rows])
save('HOST_PORTABILITY_BINDINGS.json',dict(UTC=datetime.now(timezone.utc).isoformat(),original_checker=dict(path=str((ORIGINAL/'check_one.py').relative_to(P)),sha256=sha(ORIGINAL/'check_one.py')),ported_checker_sha256=sha(HERE/'check_one.py'),neural_package_manifest_sha256=sha(SOURCE/'MANIFEST.json'),protocol_sha256=sha(SOURCE/'PROTOCOL.json'),coverage_sha256=sha(COVERAGE),stage_scope='Only tied_persistent local/global; three TRAIN-only engineering updates each',scientific_source_or_constants_or_losses_changed=False,predictive_data_access=False,paths_staged_inside_owned_remote_packet_only=True,physical_UUIDs=UUIDS,selected_UUID=UUID,python=PYTHON))
compressed=zlib.compress(json.dumps(rows).encode(),9);encoded=base64.b64encode(compressed).decode();chunks=[encoded[i:i+40000] for i in range(0,len(encoded),40000)]
for n,chunk in enumerate(chunks,1):
 code='from pathlib import Path\nimport json,os,hashlib\n'
 code+='repo=Path('+repr(str(REPO))+');root=Path('+repr(str(REMOTE/'stage_chunks'))+')\n'
 code+='assert Path.cwd()==repo and os.uname().nodename=="peptide";root.mkdir(parents=True,exist_ok=True)\n'
 code+='raw='+repr(chunk)+'.encode();path=root/'+repr('part%03d.b64'%n)+'\n'
 code+='with path.open("xb") as f:f.write(raw)\nprint(json.dumps({"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}))\n'
 value=run('graph_view_gpu77_stage_chunk%03d_20261004_v1'%n,code);assert value['sha256']==hashlib.sha256(chunk.encode()).hexdigest()
code='from pathlib import Path\nfrom datetime import datetime,timezone\nimport json,os,base64,zlib,hashlib\n'
code+='repo=Path('+repr(str(REPO))+');root=Path('+repr(str(REMOTE))+');parts=root/"stage_chunks"\n'
code+='assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code+='raw=base64.b64decode(b"".join((parts/("part%03d.b64"%n)).read_bytes() for n in range(1,'+str(len(chunks)+1)+')),validate=True)\n'
code+='assert hashlib.sha256(raw).hexdigest()=='+repr(hashlib.sha256(compressed).hexdigest())+'\nrows=json.loads(zlib.decompress(raw))\n'
code+='for r in rows:\n f=root/r["path"];assert f.resolve().is_relative_to(root) and not f.is_symlink();raw=base64.b64decode(r["data"],validate=True);assert len(raw)==r["bytes"] and hashlib.sha256(raw).hexdigest()==r["sha256"]\n if f.exists():assert f.read_bytes()==raw\n else:\n  f.parent.mkdir(parents=True,exist_ok=True)\n  with f.open("xb") as out:out.write(raw)\n'
code+='print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="EXACT_PAYLOADS_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),model_execution=False)))\n'
value=run('graph_view_gpu77_stage_join_20261004_v1',code);save('STAGE_RECEIPT.json',value);print(json.dumps(value))
