"""Guarded metadata staging and literal disabled loader; numerical imports denied."""
import base64,hashlib,json,zlib,textwrap
from pathlib import Path
from transport import OUT,helper,GUARD

def main():
 label='PHYSICS_OIPG_METADATA_ATTEMPT1';directory=OUT/'transport'
 if (directory/(label+'_REMOTE.json')).exists():raise FileExistsError('Preserve prior attempt')
 names=['runtime/engine.py','runtime/spec.py','runtime/run_study.py','runtime/compile_master_plan.py','runtime/tabm_stopping.py','runtime/oipg_typing.py',
  'runtime/adapters/source_endpoints.py','runtime/compile_physics_oipg_packet.py','preparation/amend_controller.py','preparation/compile_and_check.py',
  'CANONICAL_SOURCE_COMPLETION.json','CONTROLLER_COMPOSITION_METADATA.json','CONTROLLER_TYPE_AMENDMENT.json']
 files=[]
 for name in names:
  raw=(OUT/name).read_bytes();files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 encoded=base64.b64encode(zlib.compress(json.dumps({'destination':OUT.name,'files':files}).encode(),9)).decode()
 for offset in range(0,len(encoded),16000):
  result=helper.invoke(helper.TRANSFER,[label,str(offset),encoded[offset:offset+16000]],directory,label+'_TRANSFER_'+str(offset),60)
  if result['exit_code']:raise RuntimeError('Guarded source transport failed; retained')
 code=GUARD+textwrap.dedent(r'''
 encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text();assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2];payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert stage.exists()
 changes=[]
 for row in payload['files']:
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;q=stage/rel;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  if q.exists() and q.read_bytes()!=raw:
   old=q.read_bytes();saved=stage/'source_revisions'/sys.argv[1]/rel;saved.parent.mkdir(parents=True,exist_ok=True);assert not saved.exists();saved.write_bytes(old);saved.chmod(0o444);changes.append({'path':row['path'],'prior_sha256':hashlib.sha256(old).hexdigest(),'next_sha256':row['sha256']})
  q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
 (stage/'METADATA_SOURCE_REVISION.json').write_text(json.dumps(changes,indent=2,sort_keys=True)+'\n')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 command=[str(repo/'.venv/bin/python'),'-B',str(stage/'runtime/compile_physics_oipg_packet.py'),'--project-root',str(phase),'--packet',stage.name]
 start=time.monotonic();child=subprocess.run(command,cwd=stage/'runtime',env=env,capture_output=True,text=True,timeout=180)
 result={'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-start,'stdout':child.stdout,'stderr':child.stderr,'target_arrays_decoded':False,'actual_input_models':0,'scientific_fits':0,'GO':False}
 target=stage/'METADATA_COMPILATION_ATTEMPT1.json';assert not target.exists();target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 if child.returncode:print(json.dumps({'compile':result}));sys.exit(0)
 audit_script=r"""
 import sys,json,importlib.abc
 from pathlib import Path
 stage=Path(sys.argv[1]);phase=stage.parent;sys.path.insert(0,str(stage/'runtime'))
 class DenyNumerical(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in {'torch','numpy','scipy','torch_geometric','torch_sparse','dgl','ogb','pickle'}:raise RuntimeError('Numerical/deserializer import denied: '+fullname)
 sys.meta_path.insert(0,DenyNumerical())
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and str(args[0]).endswith(('.pt','.npz','.npy')):raise RuntimeError('Array reads denied during exact loader audit')
  if event in {'subprocess.Popen','os.system','socket.connect'}:raise RuntimeError('Process/network denied during exact loader audit')
 sys.addaudithook(audit)
 from launcher import load_plan
 from compile_master_plan import load_packet,typed_counts
 plan,digest,studies,runs,placement,deps=load_plan(phase,stage/'matrix/PLAN.json')
 entry=json.loads((stage/'MASTER_INPUT_ENTRIES.json').read_text())[0];packet=load_packet(phase,entry);counts=typed_counts([s['spec'] for s in studies.values()]);assert counts==packet['counts']
 assert len(studies)==2 and len(runs)==1830 and not plan['scientific_launch_enabled'] and plan['human_GO_required']
 assert not any(n in sys.modules for n in ('torch','numpy','dgl','ogb'))
 print(json.dumps({'schema':'Physics-OIPG-exact-frozen-disabled-loader-audit-v1','verified':True,'plan_hash':digest,'entry':entry,'counts':counts,
  'source_files_bound_by_exact_launcher':True,'target_arrays_decoded':False,'numerical_or_provider_imports':False,'array_data_reads':False,'models_constructed':0,'optimizer_updates':0,'scores':0,'scientific_fits':0,'GO':False}))
 """
 start=time.monotonic();child=subprocess.run([str(repo/'.venv/bin/python'),'-B','-c',audit_script,str(stage)],cwd=stage/'runtime',env=env,capture_output=True,text=True,timeout=180)
 audit_result={'exit_code':child.returncode,'stdout':child.stdout,'stderr':child.stderr,'wall_seconds':time.monotonic()-start,'scientific_fits':0,'GO':False}
 target=stage/'FROZEN_LOADER_ATTEMPT1.json';assert not target.exists();target.write_text(json.dumps(audit_result,indent=2,sort_keys=True)+'\n')
 if child.returncode:print(json.dumps({'compile':result,'loader':audit_result}));sys.exit(0)
 audit=json.loads(child.stdout);(stage/'FINAL_TYPED_INGESTION.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
 artifacts=[]
 for p in list((stage/'matrix').rglob('*.json'))+list((stage/'matrix').rglob('*.gz'))+[stage/'MASTER_INPUT_ENTRIES.json',stage/'COMPILATION_METADATA.json',stage/'FINAL_TYPED_INGESTION.json',stage/'METADATA_COMPILATION_ATTEMPT1.json',stage/'FROZEN_LOADER_ATTEMPT1.json',stage/'METADATA_SOURCE_REVISION.json']:
  raw=p.read_bytes();artifacts.append({'path':str(p.relative_to(stage)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 print(json.dumps({'compile':result,'loader':audit_result,'audit':audit,'artifacts':artifacts,'science':False,'GO':False}))
 ''')
 r=helper.invoke(code,[label,hashlib.sha256(encoded.encode()).hexdigest()],directory,label+'_REMOTE',420)
 if r['exit_code']:raise RuntimeError('Guarded metadata attempt failed; retained')
 value=json.loads(r['stdout'].strip())
 for row in value.pop('artifacts',[]):
  raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];p=OUT/row['path'];p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(raw)
 (OUT/'COMPILER_INGESTION_RESULT.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'compile_exit':value['compile']['exit_code'],'compile_stderr':value['compile']['stderr'],'loader':value.get('loader'),'verified':value.get('audit',{}).get('verified'),'GO':False}))
if __name__=='__main__':main()
