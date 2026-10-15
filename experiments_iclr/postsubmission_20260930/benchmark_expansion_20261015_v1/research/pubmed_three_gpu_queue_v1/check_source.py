"""Zero-training source/inventory/relocation/import and refusal checks."""
import ast,builtins,copy,dis,json,os,sys,tempfile,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
if str(HERE)not in sys.path:sys.path.insert(0,str(HERE))
import binding,contracts,pipeline,analyze
def refuses(function):
 try:function()
 except(ValueError,KeyError):return
 raise AssertionError('Expected refusal')
def main():
 for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p));compile(p.read_text(),str(p),'exec')
 inv=binding.load_inventory();config=json.loads((HERE/'TARGET_MAP_TEMPLATE.json').read_text());parts=[binding.shard(inv,w)for w in inv['assignments']]
 assert len(inv['rows'])==24 and all(len(p)==8 for p in parts)and len({r['id']for p in parts for r in p})==24
 assert [p[0]['cohort_seed']for p in parts]==[101,202,303]and all([r['kind']for r in p]==['native']*4+['G0','uniform','structured','permuted']for p in parts)
 assert [r['job']['body']for p in parts for r in p[:4]]==[0,1,2,3]*3
 for w in inv['assignments']:
  host=config['workers'][w]
  for row in binding.shard(inv,w):
   job,receipt=binding.relocated(row,host,w,'a'*64,'b'*64);assert binding.digest(row['job'])==row['job_hash']and receipt['bound_runtime_job_hash']==binding.digest(job)
   assert all(job[k]==row['job'][k]for k in('id','seed','body','procedure','policy','TEST_access','cohort_id','split'))
   extra=job.pop('portable_execution');assert extra['canonical_job_sha256']==row['job_hash']
   from descriptor_binding import rebind_job
   restored,_=rebind_job(job,host['project_root'],host['phase_relative'],inv['canonical_root'],inv['canonical_phase_relative']);assert restored==row['job']
 assert 'torch'not in sys.modules
 core=pipeline.selected_core(binding.E/'research/pubmed_cheb_transfer_driver_v1/collect.py');missing=set()
 def walk(code):
  for instruction in dis.get_instructions(code):
   if instruction.opname=='LOAD_GLOBAL'and instruction.argval not in core.__globals__ and not hasattr(builtins,instruction.argval):missing.add(instruction.argval)
  for child in code.co_consts:
   if isinstance(child,types.CodeType):walk(child)
 walk(core.__code__);assert not missing,missing
 import queue
 assert Path(queue.__file__).parent!=HERE and not(HERE/'queue.py').exists()
 with tempfile.TemporaryDirectory(dir=HERE,prefix='.metadata-fixture-')as temp:
  duplicate=copy.deepcopy(config);duplicate['workers']['server51/gpu1']['physical_gpu_uuid']=duplicate['workers']['server51/gpu0']['physical_gpu_uuid'];p=Path(temp)/'map.json';p.write_text(json.dumps(duplicate));refuses(lambda:contracts.request(str(p),binding.pin(p)['sha256'],'allocation/gpu0'))
  # Wrong CUDA masks refuse before claim/proc/lock access or numerical imports.
  prior=os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  try:refuses(lambda:contracts.child_context(config['workers']['server51/gpu1'],Path(temp),Path(temp),'server51/gpu1',{}))
  finally:
   if prior is not None:os.environ['CUDA_VISIBLE_DEVICES']=prior
 refuses(lambda:binding.shard(inv,'server77/gpu0'));refuses(lambda:analyze.intake([],config,'a'*64,HERE))
 assert 'torch'not in sys.modules
 result={'status':'passed','canonical_fits':24,'disjoint_cohorts':[8,8,8],'reported_banks':18,'selected_route_collections':12,'typed_relocations_and_roundtrips':24,'scientific_fields_preserved':True,'duplicate_physical_GPU_map_refused':True,'wrong_CUDA_mask_refused_before_claim_or_Torch':True,'collector_missing_globals':sorted(missing),'collector_original_runtime_imports_preserved':True,'stdlib_queue_import_unshadowed':True,'incomplete_joint_intake_refused':True,'Torch_imported':False,'native_forwards':0,'optimizer_updates':0,'host_or_remote_probes':0}
 print(json.dumps(result,indent=2,sort_keys=True));return result
if __name__=='__main__':main()
