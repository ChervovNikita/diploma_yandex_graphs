"""Finite allocation owner for exact final-source sync tooling; no scientific workers."""
import argparse,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
TOOLING='final_master_sync_author_runtime_scope_20261013_v4'
TOOLING_INVENTORY='final_master_sync_owner_20261013_v4/SYNC_TOOLING_INVENTORY.json'
TOOLING_INVENTORY_SHA='9d5e39a561c7ffa8d8ee6523eb730b2a459af32e208dd40268b8d45b9521f82f'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def identity(pid):
 fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 return {'pid':pid,'start_ticks':int(fields[19]),'pgid':int(fields[2])}
def write(path,value):
 with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
 path.chmod(0o444)
def run_child(command,folder,label,env,limit):
 """A bounded owned child; timeout preserves logs/partial output and identity."""
 started=time.monotonic();stdout=folder/(label+'_STDOUT.log');stderr=folder/(label+'_STDERR.log')
 with stdout.open('x') as out,stderr.open('x') as err:
  child=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
  owner=identity(child.pid);write(folder/(label+'_START.json'),{'command':command,'owner':owner,'maximum_wall_seconds':limit,'scientific_fits':0,'GO':False})
  timed_out=False
  try:child.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   timed_out=True
   if identity(child.pid)==owner:os.killpg(owner['pgid'],signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:
    if identity(child.pid)==owner:os.killpg(owner['pgid'],signal.SIGKILL)
    child.wait(timeout=5)
 result={'label':label,'command':command,'owner':owner,'exit_code':child.returncode,'timed_out':timed_out,'wall_seconds':time.monotonic()-started,
  'stdout_path':str(stdout),'stderr_path':str(stderr),'partial_outputs_preserved':True,'scientific_fits':0,'GO':False}
 write(folder/(label+'_RESULT.json'),result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('manifest_audit','bundle'));p.add_argument('--project-root',required=True);p.add_argument('--owner-directory',required=True)
 p.add_argument('--master',required=True);p.add_argument('--master-hash',required=True);p.add_argument('--summary',required=True);p.add_argument('--summary-sha256',required=True)
 p.add_argument('--manifest-directory');p.add_argument('--manifest');p.add_argument('--manifest-sha256');p.add_argument('--audit');p.add_argument('--audit-sha256');p.add_argument('--bundle-output');p.add_argument('--bundle-receipt')
 p.add_argument('--maximum-child-wall-seconds',type=int,default=1800);a=p.parse_args();root=Path(a.project_root).absolute();sys.path.insert(0,str(root/TOOLING/'runtime'))
 from sync_common import allocation_guard,safe,read_json,bound_master,receipt
 allocation_guard()
 if not 1<=a.maximum_child_wall_seconds<=1800:raise ValueError('Bounded maximum child window1..1800 required')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise ValueError('Hidden CUDA and no bytecode writes required')
 inventory,inventory_receipt=read_json(root,TOOLING_INVENTORY,sha256=TOOLING_INVENTORY_SHA)
 for row in inventory:
  if receipt(root,row['path'])!=row:raise ValueError('Sealed tooling differs')
 plan,summary,master_receipt,summary_receipt=bound_master(root,a.master,a.master_hash,a.summary,a.summary_sha256)
 folder=safe(root,a.owner_directory,file=False)
 if folder.exists():raise FileExistsError('Fresh finite owner directory required; inspect prior owner and outputs')
 if a.mode=='manifest_audit':
  if not a.manifest_directory:raise ValueError('Exact fresh manifest directory required')
  if safe(root,a.manifest_directory,file=False).exists():raise FileExistsError('Prior manifest must be preserved; no rerun into it')
 else:
  if not all((a.manifest,a.manifest_sha256,a.audit,a.audit_sha256,a.bundle_output,a.bundle_receipt)):raise ValueError('Bundle needs exact completed manifest/audit receipts and fresh output/receipt')
  for name in (a.bundle_output,a.bundle_receipt):
   if safe(root,name,file=False).exists():raise FileExistsError('Prior partial bundle/output retained; choose fresh output')
 folder.mkdir(parents=True);request={k:v for k,v in vars(a).items() if k!='project_root'}
 write(folder/'OWNER_START.json',{'owner':identity(os.getpid()),'hostname':socket.gethostname(),'request':request,'request_sha256':hashlib.sha256(canonical(request).encode()).hexdigest(),
  'exact_final_master':master_receipt,'exact_summary':summary_receipt,'tooling_inventory':inventory_receipt,'scientific_workers_allowed':False,'GO':False})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');runtime=root/TOOLING/'runtime';records=[];status='failed'
 bound=['--project-root',str(root),'--master',a.master,'--master-hash',a.master_hash,'--summary',a.summary,'--summary-sha256',a.summary_sha256]
 try:
  if a.mode=='manifest_audit':
   commands=[('FINAL_MANIFEST_BUILD',[sys.executable,'-B',str(runtime/'build_final_master_sync.py'),*bound,'--tooling-inventory',TOOLING_INVENTORY,'--tooling-inventory-sha256',TOOLING_INVENTORY_SHA,'--output',a.manifest_directory])]
   result=run_child(commands[0][1],folder,commands[0][0],env,a.maximum_child_wall_seconds);records.append(result)
   if result['exit_code']:raise RuntimeError('Manifest build failed; logs and outputs retained')
   manifest=a.manifest_directory+'/MANIFEST.json';row=receipt(root,manifest)
   result=run_child([sys.executable,'-B',str(runtime/'audit_final_master_source_loading.py'),*bound,'--manifest',manifest,'--manifest-sha256',row['sha256'],
    '--output',a.manifest_directory+'/SOURCE_LOADING_AUDIT.json'],folder,'ACTUAL_FINAL_SOURCE_LOADING_AUDIT',env,a.maximum_child_wall_seconds);records.append(result)
   if result['exit_code']:raise RuntimeError('Exact source loading audit failed; logs and outputs retained')
   output={'manifest':row,'audit':receipt(root,a.manifest_directory+'/SOURCE_LOADING_AUDIT.json'),'summary':receipt(root,a.manifest_directory+'/SUMMARY.json')}
  else:
   manifest,mr=read_json(root,a.manifest,sha256=a.manifest_sha256);audit,ar=read_json(root,a.audit,sha256=a.audit_sha256)
   if manifest['plan_hash']!=a.master_hash or manifest['plan']!=master_receipt or manifest['exact_summary_binding']!=summary_receipt or audit['manifest']!=mr or audit['plan_hash']!=a.master_hash:raise ValueError('Exact final master/manifest/audit binding differs')
   result=run_child([sys.executable,'-B',str(runtime/'make_final_master_bundle.py'),'--project-root',str(root),'--manifest',a.manifest,'--manifest-sha256',a.manifest_sha256,
    '--audit',a.audit,'--audit-sha256',a.audit_sha256,'--output',a.bundle_output,'--receipt-output',a.bundle_receipt],folder,'FINAL_SOURCE_TRANSFER_BUNDLE',env,a.maximum_child_wall_seconds);records.append(result)
   if result['exit_code']:raise RuntimeError('Bundle creation failed; partial archive and logs retained')
   value,rr=read_json(root,a.bundle_receipt);output={'bundle_receipt':rr,'bundle':{'path':a.bundle_output,'bytes':value['bytes'],'sha256':value['sha256']}}
  status='complete';failure=None
 except BaseException as error:output={};failure={'type':type(error).__name__,'message':str(error)}
 final={'schema':'allocation-final-master-finite-source-sync-owner-v1','status':status,'mode':a.mode,'request_sha256':hashlib.sha256(canonical(request).encode()).hexdigest(),'plan_hash':a.master_hash,
  'records':records,'output':output,'failure':failure,'manifest_opaque_hash_passes_per_unique_input':1,'bundle_source_stream_hash_passes_per_unique_input':1,
  'exact_launcher_audit_uses_original_hash_validator':'Source bindings may be reread across studies; measured phase_file_open_counts preserved; not claimed one-pass.',
  'old_array_copies':0,'server77_access':False,'package_installs':0,'models_constructed':0,'scientific_fits':0,'GO':False}
 write(folder/'OWNER_RESULT.json',final);print(canonical(final),flush=True);return 0 if status=='complete' else 1
if __name__=='__main__':sys.exit(main())
