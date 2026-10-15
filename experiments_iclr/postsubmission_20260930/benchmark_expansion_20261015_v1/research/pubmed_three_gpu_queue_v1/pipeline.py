"""Original fit API, original selected-collection core and portable saved banks."""
import argparse,ast,copy,importlib.util,json,shutil,signal,sys,time,types
from pathlib import Path
from binding import HERE,E,require,pin,digest,relocated,install_residual_descriptor_view,restore_descriptor_view
from contracts import request,read,write,verify,child_context
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def interrupted(*_):raise InterruptedError('Owned child interrupted; no automatic retry')
def selected_core(source):
 """Retain the original collector's entire scientific/work-count/cleanup body."""
 module=load(source,'_pubmed_original_selected_collector');main=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
 start=next(i for i,n in enumerate(main.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='started'for t in n.targets))
 names=['s','h','root','here','output','row','fit','folder','spec','args','a_path','end_path','out_path','outcomes']
 imports=[n for n in main.body[:start]if isinstance(n,ast.ImportFrom)and n.module in('adapters','bookkeeping','common')]
 require(len(imports)==3,'Original collector runtime imports changed')
 function=ast.FunctionDef(name='_selected_core',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n)for n in names],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(imports+main.body[start:]),decorator_list=[])
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),env)
 return env['_selected_core']
def fit_authority(folder,row,job,root):
 s=load(E/'research/pubmed_cheb_transfer_driver_v1/study.py','_pubmed_original_fit_custody')
 result,original_folder=s.fit(root,folder.parent.parent,row|{'job':job,'job_hash':digest(job)},{'RESULT':pin(folder/'RESULT.json')})
 require(original_folder==folder,'Exact own fit folder required');return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--target-map',required=True);p.add_argument('--target-map-sha256',required=True);p.add_argument('--mode',choices=('fit','collect','export'),required=True);p.add_argument('--index',type=int);args=p.parse_args()
 a,host,root,rows,ap=request(args.target_map,args.target_map_sha256,args.worker);output=E/'results/pubmed_three_gpu_v1'/args.worker.replace('/','_')
 child_context(host,root,output,args.worker,pin(ap))
 sys.path.insert(0,str(E/'runtime'));import run_fit
 import torch
 require(str(torch.__version__)==host['expected_torch_version'],'Reviewed Torch build differs');torch.set_num_threads(2)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 if args.mode=='export':return export(root,rows,output,args.worker,a,args.target_map_sha256)
 selected=[r for r in rows if r['index']==args.index];require(len(selected)==1,'Index outside assigned cohort');row=selected[0]
 job,binding=relocated(row,host,args.worker,a['queue_manifest_sha256'],args.target_map_sha256);folder=output/'fits'/row['token']
 view=install_residual_descriptor_view(job,row,root)
 try:
  if args.mode=='fit':
   result=run_fit.run(job,root,folder)
   write(folder/'EXECUTION_BINDING.json',{'canonical':{'id':row['id'],'job_sha256':row['job_hash']},'execution':binding,'worker':args.worker,'target_map':pin(ap),'source_manifest':pin(HERE/'MANIFEST.json')})
   return 0 if result['status']=='complete'else 1
  require(row['kind']!='native','Only four shared banks require route collection')
  end,outcomes=read(output/'FIT_END.json'),read(output/'FIT_OUTCOMES.json')
  require(end['status']=='complete'and len(outcomes)==8 and [r['id']for r in outcomes]==[r['id']for r in rows]and all(r['status']=='complete'for r in outcomes),'All eight local fits must complete before collection')
  fit=fit_authority(folder,row,job,root);s=load(E/'research/pubmed_cheb_transfer_driver_v1/study.py','_pubmed_original_study')
  h=types.SimpleNamespace(require=require,pin=pin,write=write,interrupted=interrupted)
  target=output/'collections'/row['token'];source=E/'research/pubmed_cheb_transfer_driver_v1/collect.py'
  code=selected_core(source);status=code(s,h,root,source.parent,target,row|{'job':job,'job_hash':digest(job)},fit,folder,read(HERE/'PARENT_INPUTS.json'),types.SimpleNamespace(index=row['index']),ap,output/'FIT_END.json',output/'FIT_OUTCOMES.json',{r['index']:r for r in outcomes})
  write(target/'PORTABLE_BINDING.json',{'source':pin(source),'core':'unchanged original started-through-return AST','canonical_job_sha256':row['job_hash'],'execution_job_sha256':digest(job),'worker':args.worker,'local_complete_fits':8,'joint_complete_fits_required_for_analysis':24,'execution':binding,'queue_manifest':pin(HERE/'MANIFEST.json')})
  return status
 finally:restore_descriptor_view(view)
def export(root,rows,output,worker,a,map_sha):
 import torch
 s=load(E/'research/pubmed_cheb_transfer_driver_v1/study.py','_pubmed_bundle_study');parent=read(HERE/'PARENT_INPUTS.json');host=a['workers'][worker]
 outcomes=read(output/'FIT_OUTCOMES.json');collections=read(output/'COLLECTION_OUTCOMES.json')
 require(len(outcomes)==8 and len(collections)==4 and all(r['status']=='complete'for r in outcomes)and all(r['status']in('complete','complete_with_sanity_warning')for r in collections),'Complete eight fits and four collections required')
 target=output/'bundle';target.mkdir(exist_ok=False);exports={};fits={};sources=[];cost=[]
 for row in rows:
  job,_=relocated(row,host,worker,a['queue_manifest_sha256'],map_sha);folder=output/'fits'/row['token'];fit=fit_authority(folder,row,job,root);fits[row['id']]=fit
  saved=torch.load(folder/'selected_VALID.pt',map_location='cpu',weights_only=False);s.accept_export(torch,saved,fit,job,parent['provider']);exports[row['id']]=saved
  sources.extend(folder/name for name in('JOB.json','RESULT.json','EXECUTION_BINDING.json','selected_VALID.pt'));cost.append({'canonical_id':row['id'],'cohort_seed':row['cohort_seed'],'kind':row['kind'],'fit_RESULT':pin(folder/'RESULT.json'),'result':fit})
 native=sorted([r for r in rows if r['kind']=='native'],key=lambda r:r['job']['body']);first=exports[native[0]['id']]
 aggregate=load(E/'runtime/aggregate_selected.py','_pubmed_original_I4').aggregate([fits[r['id']]['selected_VALID_export']['path']for r in native],root,target/'I4')
 bank=torch.load(verify(aggregate['bank_receipt'],root),map_location='cpu',weights_only=False)
 require(bank['M1_exact_member0_job_hash']==first['job_hash']and bank['extra_training_fits']==0 and bank['pooled_epoch_reselection']is False and torch.equal(bank['ids'],first['ids'])and torch.equal(bank['labels'],first['labels']),'Genuine I4/body0 ownership differs')
 banks={'M1':{'pool':first['raw_logits'],'members':first['raw_logits'].unsqueeze(0),'saved_native_CE':first['validation']['loss'],'canonical_ids':[native[0]['id']],'epochs':[first['selected_epoch']]},'I4':{'pool':bank['raw_logits'],'members':bank['member_raw_logits'],'saved_native_CE':bank['loss'],'canonical_ids':[r['id']for r in native],'epochs':[exports[r['id']]['selected_epoch']for r in native]}}
 for row in rows:
  if row['kind']=='native':continue
  folder=output/'collections'/row['token'];r=read(folder/'RESULT.json');saved=exports[row['id']];raw=torch.load(verify(r['raw_tensor_receipt'],root),map_location='cpu',weights_only=False)
  require(r['status']in('complete','complete_with_sanity_warning')and r['job_hash']==saved['job_hash']and r['selected_checkpoint']==saved['selected_checkpoint']and r['actual_work']['TRAIN_callbacks']==r['actual_work']['optimizer_steps']==0 and r['actual_work']['native_forwards_completed']==4 and raw['job_hash']==saved['job_hash']and torch.equal(raw['ids'],first['ids'])and torch.equal(raw['labels'],first['labels'])and tuple(raw['member_raw_logits'].shape)==(4,3943,3),'Aligned zero-training selected routes required')
  banks[row['kind']]={'pool':saved['raw_logits'],'members':raw['member_raw_logits'],'saved_native_CE':saved['validation']['loss'],'canonical_ids':[row['id']],'epochs':[saved['selected_epoch']],'sanity':r['sanity']}
  sources.extend(folder/name for name in('RESULT.json','PORTABLE_BINDING.json','ROUTE_VALID.pt'))
 require(set(banks)=={'M1','I4','G0','uniform','structured','permuted'},'Six complete portable banks required')
 torch.save({'schema':'pubmed-portable-selected-six-banks-v1','worker':worker,'seed':rows[0]['cohort_seed'],'ids':first['ids'],'labels':first['labels'],'banks':banks,'TEST_accessed':False},target/'PREDICTIONS.pt')
 files=[]
 for source in sources:
  rel=source.relative_to(output);dest=target/'evidence'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);value=pin(dest);files.append({'relative':str(dest.relative_to(target)),'origin':pin(source),'bytes':value['bytes'],'sha256':value['sha256']})
 for source in(output/'FIT_END.json',output/'FIT_OUTCOMES.json',output/'COLLECTION_OUTCOMES.json'):
  shutil.copyfile(source,target/source.name)
 for source in sorted((target/'I4').rglob('*')):
  if source.is_file():
   value=pin(source);files.append({'relative':str(source.relative_to(target)),'origin':value,'bytes':value['bytes'],'sha256':value['sha256']})
 write(target/'BUNDLE.json',{'schema':'pubmed-portable-cohort-bundle-v1','status':'complete','worker':worker,'seed':rows[0]['cohort_seed'],'canonical_ids':[r['id']for r in rows],'canonical_inventory_sha256':pin(HERE/'CANONICAL.json')['sha256'],'source_manifest':pin(HERE/'MANIFEST.json'),'target_map_sha256':map_sha,'predictions':{'relative':'PREDICTIONS.pt',**{k:pin(target/'PREDICTIONS.pt')[k]for k in('bytes','sha256')}},'files':files,'training_outcomes':cost,'collection_outcomes':collections,'I4_aggregation':aggregate,'TEST_accessed':False,'host':{'hostname':host['expected_hostname'],'physical_GPU_UUID':host['physical_gpu_uuid'],'Torch':str(torch.__version__)},'joint_analysis_requires_all_three_bundles':True})
 return 0
if __name__=='__main__':raise SystemExit(main())
