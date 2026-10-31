"""Root-admitted finite HGT source callbacks, label-free serving and state sizing.

No numerical import/model construction at module import. No scientific runner,
registry, source optimizer commit or real SELECT/CAL/evaluation target is used.
"""
import copy,gzip,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
PACKET='native_hgt_mixed_newdata_packet_20261012_v2'
class CommitDenied(RuntimeError):pass
def allocation_guard():
 if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Exact allocation route required')
def receipt(root,path):
 path=Path(path);raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
 path=root/row['path']
 if Path(row['path']).is_absolute() or '..' in Path(row['path']).parts or path.is_symlink() or receipt(root,path)!=row:raise ValueError('Exact resource source/role receipt required')
 return path
def write(path,value):
 with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 path.chmod(0o444)
def restore_payload(torch,adapter,payload,grads):
 from engine import capture_checkpoint,_equal_tree
 modules=dict(adapter.model.named_modules());parameters=dict(adapter.model.named_parameters())
 adapter.model.load_state_dict(payload['model_state'],strict=True);adapter.optimizer.load_state_dict(payload['optimizer_state'])
 for name,flag in payload['requires_grad'].items():parameters[name].requires_grad_(flag)
 for name,flag in payload['module_modes'].items():modules[name].training=flag
 adapter.restore_state(copy.deepcopy(payload['adapter_state']))
 for name,p in parameters.items():p.grad=None if grads[name] is None else grads[name].to(p.device).clone()
 actual=capture_checkpoint(torch,adapter,fit_id=payload['fit_id'],step=payload['step'],validation=payload['validation'],decision=payload['decision'])
 if not _equal_tree(torch,actual,payload):raise ValueError('Full source state did not restore exactly')
def run_protocol(torch,adapter,folder,case_id,device):
 """Exactly two whole source TRAIN gradients at unchanged weights; all maps denied."""
 from engine import AdapterStepError,capture_checkpoint,restore_checkpoint,_equal_tree,_clone_tree
 baseline=capture_checkpoint(torch,adapter,fit_id='resource-only-'+case_id,step=0,validation={},decision={})
 grads={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in adapter.model.named_parameters()}
 parameter_ids={n:id(p) for n,p in adapter.model.named_parameters()};optimizer_ids=[id(p) for g in adapter.optimizer.param_groups for p in g['params']]
 original_step=adapter.optimizer.step;scheduler_class=type(adapter.scheduler);schedule_owned='step' in scheduler_class.__dict__;original_schedule=scheduler_class.__dict__.get('step');calls={'optimizer_denials':0,'scheduler_denials':0};rows=[];active=None;mature=None;moment_elements=0
 def deny_optimizer(*a,**kw):calls['optimizer_denials']+=1;raise CommitDenied('Exact complete source old-state gradient reached optimizer; commit denied')
 def deny_scheduler(*a,**kw):calls['scheduler_denials']+=1;raise CommitDenied('Source scheduler transition denied')
 def synchronize():
  if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
 adapter.optimizer.step=deny_optimizer;scheduler_class.step=deny_scheduler
 try:
  for index in range(2):
   callback_baseline=baseline if index==0 else mature
   restore_payload(torch,adapter,callback_baseline,grads);synchronize()
   if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
   start=time.monotonic()
   try:adapter.train_step(1)
   except AdapterStepError as error:
    if not isinstance(error.__cause__,CommitDenied):raise
    counts=copy.deepcopy(error.operation_counts)
   else:raise AssertionError('Complete source callback crossed denied optimizer boundary')
   synchronize();seconds=time.monotonic()-start
   if counts.get('optimizer_updates')!=0 or counts.get('optimizer_step_attempts')!=1 or counts.get('source_scheduler_steps')!=0:raise AssertionError('A scientific optimizer or scheduler transition occurred')
   current={n for n,p in adapter.model.named_parameters() if p.grad is not None}
   if not current or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter.model.parameters()):raise ValueError('Source gradient presence/finiteness failed')
   if active is not None and current!=active:raise ValueError('Repeated unchanged source callback changed None-gradient permission leaves')
   active=current
   if not _equal_tree(torch,_clone_tree(torch,adapter.model.state_dict()),callback_baseline['model_state']) or not _equal_tree(torch,_clone_tree(torch,adapter.optimizer.state_dict()),callback_baseline['optimizer_state']):raise ValueError('Source model/optimizer state mutated despite denied commit')
   rows.append({'callback':index+1,'wall_seconds':seconds,'operation_counts':counts,'active_gradient_parameter_objects':len(active),
    'source_None_gradient_parameter_objects':len(parameter_ids)-len(active),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0,
    'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else 0,'source_complete_callback_reached_denied_optimizer':True,'moment_scope':'fresh' if index==0 else 'synthetic_active_only_mature_buffers','scientific_optimizer_updates':0})
   restore_payload(torch,adapter,callback_baseline,grads)
   if index==0:
    if adapter.optimizer.defaults.get('amsgrad',False):raise ValueError('Exact native AdamW non-AMSGrad expected')
    for name,param in adapter.model.named_parameters():
     if adapter.optimizer.state.get(param):raise ValueError('Fresh source optimizer state unexpectedly learned')
     if name not in active:continue
     adapter.optimizer.state[param]={'step':torch.tensor(100.,dtype=torch.float32),'exp_avg':torch.zeros_like(param,memory_format=torch.preserve_format),'exp_avg_sq':torch.zeros_like(param,memory_format=torch.preserve_format)}
     moment_elements+=2*param.numel()
    mature=capture_checkpoint(torch,adapter,fit_id=baseline['fit_id'],step=0,validation={},decision={})
  # Full native serving consumes no target, metric or selector. Preserve its state.
  adapter.model.eval();synchronize();start=time.monotonic()
  with torch.no_grad():
   logits=adapter.logits_fn(adapter.model,False,0)
   if logits.dtype!=torch.float32 or logits.ndim!=3 or not torch.isfinite(logits).all():raise ValueError('Full label-free native serving shape/finiteness failed')
   native_pool=logits.mean(0)
   if not torch.isfinite(native_pool).all():raise ValueError('Native FP32 mean-logit serving nonfinite')
   serving={'wall_seconds':time.monotonic()-start,'logit_shape':list(logits.shape),'native_pool_shape':list(native_pool.shape),'labels_read':False,'raw_predictions_persisted':False,'predictive_scores':0}
  del logits,native_pool;restore_payload(torch,adapter,mature,grads)
  # Nonlearned buffer-size proxy only for actually active leaves. None leaves
  # retain absent source AdamW state; their decay and moments are never inserted.
  if any(adapter.optimizer.state.get(param) for name,param in adapter.model.named_parameters() if name not in active):raise ValueError('Unused None leaves must retain absent moments')
  proxy=capture_checkpoint(torch,adapter,fit_id=baseline['fit_id'],step=0,validation={},decision={});proxy['resource_only_nonlearned_mature_buffer_proxy']=True
  path=folder/'NONLEARNED_MATURE_MOMENT_SERIALIZER_PROXY.pt';start=time.monotonic();torch.save(proxy,path);serialization_seconds=time.monotonic()-start
  restore_checkpoint(torch,path,adapter,expected_fit_id=baseline['fit_id'],verify=True)
  proxy_loaded=capture_checkpoint(torch,adapter,fit_id=baseline['fit_id'],step=0,validation={},decision={})
  if not _equal_tree(torch,proxy_loaded,{k:proxy[k] for k in proxy_loaded}):raise ValueError('Mature proxy exact serializer restore failed')
  moment_proxy={'synthetic_active_only_AdamW_moment_elements':moment_elements,'moment_values':'zeros; synthetic step100; never learned or stepped','unused_None_leaves_remain_state_absent':True,
   'source_OneCycle_state_unchanged':True,'bytes':path.stat().st_size,'serialization_seconds':serialization_seconds,'exact_engine_restore_verified':True,'paid_parent_eligible':False,'resource_fit_id':baseline['fit_id']}
  restore_payload(torch,adapter,baseline,grads)
 finally:
  adapter.optimizer.step=original_step
  if schedule_owned:scheduler_class.step=original_schedule
  else:delattr(scheduler_class,'step')
  restore_payload(torch,adapter,baseline,grads)
 if calls!={'optimizer_denials':2,'scheduler_denials':0} or parameter_ids!={n:id(p) for n,p in adapter.model.named_parameters()} or optimizer_ids!=[id(p) for g in adapter.optimizer.param_groups for p in g['params']]:raise ValueError('Source update count or owned parameter identities changed')
 return {'schema':'native-HGT-denied-update-resource-protocol-v1','callbacks':rows,'optimizer_denials':2,'source_scheduler_denials':0,'label_free_serving':serving,'nonlearned_mature_serializer_proxy':moment_proxy,
  'full_source_model_optimizer_OneCycle_private_global_RNG_modes_flags_and_grad_state_restored':True,'parameter_and_optimizer_object_ownership_unchanged':True,
  'scientific_optimizer_updates':0,'predictive_scores':0,'CAL_updates':0,'paid_warm_or_selected_parent_created':False,'GO':False}
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--proposal',required=True);p.add_argument('--proposal-sha256',required=True);p.add_argument('--engineering-admission',required=True)
 p.add_argument('--case-id',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0');a=p.parse_args();allocation_guard();root=Path(a.project_root).resolve()
 from pathlib import PurePath
 def confined(relative):
  if PurePath(relative).is_absolute() or '..' in PurePath(relative).parts:raise ValueError('Exact phase-relative resource paths required')
  q=root/relative
  if not q.resolve().is_relative_to(root):raise ValueError('Resource path escapes phase')
  return q
 proposal_path=confined(a.proposal);raw=proposal_path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=a.proposal_sha256:raise ValueError('Exact HGT13 proposal binding changed')
 proposal=json.loads(raw);admission=json.loads(confined(a.engineering_admission).read_text());helper_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 if admission.get('root_review_complete') is not True or admission.get('proposal_sha256')!=a.proposal_sha256 or admission.get('helper_sha256')!=helper_hash or a.case_id not in admission.get('admitted_case_ids',[]) or admission.get('scientific_fits_allowed') is not False:raise ValueError('Root-reviewed exact helper/proposal/case engineering admission required')
 cases=[r for r in proposal['cases'] if r['case_id']==a.case_id]
 if len(cases)!=1 or cases[0]['repetitions']!=2:raise ValueError('Exactly one frozen two-callback HGT source case required')
 case=cases[0];folder=confined(a.output)
 if folder.exists():raise FileExistsError('Preserve prior resource case and owner; fresh output required')
 folder.mkdir(parents=True);write(folder/'START.json',{'case_id':a.case_id,'pid':os.getpid(),'source_proposal_sha256':a.proposal_sha256,'helper_sha256':helper_hash,'device':a.device,'scientific_fits':0,'GO':False})
 sys.path.insert(0,str(root/PACKET/'runtime'));from spec import validate,fingerprint
 study=json.loads(gzip.decompress(bound(root,case['study_receipt']).read_bytes()));validate(study)
 if fingerprint(study)!=case['study_spec_hash']:raise ValueError('Exact HGT source study canonical binding differs')
 original=next(r for r in study['runs'] if r['run_id']==case['run_id']);run=copy.deepcopy(original)
 if fingerprint(run)!=case['run_contract_hash'] or run['config']['hgt_arm']!=case['source_arm'] or run['body']!=case['source_body_index']:raise ValueError('Exact source case differs')
 allowed={str((root/r['path']).resolve()) for r in (run['config']['node_inputs']['graph'],run['config']['node_inputs']['train'])}
 allowed.update(str((folder/n).resolve()) for n in ('SYNTHETIC_UNUSED_SELECT.pt','NONLEARNED_MATURE_MOMENT_SERIALIZER_PROXY.pt'))
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   q=Path(os.fsdecode(args[0])).absolute()
   if q.is_relative_to(root) and q.suffix in ('.pt','.npy','.npz','.zip','.pkl') and str(q.resolve()) not in allowed:raise RuntimeError('Real SELECT/CAL/evaluation/raw archive access denied in HGT resource helper')
 sys.addaudithook(audit)
 import torch
 torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
 train=torch.load(bound(root,run['config']['node_inputs']['train']),map_location='cpu',weights_only=True);used=set(train['ids'].tolist());stub_id=next(i for i in range(4057) if i not in used)
 stub=folder/'SYNTHETIC_UNUSED_SELECT.pt';torch.save({'ids':torch.tensor([stub_id],dtype=torch.long),'y':torch.zeros(1,dtype=torch.long)},stub);stub_receipt=receipt(root,stub)
 study=copy.deepcopy(study);study['stage']='engineering';study['data']['roles']['select'][str(run['split'])]=stub_receipt;run['config']['node_inputs']['select']=stub_receipt
 from adapters.hgt_native import make_adapter
 adapter=make_adapter(torch=torch,project_root=root,study=study,run=run,device=a.device)
 schema=adapter.custody['graph_schema']
 if schema['node_counts']!={'0':4057,'1':14328,'2':7723,'3':20} or schema['input_dims']!={'0':334,'1':14328,'2':7723,'3':20}:raise ValueError('Full actual DBLP feature2 graph geometry required')
 adapter.validate=lambda *_:(_ for _ in ()).throw(RuntimeError('SELECT metrics/labels are unused and forbidden in resource helper'))
 result=run_protocol(torch,adapter,folder,a.case_id,a.device);result.update(case_id=a.case_id,source_arm=case['source_arm'],source_own_body_index=run['body'],source_study_hash=case['study_spec_hash'],source_run_contract_hash=case['run_contract_hash'],
  source_complete_body_constructors=case['native_constructor_count'],full_GRAPH_TRAIN=True,real_SELECT_CAL_evaluation_targets_opened=False,synthetic_unused_SELECT=stub_receipt,resource_profile_qualified=False)
 write(folder/'RESULT.json',result);print(json.dumps({'case_id':a.case_id,'callbacks':2,'optimizer_updates':0,'scores':0,'resource_profile_qualified':False,'result':str((folder/'RESULT.json').relative_to(root)),'GO':False}))
if __name__=='__main__':main()
