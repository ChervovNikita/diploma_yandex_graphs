"""Exactly one common17forward TRAIN-only live-action screen per newly paid warm."""
import copy,hashlib,itertools,json
from pathlib import Path
from .native_label_families import bound
from .nsd import factory,construct,helpers


def receipt(root,path):
 b=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def run_screen(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch,numpy as np
 from engine import checkpoint_receipt
 root=Path(project_root).resolve();c=run['config'];pid=c['warm_parent'];parent=registry.run(study['study_id'],pid);result=parent['result'];fit=parent['contract']
 if parent['state']!='complete' or result['completed_steps']!=100 or fit['config']['nsd_arm']!='geometry_warm':raise ValueError('One newly paid exact100identity warm source')
 checkpoint=Path(output_root)/study['study_id']/pid/'final_training.pt';authority=result['final_training_checkpoint']
 if checkpoint_receipt(checkpoint,fit_id=pid,step=100)!=authority:raise ValueError('Terminal100 byte-authorized warm')
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 graph=torch.load(bound(root,c['node_inputs']['graph']),map_location='cpu',weights_only=True);train=torch.load(bound(root,c['node_inputs']['train']),map_location='cpu',weights_only=True)
 if state['fit_id']!=pid or state['step']!=100:raise ValueError('Terminal100 source identity')
 e,f,args=factory(torch,root,c['nsd_sources'],c,graph,device);h=helpers(torch,root,c['nsd_sources']);before=h.capture_rng(np,torch)
 counts={k:0 for k in ('screen_native_builder_returns','extra_offnode_spmm_attempts','extra_offnode_spmm_completed','offnode_sparse_copy_bytes_total','screen_action_CPU_bytes_total')}
 model=None
 try:
  cc={**c,'nsd_arm':'geometry_warm','device':str(device)};model=construct(torch,root,c['nsd_sources'],e,f,cc,c['initialization_seed']);model.load_state_dict(state['model_state']);model.eval();h.restore_rng(np,torch,state['adapter_state']['external']['source_rng'])
  member=model.members[0];factor=member.sheaf_learners[0].linear1.r;directions=e.init._directions(torch,factor,c['initialization_seed'],8);common=None;ids=train['ids'].to(device);y=train['y'].to(device);x=graph['x'].to(device);trials=[]
  def observe():
   nonlocal common
   with torch.no_grad(),e.init.FirstActionCapture(torch,member,ids,counts,common) as capture:
    logp=member(x)
    if logp.shape!=(len(x),2) or not torch.isfinite(logp).all():raise ValueError('Exact source full binary screen')
    nll=torch.nn.functional.nll_loss(logp[ids],y);margin=(logp[ids,1]-logp[ids,0]).detach().cpu().double()
    if common is None:common=capture.baseline_value
   return {'nll':float(nll),'action':capture.action,'margin':margin,'common_value_materiality':capture.value_materiality}
  base=observe();ascal=max(e.init._norm(torch,base['action']),1.);mscal=max(e.init._norm(torch,base['margin']),1.);limit=base['nll']+.001*max(base['nll'],.01);candidates={}
  for i,direction in enumerate(directions):
   observed=[]
   for sign in (1,-1):
    with torch.no_grad():factor.copy_(1+sign*.01*direction)
    value=observe();a=(value['action']-base['action'])/ascal;b=(value['margin']-base['margin'])/mscal
    passed=value['nll']<=limit and e.init._norm(torch,a)>1e-6 and e.init._norm(torch,b)>1e-6
    observed.append(value);trials.append({'direction':i,'sign':sign,'TRAIN_nll':value['nll'],'action_relative':e.init._norm(torch,a),'classifier_relative':e.init._norm(torch,b),'admissible':passed})
   a=.5*(observed[0]['action']-observed[1]['action']);b=.5*(observed[0]['margin']-observed[1]['margin'])
   if all(t['admissible'] for t in trials[-2:]) and e.init._norm(torch,a)/ascal>1e-6 and e.init._norm(torch,b)/mscal>1e-6:candidates[i]={'action':a,'classifier':b}
  with torch.no_grad():factor.fill_(1)
  if len(candidates)<2:raise ValueError('Whole four-arm initialization cohort unqualified: fewer than2common admissible directions;no rescue/redraw')
  random_pair=sorted(sorted(candidates,key=lambda i:(e.init._hash('random-order',c['initialization_seed'],i),i))[:2]);action_pair,action_scores=e.init._pair_choice(torch,c['initialization_seed'],candidates,'action');classifier_pair,classifier_scores=e.init._pair_choice(torch,c['initialization_seed'],candidates,'classifier')
  value={'choices':{'identity':None,'random_admissible':random_pair,'live_action_rank':action_pair,'classifier_response_rank':classifier_pair},'directions_CPU':[d.detach().cpu() for d in directions],'warm_fit':pid,'terminal_warm_checkpoint':authority,'trials':trials,'candidate_count':8,'complete_native_forwards':17,'extra_offnode_spmm':17,'source_scope':'TRAIN only;no SELECT/CAL/evaluate;common baseline actualV;source rank/no-redraw guards retained','counts':counts,'action_pair_scores':action_scores,'classifier_pair_scores':classifier_scores}
  path=Path(output_dir)/'SCREEN.pt';torch.save(value,path)
  return {'status':'complete','screen_file':{'path':'SCREEN.pt',**{k:v for k,v in receipt(root,path).items() if k!='path'}},'warm_parent':pid,'terminal_warm_checkpoint':authority,'complete_native_forwards':17,'extra_offnode_spmm':17,'source_constructor_count':1,'optimizer_constructors':0,'TRAIN_only':True,'scientific_fits':0}
 finally:h.restore_rng(np,torch,before)
