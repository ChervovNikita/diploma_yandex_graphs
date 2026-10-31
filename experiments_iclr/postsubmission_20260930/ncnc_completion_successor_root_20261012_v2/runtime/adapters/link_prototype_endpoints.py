"""Raw-only complete selected prototype bank, sealed source link endpoint."""
import gc
from pathlib import Path
from . import link_boundary_endpoints as native

@native.retain_costs
def export_bank(*,project_root,study,run,output_root,output_dir,registry,device,operation_counts):
 import torch
 config=run['config'];native._inputs(study,config);native.whole_fit_roster_closed(study,registry)
 pack=native.query_pack(torch,native.load_bound(torch,project_root,config['query_input']),study['data']['evidence']['endpoint_nodes']);parents=config['fit_parents']
 if len(parents) not in (1,4) or len(set(parents))!=len(parents):raise ValueError('One model bank or four distinct own selected source learners required')
 outputs=[];ancestry=[];contracts=[]
 for identifier in parents:
  actual,selected,checkpoint,member=native._selected_authority(project_root,output_root,study,run,registry,identifier)
  if 'link_prototype_arm' not in actual['config']:raise ValueError('Exact source prototype selected parent required')
  adapter=native.load_factory(project_root,actual['adapter'])(torch=torch,project_root=project_root,study=study,run=actual,device=device);operation_counts['model_constructions']+=1;operation_counts['checkpoint_restore_attempts']+=1
  restored=native.restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=identifier);operation_counts['checkpoint_restores']+=1
  if restored['step']!=selected['step']:raise ValueError('Exact own selected epoch differs')
  operation_counts['prediction_pack_attempts']+=1;score=adapter.predict_pairs(pack['pairs']);operation_counts['prediction_packs_completed']+=1
  if score.shape[1]!=len(pack['ids']) or not torch.isfinite(score).all():raise ValueError('Source serving omitted a query row')
  if len(parents)==4 and score.shape[0]!=1:raise ValueError('Genuine I4 requires one served trajectory per separately selected native source body')
  outputs.append(score);ancestry.append(member);contracts.append(actual);del adapter;gc.collect()
 if len(parents)==4:
  if {r['body'] for r in contracts}!={0,1,2,3}:raise ValueError('Complete four-body source bank required')
  order=sorted(range(4),key=lambda i:contracts[i]['body']);outputs=[outputs[i] for i in order];ancestry=[ancestry[i] for i in order]
 scores=torch.cat(outputs,0);native.pool_scores(torch,scores);path=Path(output_dir)/'raw_query_bank.pt';torch.save({'ids':pack['ids'],'scores':scores},path)
 return {'status':'complete','bank':native.receipt(project_root,path),'ancestry':ancestry,'query_input':config['query_input'],'fusion':'mean_raw_scores','support_policy':'full_TRAIN_only','evaluation_targets_read':False,'scientific_backbone_fits':0,'source_prototype_bank':True}

evaluate_bank=native.evaluate_bank
