"""D4 and same-selected-bank M4 query-only export; raw-only shared-pool evaluation."""
import copy,gc
from pathlib import Path
try:
 from ..engine import restore_checkpoint
 from ..run_study import load_factory
except ImportError:
 from engine import restore_checkpoint
 from run_study import load_factory
from .link_boundary_endpoints import receipt,load_bound,query_pack,pool_scores,whole_fit_roster_closed,predict_unlabelled,retain_costs,evaluate_bank

@retain_costs
def export_bank(*,project_root,study,run,output_root,output_dir,registry,device,operation_counts):
 import torch
 from structural_cardinality_semantics import validate_run
 c=run['config'];whole_fit_roster_closed(study,registry);fits=c['fit_parents'];pack=query_pack(torch,load_bound(torch,project_root,c['query_input']),235868);outputs=[];ancestry=[];diagnostic=None
 if len(fits) not in (1,4) or len(set(fits))!=len(fits):raise ValueError('One own bank or four whole independent fits required')
 for id in fits:
  row=registry.run(study['study_id'],id);fit=row['contract'];validate_run(fit)
  if row['state']!='complete' or row['result']['completed_steps']!=100 or fit['budget']['maximum']!=100:raise ValueError('Complete100state own source fit required')
  selected=row['result']['selected'];checkpoint=Path(output_root)/study['study_id']/id/selected['path'];pin=receipt(project_root,checkpoint)
  a=load_factory(project_root,fit['adapter'])(torch=torch,project_root=project_root,study=study,run=fit,device=device);operation_counts['model_constructions']+=1;operation_counts['checkpoint_restore_attempts']+=1
  restored=restore_checkpoint(torch,checkpoint,a,expected_fit_id=id);operation_counts['checkpoint_restores']+=1
  if restored['step']!=selected['step']:raise ValueError('Exact own selected epoch required')
  route=c['serving_route'];operation_counts['prediction_pack_attempts']+=1
  if 'cardinality_source_contract' in fit['config']:
   if len(fits)!=1 or route not in ('D4','M4'):raise ValueError('C64 single fixed-bank routes required')
   if route=='M4':
    d4=registry.run(study['study_id'],c['D4_export_parent'])
    if d4['state']!='complete' or d4['contract']['config']['serving_route']!='D4' or d4['contract']['config']['fit_parents']!=fits or d4['result']['ancestry'][0]['selected_checkpoint']!=pin:raise ValueError('M4 same completeD4 ownselected bank required')
    bank=a.modules['serving'].SelectedD4Bank(pin['sha256'],selected['step'],100);a._M4_authority=bank;diagnostic=a.complete_M4_SELECT(bank,pin['sha256']);operation_counts['M4_complete_SELECT_routes']=1
    fixed=a.fixed_diagnostic_routes(bank,pin['sha256']);fixed_path=Path(output_dir)/'fixed_TRAIN_D4_M4_diagnostic.pt';torch.save(fixed,fixed_path);diagnostic['fixed_TRAIN_routes']=receipt(project_root,fixed_path);diagnostic['pre_model_support_seal']=a.diagnostic['source_receipt'];operation_counts['fixed_TRAIN_D4_M4_routes']=2
   z=a.predict_pairs(pack['pairs'],route=route)
  elif 'conditional_source_contract' in fit['config']:
   if len(fits)!=1 or route!='native_pattern':raise ValueError('Own selected nativePatternTwin serving required')
   first=a.validate(selected['step']);b=load_factory(project_root,fit['adapter'])(torch=torch,project_root=project_root,study=study,run=fit,device=device);operation_counts['model_constructions']+=1;operation_counts['checkpoint_restore_attempts']+=1;restore_checkpoint(torch,checkpoint,b,expected_fit_id=id);operation_counts['checkpoint_restores']+=1;second=b.validate(selected['step'])
   if first['selection_metric']!=second['selection_metric'] or first['selection_metric']!=selected['validation']['selection_metric']:raise ValueError('Own selected completeVALID metric replay differs')
   diagnostic={'selected_state_replays':[first,second],'fresh_reconstruction':True,'extra_complete_SELECT_routes':2};operation_counts['CB_selected_complete_SELECT_replays']=2;del a;a=b;z=a.predict_pairs(pack['pairs'],route=route)
  else:
   if route!='native' or fit['config'].get('cardinality_control') not in ('native_M1','original_GNNM'):raise ValueError('Freshdeclared nativecontrol route required')
   scores,counts=predict_unlabelled(a,pack['pairs']);z=scores[None,:]
   for key,value in counts.items():operation_counts[key]=operation_counts.get(key,0)+value
  operation_counts['prediction_packs_completed']+=1;outputs.append(z);ancestry.append({'run_id':id,'body':fit['body'],'actual_model_seed':fit['config']['initialization_seed'],'selected_checkpoint':pin,'selected_epoch':selected['step'],'serving_route':route});del a;gc.collect()
  if str(device).startswith('cuda'):torch.cuda.empty_cache()
 if len(fits)==4 and (len({v['actual_model_seed'] for v in ancestry})!=4 or {v['body'] for v in ancestry}!={0,1,2,3} or any(z.shape[0]!=1 for z in outputs)):raise ValueError('GenuineI4 four ownselected oneview wholebodies required')
 scores=torch.cat(outputs);pool_scores(torch,scores);q=Path(output_dir)/'raw_query_bank.pt';torch.save({'ids':pack['ids'],'scores':scores},q)
 return {'status':'complete','bank':receipt(project_root,q),'ancestry':ancestry,'query_input':c['query_input'],'fusion':'mean_raw_scores','support_policy':'full_TRAIN_only','evaluation_targets_read':False,'CAL_used':False,'confidence_claims_available':False,'scientific_backbone_fits':0,'M4_complete_SELECT_diagnostic':diagnostic,'same_selected_D4_state':c['serving_route']=='M4'}
