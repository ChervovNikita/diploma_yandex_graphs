"""Primary mean banks and literal12raw selected diagnostics;label-free export."""
import gc,math
from pathlib import Path
from . import node_endpoints as primary
from .node_endpoints import scalar_calibration,evaluate_bank,receipt
try:
 from ..engine import restore_checkpoint
 from ..run_study import load_factory,resolve_selected_config
 from ..spec import confined
except ImportError:
 from engine import restore_checkpoint
 from run_study import load_factory,resolve_selected_config
 from spec import confined

def secondary_draws(torch,adapters,query,seed):
 outputs=[];components=[];generator=torch.Generator(device=next(adapters[0].model.parameters()).device).manual_seed(seed+6000011)
 members=4 if len(adapters)==4 or adapters[0].posterior_rows.members==4 else 1
 before=[a.posterior_source.rng_hashes(a.factor_rng) for a in adapters]
 for block in range(3):
  for position in range(4):
   route=position if members==4 else 0;a=adapters[position if len(adapters)==4 else 0];rows=a.posterior_rows;active=0 if len(adapters)==4 or members==1 else position;body=a.model.body if rows.members==4 else a.model
   with torch.no_grad():draw=rows.draw(active,generator)
   with rows.effective({active:draw}),a.source.factors.member_context(body,active),torch.no_grad():
    full=body(a.source.batch['graph'],a.source.batch['x']);z=full[query.to(full.device)]
    if z.shape!=(len(query),10) or not torch.isfinite(z).all():raise ValueError('Complete finite selected source draw required')
    outputs.append(z.detach().cpu());components.append(route)
 for a,b in zip(adapters,before):
  a.posterior_rows.mean_scope_closed()
  if a.posterior_source.rng_hashes(a.factor_rng)!=b:raise ValueError('Secondary read advanced acquisition factor generator')
 return torch.stack(outputs),components

def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 value=primary.export_bank(project_root=project_root,study=study,run=run,output_root=output_root,output_dir=output_dir,registry=registry,device=device)
 config=run['config']
 if not config.get('source_secondary_raw12'):return value
 if config.get('source_secondary_seed')!=run['seed']+6000011 or config.get('fusion')!='mean_probabilities':raise ValueError('Exact source secondary seed/raw law required')
 import torch
 fits=config['fit_parents'];ids={role:torch.load(confined(project_root,config['endpoint_inputs'][role+'_ids']),map_location='cpu',weights_only=True) for role in ('cal','evaluate')};query=torch.cat(list(ids.values()));adapters=[];components=[];outputs=[];generator=torch.Generator(device=device).manual_seed(run['seed']+6000011)
 for identifier in fits:
  row=registry.run(study['study_id'],identifier);actual=resolve_selected_config(study,row['contract'],registry);factory=load_factory(project_root,actual['adapter']);adapter=factory(torch=torch,project_root=project_root,study=study,run=actual,device=device);checkpoint=output_root/study['study_id']/identifier/row['result']['selected']['path'];restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=identifier);adapter.model.eval()
  if adapter.posterior_rows is None:raise ValueError('Gaussian complete selected source fit required')
  adapters.append(adapter)
 raw,components=secondary_draws(torch,adapters,query,run['seed'])
 prob=raw.softmax(-1);comp=torch.tensor(components,dtype=torch.long);offset=0;files={}
 for role,row_ids in ids.items():
  end=offset+len(row_ids);z,p=raw[:,offset:end],prob[:,offset:end];q=output_dir/(role+'_secondary_raw12.pt');torch.save({'ids':row_ids,'draw_raw_logits':z,'draw_probability':p,'draw_component':comp,'component_integrated_probabilities':torch.stack([p[comp==m].mean(0) for m in range(members)]),'all12_probability_mean':p.mean(0),'fixed_four_forward_block_probability_means':p.reshape(3,4,len(row_ids),10).mean(1)},q);files[role]=receipt(q,project_root);offset=end
 value['source_secondary_raw12']={'files':files,'seed':run['seed']+6000011,'native_selected_forward_calls':12,'component_sequence':components,'primary':False,'raw_only':True,'blocks':3,'best_block_selection':False,'calibration':False,'native_dropout':False,'acquisition_rng_unchanged':True,'model_updates':0,'label_targets_read':False}
 del adapters,raw,prob;gc.collect()
 if str(device).startswith('cuda'):torch.cuda.empty_cache()
 return value
