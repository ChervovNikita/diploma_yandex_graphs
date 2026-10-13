"""Read already paid synthetic checkpoints; no constructors, forwards or updates."""
import gzip,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;sys.path.insert(0,str(P/'runtime'))
import torch
torch.set_num_threads(1)
from postfit_semantics import validate_postfit_run
specs=[json.loads(gzip.decompress(q.read_bytes())) for q in sorted((P/'matrix/studies').glob('*.gz'))];records=[]
for family,name,rule in [('direct','direct_source_solver_bridge.pt',None),('affine','affine_delta_bridge.pt',None)]+[('serving','fold_'+rule+'.pt',rule) for rule in ('temperature_global','temperature_member','reliability_self','linear_stacking','reliability_graph')]:
 p=R/'native_postfit_serving_ports_20261013_v1/synthetic_bridge_qa_01'/name;state=torch.load(p,map_location='cpu',weights_only=False);group=state['optimizer_state']['param_groups'][0]
 r=next(r for s in specs for r in s['runs'] if r['kind']=='model_fit' and r['config'].get('postfit_phase')=='head' and r['config'].get('postfit_family')==family and (family!='serving' or r['config'].get('postfit_rule')==rule));validate_postfit_run(r);c=r['config'];assert group['lr']==c['learning_rate']==c['postfit_optimizer']['learning_rate'];assert c['donor_learning_rate']==.001
 if family!='direct':assert tuple(group['betas'])==tuple(c['postfit_optimizer']['betas']) and group['eps']==c['postfit_optimizer']['eps'] and group['weight_decay']==0
 dtype=torch.float32 if family=='affine' else torch.float64;assert all(v.dtype==dtype for v in state['model_state'].values() if torch.is_tensor(v))
 records.append({'family':family,'rule':rule,'paid_synthetic_checkpoint':str(p.relative_to(R)),'actual_optimizer_LR':group['lr'],'reported_head_LR':c['learning_rate'],'donor_LR':c['donor_learning_rate'],'dtype':str(dtype),'source_optimizer':c['postfit_optimizer']})
report={'schema':'postfit-v2-actual-paid-optimizer-reporting-readback-v1','status':'passed','records':records,'new_optimizer_maps':0,'new_model_forwards':0,'closed_kernel_repeats':0,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False};(P/'OPTIMIZER_REPORTING_READBACK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','optimizer_states_read':7,'new_optimizer_maps':0,'GO':False}))
