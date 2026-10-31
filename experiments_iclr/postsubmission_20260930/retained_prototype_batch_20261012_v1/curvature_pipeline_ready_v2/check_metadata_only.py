"""Actual frozen-v10 plus exact proposed delta ingestion; metadata only."""
import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(P/'controller_delta_v10'),str(ROOT/'master_controller_source_20261012_v10/runtime')]
import compile_master_plan as controller
from run_study import require_curvature_paid_parent,paid_parent_ids

def main():
 entry=json.loads((P/'MASTER_INPUT_ENTRY.json').read_text());packet=controller.load_packet(ROOT,entry)
 c=packet['counts'];t=c['totals'];study=packet['studies'][0]
 assert t['atomic_fit_run_groups']==762 and t['known_registered_optimizer_step_calls']==2112000
 assert c['model_optimizer_step_call_minimum']==2112168 and c['model_optimizer_step_call_maximum']==2113680
 assert c['model_optimizer_step_call_total'] is None and not c['model_optimizer_step_call_total_fully_determined']
 assert c['finite_bounded_trial_groups']==168 and c['epoch_fit_groups_with_unresolved_minibatch_update_count']==0
 assert t['served_banks']==240 and t['reused_exact_member0_fit_references']==30
 assert t['registered_scalar_CAL_updates']==120000 and c['paid_stepped_Adam_starts_minimum']==1116 and c['paid_stepped_Adam_starts_maximum']==2628
 records={r['run_id']:r for r in study['runs']};rejects=0
 cont=next(r for r in study['runs'] if r['config'].get('curvature_stage')=='continuation')
 for key,value in (('members',1),('local_transition_updates',100),('source_optimizer_count',4),('curvature_arm','other'),('is_original_gnnm',True)):
  bad=copy.deepcopy(cont);bad['config'][key]=value
  try:controller.fit_semantics(bad)
  except ValueError:rejects+=1
  else:raise AssertionError('Invalid curvature semantics admitted')
 for stage in ('warm','trials','continuation','cold'):
  bad=copy.deepcopy(next(r for r in study['runs'] if r['config'].get('curvature_stage')==stage));bad['budget']['maximum']+=1
  try:controller.fit_semantics(bad)
  except ValueError:rejects+=1
  else:raise AssertionError('Wrong paid stage budget admitted')
 bad=copy.deepcopy(cont);bad['selection']['checkpoint0']=False
 try:controller.fit_semantics(bad)
 except ValueError:rejects+=1
 else:raise AssertionError('Source epoch0 exception silently removed')
 value={'schema':'actual-v10-plus-curvature-proposed-delta-metadata-ingest-v2','status':'passed',
  'actual_v10_base':json.loads((P/'controller_delta_v10/DELTA_MANIFEST.json').read_text()),
  'packet_plan_hash':packet['plan_hash'],'studies':len(packet['studies']),'run_records':t['run_records'],
  'rejection_cases':rejects,'typed_counts':c,'reference_matches':packet['references'],
  'source_data_arrays_rehashed':False,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False,
  'root_merge_is_pending':True}
 (P/'ACTUAL_CONTROLLER_V10_DELTA_INGEST.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps({'status':'passed','groups':t['atomic_fit_run_groups'],'minimum_maps':c['model_optimizer_step_call_minimum'],'maximum_maps':c['model_optimizer_step_call_maximum'],'rejects':rejects,'scientific_fits':0}))

if __name__=='__main__':main()
