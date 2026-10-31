"""Pure metadata composition check on exact v11; no source or numerical tests."""
import importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(P),str(ROOT/'master_controller_source_20261012_v11/runtime')]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
base=module('curvature_actual_v11_base',ROOT/'master_controller_source_20261012_v11/runtime/compile_master_plan.py')
composed=module('curvature_v11_owned_composition',P/'compile_master_plan.py')
entry=json.loads((ROOT/'retained_prototype_batch_20261012_v1/curvature_pipeline_ready_v2/MASTER_INPUT_ENTRY.json').read_text())
packet=composed.load_packet(ROOT,entry);counts=packet['counts']
assert counts['totals']['atomic_fit_run_groups']==762 and counts['model_optimizer_step_call_maximum']==2113680
assert counts['model_optimizer_step_call_minimum']==2112168
nsd=json.loads((ROOT/'native_nsd_five_new_data_packet_20261012_v2/ALLOCATION_HANDOFF.json').read_text())
results=[]
for entry in nsd['entries']:
 a=base.load_packet(ROOT,entry);b=composed.load_packet(ROOT,entry)
 assert a['counts']['totals']==b['counts']['totals'] and a['counts']['by_stage']==b['counts']['by_stage']
 assert all(b['counts'][k]==v for k,v in a['counts'].items())
 assert a['references']==b['references'] and a['plan_hash']==b['plan_hash']
 assert [base.fit_semantics(r) for s in a['studies'] for r in s['runs'] if r['kind']=='model_fit']==[composed.fit_semantics(r) for s in b['studies'] for r in s['runs'] if r['kind']=='model_fit']
 results.append({'id':entry['id'],'model_fit_groups':a['counts']['totals']['atomic_fit_run_groups'],
  'model_optimizer_maps':a['counts']['model_optimizer_step_call_total'],'all_existing_v11_typed_fields_exact':True})
value={'schema':'curvature-exact-v11-composition-metadata-proof-v1','status':'passed',
 'curvature_plan_hash':packet['plan_hash'],'curvature_typed_counts':counts,'NSD_metadata_preserved':results,
 'NSD_numerical_QA_rerun':False,'source_CPU_QA_rerun':False,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False}
(P/'V11_COMPOSITION_METADATA_QA.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':'passed','curvature_groups':762,'curvature_maximum_maps':2113680,
 'NSD_groups_preserved':sum(r['model_fit_groups'] for r in results),'NSD_maps_preserved':sum(r['model_optimizer_maps'] for r in results),'scientific_fits':0}))
