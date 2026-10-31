"""New eight-view wrapper/controller custody fixtures, not prior unit tests."""
import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(P/'runtime'))
from adapters import single8_attribution as port
def equal(torch,a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(torch,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(torch,x,y) for x,y in zip(a,b))
 if hasattr(a,'shape'):return bool((a==b).all())
 return a==b

def main():
 import torch,gzip
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 predecessor=ROOT/'retained_prototype_batch_20261012_v1/polynormer_eligible_successor_v2'
 plan=json.loads((predecessor/'matrix/PLAN.json').read_text());row=next(r for r in plan['studies'] if 'wikics' in r['path'] and 'mechanisms' in r['path'])
 study=json.loads(gzip.decompress((ROOT/row['path']).read_bytes()))
 c=next(r['config'] for r in study['runs'] if r['kind']=='model_fit' and r['config'].get('mechanism_arm')=='native_M1')
 binding=json.loads((P/'SINGLE8_SOURCE_BINDING.json').read_text())
 n=12;torch.manual_seed(3437);x=torch.randn(n,300);ids=torch.arange(n)
 edge=torch.stack((ids.repeat(2),torch.cat((ids,(ids+1)%n))))
 cases=P/'CPU_cases';cases.mkdir(exist_ok=True);rows=[]
 for control in port.CONTROLS:
  done=cases/(control+'.json')
  if done.exists() and json.loads(done.read_text()).get('status')=='passed':rows.append(json.loads(done.read_text()));continue
  session,details=port.construct(torch,ROOT,sources=c['mechanism_sources'],single8_source=binding,control=control,
   features=300,classes=10,seed=88101,dropout=.5,learning_rate=.001,train_count=10,device='cpu')
  adapter=port.attach(torch,session,details,x=x,edges=edge,train_ids=ids[:10],train_y=ids[:10]%10,
   select_ids=ids[10:],select_y=ids[10:]%10,provenance={'fixture':control,'not_scientific_parent':True})
  adapter.local_updates=1
  updates=[]
  for step in (1,2):
   result=adapter.train_step(step);metrics=adapter.validate(step);adapter.observe_validation(step,metrics)
   assert result['operation_counts']['optimizer_updates']==1 and result['operation_counts']['training_body_forwards']==8
   updates.append(result['operation_counts'])
  assert adapter.transition_completed and session.model.members==1 and len(session.optimizers)==1
  state=copy.deepcopy(adapter.capture_state());assert len(state['external']['training_view_streams'])==4
  assert session.control_counts['Adam_step_calls']==2 and session.control_counts['training_body_forwards']==16
  session.training_view_streams[2]['cpu'].zero_();session.control_counts['training_body_forwards']=0
  adapter.restore_state(state);assert equal(torch,state,adapter.capture_state())
  with torch.no_grad():served=adapter.predict_rows(ids[10:],2)
  assert served.shape==(1,2,10) and torch.isfinite(served).all()
  value={'control':control,'status':'passed','toy_optimizer_maps':2,'source_training_forwards':16,
   'literal_source_one_body_Adam_and_single_serving':True,'four_TRAIN_streams_counts_local_transition_restored':True,
   'operation_counts':updates,'actual_input_arrays_opened':0,'scientific_fits':0,'CUDA_compute':0,'GO':False}
  done.write_text(json.dumps(value,indent=2)+'\n');rows.append(value)
 value={'schema':'new-Single8-source-wrapper-controller-CPU-QA-v1','status':'passed','cases':rows,
  'toy_optimizer_maps':sum(r['toy_optimizer_maps'] for r in rows),'old_unit_private_kernel_checks_rerun':False,
  'actual_input_arrays_opened':0,'scientific_fits':0,'CUDA_compute':0,'GO':False}
 (P/'SINGLE8_PIPELINE_CPU_QA.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(rows),'toy_maps':value['toy_optimizer_maps'],'scientific_fits':0}))

if __name__=='__main__':main()
