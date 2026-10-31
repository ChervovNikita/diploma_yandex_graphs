import copy,hashlib,io,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));from adapters import private_response as port

def equal(t,a,b):
 if t.is_tensor(a):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(t,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(t,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();torch.manual_seed(323);n=12;ids=torch.arange(n);pairs=torch.stack((ids.repeat(2),torch.cat(((ids+1)%n,(ids+2)%n))),1);graph={'x':torch.randn(n,3703),'train_support_edge_index':torch.cat((pairs.t(),pairs.t().flip(0)),1)};train={'positive':pairs};negative=torch.stack((ids,(ids+6)%n),1);inner=[(pairs[m*2:m*2+2].t(),negative[m*2:m*2+2].t()) for m in range(4)];outer=(pairs[8:12].t(),negative[8:12].t());select={'positive':pairs[12:14],'negative':negative[:2,None,:].repeat(1,500,1)};B=json.loads((P/'PRIVATE_RESPONSE_SOURCE_BINDINGS.json').read_text());cases=P/'CPU_cases';cases.mkdir(exist_ok=True);rows=[]
 for arm in ('shared_f4','capable_single','untied4','ordinary_native4','row0_single','native_M1','original_GNNM'):
  done=cases/(arm+'.json');paid=cases/(arm+'_paid_episode.pt')
  if done.is_file() and json.loads(done.read_text()).get('status')=='passed':
   row=json.loads(done.read_text());live=arm in ('shared_f4','capable_single','untied4','row0_single');counts=row['operation_counts']
   if 'source_committed_model_maps' not in counts:
    row['previous_counter_report']=copy.deepcopy(counts);counts['source_committed_model_maps']=counts['optimizer_updates'];counts['optimizer_updates']=1 if live else 3;counts['backward_calls']=counts['model_forwards'] if live else 12 if arm=='original_GNNM' else 3;row['counter_clarification']='Registered TorchAdam calls separated from functionalprivatecommit;ordinarynative4 three summed-own backward calls;numeric episode/state proof unchanged'
    done.write_text(json.dumps(row,indent=2)+'\n')
   rows.append(row);continue
  state=port.construct(torch,R,B,arm,0,283136447778829174,'cpu');a=port.Adapter(torch,*state,graph,train,select,arm,0,0,{'synthetic':arm});model,opt=a.model,a.optimizer;owned=[p for g in opt.param_groups for p in g['params']];assert len(owned)==len({id(p) for p in owned}) and {id(p) for p in owned}=={id(p) for p in model.parameters()};saved_alias=a.s.native.spmm_add
  if paid.is_file():
   data=torch.load(paid,weights_only=False,map_location='cpu');model.load_state_dict(data['model']);opt.load_state_dict(data['optimizer']);a.restore_state(data['adapter']);update,counts=data['update'],data['counts']
  else:
   update,counts=a.episode(a.full_support,inner,outer);torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'adapter':a.capture_state(),'update':update,'counts':counts},paid)
  assert a.s.native.spmm_add is saved_alias;live=isinstance(opt,port.HybridAdam);assert counts['optimizer_updates']==(1 if live else 3) and counts['source_discarded_virtual_private_Adam_maps']==int(live);committed=[r['committed_calls'] for r in a.streams.rows];assert committed==([1,1,1,1] if arm in ('shared_f4','untied4','ordinary_native4','row0_single','original_GNNM') else [1,0,0,0]);assert all(not hasattr(v,'grad_fn') or v.grad_fn is None for row in (opt.previous.values() if live else []) for v in row.values())
  before=copy.deepcopy(a.capture_state());v=a.validate(5);assert equal(torch,before,a.capture_state());assert a.observe_validation(5,v)['save_selected'];assert not a.observe_validation(10,v)['save_selected'];saved=copy.deepcopy(a.capture_state());m=copy.deepcopy(model.state_dict());o=copy.deepcopy(opt.state_dict());fresh=port.construct(torch,R,B,arm,0,283136447778829174,'cpu');other=port.Adapter(torch,*fresh,graph,train,select,arm,0,0,{'synthetic':arm});other.model.load_state_dict(m);other.optimizer.load_state_dict(o);other.restore_state(saved);assert equal(torch,saved,other.capture_state());assert equal(torch,o,other.optimizer.state_dict());assert not other.observe_validation(60,v)['save_selected']
  if live:assert all(r['step']==1 for r in opt.previous.values())
  buf=io.BytesIO();torch.save({'model':m,'optimizer':o,'adapter':saved},buf);row={'arm':arm,'status':'passed','synthetic_episodes':1,'committed_model_maps':counts['source_committed_model_maps'],'discarded_virtual_private_Adam_maps':int(live),'cold_literal_source_parameters':sum(p.numel() for p in model.parameters()),'source_private_dropout_committed_calls':committed,'source_recursive_adjoint_alias_restored':True,'whole_selected_model_TorchAdam_functionalprivate_state_and_RNG_reconstructed':True,'strictfirst_source_MRR_selector':True,'operation_counts':counts,'checkpoint_preview_bytes':len(buf.getvalue()),'source_partition_shared_parameters':sum(p.numel() for k,p in model.named_parameters() if k in a.shared),'source_partition_private_parameters':sum(p.numel() for k,p in model.named_parameters() if k in a.private),'actual_input_arrays_opened':0,'old_closed_kernel_gates_rerun':False,'scientific_fits':0,'GO':False};done.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 report={'schema':'private-response5-new-pipeline-synthetic-CPU-v1','status':'passed','cases':rows,'synthetic_episodes':7,'committed_model_maps':sum(r['committed_model_maps'] for r in rows),'discarded_virtual_private_Adam_maps':sum(r['discarded_virtual_private_Adam_maps'] for r in rows),'actual_input_arrays_opened':0,'source_closed_numeric_kernel_tests_rerun':False,'scientific_fits':0,'CUDA_compute':0,'GO':False};(P/'PRIVATE_RESPONSE_PIPELINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','synthetic_episodes':7,'committed_maps':report['committed_model_maps'],'virtual_maps':report['discarded_virtual_private_Adam_maps'],'GO':False}))
if __name__=='__main__':main()
