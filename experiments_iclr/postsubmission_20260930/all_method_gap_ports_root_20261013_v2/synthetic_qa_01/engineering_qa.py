"""Source adapter restore checks on explicitly synthetic CPU tensor fixtures."""
import ast,copy,gzip,json,sys,types
from pathlib import Path

ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);STAGE=ROOT/'all_method_gap_ports_root_20261013_v2'
sys.path.insert(0,str(ROOT/'master_controller_source_20261013_v23/runtime'))
import run_study,engine
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
actual_load=run_study.load_factory

def receipt(p):
 import hashlib
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def geometry_function(factory,changes):
 """Change only fixed synthetic node/count guards in the source factory."""
 import inspect,textwrap
 tree=ast.parse(textwrap.dedent(inspect.getsource(factory)))
 edits=[]
 class Provider(ast.NodeTransformer):
  def visit_Constant(self,n):
   if type(n.value) is int and n.value in changes:
    edits.append(n.value);return ast.copy_location(ast.Constant(changes[n.value]),n)
   return n
 tree=ast.fix_missing_locations(Provider().visit(tree));space=dict(factory.__globals__)
 exec(compile(tree,'<synthetic_guard_and_count_binding_only>','exec'),space)
 return space[factory.__name__],edits

def source_fixture(root,entry):
 source=actual_load(root,entry)
 if entry.endswith('/private_hop_port.py:source_environment'):
  space=dict(source.__globals__)
  fixture=types.FunctionType(source.__code__,space,source.__name__,(12,),source.__closure__)
  space['source_environment']=fixture
  base,edits=geometry_function(space['make_adapter'],{19717:12})
  base.__globals__['source_environment']=fixture;space['make_adapter']=base
  return fixture
 return source

def fixture_run(dataset,n,features,classes,train_count,original):
 run=copy.deepcopy(original);q=OUT/dataset;q.mkdir(exist_ok=True)
 torch.manual_seed(11677);x=torch.randn(n,features);ids=torch.arange(n)
 graph={'x':x,'edge_index':torch.stack([ids,torch.roll(ids,1)])}
 values={'graph':graph,'train':{'ids':ids[:train_count],'y':ids[:train_count]%classes},'select':{'ids':ids[train_count:],'y':ids[train_count:]%classes}}
 for k,v in values.items():
  path=q/(k+'.pt')
  if not path.exists():torch.save(v,path)
  run['config']['node_inputs'][k]=receipt(path)
 return run

def snapshot(adapter):
 return engine._clone_tree(torch,{'model':adapter.model.state_dict(),'optimizer':adapter.optimizer.state_dict(),'adapter':adapter.capture_state()})

def restore(adapter,value):
 adapter.model.load_state_dict(value['model']);adapter.optimizer.load_state_dict(value['optimizer']);adapter.restore_state(value['adapter'])

def exercise(adapter,name,transition=False):
 before=snapshot(adapter);first=adapter.train_step(1);after=snapshot(adapter)
 restore(adapter,before);replay=adapter.train_step(1)
 assert equal(first,replay) and equal(after,snapshot(adapter)),name+' exact Adam/stream replay'
 assert first['operation_counts']['optimizer_updates']==1
 if transition:
  assert first['operation_counts']['output_cotangent_collections']==2
  assert first['operation_counts']['member_reverse_collections']==16
  adapter.observe_validation(1,{'selection_metric':.5})
  adapter.local_updates=1;adapter.train_step(2)
  assert adapter.transition_completed and adapter.session.model.models[0].body._global
  state=snapshot(adapter);adapter.session.steps=-1;adapter.session.model.set_global(False);restore(adapter,state)
  assert adapter.session.model.models[0].body._global and equal(state,snapshot(adapter))
 else:
  adapter.observe_validation(1,{'selection_metric':.5});decision=adapter.observe_validation(251,{'selection_metric':.5})
  assert decision['stop_requested'] and decision['stale_steps']==250
 row={'case':name,'exact_model_Adam_rng_restore':True,'plain_controller_clone':True,'source_local_handoff':transition,'source250_selector_stop':not transition,'synthetic_optimizer_maps':3 if transition else 2}
 print(json.dumps({'case':row}),flush=True);return row

def studies(name):
 p=json.loads((STAGE/name/'PLAN.json').read_text())
 return [json.loads(gzip.decompress((ROOT/r['path']).read_bytes())) for r in p['studies']]

rows=[]
relation=studies('relation_successor')[0]
for policy in ('alphaF','allJ','phiJ','relationJ'):
 original=next(r for r in relation['runs'] if r['kind']=='model_fit' and r['config'].get('configured_relation_policy')==policy)
 run=fixture_run('synthetic_wikics',24,300,10,20,original)
 actual=actual_load(ROOT,run['adapter']);factory,edits=geometry_function(actual,{11701:24,580:20,3505:4})
 assert set(edits)=={11701,580,3505}
 adapter=factory(torch=torch,project_root=ROOT,study=relation,run=run,device='cpu')
 assert adapter.provenance['source_contract']==actual.__globals__['CONTRACT']
 assert adapter.provenance['bindings']['source_recipe']['training']['epochs']==1100
 rows.append(exercise(adapter,'copied_relation_'+policy,True));del adapter

hop=studies('complete_hop_packet')[0]
run_study.load_factory=source_fixture
for arm in ('shared4_own','private_missinghop','full_aux','common_nonfull','allblock_missinghop','factor1_allview','native_M1','original_GNNM'):
 original=next(r for r in hop['runs'] if r['kind']=='model_fit' and r['config']['complete_hop_arm']==arm)
 run=fixture_run('synthetic_pubmed',12,500,3,8,original);factory,edits=geometry_function(actual_load(ROOT,run['adapter']),{19717:12});assert edits==[19717]
 adapter=factory(torch=torch,project_root=ROOT,study=hop,run=run,device='cpu');rows.append(exercise(adapter,'complete_hop_'+arm));del adapter
run_study.load_factory=actual_load

# Reuse the source eight-view and ordinary-mechanism constructors while checking
# the new wrapper through its real make_adapter path on synthetic role files.
for kind in ('single8_control','mechanism_arm'):
 s=next(s for s in studies('plain_state_exclusive_successor') if any(r['kind']=='model_fit' and kind in r['config'] for r in s['runs']))
 original=next(r for r in s['runs'] if r['kind']=='model_fit' and kind in r['config'])
 run=fixture_run('synthetic_plain_wrapper',24,300,10,20,original)
 c=run['config'];c['provider_graph'].update(nodes=24,features=300,edges=24)
 adapter=actual_load(ROOT,run['adapter'])(torch=torch,project_root=ROOT,study=s,run=run,device='cpu')
 before=snapshot(adapter);value=adapter.train_step(1);after=snapshot(adapter);restore(adapter,before);again=adapter.train_step(1)
 assert equal(value,again) and equal(after,snapshot(adapter))
 rows.append({'case':'plain_state_'+kind,'actual_wrapper_make_adapter':True,'exact_model_Adam_rng_restore':True,'plain_controller_clone':True,'synthetic_optimizer_maps':2});del adapter
assert len(rows)==14 and not torch.cuda.is_initialized()
result={'status':'passed','cases':rows,'synthetic_optimizer_maps':sum(r['synthetic_optimizer_maps'] for r in rows),'scientific_fits':0,'benchmark_tensor_decodes':0,'heldout_access':False,'CUDA':False,'scope':'Source guards/counts explicitly reduced for synthetic fixture only; native learned geometry and objectives retained; engineering states never donors','GO':False}
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
