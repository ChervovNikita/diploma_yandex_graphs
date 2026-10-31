"""Source endpoint E VALUE point/Gaussian acquisition under new disabled custody."""
import ast,copy,hashlib,inspect,math,types
from types import SimpleNamespace
from pathlib import Path
from .common_native_wrapper import bound,load_module
from .film_native import attach as attach_point,construct as construct_original
CONTRACT={'schema':'literal-E-Rank1-BNN-WikiCS-minimal-v1','native':'original Model SAGE plus literal endpoint-conditioned E VALUE;native/all-builder FactorLinear retained','gaussian':'all3958 coordinates perroute;immutable constructor centres;rho=log(.1),std=exp(rho),no clamp/floor/annealing;one vector perroute/site/update reused graph-wide','objective':'source mean own sampled CE + average exact conditional Gaussian KL/580;one backward/AdamW WD0','point_controls':'exact endpoint E point4,pointM1,genuine pointI4','selector':'strict-first mean-factor probability-mean SELECT accuracy;cap1000/patience300;no sampled selector','secondary':'12raw selected draw forwards;seed+6000011;three fixed4draw blocks;no dropout,selection or secondary calibration','adaptations':['same literal GRAPH/TRAIN;new SELECT3505/CAL1769 rather than merged historical VALID5274','one public source-compatible mask0 x three literal source seeds','exact pointM1 and qM1 member0 reuse in respective genuineI4','fresh original GNNM SAGE comparator','after-serving-pool CAL500;rawprimary and12rawdraws retained;no source old5foldprepool calibration equivalence'],'historical_state_inherited':False}
METHODS=('seeds','native_body','make','scope','logits','training_outputs','metrics')

def construct(torch,root,B,graph,arm,seed,body,device):
 if arm=='original_GNNM':
  temp={'common':B['common']};return construct_original(torch,root,temp,graph,arm,seed,body,device)+(None,[],None)
 import torch_geometric
 from torch_geometric.nn.conv import SAGEConv,MessagePassing
 if torch_geometric.__version__!='2.7.0':raise ValueError('Pinned source PyG2.7.0 required')
 for key,obj in [('sage_conv',SAGEConv),('message_passing',MessagePassing)]:
  raw=Path(inspect.getfile(obj)).read_bytes()
  if len(raw)!=B[key]['bytes'] or hashlib.sha256(raw).hexdigest()!=B[key]['sha256']:raise ValueError('Installed PyG source differs')
 t=ast.parse(bound(root,B['family']).read_text());f=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Family');methods=[copy.deepcopy(next(n for n in f.body if isinstance(n,ast.FunctionDef) and n.name==k)) for k in METHODS];mod=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='SourceBuild',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]));ns={'math':math};exec(compile(mod,'<literal_endpoint_E_native_Family_AST>','exec'),ns);source=ns['SourceBuild']();source.torch=torch;source.F=torch.nn.functional;source.device=torch.device(device);source.cuda=source.device.index or 0
 source.cfg={'depth':2,'hidden':128,'dropout':.2,'rank':16,'learning_rate':.001,'block_seed_offset':4000037};source.set_seed=load_module(root,B['common']['driver']).set_seed;source.Model=load_module(root,B['common']['native']).Model;source.routes=load_module(root,B['common']['routes']);source.factors=load_module(root,B['common']['factors']);source.value_installer=load_module(root,B['value_installer']);source.value_helper=load_module(root,B['value_helper']);source.edge_metadata={'nodes':11701,'edges':graph['edge_index'].shape[1],'self_edges':int(graph['edge_index'][0].eq(graph['edge_index'][1]).sum())};source._single_adapter=None;source._operation='endpoint'
 posterior=load_module(root,B['posterior']);delta=load_module(root,B['delta']);posterior.bind(source,delta);law='fixed_noise' if arm=='fixed_noise4' else 'learned_q' if arm in ('q4','q_M1') else 'point';members=1 if arm in ('q_M1','point_M1') else 4
 a=ast.parse(bound(root,B['acquire']).read_text());functions=[copy.deepcopy(next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==k)) for k in ('require','gaussian_make','sampled_update')];amod=ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]));primitives={};exec(compile(amod,'<literal_Rank1_acquisition_primitives_AST>','exec'),primitives)
 if law=='point':model,opt,streams=source.make(seed,body,'baseline',True,members,'shared' if members==4 else 'single');rows=None;rng=[]
 else:model,opt,streams,rows,rng=primitives['gaussian_make'](source,posterior,seed,body,members,law)
 rendered={'native_source_method_AST_hashes':{m.name:hashlib.sha256(ast.dump(m,include_attributes=False).encode()).hexdigest() for m in methods},'acquisition_primitive_AST_hashes':{m.name:hashlib.sha256(ast.dump(m,include_attributes=False).encode()).hexdigest() for m in functions},'source_AST_edits':[],'endpoint_VALUE_installed':True}
 return source,model,opt,streams,primitives,rendered,rows,rng,posterior

def attach(torch,source,model,opt,streams,primitives,rendered,rows,rng,posterior,graph,train,select,arm,seed,body,provenance):
 mapped='original_GNNM' if arm=='original_GNNM' else 'native_M1' if arm in ('point_M1','q_M1') else 'shared4';adapter=attach_point(torch,source,model,opt,streams,primitives,rendered,graph,train,select,mapped,seed,body,provenance)
 adapter.posterior_rows=rows;adapter.factor_rng=rng;adapter.posterior_source=posterior;adapter.rank1_arm=arm
 if rows is None:return adapter
 def update(adapter,step):
  value=primitives['sampled_update'](source,model,opt,streams,rows,rng);return {'objective':value['objective'],'ce':value['mean_sampled_own_CE'],'regularization':value['mean_KL_per580'],'objective_components':value,'operation_counts':{'optimizer_updates':1,'backward_calls':1,'model_forwards':rows.members,'route_predictions':rows.members,'supervised_examples':580,'native_graph_block_calls':2*rows.members,'source_factor_vector_draws':rows.members,'source_factor_coordinate_draws':3958*rows.members}}
 adapter.custom_train=update;original_capture=adapter.external_capture;original_restore=adapter.external_restore;original_validate=adapter.validate
 def capture():return {'native':original_capture(),'posterior':rows.checkpoint(),'factor_rng':[g.get_state().clone() for g in rng]}
 def restore(state):
  rows.restore_constants(state['posterior']);original_restore(state['native'])
  if len(state['factor_rng'])!=len(rng):raise ValueError('Exact factor generator roster required')
  for g,v in zip(rng,state['factor_rng']):g.set_state(v)
  rows.mean_scope_closed()
 def validate(step):
  rows.mean_scope_closed();before=posterior.rng_hashes(rng);value=original_validate(step);rows.mean_scope_closed()
  if posterior.rng_hashes(rng)!=before:raise ValueError('Mean selector consumed factor draw')
  return value
 adapter.external_capture=capture;adapter.external_restore=restore;adapter.validate=validate
 return adapter

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config'];arm=c['rank1_arm']
 if study['dataset']!='wikics' or c['rank1_source_contract']!=CONTRACT or arm not in ('point4','point_M1','fixed_noise4','q4','q_M1','original_GNNM'):raise ValueError('Exact finite adopted source condition required')
 from ..native_stopping import validate_native_run
 validate_native_run(run)
 def load(row):return torch.load(bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=[load(c['node_inputs'][k]) for k in ('graph','train','select')]
 if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'} or graph['x'].shape!=(11701,300) or graph['x'].dtype!=torch.float32 or graph['edge_index'].shape!=(2,c['provider_graph']['edges']) or train['ids'].shape!=(580,):raise ValueError('Literal source GRAPH/TRAIN and new SELECT required')
 if torch.isin(train['ids'],select['ids']).any():raise ValueError('Role overlap')
 state=construct(torch,project_root,c['rank1_sources'],graph,arm,run['seed'],run['body'],device)
 return attach(torch,*state,graph,train,select,arm,run['seed'],run['body'],{'actual_model_seed':c['initialization_seed'],'study':study['study_id'],'arm':arm,'source_bindings':c['rank1_sources'],'node_inputs':c['node_inputs'],'contract':CONTRACT})
