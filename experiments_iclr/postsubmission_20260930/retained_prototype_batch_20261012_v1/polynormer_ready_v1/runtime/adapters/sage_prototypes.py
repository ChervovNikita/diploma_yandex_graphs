"""Source-faithful SAGE path and equal-slow-start joint prototypes; lazy numerics."""
import ast,copy,hashlib,math
from contextlib import nullcontext
from types import SimpleNamespace
from . import common_native_wrapper as common
from .classifier import ClassifierAdapter
PINS={
 'matched':'SAGE_matched_joint_reference_77_source_20261010_v1/run_family.py',
 'ordered_native':'shared_ordered_path_operator_source_20261011_v1/ordered_path_native.py',
 'ordered_operator':'shared_ordered_path_operator_source_20261011_v1/ordered_path_operator.py',
 'ordered_sampler':'shared_ordered_path_operator_source_20261011_v1/ordered_path_sampler.py'}
SHA={'matched':'5ea88c5eba68df512c20ea10ee778d25f46499705059a91cb80104c4fc8943c4','ordered_native':'e04c19d01df1e54d0432f684510d6720b00a5cc3db1f71d47e78ea92aad1bd27','ordered_operator':'02cd78c31c68e69aaa910ea627c8e16143c18d7315f692d6c2c1512d3ed89cd4','ordered_sampler':'780c292f1d30a1e928252fd2c857696292869581c1a443982e25eb1d868b71b2'}
ARMS=('matched_joint_untied4','ordered_path_M1','ordered_path_shared4','depth3_M1','depth3_shared4')
I4={'ordered_path_I4':'ordered_path_M1','depth3_I4':'depth3_M1'}
CONTRACT={'schema':'SAGE-retained-source-prototype-port-v1','source_exact_recipe':False,
 'matched_joint':'literal native_body(seed,0,row,True,1); same initial native slow values with disjoint parameter storage, row-specific first-r/dropout; one AdamW and cohort mean-own CE/probability pool',
 'ordered':'literal SAGE depth2 plus ordered four-position path map on8 fixed incoming nonself-record three-edge walks; exact native nonlinear stem reused once, residual before original final normalization; second W/b zero, extra fast vectors one; no source teacher/loss/dropout added',
 'source_protection':'SAGE-only hard source checks; depth3 matched radius control retained; no old K/source quality lock or scores invoked',
 'prospective_initialization':'source interface leaves native/factor init root-owned: bind exact common native seed/body stride and first-r Rademacher; path CPU generator native_seed+6000017, fork_rng restores ambient stream; declared new-source design binding',
 'controller':'complete new WikiCS20/Photo10 GRAPH/TRAIN/SELECT/CAL/evaluate; full5000 strict-first every-update SELECT accuracy replaces historical joint1000/patience300; ordered source had no fit budget',
 'HPO':'nine LR/dropout anchor neighbors around.001/.2, same paired common-v2 seeds; I4 component M1 HPO then four distinct complete learners with proven same M1 body0 checkpoint reference',
 'controls':'import exact planned common-v2 source kernels, no old outcomes/selected states; literal original R/S/B GNNM mean-logit law retained; new exact-kernel replacement mapping required before master admission'}

def render_matched(root,b,features,classes):
 path=common.bound(root,b['matched']);assert b['matched']['sha256']==SHA['matched'];tree=ast.parse(path.read_text());family=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Family');m={n.name:n for n in family.body if isinstance(n,ast.FunctionDef)}
 functions=[copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='wrap_joint_untied'))]
 names=('seeds','native_body','make','scope','logits','training_outputs','within_route_supcon','probability_pool_ce');methods=[copy.deepcopy(m[k]) for k in names];changes=[]
 class Bind(ast.NodeTransformer):
  def visit_Call(self,node):
   self.generic_visit(node)
   if isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='self.Model':
    assert node.args[2].value==300 and node.args[4].value==10;node.args[2].value=features;node.args[4].value=classes;changes.extend(['input300->features','output10->classes'])
   if isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='torch.randint':
    assert ast.unparse(node.args[2])=='(1, 300)';node.args[2].elts[1].value=features;changes.append('first_r_shape300->features')
   return node
 methods=[Bind().visit(n) for n in methods];assert changes==['input300->features','output10->classes','first_r_shape300->features']
 loop=next(n for n in ast.walk(m['fit']) if isinstance(n,ast.For) and ast.unparse(n.target)=='step');stop=next(i for i,n in enumerate(loop.body) if ast.unparse(n)=='model.eval()')
 train=ast.FunctionDef(name='train_source',args=ast.parse('def f(self,model,optimizer,streams,bank,members,objective="own_only",use_supcon=False):pass').body[0].args,
  body=[copy.deepcopy(m['fit'].body[0])]+copy.deepcopy(loop.body[:stop])+ast.parse('return train_loss').body,decorator_list=[])
 code=ast.fix_missing_locations(ast.Module(body=functions+[ast.ClassDef(name='MatchedSource',bases=[],keywords=[],body=methods+[train],decorator_list=[])],type_ignores=[]))
 return {'code':ast.unparse(code),'provider_edits':changes,'method_AST':{name:common.ast_hash(m[name]) for name in names},'TRAIN_loop_AST':common.ast_hash(ast.Module(body=loop.body[:stop],type_ignores=[]))}

def operator_provider(root, binding):
    """Same source Kaiming draw on Torch2.1, whose init helper lacks generator."""
    module=common.load_module(root,binding);tree=ast.parse(common.bound(root,binding).read_text());function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_path_block'));count=0
    for node in ast.walk(function):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.nn.init.kaiming_uniform_':
            assert ast.unparse(node)=='torch.nn.init.kaiming_uniform_(first.weight, a=math.sqrt(5), generator=generator)'
            # Exact nn.init native bound: sqrt(3)*gain/sqrt(fan_in), gain=sqrt(2/(1+a*a)).
            replacement=ast.parse('first.weight.uniform_(-math.sqrt(3.0)*math.sqrt(2.0/(1+math.sqrt(5)**2))/math.sqrt(4*width), math.sqrt(3.0)*math.sqrt(2.0/(1+math.sqrt(5)**2))/math.sqrt(4*width), generator=generator)').body[0].value
            node.func,node.args,node.keywords=replacement.func,replacement.args,replacement.keywords;count+=1
    assert count==1
    compiled=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));namespace=dict(vars(module));exec(compile(compiled,'<source_Kaiming_generator_API_binding>','exec'),namespace)
    result=SimpleNamespace(**vars(module));result.make_path_block=namespace['make_path_block'];result.generator_API_binding={'source_sha256':binding['sha256'],'bound_AST_sha256':common.ast_hash(compiled),'edit':'nn.init helper to its exact Tensor.uniform_ primitive; same native gain/fan bound, generator and draw shape; no learned law change'}
    return result

def construct(torch,root,common_bindings,prototype_bindings,*,arm,features,classes,seed,body,dropout,learning_rate,device):
 if arm not in ARMS:raise ValueError('Explicit SAGE prototype arm required')
 rendered=common.render_sources(root,common_bindings,features,classes);primitives={'F':torch.nn.functional,'TABM_INITS':4}
 exec(compile(rendered['build'],'<literal_common_source_context>','exec'),primitives);exec(compile(rendered['train'],'<literal_common_TRAIN>','exec'),primitives)
 base=primitives['SourceBuild']();base.torch,base.F=torch,torch.nn.functional;base.device=torch.device(device);base.cuda=base.device.index or 0
 base.cfg={'backbone':'SAGE','depth':2,'hidden':128,'rank':16,'dropout':dropout,'learning_rate':learning_rate}
 base.Model=common.load_module(root,common_bindings['native']).Model;base.set_seed=common.load_module(root,common_bindings['driver']).set_seed
 base.routes=common.load_module(root,common_bindings['routes']);base.factors=common.load_module(root,common_bindings['factors'])
 # Source context extraction constructs no model/optimizer or extra RNG draw.
 if arm=='matched_joint_untied4':
  bound=render_matched(root,prototype_bindings,features,classes);namespace={'nullcontext':nullcontext,'math':math};exec(compile(bound['code'],'<literal_matched_joint_source>','exec'),namespace)
  source=namespace['MatchedSource']();source.__dict__.update(base.__dict__);source.device=torch.device(device);source.cuda=source.device.index or 0
  model,opt,streams=source.make(seed,0,'baseline',True,4,architecture='joint_untied')
  return source,model,opt,streams,{'source':bound,'train':lambda:source.train_source(model,opt,streams,True,4)}
 source=base;source.device=torch.device(device);source.cuda=source.device.index or 0;source.set_seed(source.seeds(seed,body)[0])
 native=common.load_module(root,prototype_bindings['ordered_native']);operator=operator_provider(root,prototype_bindings['ordered_operator']);models=common.load_module(root,common_bindings['native'])
 members=4 if arm.endswith('shared4') else 1
 if arm.startswith('ordered_path'):
  model=native.build_path_shared(torch,models,source.factors,source.routes,operator.make_path_block,feature_dim=features,classes=classes,
      init_seed=source.seeds(seed,body)[0]+6000017,mode='ordered',width=128,dropout=dropout,members=members)
 else:
  model=native.build_depth3_shared(torch,models,source.factors,source.routes,feature_dim=features,classes=classes,width=128,dropout=dropout,members=members)
 with torch.no_grad():
  for row in range(members):
   generator=torch.Generator(device='cpu').manual_seed(source.seeds(seed,body+row)[1]);signs=torch.randint(0,2,(1,features),generator=generator)*2-1;model.body.input_linear.r[row].copy_(signs[0].to(model.body.input_linear.r))
 model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.);model.check_optimizer_ownership(opt)
 streams=[]
 for row in range(members):
  source.set_seed(source.seeds(seed,body+row)[2]);streams.append({'cpu':torch.get_rng_state(),**({'cuda':torch.cuda.get_rng_state(source.device)} if source.device.type=='cuda' else {})})
 return source,model,opt,streams,{'source':rendered,'train':lambda:primitives['packed_train_step'](source,model,opt,streams,members)}


def attach(torch,source,model,opt,streams,extras,*,x,edges,train_ids,train_y,select_ids,select_y,arm,walks,swap,provenance):
 labels=torch.zeros(len(x),dtype=torch.long,device=x.device);mask=torch.zeros(len(x),dtype=torch.bool,device=x.device);labels[train_ids]=train_y;mask[train_ids]=True
 source.data=SimpleNamespace(x=x,edge_index=edges,y=labels,train_mask=mask);source.train_ids=train_ids
 source.batch={'x':x,'edge_index':edges,'graph':SimpleNamespace(edge_index=edges)}
 if arm.startswith('ordered_path'):source.batch.update(walk_indices=walks,walk_swap_mask=swap)
 def logits(current,training,step):return current(source.batch,route_scope=lambda member:source.scope(streams,member))[0]
 def train(adapter,step):
  loss=extras['train']();members=4 if arm=='matched_joint_untied4' or arm.endswith('shared4') else 1
  counts={'optimizer_updates':1,'backward_calls':1,'fullgraph_native_member_trajectories':members,'native_graph_block_calls':(3 if arm.startswith('depth3') else 2)*members,'physical_slow_bodies':4 if arm=='matched_joint_untied4' else 1}
  if arm.startswith('ordered_path'):counts.update(path_walk_slots_gathered=members*len(x)*8*4,path_dense_MACs=members*len(x)*(8*4+1)*128**2)
  return {'objective':loss,'ce':loss,'regularization':0.,'operation_counts':counts}
 adapter=ClassifierAdapter(torch,torch.nn.functional,model,opt,logits_fn=logits,train_ids=train_ids,train_y=train_y,select_ids=select_ids,select_y=select_y,regularizer=lambda m:x.new_zeros(()),fusion='mean_probabilities',
  selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5000,'save_every':1},provenance=provenance,custom_train=train,
  external_capture=lambda:{'streams':copy.deepcopy(streams)},external_restore=lambda value:streams.__setitem__(slice(None),copy.deepcopy(value['streams'])))
 adapter.source=source;adapter.streams=streams;return adapter


def make_adapter(*,torch,project_root,study,run,device):
 c=run['config'];arm=c['prototype_arm'];assert study['dataset'] in common.DATASETS and c['backbone']=='SAGE'
 def read(row):return torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=(read(c['node_inputs'][k]) for k in ('graph','train','select'));x,edges=graph['x'].to(device),graph['edge_index'].to(device)
 metadata=c['provider_graph'];assert x.dtype==torch.float32 and tuple(x.shape)==(metadata['nodes'],metadata['features']) and edges.dtype==torch.long and tuple(edges.shape)==(2,metadata['edges'])
 for role in (train,select):
  ids,y=role['ids'],role['y'];assert ids.dtype==y.dtype==torch.long and ids.ndim==1 and y.shape==ids.shape and len(ids)>0 and not (ids.min()<0 or ids.max()>=len(x) or y.min()<0 or y.max()>=c['classes']) and len(torch.unique(ids))==len(ids) and torch.equal(ids,ids.sort().values)
 assert not torch.isin(train['ids'],select['ids']).any()
 source,model,opt,streams,extras=construct(torch,project_root,c['common_sources'],c['prototype_sources'],arm=arm,features=x.shape[1],classes=c['classes'],seed=run['seed'],body=run['body'],dropout=c['dropout'],learning_rate=c['learning_rate'],device=device)
 walks=swap=None
 if arm.startswith('ordered_path'):
  import gzip,json
  artifact=json.loads(gzip.decompress(common.bound(project_root,c['walk_metadata']).read_bytes()));walks=torch.tensor(artifact['indices'],dtype=torch.long,device=device);swap=torch.tensor(artifact['swap_mask'],dtype=torch.bool,device=device)
  common.load_module(project_root,c['prototype_sources']['ordered_operator']).validate_walk_tensors(torch,len(x),walks,swap,x.device)
 return attach(torch,source,model,opt,streams,extras,x=x,edges=edges,train_ids=train['ids'].to(device),train_y=train['y'].to(device),select_ids=select['ids'].to(device),select_y=select['y'].to(device),
  arm=arm,walks=walks,swap=swap,provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'actual_model_seed':source.seeds(run['seed'],run['body'])[0],'source_contract':CONTRACT})
