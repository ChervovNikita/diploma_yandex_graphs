"""Pinned fixed-context-key SAGE mechanisms on explicit standardized providers.

No numerical import occurs here. The source target algebra is reused; only
its declared row/key/hidden shape constants are relocated. This is neither
the archived 580-row/1000-update recipe nor a replacement for other families.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
import types
from types import SimpleNamespace

try:
    from .classifier import ClassifierAdapter
    from ..engine import AdapterStepError
    from ..spec import ORIGINAL_PAPER_DATASETS,normalized_id,fingerprint
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from engine import AdapterStepError
    from spec import ORIGINAL_PAPER_DATASETS,normalized_id,fingerprint

SOURCE_PINS={
    'native':'07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
    'routes':'84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773',
    'factors':'9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3',
    'objective':'4e21ed2968e2f9c3fb97aa17bdf8370c6a52f9f9338cd6d760492ca235446944',
    'driver':'b24256e07af818f88d358b55f6cbd58cc61a8ab4b3c4042002fbeb5d16dc3c6e',
}
ARMS=('ordinaryM1_key_mixture','factorizedM1_key_mixture','genuinefactorizedI4_key_mixture',
    'genuinefactorizedI4_key_assigned','shared4_key_private','shared4_key_mixture','shared4_key_uniform',
    'shared4_projected_peer_private','shared4_key_moment_private','shared4_key_shuffled_private')
MODES=dict(zip(ARMS,('key_mixture','key_mixture','key_mixture','key_assigned','key_private',
    'key_mixture','key_uniform','projected_peer_private','key_moment_private','key_shuffled_private')))
REFERENCES=('reference_ordinary_M1','reference_factorized_M1','reference_original_GNNM')
CONTRACT={'source_arms':ARMS,'native_backbone':'SAGE','source_horizon':1000,'standardized_horizon':5000,
    'target_policy':'continuous_all_classmate_full_denominator_v1',
    'key_policy':'fixed_normalized_graph_context_keys_unit_logZ_v1','coefficient':.05,'temperature':.2,
    'query_map':'one bare trainable no-bias hidden->feature map per optimizer body; shared across shared routes',
    'query_initialization':'private CPU float64 QR/sign then float32; same paired-seed initial bytes across arms/bodies',
    'targets':'detached factual fixed keys; literal float32 Q@K moments without normalization',
    'denominator':'all nonself TRAIN identities, literal unit logZ coefficient',
    'peer':'same-route live projected query on both sides, both gradients',
    'shuffle':'class-preserving permutation on both Q axes only; keys stay factual',
    'row_order':'shared4 input TRAIN bank; M1/I4 stable sorted boolean-mask TRAIN order',
    'gradient_permissions':'auxiliary trains query map and live pre-head native trunk/factors; fixed targets receive none',
    'serving':'source arms probability mean; original GNNM reference raw-logit mean; query map never served',
    'shape_adaptation':'TRAIN rows and key/hidden constants relocated; provider classes passed to native constructor; key width>=hidden width',
    'archived_recipe_replication':False}


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(root,row):
    raw=Path(row['path']);root=Path(root).resolve();path=root/raw
    if raw.is_absolute() or '..' in raw.parts or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):raise ValueError('Confine immutable context source/input')
    if path.stat().st_size!=row['bytes'] or file_sha(path)!=row['sha256']:raise ValueError('Context source/input receipt changed')
    return path


def source_module(root,row,name):
    path=bound(root,row);identifier='context_keys_'+name+'_'+row['sha256'][:20]+'_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if identifier not in sys.modules:
        spec=importlib.util.spec_from_file_location(identifier,path);module=importlib.util.module_from_spec(spec)
        sys.modules[identifier]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(identifier,None);raise
    return sys.modules[identifier]


def relocate_objective(module,rows,features,width):
    """Keep every pinned equation and replace only audited dimension literals."""
    if min(rows,width)<2 or features<width:raise ValueError('At least two TRAIN rows; key width must cover hidden QR columns')
    namespace=dict(module.__dict__,TRAIN_ROWS=rows,QUERY_WIDTH=features,HIDDEN_WIDTH=width)
    make=types.FunctionType(module.make_query_map.__code__,namespace,module.make_query_map.__name__)
    tree=ast.parse(Path(module.__file__).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='context_auxiliary')
    class Shapes(ast.NodeTransformer):
        def visit_Constant(self,node):
            names={580:'TRAIN_ROWS',128:'HIDDEN_WIDTH',300:'QUERY_WIDTH'}
            return ast.copy_location(ast.Name(id=names[node.value],ctx=ast.Load()),node) if type(node.value)==int and node.value in names else node
    adapted=ast.fix_missing_locations(ast.Module(body=[Shapes().visit(function)],type_ignores=[]))
    exec(compile(adapted,module.__file__+'::explicit_shape_adaptation','exec'),namespace)
    return SimpleNamespace(prepare_targets=module.prepare_targets,tensor_bank=module.tensor_bank,
        make_query_map=make,context_auxiliary=namespace['context_auxiliary'],COEFFICIENT=module.COEFFICIENT)


def rng_state(torch,np):
    name,keys,position,gauss,cached=np.random.get_state()
    return {'python':random.getstate(),'numpy':(name,keys.tolist(),position,gauss,cached),
        'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}


def restore_rng(torch,np,state):
    random.setstate(state['python']);name,keys,position,gauss,cached=state['numpy']
    np.random.set_state((name,np.asarray(keys,dtype='uint32'),position,gauss,cached));torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


def validate_config(study,run):
    config=copy.deepcopy(run['config']);arm=config['context_key_arm']
    if arm not in ARMS+REFERENCES or config['backbone']!='SAGE':raise ValueError('Exact named fixed-key SAGE arm required')
    if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS:raise ValueError('Original paper datasets are excluded')
    if config['target_law']!='multiclass_ce' or config['classes']<2:raise ValueError('This port supports multiclass only')
    if config['width']!=128 or config['depth']!=2 or config['weight_decay']!=0 or config['context_coefficient']!=.05:
        raise ValueError('Fixed width128/depth2/WD0/auxiliary.05 required')
    if config['learning_rate'] not in (.0005,.001,.002) or config['dropout'] not in (.1,.2,.4):raise ValueError('Frozen nine-point LR/dropout domain required')
    expected=4 if arm.startswith('shared4_') or arm=='reference_original_GNNM' else 1
    if config['members']!=expected:raise ValueError('Native source route ownership changed')
    if set(config['node_inputs'])!={'graph','train','select'}:raise ValueError('Fit opens GRAPH/TRAIN/SELECT only')
    for role in ('train','select'):
        if study['data']['roles'][role][str(run['split'])]!=config['node_inputs'][role]:raise ValueError('Context fit role differs from the declared '+role+' authority')
    if set(config['context_key_sources'])!=set(SOURCE_PINS):raise ValueError('Bind every exact native/route/factor/objective/driver source')
    for name,row in config['context_key_sources'].items():
        if SOURCE_PINS[name] and row['sha256']!=SOURCE_PINS[name]:raise ValueError('Pinned context-key source differs: '+name)
    if config['initialization_seed']!=config['paired_seed']+1000003*run['body']:raise ValueError('Own paid member seed ancestry changed')
    if arm=='genuinefactorizedI4_key_assigned' and run['body'] not in range(4):raise ValueError('Assigned context is exactly own body0..3')
    if run['budget']['unit']!='optimizer_updates' or run['budget']['policy']!='full_budget':raise ValueError('Full optimizer-update horizon required')
    if study['stage']!='engineering' and run['budget']['maximum']!=5000:raise ValueError('Standardized full5000 budget required')
    if run['selection']['metric']!='accuracy' or run['selection']['direction']!='max' or run['selection']['ties']!='first':raise ValueError('Own strict earliest SELECT accuracy selector required')
    return config


def tensor_hash(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ContextKeysAdapter(ClassifierAdapter):
    def train_step(self,step):
        counts={'optimizer_updates':0,'optimizer_step_attempts':0,'backward_attempts':0,'backward_calls':0,
            'model_forwards':0,'route_predictions':0,'supervised_examples':0}
        try:
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            logits=self.logits_fn(self.model,True,step);selected=logits[:,self.train_ids]
            counts.update(model_forwards=self.model.members,route_predictions=self.model.members,
                supervised_examples=self.model.members*len(self.train_ids))
            ce=self.F.cross_entropy(selected.flatten(0,1),self.train_y.repeat(self.model.members))
            auxiliary=ce.new_zeros(());route_losses=[];pair_losses=[]
            if self.targets is not None:
                auxiliary,route,pair,paid=self.objective.context_auxiliary(self.t,self.F,self.model.last_hidden[:,self.train_ids],
                    self.targets,self.mode,context_index=self.context_index,query_map=self.model.query_map)
                counts.update(paid);route_losses=route.detach().cpu().tolist();pair_losses=pair.detach().cpu().tolist()
            if not self.preparation_charged:
                for key,value in self.preparation_counts.items():counts[key]=counts.get(key,0)+value
                self.preparation_charged=True
            loss=ce+self.coefficient*auxiliary
            if not bool(self.t.isfinite(loss)):raise FloatingPointError('Nonfinite source fixed-key objective')
            counts['backward_attempts']+=1;loss.backward();counts['backward_calls']+=1
            if any(p.grad is not None and not bool(self.t.isfinite(p.grad).all()) for p in self.model.parameters()):raise FloatingPointError('Nonfinite fixed-key gradient')
            counts['optimizer_step_attempts']+=1;self.optimizer.step();counts['optimizer_updates']+=1
            return {'objective':self._scalar(loss),'ce':self._scalar(ce),'regularization':self._scalar(self.coefficient*auxiliary),
                'context_auxiliary':self._scalar(auxiliary),'context_route_losses':route_losses,'context_pair_losses':pair_losses,'operation_counts':counts}
        except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error

    def capture_state(self):
        value=super().capture_state()
        value['context_keys']={'streams':copy.deepcopy(self.model.streams),'target_tensors':{k:v.detach().cpu().clone() for k,v in (self.targets or {}).items()},
            'target_hashes':self.target_hashes,'preparation_charged':self.preparation_charged,'numpy_rng':rng_state(self.t,self.np)['numpy']}
        return value

    def restore_state(self,state):
        actual=state['context_keys']
        if actual['target_hashes']!=self.target_hashes or set(actual['target_tensors'])!=set(self.targets or {}):raise ValueError('Context target/order ancestry changed')
        if any(not self.t.equal(actual['target_tensors'][k],v.detach().cpu()) for k,v in (self.targets or {}).items()):raise ValueError('Fixed target/key/moment bytes changed')
        super().restore_state(state);self.model.streams=copy.deepcopy(actual['streams']);self.preparation_charged=actual['preparation_charged']
        name,keys,position,gauss,cached=actual['numpy_rng'];self.np.random.set_state((name,self.np.asarray(keys,dtype='uint32'),position,gauss,cached))


def make_adapter(*,torch,project_root,study,run,device):
    import numpy as np
    from torch.nn import functional as F
    config=validate_config(study,run)
    sources={name:source_module(project_root,row,name) for name,row in config['context_key_sources'].items() if name!='driver'}
    bound(project_root,config['context_key_sources']['driver'])  # The archived trainer is custody, never executed.
    inputs={name:torch.load(bound(project_root,row),map_location='cpu',weights_only=True) for name,row in config['node_inputs'].items()}
    graph,train,select=inputs['graph'],inputs['train'],inputs['select']
    if set(graph)!={'x','edge_index'} or any(set(pack)!={'ids','y'} for pack in (train,select)):raise ValueError('Exact separated node schemas required')
    x,edges=graph['x'],graph['edge_index']
    if x.ndim!=2 or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or edges.dtype!=torch.long or edges.ndim!=2 or edges.shape[0]!=2:raise ValueError('Plain finite float32 graph and int64 COO required')
    for pack in (train,select):
        if pack['ids'].dtype!=torch.long or pack['y'].dtype!=torch.long or pack['ids'].ndim!=1 or pack['y'].shape!=pack['ids'].shape or not len(pack['ids']) or len(torch.unique(pack['ids']))!=len(pack['ids']):raise ValueError('Complete unique role IDs and int64 labels required')
        if int(pack['ids'].min())<0 or int(pack['ids'].max())>=len(x) or int(pack['y'].min())<0 or int(pack['y'].max())>=config['classes']:raise ValueError('Role endpoint/class outside native graph')
    if torch.isin(train['ids'],select['ids']).any():raise ValueError('TRAIN/SELECT overlap')
    arm=config['context_key_arm'];has_context=arm in ARMS;members=config['members'];factorized=arm not in ('ordinaryM1_key_mixture','reference_ordinary_M1','reference_original_GNNM')
    if has_context and bool((torch.bincount(train['y'],minlength=config['classes'])<2).any()):raise ValueError('Every TRAIN class needs at least two examples; no filtering/repair')
    order=np.arange(len(train['ids']),dtype=np.int64) if members==4 else np.argsort(train['ids'].numpy(),kind='stable')
    ordered_ids=train['ids'][torch.from_numpy(order)].to(device);ordered_y=train['y'][torch.from_numpy(order)].to(device)
    objective=None;targets=None;preparation={};preparation_counts={}
    if has_context:
        objective=relocate_objective(sources['objective'],len(train['ids']),x.shape[1],config['width'])
        arrays,preparation=objective.prepare_targets(np,x.numpy(),edges.numpy(),train['ids'].numpy(),train['y'].numpy())
        targets=objective.tensor_bank(torch,arrays,order,device)
        preparation_counts={key:int(value) for key,value in preparation.items() if type(value)==int and key not in ('TRAIN_rows','key_width','shuffle_seed')}
        preparation_counts.update(context_target_preparations=1,context_moment_bmm_calls=3,
            context_moment_matrix_products=12,context_moment_MACs=12*len(train['ids'])**2*x.shape[1])
    actual_graph=SimpleNamespace(edge_index=edges.to(device));actual_x=x.to(device)
    native,routes,factors=sources['native'],sources['routes'],sources['factors']
    outer=rng_state(torch,np)
    try:
        seed=config['initialization_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
        args=('SAGE',config['depth'],x.shape[1],config['width'],config['classes'],1,8,'LayerNorm',config['dropout'])
        body=native.TABMModel(*args,tabm_inits=4,device='cpu') if arm=='reference_original_GNNM' else native.Model(*args)
        wrapper=None
        if factorized:
            wrapper=routes.wrap_shared(torch,factors,body,routes.NativeModelAdapter(torch,body),members=members,kind='baseline',rank=16,block_seed=config['paired_seed']+4000037)
            with torch.no_grad():
                for row in range(members):
                    generator=torch.Generator(device='cpu').manual_seed(seed+1000003*row+2000003)
                    signs=torch.randint(0,2,(1,x.shape[1]),generator=generator)*2-1
                    body.input_linear.r[row].copy_(signs[0].to(body.input_linear.r))
        query=objective.make_query_map(torch,F,config['paired_seed'],device) if has_context else None
        streams=[]
        for row in range(members):
            torch.manual_seed(seed+1000003*row+3000007)
            stream={'cpu':torch.get_rng_state().clone()}
            if str(device).startswith('cuda'):
                torch.cuda.manual_seed_all(seed+1000003*row+3000007);stream['cuda']=torch.cuda.get_rng_state(torch.device(device)).clone()
            streams.append(stream)
    finally:restore_rng(torch,np,outer)
    class ContextBank(torch.nn.Module):
        def __init__(self):
            super().__init__();self.body=wrapper if members>1 and factorized else body;self.query_map=query
            self.streams=streams;self.members=members;self.last_hidden=None
        def forward(self):
            batch={'x':actual_x,'edge_index':actual_graph.edge_index,'graph':actual_graph}
            cuda=torch.device(device).index or 0 if str(device).startswith('cuda') else None
            scope=lambda m:routes.existing_member_rng_scope(torch,self.streams,m,cuda)
            if arm=='reference_original_GNNM':
                outputs=[]
                for member in range(4):
                    with scope(member):outputs.append(self.body(actual_graph,actual_x,tabm_seed=member))
                self.last_hidden=None;return torch.stack(outputs)
            if members==4:
                logits,hidden=self.body(batch,route_scope=scope)
            else:
                with scope(0):logits,hidden=routes.NativeModelAdapter(torch,self.body).native(batch)
                logits,hidden=logits.unsqueeze(0),hidden.unsqueeze(0)
            self.last_hidden=hidden;return logits
    model=ContextBank().to(device)
    body_parameters=list(model.body.parameters());groups=[{'params':body_parameters,'lr':config['learning_rate'],'weight_decay':0}]
    if query is not None:groups.append({'params':list(model.query_map.parameters()),'lr':config['learning_rate'],'weight_decay':0})
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
    owned=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(owned)!=len(set(owned)) or set(owned)!={id(p) for p in model.parameters()}:raise ValueError('Every native/query parameter must have exactly one optimizer owner')
    selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':run['budget']['maximum']+1,'save_every':1}
    fusion='mean_logits' if arm=='reference_original_GNNM' else 'mean_probabilities'
    hashes={key:tensor_hash(value) for key,value in (targets or {}).items()}
    adapter=ContextKeysAdapter(torch,F,model,optimizer,logits_fn=lambda m,training,step:m(),train_ids=ordered_ids,train_y=ordered_y,
        select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda m:actual_x.new_zeros(()),fusion=fusion,selector=selector,
        provenance={'run_id':run['run_id'],'source':study['source'],'data':study['data'],'config':config,'actual_model_seed':config['initialization_seed'],
            'query_initial_sha256':tensor_hash(query.weight) if query is not None else None,'target_hashes':hashes,'source_contract':CONTRACT,
            'source_archived_recipe_reproduction':False,'target_preparation':preparation,'optimizer_objects':1,'physical_slow_bodies':1,
            'native_parameter_tensors':len(body_parameters),'query_parameter_tensors':1 if query is not None else 0})
    adapter.np=np;adapter.targets=targets;adapter.target_hashes=hashes;adapter.objective=objective
    adapter.mode=MODES.get(arm);adapter.context_index=run['body'] if arm=='genuinefactorizedI4_key_assigned' else None
    adapter.coefficient=.05;adapter.preparation_counts=preparation_counts;adapter.preparation_charged=False
    return adapter
