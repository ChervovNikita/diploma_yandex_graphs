"""Private reset-after GRU source algebra with explicit standardized lifecycle.

Archived TF1Adam/100k/latest-minimum/checkpoint0 procedure remains separate.
This module performs no numerical import or preparation at import time.
"""
from __future__ import annotations
import ast
import copy
import math
from pathlib import Path
import types

try:
    from .classifier import ClassifierAdapter
    from .retained_pt_routes import bound, receipt, tensor_hash, Streams, compile_source
    from ..engine import AdapterStepError, restore_checkpoint
    from ..run_study import resolve_selected_config, paid_parent_checkpoints
    from ..spec import fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from adapters.retained_pt_routes import bound, receipt, tensor_hash, Streams, compile_source
    from engine import AdapterStepError, restore_checkpoint
    from run_study import resolve_selected_config, paid_parent_checkpoints
    from spec import fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS

PINS = {
    'gru':'97f47c6671e83f0122ff3539f5c07093f8d37cd0ea641ff18d6837b557fd050a',
    'backend':'a3a752e192a67f0c6b302ceffe3a09833a4e33333b3602c0a488c2c9489b4228',
    'tours':'86f6504de408ec07e10fa109946e6bd0b1acb22753e1dc1b8c7b8947ba3461ea',
    'native':'242396b38ab43a99ac9f0b00f63424569688a223c0fc3838fa5aeb94e3b5a301',
    'original':'07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
    'original_driver':'e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc',
}
SOURCE_ARMS = ('ordinary_p1','factor_f1','factor_s4','original_gnnm')
MEMORY_ARMS = ('B4','G1','GF1','GM4','GN4')
NATIVE_CONTROLS = ('continued_p1','continued_f1')
CONTRACT = {'schema':'private-GRU-standardized-port-v1','architecture':'private_gru_memory',
    'source_core':PINS['gru'],'backend':PINS['backend'],'torch_backend_version':'2.1.2',
    'source_horizon':5000,'continuation_horizon':5000,'optimizer':'AdamW','weight_decay':0,
    'betas':[.9,.999],'epsilon':1e-8,'width':32,'native_first_penalty':.05,
    'memory':'reset-after r/z/n; input z bias+log99; uniform slow; ones factors; shared zero Wo; residual.25; native clip',
    'tours':'four complete private randomized DFS forest arrival/return tours; two reset directions/component; occurrence mean; no truncation',
    'training':'four streamed own CE/4 backwards then one/mean first-layer penalty; one fresh optimizer update',
    'GN4':'live native first factors/offsets; no fixed H0 cache or first dropout; ordinary shared memory',
    'GI16_GFI16':'four disjoint separately selected G1/GF1 learners, four views each, mean16 probabilities; fixed body0 reused',
    'serving':'probability mean; original GNNM raw-logit mean',
    'standardized_adaptations':['provider N/F/C and edge count','explicit paired seeds','tail dropout .2/.4/.6 with.4 anchor',
        'full5000 AdamW WD0/no checkpoint0/own earliest SELECT accuracy','hash-bound nonpersistent external H instead of legacy special buffer restore'],
    'archived_recipe_replication':False,'novelty_cleared':False,'capacity_matched_GM4_GN4':False,
    'archived_policy':'TF1Adam; max100k/p50; latest tied SELECT CE+L2 minimum; stage2checkpoint0; member-temperature before pool',
    'new_native_continued_controls':'all native parameters live; selected source then fresh5000 optimizer; four TRAIN views; optimizer-budget control only; unequal FLOPs/label reduction disclosed'}


def tree_hash(torch, value):
    if torch.is_tensor(value): return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':tensor_hash(value)}
    if isinstance(value,dict): return {k:tree_hash(torch,v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [tree_hash(torch,v) for v in value]
    return value


def load_source(root,row):
    path = bound(root,row); namespace = {'__name__':'gru_source_isolated','__file__':str(path)}
    exec(compile(path.read_text(),str(path),'exec'),namespace)
    return types.SimpleNamespace(**namespace)


def graph_check(torch, x, edge, adjacency, degrees):
    n = len(x)
    if x.dtype!=torch.float32 or x.ndim!=2 or x.requires_grad or (x.is_sparse and not x.is_coalesced()) or not bool(torch.isfinite(x.values() if x.is_sparse else x).all()): raise ValueError('Unchanged finite native FP32 X required')
    if edge.dtype!=torch.long or edge.ndim!=2 or edge.shape[0]!=2 or edge.numel()==0 or int(edge.min())<0 or int(edge.max())>=n: raise ValueError('Qualified complete native edge_index required')
    if adjacency.layout!=torch.sparse_coo or not adjacency.is_coalesced() or adjacency.shape!=(n,n) or adjacency.requires_grad or not torch.equal(adjacency.values(),torch.ones_like(adjacency.values())): raise ValueError('Binary complete coalesced graph; no duplicate/weighted support')
    rows,cols = adjacency.indices()
    if int((rows==cols).sum())!=n or degrees.shape!=(n,1) or degrees.requires_grad or not bool((degrees>0).all()) or not torch.equal(degrees,torch.bincount(rows,minlength=n).float()[:,None]): raise ValueError('Exact one self per node and exact positive degree; no clamp')


def compile_gru(root,row,*,nodes,features,classes,seed,dropout):
    path = bound(root,row); tree = ast.parse(path.read_text()); drops = []
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,node):
            names = {34493:'BOUND_NODES',8415:'BOUND_FEATURES',5:'BOUND_CLASSES'}
            return ast.copy_location(ast.Name(id=names[node.value],ctx=ast.Load()),node) if type(node.value)==int and node.value in names else node
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='F' and node.func.attr=='dropout':
                p = next(k for k in node.keywords if k.arg=='p')
                if not isinstance(p.value,ast.Constant) or p.value.value!=.4: raise ValueError('Exact GRU source tail dropout anchor changed')
                p.value = ast.copy_location(ast.Constant(dropout),p.value); drops.append(node.lineno)
            return node
    tree = ast.fix_missing_locations(Binding().visit(tree))
    if len(drops)!=1: raise ValueError('One exact GRU native tail dropout site required')
    namespace = {'__name__':'standardized_GRU_core','__file__':str(path),'BOUND_NODES':nodes,'BOUND_FEATURES':features,'BOUND_CLASSES':classes}
    exec(compile(tree,str(path)+'::declared_shape_seed_dropout','exec'),namespace); namespace['BLOCKS']=(seed,)
    d = 32; source = 2*features*d+d+2*d*classes+classes; head = 2*d*classes+classes
    ordinary = 7*d*d+6*d; factor = ordinary+9*d; private = 7*d*d+60*d; native = ordinary+8*features+20*d+12*classes
    namespace['COUNTS'] = {'B4':(source,head),'G1':(source+ordinary,head+ordinary),'GF1':(source+factor,head+factor),
        'GM4':(source+private,head+private),'GN4':(source+native,head+native),'GI16':(4*(source+ordinary),4*(head+ordinary)),
        'GFI16':(4*(source+factor),4*(head+factor))}
    return types.SimpleNamespace(**namespace)


def compile_tours(root,row,*,nodes,edges,seed):
    path = bound(root,row); tree = ast.parse(path.read_text())
    class Shapes(ast.NodeTransformer):
        def visit_Constant(self,node):
            names = {34493:'BOUND_NODES',530417:'BOUND_EDGES',68985:'BOUND_MAX_TOKENS'}
            return ast.copy_location(ast.Name(id=names[node.value],ctx=ast.Load()),node) if type(node.value)==int and node.value in names else node
    namespace = {'__name__':'standardized_GRU_tours','__file__':str(path),'BOUND_NODES':nodes,'BOUND_EDGES':edges,'BOUND_MAX_TOKENS':2*nodes-1}
    exec(compile(ast.fix_missing_locations(Shapes().visit(tree)),str(path)+'::declared_shape_seed','exec'),namespace)
    namespace['BLOCKS']=(seed,)
    return types.SimpleNamespace(**namespace)


def component_count(torch,edge,nodes):
    neighbors = [[] for _ in range(nodes)]
    for u,v in zip(*edge.tolist()):
        if u!=v: neighbors[u].append(v)
    visited = [False]*nodes; components = 0
    for start in range(nodes):
        if visited[start]: continue
        components += 1; visited[start] = True; pending = [start]
        while pending:
            for v in neighbors[pending.pop()]:
                if not visited[v]: visited[v] = True; pending.append(v)
    return components


def prepare_tour_capsule(torch,root,graph_binding,sources,seed):
    """Only target-free provider GRAPH; no source model, labels or optimizer."""
    graph = torch.load(bound(root,graph_binding),map_location='cpu',weights_only=True)
    if set(graph)!={'x','edge_index'}: raise ValueError('Tour prep opens target-free GRAPH only')
    x,edge = graph['x'],graph['edge_index']; n = len(x)
    A = torch.sparse_coo_tensor(edge.flip(0),torch.ones(edge.shape[1]),(n,n)).coalesce(); degree = torch.bincount(A.indices()[0],minlength=n).float()[:,None]
    graph_check(torch,x,edge,A,degree)
    helper = compile_tours(root,sources['tours'],nodes=n,edges=edge.shape[1],seed=seed); backend = load_source(root,sources['backend'])
    backend.require_runtime(torch); outer = torch.get_rng_state().clone(); components = component_count(torch,edge,n)
    tours = helper.build_tours(torch,edge,seed,expected_components=components); packed = tuple(backend.pack_tour(torch,t) for t in tours)
    if not torch.equal(outer,torch.get_rng_state()): raise ValueError('Private tour prep changed global RNG')
    capsule = {'schema':'standardized-private-GRU-tours-v1','graph':graph_binding,'seed':seed,'nodes':n,'components':components,
        'source_tours':sources['tours'],'source_backend':sources['backend'],'tours':tours,'packed':packed,
        'labels_or_predictions_used':False,'models_constructed':0,'optimizers_created':0}
    capsule['payload_hash'] = fingerprint(tree_hash(torch,capsule)); return capsule


def validate_config(study,run):
    c = copy.deepcopy(run['config']); arm = c['gru_arm']; stage = c['gru_stage']
    if arm not in SOURCE_ARMS+MEMORY_ARMS+NATIVE_CONTROLS or stage not in ('source','continuation') or (arm in SOURCE_ARMS)!=(stage=='source'): raise ValueError('Separate typed GRU source/continuation arm required')
    if c['architecture'] != ('original_gnnm' if arm=='original_gnnm' else 'private_gru_memory') or c['source_optimizer_count']!=1 or c['physical_slow_bodies']!=1: raise ValueError('One physical body and one fresh optimizer per GRU fit')
    if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS: raise ValueError('Original paper reruns excluded')
    if c['width']!=(128 if arm=='original_gnnm' else 32) or c['depth']!=2 or c['classes']<2 or c['weight_decay']!=0 or c['learning_rate'] not in (.0005,.001,.002) or c['dropout'] not in (.2,.4,.6): raise ValueError('Explicit standardized dimensions/domain/WD0 required')
    expected = 4 if arm in MEMORY_ARMS+('factor_s4','original_gnnm') else 1
    if c['members']!=expected or c['initialization_seed']!=c['paired_seed']+1000003*run['body']: raise ValueError('Four views per actual GRU learner and exact paid body seed required')
    if set(c['gru_sources'])!=set(PINS) or any(c['gru_sources'][k]['sha256']!=v for k,v in PINS.items()): raise ValueError('All exact GRU/native/original/tour/backend pins required')
    if set(c['node_inputs'])!={'graph','train','select'}: raise ValueError('Fit opens GRAPH/TRAIN/SELECT only')
    for role in ('train','select'):
        if c['node_inputs'][role]!=study['data']['roles'][role][str(run['split'])]: raise ValueError('GRU role authority differs')
    if run['budget']['unit']!='optimizer_updates' or run['budget']['policy']!='full_budget' or (study['stage']!='engineering' and run['budget']['maximum']!=5000): raise ValueError('Full5000 standardized optimizer horizon required')
    if any(run['selection'][k]!=v for k,v in {'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False}.items()): raise ValueError('Standard own SELECT/no0 differs from archived GRU recipe')
    if stage=='continuation':
        from_life = c.get('lifecycle',{})
        if set(from_life)!={'schema','parent_fit_ids','new_optimizer','parameter_initialization'} or from_life.get('schema')!='paid-parent-continuation-v1' or from_life.get('new_optimizer') is not True or from_life.get('parameter_initialization')!='selected_parent_no_optimizer_reuse' or len(from_life['parent_fit_ids'])!=1 or not set(from_life['parent_fit_ids']).issubset(run['parents']): raise ValueError('Explicit one paid source/fresh continuation lifecycle required')
    if (c.get('gru_tours') is not None)!=(arm in MEMORY_ARMS): raise ValueError('Only GRU memory/head interfaces bind four complete tour custody')
    return c


def restore_source(torch,root,study,run,c,device,context):
    if not isinstance(context,dict) or not {'registry','output_root'}.issubset(context): raise ValueError('Explicit paid-source execution_context required')
    registry = context['registry']; identifier = c['lifecycle']['parent_fit_ids'][0]
    verified = paid_parent_checkpoints(root,study,run,registry,context['output_root'])[identifier]
    parent = verified['contract']; actual = resolve_selected_config(study,parent,registry); p = actual['config']
    expected = 'factor_f1' if c['gru_arm']=='continued_f1' else 'ordinary_p1'
    if p.get('gru_stage')!='source' or p.get('gru_arm')!=expected or parent['budget']!=run['budget'] or any(parent[k]!=run[k] for k in ('split','seed','body')) or any(p[k]!=c[k] for k in ('node_inputs','gru_sources','width','classes','stage','paired_seed','initialization_seed')): raise ValueError('Complete matching GRU source contract/roles/seed/horizon required')
    adapter = make_adapter(torch=torch,project_root=root,study=study,run=actual,device=device)
    restored = restore_checkpoint(torch,verified['checkpoint'],adapter,expected_fit_id=identifier)
    if restored['step']!=verified['result']['selected']['step'] or restored['validation']!=verified['result']['selected']['validation']: raise ValueError('Authenticated own selected source state differs')
    evidence = {'fit_id':identifier,'contract_hash':fingerprint(parent),'resolved_config_hash':fingerprint(p),
        'checkpoint':receipt(root,verified['checkpoint']),'selected_step':restored['step'],'paid_source_updates':parent['budget']['maximum'],'optimizer_reused':False}
    return adapter,evidence


class GRUMemoryAdapter(ClassifierAdapter):
    def train_step(self,step):
        counts = {'optimizer_updates':0,'optimizer_step_attempts':0,'backward_calls':0,'backward_attempts':0,'model_forwards':0,
            'forward_attempts':0,'route_predictions':0,'supervised_examples':0}
        try:
            self.model.train(); self.optimizer.zero_grad(set_to_none=True); losses = []
            if self.model.is_memory:
                for view in range(4):
                    counts['forward_attempts']+=1; z = self.model.view(view); counts['model_forwards']+=1; counts['route_predictions']+=1
                    counts['supervised_examples']+=len(self.train_ids); ce = self.F.cross_entropy(z[self.train_ids],self.train_y)
                    if not bool(self.t.isfinite(ce)): raise FloatingPointError('Nonfinite full own GRU view CE')
                    losses.append(self._scalar(ce)); counts['backward_attempts']+=1; (ce/4).backward(); counts['backward_calls']+=1
                ce_value = sum(losses)/4; penalty = self.regularizer(self.model)
                if penalty.requires_grad: counts['backward_attempts']+=1; penalty.backward(); counts['backward_calls']+=1
            else:
                counts['forward_attempts']+=1; logits = self.model(); counts.update(model_forwards=self.model.train_forwards,route_predictions=len(logits),supervised_examples=len(logits)*len(self.train_ids))
                ce = self.t.stack([self.F.cross_entropy(z[self.train_ids],self.train_y) for z in logits]).mean(); penalty = self.regularizer(self.model)
                ce_value = self._scalar(ce); counts['backward_attempts']+=1; (ce+penalty).backward(); counts['backward_calls']+=1
            counts.update(self.model.operation_counts(True))
            if not self.preparation_charged: counts.update(self.preparation_counts); self.preparation_charged=True
            penalty_value = self._scalar(penalty)
            if any(p.grad is not None and not bool(self.t.isfinite(p.grad).all()) for p in self.model.parameters()): raise FloatingPointError('Nonfinite live GRU/native gradient')
            if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad): raise ValueError('Frozen acquired source has gradients')
            counts['optimizer_step_attempts']+=1; self.optimizer.step(); counts['optimizer_updates']+=1; self.model.guard()
            return {'objective':ce_value+penalty_value,'ce':ce_value,'regularization':penalty_value,'view_CE':losses,'operation_counts':counts}
        except BaseException as error: raise AdapterStepError(str(error),operation_counts=counts) from error

    def validate(self,step):
        self.model.eval()
        with self.t.no_grad():
            logits=self.model()[:,self.select_ids].double(); ce=self.t.stack([self.F.cross_entropy(z,self.select_y) for z in logits]).mean(); penalty=self.regularizer(self.model)
            logp=self.t.logsumexp(logits.log_softmax(-1),dim=0)-math.log(len(logits)) if self.fusion=='mean_probabilities' else logits.mean(0).log_softmax(-1)
            correct=int(logp.argmax(-1).eq(self.select_y).sum()); accuracy=correct/len(self.select_y)
            return {'objective':self._scalar(ce+penalty),'ce':self._scalar(ce),'regularization':self._scalar(penalty),'correct':correct,'examples':len(self.select_y),
                'accuracy':accuracy,'selection_metric':accuracy,'pooled_nll':self._scalar(-logp.gather(1,self.select_y[:,None]).mean()),'fusion':self.fusion,
                'operation_counts':{'optimizer_updates':0,'backward_calls':0,'model_forwards':self.model.eval_forwards,'route_predictions':len(logits),'supervised_examples':0,**self.model.operation_counts(False)}}

    def capture_state(self):
        state=super().capture_state(); state['gru_memory']={'streams':copy.deepcopy(self.streams.states),'custody':copy.deepcopy(self.custody),'preparation_charged':self.preparation_charged}; return state
    def restore_state(self,state):
        if state['gru_memory']['custody']!=self.custody: raise ValueError('GRU paid source/H/tour/graph ancestry changed')
        super().restore_state(state); self.streams.states=copy.deepcopy(state['gru_memory']['streams']); self.preparation_charged=state['gru_memory']['preparation_charged']


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    from torch.nn import functional as F
    root=Path(project_root).resolve(); c=validate_config(study,run); arm=c['gru_arm']; seed=c['initialization_seed']
    for row in c['gru_sources'].values(): bound(root,row)
    inputs={k:torch.load(bound(root,row),map_location='cpu',weights_only=True) for k,row in c['node_inputs'].items()}
    graph,train,select=inputs['graph'],inputs['train'],inputs['select']
    if set(graph)!={'x','edge_index'} or any(set(p)!={'ids','y'} for p in (train,select)): raise ValueError('Separated exact node schemas required')
    x,edge=graph['x'].to(device),graph['edge_index'].to(device); A=torch.sparse_coo_tensor(edge.flip(0),torch.ones(edge.shape[1],device=device),(len(x),len(x))).coalesce(); degree=torch.bincount(A.indices()[0],minlength=len(x)).float()[:,None]
    graph_check(torch,x,edge,A,degree)
    for p in (train,select):
        if p['ids'].dtype!=torch.long or p['y'].dtype!=torch.long or p['ids'].ndim!=1 or p['y'].shape!=p['ids'].shape or len(p['ids'])==0 or len(torch.unique(p['ids']))!=len(p['ids']) or int(p['ids'].min())<0 or int(p['ids'].max())>=len(x) or int(p['y'].min())<0 or int(p['y'].max())>=c['classes']: raise ValueError('Complete unique valid int64 role IDs/labels required')
    if torch.isin(train['ids'],select['ids']).any(): raise ValueError('TRAIN/SELECT overlap')
    native_source=compile_source(root,c['gru_sources']['native'],dropout=c['dropout'],dimensions=(len(x),x.shape[1],c['width'],c['classes']),seed=seed)
    custody={'inputs':copy.deepcopy(c['node_inputs']),'graph_hash':tensor_hash(A),'X_hash':tensor_hash(x),'degree_hash':tensor_hash(degree),'sources':copy.deepcopy(c['gru_sources']),'parent':None,'H_hash':None,'tours':None}
    source_adapter=None; memory_model=None; native_model=None; packed=None; fixed_tensors=[x,A,degree]
    if c['gru_stage']=='continuation': source_adapter,custody['parent']=restore_source(torch,root,study,run,c,device,execution_context)
    if arm in MEMORY_ARMS:
        backend=load_source(root,c['gru_sources']['backend']); backend.require_runtime(torch)
        capsule=torch.load(bound(root,c['gru_tours']['capsule']),map_location='cpu',weights_only=True); stored=capsule.pop('payload_hash')
        observed=fingerprint(tree_hash(torch,capsule))
        if capsule['schema']!='standardized-private-GRU-tours-v1' or capsule['graph']!=c['node_inputs']['graph'] or capsule['seed']!=c['paired_seed'] or capsule['nodes']!=len(x) or capsule['source_tours']!=c['gru_sources']['tours'] or capsule['source_backend']!=c['gru_sources']['backend'] or stored!=observed or observed!=c['gru_tours']['payload_hash'] or capsule['labels_or_predictions_used'] is not False or len(capsule['tours'])!=4 or len(capsule['packed'])!=4: raise ValueError('Complete actual graph/seed/source/four-tour payload custody required')
        for tour,pack in zip(capsule['tours'],capsule['packed']):
            replay=backend.pack_tour(torch,tour)
            if fingerprint(tree_hash(torch,replay))!=fingerprint(tree_hash(torch,pack)): raise ValueError('Persisted full tour/packing maps differ')
        packed=tuple(backend.to_device(p,device) for p in capsule['packed']); custody['tours']=copy.deepcopy(c['gru_tours'])
        for values in list(capsule['tours'])+list(capsule['packed'])+list(packed): fixed_tensors += [v for v in values.values() if torch.is_tensor(v)]
        ancestor=source_adapter.model.native
        if arm=='GN4': H=None
        else:
            first=ancestor.bodies[0].layers[0]
            with torch.no_grad():
                z=torch.sparse.mm(A,first._transform(x,first.aggregate_weight,None,None))/degree+first._transform(x,first.skip_weight,None,None)+first.bias[0]
                H=F.relu(z/torch.linalg.vector_norm(z,ord=2,dim=1,keepdim=True).clamp_min(1.)).detach()
            custody['H_hash']=tensor_hash(H); fixed_tensors.append(H)
        source=compile_gru(root,c['gru_sources']['gru'],nodes=len(x),features=x.shape[1],classes=c['classes'],seed=seed,dropout=c['dropout'])
        core=source.build_arm(torch,F,arm,seed,[ancestor],[H],gru_backend=backend); source.inspect_parameter_ownership(core); memory_model=core['models'][0].to(device)
        if hasattr(memory_model,'fixed_H'): memory_model.register_buffer('fixed_H',memory_model.fixed_H,persistent=False)
        memory_model.paid_body_index=run['body']; train_views=eval_views=4
    elif arm=='original_gnnm':
        original=load_source(root,c['gru_sources']['original'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); native_model=original.TABMModel('SAGE',2,x.shape[1],128,c['classes'],1,8,'LayerNorm',c['dropout'],tabm_inits=4,device='cpu').to(device)
        train_views=eval_views=4
    else:
        kind='F1' if arm in ('factor_f1','continued_f1') else 'S4' if arm=='factor_s4' else 'P1'
        core=native_source.build_arm(torch,F,kind,seed); native_source.inspect_parameter_ownership(core); native_model=core['models'][0].to(device)
        if source_adapter is not None: native_model.load_state_dict(source_adapter.model.native.state_dict(),strict=True)
        train_views=4 if arm in NATIVE_CONTROLS else c['members']; eval_views=c['members']
    streams=Streams(torch,device,[seed+459000151+1009*v for v in range(train_views)])
    class Bank(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.native=native_model; self.memory_body=memory_model; self.is_memory=memory_model is not None
            self.train_forwards=4 if arm in MEMORY_ARMS+NATIVE_CONTROLS+('original_gnnm',) else 1
            self.eval_forwards=1 if arm!='original_gnnm' else 4
            self.versions=tuple(v._version for v in fixed_tensors)
        def guard(self):
            if self.versions!=tuple(v._version for v in fixed_tensors): raise ValueError('Immutable GRU raw graph/H/tours/packing mutated')
        def view(self,v):
            self.guard(); z=self.memory_body.view_forward(x,A,degree,packed,v,streams.scope); self.guard(); return z
        def forward(self):
            self.guard()
            if self.is_memory: out=self.memory_body(x,A,degree,packed,streams.scope)
            elif arm=='original_gnnm':
                values=[]; graph=types.SimpleNamespace(edge_index=edge)
                for v in range(4):
                    with streams.scope(v): values.append(self.native(graph,x,tabm_seed=v))
                out=torch.stack(values)
            elif arm in NATIVE_CONTROLS and self.training: out=torch.cat([self.native(x,A,degree,route_rng_context=lambda route,v=v:streams.scope(v)) for v in range(4)])
            else: out=self.native(x,A,degree,route_rng_context=streams.scope)
            self.guard(); return out
        def regularizer(self): return self.memory_body.regularizer() if self.is_memory else x.new_zeros(()) if arm=='original_gnnm' else self.native.regularizer()
        def operation_counts(self,training):
            routes=train_views if training else eval_views
            if self.is_memory:
                recurrent=arm!='B4'
                return {'GRU_view_forward_calls':4,'GRU_direction_forward_API_calls':8 if recurrent else 0,
                    'GRU_direction_backward_API_calls':8 if training and recurrent else 0,'GRU_reset_components':8*capsule['components'] if recurrent else 0,
                    'GRU_complete_token_occurrences':2*sum(p['length'] for p in packed) if recurrent else 0,
                    'native_first_layer_calls':4 if arm=='GN4' else 0,'native_tail_layer_calls':4,'native_tail_dropout_draws':4 if training else 0,
                    'native_graph_propagations':8 if arm=='GN4' else 4,'GRU_weight_list_constructions':4 if recurrent else 0,
                    'GRU_live_factor_product_banks':4 if arm in ('GF1','GM4') else 0}
            if arm=='original_gnnm': return {'original_TABM_forward_calls':4,'original_SAGEConv_forward_calls':8}
            return {'native_source_model_forward_calls':self.train_forwards if training else self.eval_forwards,'native_first_layer_calls':routes,
                'native_tail_layer_calls':routes,'native_graph_propagations':2*routes,'native_dropout_draws':2*routes if training else 0}
    model=Bank().to(device); optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=0,betas=(.9,.999),eps=1e-8)
    if optimizer.state: raise ValueError('New GRU/native continuation optimizer must have empty moments')
    adapter=GRUMemoryAdapter(torch,F,model,optimizer,logits_fn=lambda m,training,step:m(),train_ids=train['ids'].to(device),train_y=train['y'].to(device),select_ids=select['ids'].to(device),select_y=select['y'].to(device),
        regularizer=lambda m:m.regularizer(),fusion=c['fusion'],selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':run['budget']['maximum']+1,'save_every':1},
        provenance={'run_id':run['run_id'],'source':study['source'],'data':study['data'],'config':c,'actual_model_seed':seed,'contract':CONTRACT,'actual_parent':custody['parent'],'actual_H_hash':custody['H_hash'],'fresh_optimizer':True})
    adapter.streams=streams; adapter.custody=custody; adapter.preparation_charged=False
    adapter.preparation_counts={'selected_source_checkpoint_restores':1,'source_optimizer_restores_for_evidence_only':1,'H_preparations':int(arm in MEMORY_ARMS and arm!='GN4'),'H_native_first_map_products':2 if arm in MEMORY_ARMS and arm!='GN4' else 0,'H_graph_propagations':int(arm in MEMORY_ARMS and arm!='GN4')} if source_adapter is not None else {}
    return adapter
