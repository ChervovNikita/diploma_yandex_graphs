"""Pinned inducing-write/grouped-read landmark source; standardized lifecycle.

No numerical imports or data/model preparation at module import. The closed
PubMed LM4 failure remains closed; this is a source architecture transfer.
"""
from __future__ import annotations
import ast
from collections import deque
import copy
import math
from pathlib import Path
import types

try:
    from .classifier import ClassifierAdapter
    from .retained_pt_routes import bound, receipt, tensor_hash, Streams
    from . import polyformer as poly
    from . import polyformer_boundary as boundary
    from ..engine import AdapterStepError, restore_checkpoint
    from ..run_study import resolve_selected_config, paid_parent_checkpoints
    from ..spec import fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from adapters.retained_pt_routes import bound, receipt, tensor_hash, Streams
    from adapters import polyformer as poly, polyformer_boundary as boundary
    from engine import AdapterStepError, restore_checkpoint
    from run_study import resolve_selected_config, paid_parent_checkpoints
    from spec import fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS

SOURCE_PATHS = {
    'core':'pubmed_private_landmark_memory_execution_source_20261011_v2/core.py',
    'runtime':'pubmed_private_landmark_memory_execution_source_20261011_v2/runtime.py',
    'prepare':'pubmed_private_landmark_memory_execution_source_20261011_v2/prepare.py',
    'learning':'pubmed_private_landmark_memory_execution_source_20261011_v2/learning.py',
    'acquire':'pubmed_private_landmark_memory_execution_source_20261011_v2/acquire.py',
    'factor_adapter':'pubmed_factor1_controls_source_20261010_v1/adapter.py',
    'strong_adapter':'combination_pubmed_strong_reference_source_20261010_v2/adapter.py',
    **{k:v['path'] for k,v in poly.SOURCE_FILES.items()}, 'original':boundary.BOUNDARY_SOURCE['path'],
}
PINS = {
    'core':'e0248e310af75f177ee0fcec31a9ae4adb246af00089d1abef3d2e83685a9c4a',
    'runtime':'3126a1510f1830549ab7004e71cc53aece6fd4e8f851e8f70972b5ce8150d0a0',
    'prepare':'aa93a29da56d1a2d48fb3514fff48b8dcb0204a2495bce74aa35508e65980521',
    'learning':'12fc57d981f1bae5056ac0a793a0d537208b978379b27881f71e36a134bdaabe',
    'acquire':'cf23e2e9e0300380451344c5135fa7f6a2fdee799e17c76bb4d78ea27bed404e',
    'factor_adapter':'198393828be6fe1feabd01abd64573db3b9a48eec5d72687d0f34d0009dde22e',
    'strong_adapter':'4c5494cd89d685869cb39e6ba3e5168f7971f2463ba93a5b795aaa0738246a05',
    **{k:v['sha256'] for k,v in poly.SOURCE_FILES.items()}, 'original':boundary.BOUNDARY_SOURCE['sha256'],
}
SOURCE_ARMS = ('ordinary_m1','factor_m1','shared_factor4','original_boundary')
MEMORY_ARMS = ('B1','B4','T16','S64','FS64','LM4')
NATIVE_CONTROLS = ('continued_m1','continued_f1')
NATIVE = dict(K=2,hidden=256,nlayer=2,n_head=8,d_ffn=32,q=1.6,multi=2,dropout=.5,dprate=.8)
CONTRACT = {
    'schema':'landmark-memory-standardized-port-v1','architecture':'landmark_memory',
    'source_core':PINS['core'],'author':'air029/PolyFormer','author_commit':poly.AUTHOR_COMMIT,
    'source_horizon':5000,'continuation_horizon':5000,'optimizer':'AdamW','weight_decay':0,
    'betas':[.9,.999],'epsilon':1e-8,'native':NATIVE,'source_factors':'all23 native Linear sites; unit r/s; bias outside s',
    'features':'exact PyG NormalizeFeatures once, then pinned monoK2 gcn_norm/SciPy FP32 COO/successive spmm',
    'geometry':'64 distinct ordered anchors, seed+491001001; loop-free unique undirected support only for64 capped16 BFS; bias-distance/4; unreachable16; consecutive16 groups',
    'cache':'authenticated selected source; raw unclipped pre-lin3 H; U=H/norm.clamp_min(1) memory-only; trunk frozen except exact factorized lin3',
    'memory':'six Xavier maps in Aq Ak Av Bq Bk Bv order/private CPU generator; zero Wo and E; no discarded draws; H+.25Wo(read) unclipped; Aq and E retained',
    'training':'fresh common Ak(U)/Av(U)/Bq(U) projected tape once/update; streamed own CE/views backward retaining tape for earlier views; one fresh update; no pooled CE',
    'views':{'B1':1,'B4':4,'T16':4,'S64':1,'FS64':1,'LM4':4},
    'independent':'I4[S64]/I4[FS64] four separately optimized/selected one-view learners, distinct factor_m1 parents, mean4 probabilities; fixed body0 reuse',
    'serving':'probability mean; original boundary GNNM retains raw-logit mean',
    'common_CAL':{'placement':'after_serving_pool','optimizer':'float64 Adam','learning_rate':.01,'steps':500,'log_temperature_clamp':[-4,4],'selection':'last'},
    'historical_CAL':{'placement':'per_member_before_probability_pool','optimizer':'float64 Adam','learning_rate':.05,'steps':500,'log_temperature_clamp':[-5,5],'selection':'latest minimum including0'},
    'standardized_adaptations':['provider N/F/C and complete new TRAIN/SELECT/CAL/development roles','explicit paid body seed stride1000003',
        'full5000 AdamW WD0/source attention and continuation memory low LR=base/10','tail dropout.3/.5/.7','earliest own SELECT accuracy/no eligible0; ineligible0 proper-loss diagnostic',
        'common after-serving CAL source-law adaptation','new full-native continuation controls: four TRAIN draws/one served view; optimizer-budget-only matching; unequal label reduction/FLOPs'],
    'archived_recipe_replication':False,'novelty_cleared':False,'pure_topology_or_private_sharing_claim':False,
    'closed_PubMed_LM4':'36 groups/54 trajectories/39 CAL endpoints closed; checkpoint0 all3; raw all-positive failed; calibrated NLL worse than I4[S64] by.009752; no rescue',
}


def tree_hash(torch,value):
    if torch.is_tensor(value): return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':tensor_hash(value)}
    if isinstance(value,dict): return {k:tree_hash(torch,v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [tree_hash(torch,v) for v in value]
    return value


def isolated_helpers(sources):
    """Clone functions into owned pin namespaces, never mutate helper globals."""
    def clone(module,replacements):
        namespace=dict(module.__dict__); namespace.update(replacements)
        for name,value in module.__dict__.items():
            if isinstance(value,types.FunctionType) and value.__globals__ is module.__dict__:
                function=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
                function.__kwdefaults__=value.__kwdefaults__; namespace[name]=function
        return types.SimpleNamespace(**namespace)
    p=clone(poly,{'SOURCE_FILES':{k:sources[k] for k in poly.SOURCE_FILES}})
    b=clone(boundary,{'p':p,'BOUNDARY_SOURCE':sources['original']})
    return p,b


def compile_core(root,row,*,nodes,features,classes,seed,dropout):
    path=bound(root,row); tree=ast.parse(path.read_text()); changed=[]
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value)==int and node.value==19717: return ast.copy_location(ast.Name(id='BOUND_NODES',ctx=ast.Load()),node)
            return node
        def visit_Compare(self,node):
            self.generic_visit(node)
            # Only categorical lin3 weight/bias/s guards contain literal3.
            for item in ast.walk(node):
                if isinstance(item,ast.Tuple) and len(item.elts) in (1,2):
                    for i,v in enumerate(item.elts):
                        if isinstance(v,ast.Constant) and type(v.value)==int and v.value==3:
                            item.elts[i]=ast.copy_location(ast.Name(id='BOUND_CLASSES',ctx=ast.Load()),v)
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='F' and node.func.attr=='dropout':
                p=next(k for k in node.keywords if k.arg=='p')
                if not isinstance(p.value,ast.Constant) or p.value.value!=.5: raise ValueError('Exact single landmark tail dropout anchor changed')
                p.value=ast.copy_location(ast.Constant(dropout),p.value); changed.append(node.lineno)
            return node
    namespace={'__name__':'standardized_landmark_core','__file__':str(path),'BOUND_NODES':nodes,'BOUND_CLASSES':classes}
    exec(compile(ast.fix_missing_locations(Binding().visit(tree)),str(path)+'::declared_shape_seed_dropout','exec'),namespace)
    if len(changed)!=1: raise ValueError('One exact landmark tail dropout site required')
    namespace['SEEDS']=(seed,)
    source=2083818+257*(features-500)+258*(classes-3); head=258*classes+256
    memory={'B1':0,'B4':0,'T16':43520,'S64':45056,'FS64':46624,'LM4':45056}
    namespace['COUNTS']={a:(source+n,head+n) for a,n in memory.items()}
    namespace['COUNTS'].update({f'I4[{a}]':(4*(source+memory[a]),4*(head+memory[a])) for a in ('S64','FS64')})
    return types.SimpleNamespace(**namespace)


def prepare_graph_capsule(torch,root,graph_binding,sources,seed):
    """GRAPH-only NormalizeFeatures/monoK2/64 BFS; no labels/source model/H."""
    from torch_geometric.data import Data
    from torch_geometric.transforms import NormalizeFeatures
    import torch_geometric
    graph=torch.load(bound(root,graph_binding),map_location='cpu',weights_only=True)
    if set(graph)!={'x','edge_index'}: raise ValueError('Landmark prep opens target-free GRAPH only')
    x,edge=graph['x'],graph['edge_index']; n=len(x)
    if n<64 or x.dtype!=torch.float32 or x.ndim!=2 or x.requires_grad or not bool(torch.isfinite(x).all()) or edge.dtype!=torch.long or edge.ndim!=2 or edge.shape[0]!=2 or edge.numel()==0 or int(edge.min())<0 or int(edge.max())>=n: raise ValueError('Complete finite FP32 graph with64 distinct nodes required')
    outer=torch.get_rng_state().clone(); p,_=isolated_helpers(sources)
    normalized=NormalizeFeatures()(Data(x=x.clone(),edge_index=edge)).x
    tokens=p.mono_tokens(torch,root,normalized,edge,K=2,preprocessing='pinned_mono_base')
    support=sorted({arc for u,v in zip(*edge.tolist()) if u!=v for arc in ((u,v),(v,u))})
    support_tensor=torch.tensor(support,dtype=torch.long).reshape(-1,2).T.contiguous(); neighbors=[[] for _ in range(n)]
    for u,v in support: neighbors[u].append(v)
    generator=torch.Generator(device='cpu').manual_seed(seed+491001001); before=generator.get_state().clone()
    anchors=torch.randperm(n,generator=generator,dtype=torch.long)[:64].clone(); after=generator.get_state().clone()
    distance=torch.full((n,64),16,dtype=torch.uint8); visits=0
    # Python lists preserve the source FIFO BFS and avoid millions of scalar
    # Torch dispatches. Conversion to uint8 happens after each complete BFS.
    for column,anchor in enumerate(anchors.tolist()):
        depths=[16]*n; depths[anchor]=0; pending=deque([anchor])
        while pending:
            u=pending.popleft(); depth=depths[u]
            if depth>=15: continue
            for v in neighbors[u]:
                visits+=1
                if depths[v]==16: depths[v]=depth+1; pending.append(v)
        distance[:,column]=torch.tensor(depths,dtype=torch.uint8)
    geometry={'seed':seed,'landmark_seed':seed+491001001,'anchors':anchors,'distance':distance,'bias':distance.float().div(-4),
        'generator_before':before,'generator_after':after,'support':support_tensor,'support_hash':tensor_hash(support_tensor),'native_edge_hash':tensor_hash(edge)}
    if len(anchors.unique())!=64 or not bool((distance[anchors,torch.arange(64)]==0).all()) or not torch.equal(outer,torch.get_rng_state()): raise ValueError('Exact64/BFS/private RNG custody failed')
    capsule={'schema':'standardized-landmark-graph-v1','graph':graph_binding,'seed':seed,'nodes':n,'features':x.shape[1],
        'source_pins':sources,'raw_X_hash':tensor_hash(x),'normalized_X_hash':tensor_hash(normalized),'tokens':tokens,'geometry':geometry,
        'NormalizeFeatures':{'implementation':'torch_geometric.transforms.NormalizeFeatures','version':torch_geometric.__version__,'applications':1},
        'BFS_support_arc_visits':visits,'labels_or_predictions_used':False,'models_constructed':0,'optimizers_created':0}
    capsule['payload_hash']=fingerprint(tree_hash(torch,capsule)); return capsule


def validate_config(study,run):
    c=copy.deepcopy(run['config']); arm=c['landmark_arm']; stage=c['landmark_stage']
    if arm not in SOURCE_ARMS+MEMORY_ARMS+NATIVE_CONTROLS or stage not in ('source','continuation') or (arm in SOURCE_ARMS)!=(stage=='source'): raise ValueError('Separate typed landmark source/continuation required')
    if c['architecture']!=('original_gnnm' if arm=='original_boundary' else 'landmark_memory') or c['source_optimizer_count']!=1 or c['physical_slow_bodies']!=1: raise ValueError('One real body/one fresh optimizer per landmark fit')
    if c['is_original_gnnm']!=(arm=='original_boundary') or normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS: raise ValueError('Original paper reruns excluded; original boundary comparison retained')
    if c['width']!=256 or c['depth']!=2 or c['classes']<2 or c['weight_decay']!=0 or c['learning_rate'] not in (.0025,.005,.01) or c['dropout'] not in (.3,.5,.7): raise ValueError('Explicit fixed native recipe and useful9 domain required')
    expected=4 if arm in ('B4','T16','LM4','shared_factor4','original_boundary') else 1
    if c['members']!=expected or c['initialization_seed']!=c['paired_seed']+1000003*run['body'] or run['seed']!=c['paired_seed']: raise ValueError('Exact real view count and paid body seed required')
    if set(c['landmark_sources'])!=set(PINS) or any(c['landmark_sources'][k]['sha256']!=v for k,v in PINS.items()): raise ValueError('All exact landmark/native/factor/original source pins required')
    if set(c['node_inputs'])!={'graph','train','select'}: raise ValueError('Fit opens GRAPH/TRAIN/SELECT only')
    for role in ('train','select'):
        if c['node_inputs'][role]!=study['data']['roles'][role][str(run['split'])]: raise ValueError('Landmark role authority differs')
    if run['budget']['unit']!='optimizer_updates' or run['budget']['policy']!='full_budget' or (study['stage']!='engineering' and run['budget']['maximum']!=5000): raise ValueError('Full5000 standardized optimizer horizon required')
    if any(run['selection'][k]!=v for k,v in {'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False}.items()): raise ValueError('Own earliest SELECT accuracy/no eligible0 required')
    if c['fusion']!=('mean_logits' if arm=='original_boundary' else 'mean_probabilities'): raise ValueError('Source serving law changed')
    if stage=='continuation':
        life=c.get('lifecycle',{})
        if set(life)!={'schema','parent_fit_ids','new_optimizer','parameter_initialization'} or life.get('schema')!='paid-parent-continuation-v1' or life.get('new_optimizer') is not True or life.get('parameter_initialization')!='selected_parent_no_optimizer_reuse' or len(life['parent_fit_ids'])!=1 or not set(life['parent_fit_ids']).issubset(run['parents']): raise ValueError('One paid source/fresh continuation lifecycle required')
    return c


def restore_source(torch,root,study,run,c,device,context):
    if not isinstance(context,dict) or not {'registry','output_root'}.issubset(context): raise ValueError('Explicit paid-source execution_context required')
    registry=context['registry']; identifier=c['lifecycle']['parent_fit_ids'][0]
    verified=paid_parent_checkpoints(root,study,run,registry,context['output_root'])[identifier]
    parent=verified['contract']; actual=resolve_selected_config(study,parent,registry); p=actual['config']
    expected='ordinary_m1' if c['landmark_arm']=='continued_m1' else 'factor_m1'
    if p.get('landmark_stage')!='source' or p.get('landmark_arm')!=expected or parent['budget']!=run['budget'] or any(parent[k]!=run[k] for k in ('split','seed','body')) or any(p[k]!=c[k] for k in ('node_inputs','landmark_sources','landmark_graph','width','classes','depth','stage','paired_seed','initialization_seed')): raise ValueError('Complete matching paid factor/ordinary source contract required')
    adapter=make_adapter(torch=torch,project_root=root,study=study,run=actual,device=device)
    restored=restore_checkpoint(torch,verified['checkpoint'],adapter,expected_fit_id=identifier)
    if restored['step']!=verified['result']['selected']['step'] or restored['validation']!=verified['result']['selected']['validation']: raise ValueError('Authenticated own selected source state differs')
    evidence={'fit_id':identifier,'contract_hash':fingerprint(parent),'resolved_config_hash':fingerprint(p),'checkpoint':receipt(root,verified['checkpoint']),
        'selected_step':restored['step'],'paid_source_updates':parent['budget']['maximum'],'optimizer_reused':False}
    return adapter,evidence


class LandmarkAdapter(ClassifierAdapter):
    def train_step(self,step):
        counts={'optimizer_updates':0,'optimizer_step_attempts':0,'backward_calls':0,'backward_attempts':0,'model_forwards':0,'forward_attempts':0,'route_predictions':0,'supervised_examples':0}
        try:
            self.model.train(); self.optimizer.zero_grad(set_to_none=True); losses=[]
            constants=self.model.constants(); counts.update(self.model.projection_counts())
            for view in range(self.model.train_views):
                counts['forward_attempts']+=1; z=self.model.view(view,constants); counts['model_forwards']+=1; counts['route_predictions']+=1
                counts['supervised_examples']+=len(self.train_ids); ce=self.F.cross_entropy(z[self.train_ids],self.train_y)
                if not bool(self.t.isfinite(ce)): raise FloatingPointError('Nonfinite own landmark/native CE')
                losses.append(self._scalar(ce)); counts['backward_attempts']+=1
                (ce/self.model.train_views).backward(retain_graph=self.model.has_memory and view+1<self.model.train_views); counts['backward_calls']+=1
            counts.update(self.model.operation_counts(True))
            if not self.preparation_charged: counts.update(self.preparation_counts); self.preparation_charged=True
            if any(p.grad is not None and not bool(self.t.isfinite(p.grad).all()) for p in self.model.parameters()): raise FloatingPointError('Nonfinite live landmark gradient')
            if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad): raise ValueError('Frozen selected trunk acquired gradients')
            counts['optimizer_step_attempts']+=1; self.optimizer.step(); counts['optimizer_updates']+=1; self.model.guard(); del constants
            result={'objective':sum(losses)/len(losses),'ce':sum(losses)/len(losses),'regularization':0.,'view_CE':losses,'operation_counts':counts}
            if step==1: result['ineligible_step0_SELECT']=self.initial_diagnostic
            return result
        except BaseException as error: raise AdapterStepError(str(error),operation_counts=counts) from error

    def validate(self,step):
        self.model.eval()
        with self.t.no_grad():
            logits=self.model()[:,self.select_ids].double(); ce=self.t.stack([self.F.cross_entropy(z,self.select_y) for z in logits]).mean()
            logp=self.t.logsumexp(logits.log_softmax(-1),dim=0)-math.log(len(logits)) if self.fusion=='mean_probabilities' else logits.mean(0).log_softmax(-1)
            correct=int(logp.argmax(-1).eq(self.select_y).sum()); accuracy=correct/len(self.select_y)
            return {'objective':self._scalar(ce),'ce':self._scalar(ce),'regularization':0.,'correct':correct,'examples':len(self.select_y),'accuracy':accuracy,
                'selection_metric':accuracy,'pooled_nll':self._scalar(-logp.gather(1,self.select_y[:,None]).mean()),'fusion':self.fusion,
                'operation_counts':{'optimizer_updates':0,'backward_calls':0,'model_forwards':self.model.eval_views,'route_predictions':len(logits),'supervised_examples':0,**self.model.operation_counts(False),**self.model.projection_counts()}}

    def capture_state(self):
        self.model.guard(); state=super().capture_state(); state['landmark_memory']={'streams':copy.deepcopy(self.streams.states),'custody':copy.deepcopy(self.custody),'preparation_charged':self.preparation_charged}; return state
    def restore_state(self,state):
        self.model.guard()
        if state['landmark_memory']['custody']!=self.custody: raise ValueError('Landmark paid-source/H/U/geometry/token custody changed')
        super().restore_state(state); self.streams.states=copy.deepcopy(state['landmark_memory']['streams']); self.preparation_charged=state['landmark_memory']['preparation_charged']; self.model.guard()


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    from torch.nn import functional as F
    root=Path(project_root).resolve(); c=validate_config(study,run); arm=c['landmark_arm']; seed=c['initialization_seed']
    for row in c['landmark_sources'].values(): bound(root,row)
    inputs={k:torch.load(bound(root,row),map_location='cpu',weights_only=True) for k,row in c['node_inputs'].items()}; graph,train,select=(inputs[k] for k in ('graph','train','select'))
    if set(graph)!={'x','edge_index'} or any(set(p)!={'ids','y'} for p in (train,select)): raise ValueError('Separated exact node schemas required')
    for p in (train,select):
        if p['ids'].dtype!=torch.long or p['y'].dtype!=torch.long or p['ids'].ndim!=1 or p['y'].shape!=p['ids'].shape or len(p['ids'])==0 or len(torch.unique(p['ids']))!=len(p['ids']) or int(p['ids'].min())<0 or int(p['ids'].max())>=len(graph['x']) or int(p['y'].min())<0 or int(p['y'].max())>=c['classes']: raise ValueError('Complete unique valid int64 role IDs/labels required')
    if torch.isin(train['ids'],select['ids']).any(): raise ValueError('TRAIN/SELECT overlap')
    capsule=torch.load(bound(root,c['landmark_graph']['capsule']),map_location='cpu',weights_only=True); stored=capsule.pop('payload_hash'); observed=fingerprint(tree_hash(torch,capsule))
    if capsule['schema']!='standardized-landmark-graph-v1' or capsule['graph']!=c['node_inputs']['graph'] or capsule['seed']!=c['paired_seed'] or capsule['source_pins']!=c['landmark_sources'] or stored!=observed or observed!=c['landmark_graph']['payload_hash'] or capsule['labels_or_predictions_used'] is not False or capsule['raw_X_hash']!=tensor_hash(graph['x']) or capsule['geometry']['native_edge_hash']!=tensor_hash(graph['edge_index']): raise ValueError('Complete actual graph/seed/source/token/geometry payload custody required')
    n=len(graph['x']); geometry={k:v.to(device) if torch.is_tensor(v) else v for k,v in capsule['geometry'].items()}; tokens=[v.to(device) for v in capsule['tokens']]
    if len(tokens)!=3 or any(v.shape!=graph['x'].shape or v.dtype!=torch.float32 or v.requires_grad or not bool(torch.isfinite(v).all()) for v in tokens) or geometry['anchors'].shape!=(64,) or geometry['bias'].shape!=(n,64) or not torch.equal(geometry['bias'],geometry['distance'].float().div(-4)): raise ValueError('Exact aligned native tokens/64 capped geometry required')
    p,b=isolated_helpers(c['landmark_sources']); native={**NATIVE,'dropout':c['dropout']}; train_views=4 if arm in NATIVE_CONTROLS else c['members']; eval_views=c['members']
    custody={'inputs':copy.deepcopy(c['node_inputs']),'graph_capsule':copy.deepcopy(c['landmark_graph']),'sources':copy.deepcopy(c['landmark_sources']),'parent':None,'H_hash':None,'U_hash':None,'containment':None}
    ancestor=None; memory_model=None; native_model=None; fixed=[graph['x'],graph['edge_index']]+tokens+[v for v in geometry.values() if torch.is_tensor(v)]
    source_adapter=None
    if c['landmark_stage']=='continuation': source_adapter,custody['parent']=restore_source(torch,root,study,run,c,device,execution_context)
    if arm in MEMORY_ARMS:
        source_adapter.model.eval()
        with torch.no_grad(): source_logits,H=source_adapter.model.native_bank.forward_route(0)
        H=H.detach().clone(); U=H/torch.linalg.vector_norm(H,dim=1,keepdim=True).clamp_min(1); ancestor=source_adapter.model.native_bank.body
        with torch.no_grad(): replay=ancestor.lin3(H)
        same=torch.equal(source_logits.argmax(-1),replay.argmax(-1)) and torch.equal(source_logits[train['ids'].to(device)].argmax(-1).eq(train['y'].to(device)),replay[train['ids'].to(device)].argmax(-1).eq(train['y'].to(device)))
        if not same or not torch.allclose(source_logits,replay,atol=1e-5,rtol=1e-4): raise ValueError('Raw unclipped H/source lin3 full prediction/TRAIN-mask containment failed')
        custody.update(H_hash=tensor_hash(H),U_hash=tensor_hash(U),containment={'full_predictions_and_TRAIN_masks_equal':True,'atol':1e-5,'rtol':1e-4,'max_absolute_delta':float((source_logits-replay).abs().max()),'H_unclipped':True,'U_memory_only':True})
        source=compile_core(root,c['landmark_sources']['core'],nodes=n,features=graph['x'].shape[1],classes=c['classes'],seed=seed,dropout=c['dropout'])
        factors=p.factor_definitions(torch,root); core=source.build_arm(torch,F,arm,seed,[ancestor],[{'H':H,'U':U}],geometry,FactorLinear=factors['FactorLinear'])
        ownership=source.inspect_parameter_ownership(core); memory_model=core['models'][0].to(device); memory_model.body_index=run['body']; fixed += [H,U]
    else:
        factory=b.build_bank if arm=='original_boundary' else p.build_bank
        kwargs={} if arm=='original_boundary' else {'factorized':arm in ('factor_m1','shared_factor4','continued_f1'),'factor_initialization':'unit'}
        native_model=factory(torch,root,tokens,native=native,classes=c['classes'],dataset=study['dataset'],members=c['members'],initialization_seed=seed,
            dropout_seeds=[seed+491005001+1009*v for v in range(train_views)],**kwargs)
        if source_adapter is not None:
            native_model.body.load_state_dict(source_adapter.model.native_bank.body.state_dict(),strict=True)
            # Source PolyAttn bias is an unregistered CPU tensor. Carry it with
            # selected runtime state while keeping new continuation streams.
            state=source_adapter.model.native_bank.capture_runtime_state(); state['streams']=copy.deepcopy(native_model.streams); native_model.restore_runtime_state(state)
        ownership={'resident_parameters':sum(v.numel() for v in native_model.parameters()),'trainable_parameters':sum(v.numel() for v in native_model.parameters()),'optimizer_trajectories':1,'actual_served_views':eval_views}
    streams=Streams(torch,device,[seed+491005001+1009*v for v in range(train_views)])
    class Bank(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.native_bank=native_model; self.memory_body=memory_model; self.has_memory=memory_model is not None and memory_model.memory is not None
            self.train_views=train_views; self.eval_views=eval_views; self.versions=tuple(v._version for v in fixed)
        def guard(self):
            if self.versions!=tuple(v._version for v in fixed): raise ValueError('Immutable landmark raw graph/tokens/H/U/geometry mutated')
        def constants(self): return self.memory_body.bank_constants() if self.memory_body is not None else {}
        def view(self,v,constants):
            self.guard()
            if self.memory_body is not None: out=self.memory_body.view_forward(v,constants,streams.scope)
            else: out=self.native_bank.forward_route(v if c['members']>1 else 0,slot=v if self.training else None)[0]
            self.guard(); return out
        def forward(self):
            constants=self.constants(); out=torch.stack([self.view(v,constants) for v in range(train_views if self.training else eval_views)]); del constants; return out
        def projection_counts(self): return {'landmark_common_projected_banks':int(self.has_memory),'landmark_common_map_products':3 if self.has_memory else 0}
        def operation_counts(self,training):
            views=train_views if training else eval_views
            return {'landmark_view_forwards':views if memory_model is not None else 0,'landmark_write_read_pairs':views if self.has_memory else 0,
                'landmark_tail_dropout_draws':views if training and memory_model is not None else 0,'native_PolyFormer_forwards':0 if memory_model is not None else views,
                'native_factor_lin3_products':views if memory_model is not None else 0,'landmark_memory_map_products':4*views if self.has_memory else 0}
        def capture_runtime_state(self): return native_model.capture_runtime_state() if native_model is not None else {}
        def restore_runtime_state(self,state):
            if native_model is not None: native_model.restore_runtime_state(state)
            elif state!={}: raise ValueError('Memory head has no hidden native runtime state')
    model=Bank().to(device); groups=[]
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        low='memory_body.memory.' in name if memory_model is not None else 'attnmodule' in name
        groups.append({'params':[param],'lr':c['learning_rate']/10 if low else c['learning_rate'],'weight_decay':0})
    optimizer=torch.optim.AdamW(groups,weight_decay=0,betas=(.9,.999),eps=1e-8)
    if optimizer.state: raise ValueError('New landmark/native continuation optimizer must have empty moments')
    adapter=LandmarkAdapter(torch,F,model,optimizer,logits_fn=lambda m,training,step:m(),train_ids=train['ids'].to(device),train_y=train['y'].to(device),select_ids=select['ids'].to(device),select_y=select['y'].to(device),
        regularizer=lambda m:tokens[0].new_zeros(()),fusion=c['fusion'],selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':run['budget']['maximum']+1,'save_every':1},
        provenance={'run_id':run['run_id'],'source':study['source'],'data':study['data'],'config':c,'actual_model_seed':seed,'contract':CONTRACT,'actual_parent':custody['parent'],
            'actual_H_hash':custody['H_hash'],'actual_U_hash':custody['U_hash'],'containment':custody['containment'],'ownership':ownership,'fresh_optimizer':True},
        external_capture=model.capture_runtime_state,external_restore=model.restore_runtime_state)
    adapter.streams=streams; adapter.custody=custody; adapter.preparation_charged=False
    adapter.preparation_counts={'selected_source_checkpoint_restores':1,'source_optimizer_restores_for_evidence_only':1,'H_U_preparations':int(arm in MEMORY_ARMS),'H_native_PolyFormer_forwards':int(arm in MEMORY_ARMS)} if source_adapter is not None else {}
    adapter.initial_diagnostic=adapter.validate(0); adapter.initial_diagnostic['checkpoint_eligible']=False; model.train()
    return adapter
