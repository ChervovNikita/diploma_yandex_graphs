"""Retained supervised P/T v2, explicit paid ancestry, standardized new roles.

No numerical imports, data reads, optimizer or fit occurs at module import.
Preserved sources are compiled into isolated namespaces; only declared shape,
seed and dropout bindings change. This is not archived recipe replication.
"""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import types

try:
    from .classifier import ClassifierAdapter
    from ..engine import AdapterStepError, restore_checkpoint
    from ..run_study import resolve_selected_config
    from ..spec import fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from engine import AdapterStepError, restore_checkpoint
    from run_study import resolve_selected_config
    from spec import fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS

PINS = {
    'pt': '44d69787e3080a849d2f986ce0d5eadaf4b7ed95e3809b747b19afc20693e08f',
    'native': '242396b38ab43a99ac9f0b00f63424569688a223c0fc3838fa5aeb94e3b5a301',
    'original': '07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
    'original_driver': 'e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc',
}
SOURCE_ARMS = ('ordinary_p1', 'factor_f1', 'factor_s4', 'original_gnnm')
PT_ARMS = ('head_only', 'mixed_single', 'bias_coordinate_single', 'private4',
           'middle_state_mean', 'separable4')
NATIVE_CONTROLS = ('continued_p1', 'continued_f1')
CONTRACT = {
    'schema': 'retained-pt-standardized-port-v1', 'source_prototype': 'exact_v2',
    'source_horizon': 5000, 'continuation_horizon': 5000, 'optimizer': 'AdamW',
    'weight_decay': 0, 'betas': [.9, .999], 'epsilon': 1e-8,
    'selector': 'own strict earliest SELECT accuracy, every update, no checkpoint0',
    'serving': 'mean probabilities; original GNNM mean raw logits',
    'pt_equations': 'two tied-role retained PP/PT/TP/TT banks; private e[2,4,4]; frozen undropped native H0; zero Wo residual .25',
    'common_adaptations': ['provider dimensions/classes', 'explicit paid seed bindings',
                           'dropout .2/.4/.6 with exact .4 source anchor',
                           'full5000 AdamW WD0 controller and new roles'],
    'native_continued_controls': 'all native parameters live; selected source initialization; fresh optimizer; four own stochastic TRAIN views (NEW)',
    'separable4': 'NEW shared frozen body0 stem, four private complete mixed continuation/head bodies, one joint own-loss optimizer; greater private capacity',
    'history_controls': ['head_only', 'mixed_single', 'middle_state_mean', 'bias_coordinate_single', 'genuine own I4'],
    'source_acquisition': 'paid selected ordinary P1 per actual independent PT body; never a prepared H0 or hidden teacher',
    'novelty_cleared': False, 'archived_recipe_replication': False,
    'causal_claim': 'PT and TP use different maps; middle averaging changes forward and backward coupling',
}


def receipt(root, path):
    path = Path(path)
    if path.is_symlink(): raise ValueError('Confined real file required')
    path = path.resolve(); root = Path(root).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError('Confined real file required')
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def bound(root, row):
    raw = Path(row['path']); path = Path(root).resolve() / raw
    if raw.is_absolute() or '..' in raw.parts or path.is_symlink():
        raise ValueError('Confined immutable PT source/input required')
    if receipt(root, path) != row: raise ValueError('PT source/input bytes changed')
    return path


def tensor_hash(value):
    value = value.detach().cpu()
    if value.is_sparse:
        value = value.coalesce()
        return fingerprint({'indices': tensor_hash(value.indices()), 'values': tensor_hash(value.values()), 'shape': list(value.shape)})
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def compile_source(root, row, *, dropout, dimensions=None, seed=None):
    """Audit exactly the source dropout call sites, preserve all other AST nodes."""
    path = bound(root, row); tree = ast.parse(path.read_text()); changed = []
    class Dropout(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'F' and node.func.attr == 'dropout':
                keywords = [k for k in node.keywords if k.arg == 'p']
                if len(keywords) != 1 or not isinstance(keywords[0].value, ast.Constant) or keywords[0].value.value != .4:
                    raise ValueError('Preserved source dropout anchor changed')
                keywords[0].value = ast.copy_location(ast.Constant(dropout), keywords[0].value); changed.append(node.lineno)
            return node
    tree = ast.fix_missing_locations(Dropout().visit(tree))
    expected = 1 if row['sha256'] == PINS['pt'] else 2
    if len(changed) != expected: raise ValueError('Exact source dropout call-site census differs')
    namespace = {'__name__': 'retained_pt_isolated', '__file__': str(path)}
    exec(compile(tree, str(path)+'::explicit_dropout_binding', 'exec'), namespace)
    if dimensions is not None:
        nodes, features, width, classes = dimensions
        namespace.update(NODES=nodes, FEATURES=features, HIDDEN=width, CLASSES=classes, BLOCKS=(seed,))
        ordinary = 2*features*width + width + 2*width*classes + classes
        factors = 2*(features+width)+2*(width+classes)
        namespace['EXPECTED_ARM_PARAMETERS'] = {'P1': ordinary, 'F1': ordinary+factors,
            'S4': ordinary+3*(width+classes)+4*factors}
    return types.SimpleNamespace(**namespace)


class Streams:
    """Persistent isolated streams; checkpointed, never reseeded during a fit."""
    def __init__(self, torch, device, seeds):
        self.t, self.device = torch, torch.device(device); self.states = []
        for seed in seeds:
            cpu = torch.Generator(device='cpu').manual_seed(seed).get_state()
            cuda = torch.Generator(device=self.device).manual_seed(seed).get_state() if self.device.type == 'cuda' else None
            self.states.append({'cpu': cpu.clone(), 'cuda': cuda.clone() if cuda is not None else None})

    @contextmanager
    def scope(self, index):
        t = self.t; state = self.states[index]; cpu = t.get_rng_state()
        cuda = t.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None
        try:
            t.set_rng_state(state['cpu'])
            if cuda is not None: t.cuda.set_rng_state(state['cuda'], self.device)
            yield
        finally:
            state['cpu'] = t.get_rng_state().clone()
            if cuda is not None: state['cuda'] = t.cuda.get_rng_state(self.device).clone()
            t.set_rng_state(cpu)
            if cuda is not None: t.cuda.set_rng_state(cuda, self.device)


def validate_config(study, run):
    c = copy.deepcopy(run['config']); arm = c['pt_arm']; stage = c['pt_stage']
    if arm not in SOURCE_ARMS+PT_ARMS+NATIVE_CONTROLS or stage not in ('source', 'continuation'):
        raise ValueError('Named PT source or continuation contract required')
    if (arm in SOURCE_ARMS) != (stage == 'source'): raise ValueError('PT stage/arm differs')
    if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS: raise ValueError('Original paper experiments excluded')
    if c['weight_decay'] != 0 or c['learning_rate'] not in (.0005,.001,.002) or c['dropout'] not in (.2,.4,.6):
        raise ValueError('Fixed useful nine-point domain and WD0 required')
    if c['width'] != (128 if arm == 'original_gnnm' else 32) or c['depth'] != 2 or c['classes'] < 2:
        raise ValueError('Declared source architecture dimensions required')
    expected = 4 if arm in ('factor_s4','original_gnnm','head_only','private4','middle_state_mean','separable4') else 1
    if c['members'] != expected or c['physical_slow_bodies'] != (4 if arm == 'separable4' else 1) or c['source_optimizer_count'] != 1:
        raise ValueError('Actual model/route/optimizer ownership differs')
    if c['initialization_seed'] != c['paired_seed'] + 1000003*run['body']:
        raise ValueError('Paid body seed ancestry differs')
    if set(c['pt_sources']) != set(PINS) or any(c['pt_sources'][k]['sha256'] != v for k,v in PINS.items()):
        raise ValueError('Every exact prototype/native/original source pin required')
    if set(c['node_inputs']) != {'graph','train','select'}: raise ValueError('Only GRAPH/TRAIN/SELECT fit authorities')
    for role in ('train','select'):
        if c['node_inputs'][role] != study['data']['roles'][role][str(run['split'])]: raise ValueError('Role custody differs')
    if run['budget']['unit'] != 'optimizer_updates' or run['budget']['policy'] != 'full_budget' or (study['stage'] != 'engineering' and run['budget']['maximum'] != 5000):
        raise ValueError('Full standardized5000 optimizer-update horizon required')
    if any(run['selection'][k] != v for k,v in {'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False}.items()):
        raise ValueError('Strict own SELECT accuracy, earliest ties, each update, checkpoint0 false')
    if stage == 'continuation':
        life = c.get('lifecycle', {})
        if set(life) != {'schema','parent_fit_ids','new_optimizer','parameter_initialization'} or life['schema'] != 'paid-parent-continuation-v1' or life['new_optimizer'] is not True or life['parameter_initialization'] != 'selected_parent_no_optimizer_reuse' or len(life['parent_fit_ids']) != 1 or not set(life['parent_fit_ids']).issubset(run['parents']):
            raise ValueError('Exact paid selected-parent fresh-optimizer lifecycle required')
    return c


def selected_parent(*, torch, root, study, run, config, device, execution_context):
    if not isinstance(execution_context, dict) or not {'registry','output_root'}.issubset(execution_context) or not set(execution_context).issubset({'registry','output_root','paid_parent_checkpoints'}):
        raise ValueError('Continuation requires explicit registry/output_root execution_context')
    registry = execution_context['registry']; out = Path(execution_context['output_root']).resolve()
    if not out.is_relative_to(Path(root).resolve()): raise ValueError('Parent output authority escapes project')
    identifier = config['lifecycle']['parent_fit_ids'][0]; row = registry.run(study['study_id'], identifier)
    actual_model_parents = {i for i in run['parents'] if registry.run(study['study_id'],i)['contract']['kind'] == 'model_fit'}
    if actual_model_parents != {identifier}: raise ValueError('Every and only direct paid source ancestor must be declared')
    parent = row['contract']; actual = resolve_selected_config(study, parent, registry)
    wanted = 'factor_f1' if config['pt_arm'] == 'continued_f1' else 'ordinary_p1'
    if row['state'] != 'complete' or parent['kind'] != 'model_fit' or actual['config'].get('pt_stage') != 'source' or actual['config'].get('pt_arm') != wanted:
        raise ValueError('Complete matching paid source ancestor required')
    if (parent['split'],parent['seed'],parent['body']) != (run['split'],run['seed'],run['body']):
        raise ValueError('Source split/seed/body differs from continuation')
    if parent['budget'] != run['budget'] or any(actual['config'][k] != config[k] for k in ('node_inputs','pt_sources','classes','width','paired_seed','initialization_seed','stage')):
        raise ValueError('Complete source role/model/horizon contracts differ')
    selected = row['result'].get('selected', {}); authority = row['result'].get('selected_checkpoint', {})
    if row['result'].get('status') != 'complete' or row['result'].get('completed_steps') != run['budget']['maximum']:
        raise ValueError('Parent did not pay the complete source horizon')
    step = selected.get('step')
    if type(step) != int or not 1 <= step <= parent['budget']['maximum'] or set(authority) != {'path','bytes','sha256','fit_id','step'} or authority['fit_id'] != identifier or authority['step'] != step:
        raise ValueError('Authenticated selected source checkpoint receipt required')
    relative = Path(authority['path'])
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Selected checkpoint path escapes paid fit')
    checkpoint = out/study['study_id']/identifier/relative
    observed = receipt(root, checkpoint)
    if any(observed[k] != authority[k] for k in ('bytes','sha256')): raise ValueError('Selected source checkpoint hash changed')
    adapter = make_adapter(torch=torch,project_root=root,study=study,run=actual,device=device)
    restored = restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=identifier)
    if restored['step'] != step or restored['validation'] != selected['validation']:
        raise ValueError('Selected source state/SELECT evidence differs')
    evidence = {'fit_id':identifier, 'contract_hash':fingerprint(parent), 'resolved_config_hash':fingerprint(actual['config']),
        'checkpoint':observed, 'selected_step':step, 'paid_source_updates':parent['budget']['maximum'],
        'optimizer_reused':False}
    return adapter, evidence


class RetainedPTAdapter(ClassifierAdapter):
    def train_step(self, step):
        counts = {'optimizer_updates':0,'optimizer_step_attempts':0,'backward_calls':0,'backward_attempts':0,
                  'model_forwards':0,'forward_attempts':0,'route_predictions':0,'supervised_examples':0}
        try:
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            counts['forward_attempts'] += 1; logits = self.logits_fn(self.model, True, step)
            counts.update(model_forwards=self.model.train_forwards, route_predictions=len(logits),
                          supervised_examples=len(logits)*len(self.train_ids))
            counts.update(self.model.operation_counts(True))
            if not self.preparation_charged:
                counts.update(self.preparation_counts); self.preparation_charged = True
            ce = self.t.stack([self.F.cross_entropy(z[self.train_ids],self.train_y) for z in logits]).mean()
            penalty = self.regularizer(self.model); loss = ce+penalty
            if not bool(self.t.isfinite(loss)): raise FloatingPointError('Nonfinite own PT objective')
            counts['backward_attempts'] += 1; loss.backward(); counts['backward_calls'] += 1
            if any(p.grad is not None and not bool(self.t.isfinite(p.grad).all()) for p in self.model.parameters()): raise FloatingPointError('Nonfinite PT gradient')
            counts['optimizer_step_attempts'] += 1; self.optimizer.step(); counts['optimizer_updates'] += 1
            return {'objective':self._scalar(loss),'ce':self._scalar(ce),'regularization':self._scalar(penalty),'operation_counts':counts}
        except BaseException as error: raise AdapterStepError(str(error),operation_counts=counts) from error

    def validate(self, step):
        self.model.eval()
        with self.t.no_grad():
            logits = self.logits_fn(self.model,False,step)[:,self.select_ids].double()
            logp = (self.t.logsumexp(logits.log_softmax(-1),dim=0)-math.log(len(logits)) if self.fusion == 'mean_probabilities' else logits.mean(0).log_softmax(-1))
            return {'accuracy':self._scalar(logp.argmax(-1).eq(self.select_y).double().mean()),
                    'nll':self._scalar(-logp.gather(1,self.select_y[:,None]).mean()),
                    'operation_counts':{'optimizer_updates':0,'backward_calls':0,'model_forwards':self.model.eval_forwards,'route_predictions':len(logits),'supervised_examples':0}}

    def capture_state(self):
        state = super().capture_state()
        state['retained_pt'] = {'streams':copy.deepcopy(self.streams.states), 'custody':copy.deepcopy(self.custody), 'preparation_charged':self.preparation_charged}
        return state

    def restore_state(self, state):
        if state['retained_pt']['custody'] != self.custody: raise ValueError('Actual PT graph/cache/parent ancestry changed')
        super().restore_state(state); self.streams.states = copy.deepcopy(state['retained_pt']['streams']); self.preparation_charged = state['retained_pt']['preparation_charged']


def make_adapter(*, torch, project_root, study, run, device, execution_context=None):
    from torch.nn import functional as F
    root = Path(project_root).resolve(); c = validate_config(study,run); arm = c['pt_arm']
    for row in c['pt_sources'].values(): bound(root,row)
    inputs = {k:torch.load(bound(root,v),map_location='cpu',weights_only=True) for k,v in c['node_inputs'].items()}
    graph, train, select = inputs['graph'],inputs['train'],inputs['select']
    if set(graph) != {'x','edge_index'} or any(set(p) != {'ids','y'} for p in (train,select)): raise ValueError('Separated exact node input schemas required')
    x, edge = graph['x'].to(device),graph['edge_index'].to(device)
    if x.ndim != 2 or x.dtype != torch.float32 or not bool(torch.isfinite(x.values() if x.is_sparse else x).all()) or edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2: raise ValueError('Finite raw FP32 X and int64 COO required')
    for p in (train,select):
        if p['ids'].dtype != torch.long or p['y'].dtype != torch.long or p['ids'].ndim != 1 or p['y'].shape != p['ids'].shape or len(p['ids']) == 0 or len(torch.unique(p['ids'])) != len(p['ids']): raise ValueError('Unique complete int64 role pack required')
        if int(p['ids'].min()) < 0 or int(p['ids'].max()) >= len(x) or int(p['y'].min()) < 0 or int(p['y'].max()) >= c['classes']: raise ValueError('Role IDs/classes outside graph')
    if torch.isin(train['ids'],select['ids']).any(): raise ValueError('TRAIN/SELECT overlap')
    if edge.numel() == 0 or int(edge.min()) < 0 or int(edge.max()) >= len(x): raise ValueError('Graph endpoints outside raw X')
    # Provider uses edge_index[src,dst]. Do not add loops/union/weights/clamps.
    adjacency = torch.sparse_coo_tensor(edge.flip(0),torch.ones(edge.shape[1],device=device),(len(x),len(x))).coalesce()
    degrees = torch.bincount(adjacency.indices()[0],minlength=len(x)).float().unsqueeze(1)
    pt = compile_source(root,c['pt_sources']['pt'],dropout=c['dropout']); pt.graph_check(torch,adjacency,degrees,len(x))
    dims = (len(x),x.shape[1],c['width'],c['classes']); seed = c['initialization_seed']
    native = compile_source(root,c['pt_sources']['native'],dropout=c['dropout'],dimensions=dims,seed=seed)
    custody = {'inputs':copy.deepcopy(c['node_inputs']), 'graph_hash':tensor_hash(adjacency), 'degree_hash':tensor_hash(degrees), 'X_hash':tensor_hash(x), 'sources':copy.deepcopy(c['pt_sources']), 'parent':None, 'H0_hash':None}
    source_adapter = None; parent_evidence = None; cores = []
    if c['pt_stage'] == 'continuation':
        source_adapter,parent_evidence = selected_parent(torch=torch,root=root,study=study,run=run,config=c,device=device,execution_context=execution_context)
        custody['parent'] = parent_evidence
    if arm in PT_ARMS:
        ancestor = source_adapter.model.native
        H0 = pt.cache_stem(torch,F,ancestor,x,adjacency,degrees); custody['H0_hash'] = tensor_hash(H0)
        if arm == 'separable4':
            # Explicit new joint-capacity control: one paid stem, four private bodies.
            for body in range(4):
                cores.append(pt.build_arm(torch,F,'mixed_single',[ancestor],[H0],adjacency,degrees,initialization_seeds=[seed+1000003*body]))
        else: cores = [pt.build_arm(torch,F,arm,[ancestor],[H0],adjacency,degrees,initialization_seeds=[seed])]
        bodies = [core['models'][0] for core in cores]
        if arm == 'separable4':
            stem = bodies[0].native.bodies[0].layers[0]
            for body in bodies[1:]: body.native.bodies[0].layers[0] = stem
        train_views = 16 if arm == 'separable4' else 4
        eval_views = 4 if arm in ('head_only','private4','middle_state_mean','separable4') else 1
        native_model = None
    elif arm == 'original_gnnm':
        namespace = {'__name__':'retained_pt_original','__file__':str(bound(root,c['pt_sources']['original']))}
        exec(compile(Path(namespace['__file__']).read_text(),namespace['__file__'],'exec'),namespace)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            native_model = namespace['TABMModel']('SAGE',2,x.shape[1],128,c['classes'],1,8,'LayerNorm',c['dropout'],tabm_inits=4,device='cpu').to(device)
        bodies = []; train_views = eval_views = 4
    else:
        if arm in NATIVE_CONTROLS:
            # Rebuild with this continuation's explicit dropout choice. Copying
            # the closure would retain the selected source's dropout instead.
            core = native.build_arm(torch,F,'F1' if arm == 'continued_f1' else 'P1',seed)
            native_model = core['models'][0].to(device)
            native_model.load_state_dict(source_adapter.model.native.state_dict(),strict=True)
            for p in native_model.parameters(): p.requires_grad_(True)
        else:
            core = native.build_arm(torch,F,{'ordinary_p1':'P1','factor_f1':'F1','factor_s4':'S4'}[arm],seed)
            native.inspect_parameter_ownership(core); native_model = core['models'][0].to(device)
        bodies = []; train_views = 4 if arm in NATIVE_CONTROLS else c['members']; eval_views = c['members']
    streams = Streams(torch,device,[seed+3000007+1000033*v for v in range(train_views)])
    class Bank(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.native = native_model; self.continuations = torch.nn.ModuleList(bodies)
            self.train_forwards = len(bodies) if bodies else 4 if arm in NATIVE_CONTROLS or arm=='original_gnnm' else 1
            self.eval_forwards = len(bodies) if bodies else 4 if arm=='original_gnnm' else 1
            self.fixed_versions = (x._version,adjacency._version,degrees._version)
        def operation_counts(self, training):
            if bodies:
                bank_calls = (5 if arm=='private4' else 0 if arm=='head_only' else 2)*len(bodies)
                decodes = train_views if training else eval_views
                return {'pt_source_model_forward_calls':len(bodies),'pt_native_head_decode_calls':decodes,
                        'pt_block_bank_calls':bank_calls,'pt_T_map_calls':4*bank_calls,
                        'pt_retained_operations_evaluated':4*bank_calls,'pt_graph_propagations':3*bank_calls+decodes,
                        'pt_tail_dropout_draws':decodes if training else 0}
            forwards = self.train_forwards if training else self.eval_forwards
            routes = train_views if training else eval_views
            if arm=='original_gnnm': return {'original_TABM_forward_calls':forwards,'original_SAGEConv_forward_calls':2*routes}
            return {'native_source_model_forward_calls':forwards,'native_first_layer_forward_calls':routes,
                    'native_last_layer_forward_calls':routes,'native_graph_propagations':2*routes,
                    'native_dropout_draws':2*routes if training else 0}
        def guard(self):
            if self.fixed_versions != (x._version,adjacency._version,degrees._version): raise ValueError('Raw X/A/degree mutated')
        def forward(self):
            self.guard()
            if bodies:
                values = [body(route_rng_context=lambda view,b=b:streams.scope(4*b+view)) for b,body in enumerate(self.continuations)]
                out = torch.cat(values,dim=0)
            elif arm == 'original_gnnm':
                g = types.SimpleNamespace(edge_index=edge); values = []
                for view in range(4):
                    with streams.scope(view): values.append(self.native(g,x,tabm_seed=view))
                out = torch.stack(values)
            elif arm in NATIVE_CONTROLS and self.training:
                out = torch.cat([self.native(x,adjacency,degrees,route_rng_context=lambda route,v=v:streams.scope(v)) for v in range(4)])
            else: out = self.native(x,adjacency,degrees,route_rng_context=streams.scope)
            self.guard(); return out
        def regularizer(self):
            if bodies: return torch.stack([body.regularizer() for body in self.continuations]).mean()
            return x.new_zeros(()) if arm == 'original_gnnm' else self.native.regularizer()
    model = Bank().to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=0,betas=(.9,.999),eps=1e-8)
    if optimizer.state: raise ValueError('Fresh continuation optimizer must have no inherited moments')
    selector = {'metric':'accuracy','direction':'max','ties':'first','patience_steps':run['budget']['maximum']+1,'save_every':1}
    adapter = RetainedPTAdapter(torch,F,model,optimizer,logits_fn=lambda m,training,step:m(),
        train_ids=train['ids'].to(device),train_y=train['y'].to(device),select_ids=select['ids'].to(device),select_y=select['y'].to(device),
        regularizer=lambda m:m.regularizer(),fusion=c['fusion'],selector=selector,
        provenance={'run_id':run['run_id'],'source':study['source'],'data':study['data'],'config':c,'actual_model_seed':seed,'contract':CONTRACT,'actual_parent':parent_evidence,'actual_H0_hash':custody['H0_hash'],'fresh_optimizer':True})
    adapter.streams = streams; adapter.custody = custody; adapter.cores = cores
    adapter.preparation_counts = {'selected_source_checkpoint_restores':1,'source_optimizer_restores_for_evidence_only':1,
        'H0_preparations':1 if arm in PT_ARMS else 0,'H0_native_first_map_products':2 if arm in PT_ARMS else 0,
        'H0_graph_propagations':1 if arm in PT_ARMS else 0} if source_adapter is not None else {}
    adapter.preparation_charged = False
    return adapter
