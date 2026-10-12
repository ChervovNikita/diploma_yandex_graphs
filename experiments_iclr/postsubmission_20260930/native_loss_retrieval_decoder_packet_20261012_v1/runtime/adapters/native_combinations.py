"""Literal source loss/retrieval/decoder ports. Scientific imports are lazy.

Historical data loaders, fits, outcome writers and eligibility are never run.
Provider dimensions/counts and prospective controller/search are explicit edits.
"""
import ast
import copy
import math
from types import SimpleNamespace
from . import common_native_wrapper as common
from .classifier import ClassifierAdapter

EXTRA = {
 'gncl': '24c657ac77426a499eba095884e9763d46377c25da09beb3510248fc6d9d6d74',
 'retrieval': 'a6144527a6f5e551deeecb89cf312cb54473df69a48e4c0cd6c27a08ae01e0fc',
 'memory': 'c568c41f58e741f96b13a552bbac4df6ef137e280db8629a6165d732c760145f',
 'extension': 'ce7d9c5276595583ce579ea8a3f4be68be4fed48b6c185b68605661a36be4a98',
 'decoder': 'e2f4da1953b10f81ba20817083e11b070ef4acad419af0fe0e52badfb5b7d1f8',
}
FAMILIES = {
 'gncl_supcon': {'backbones': ('SAGE',), 'arms': ('shared4_GNCL', 'shared4_own_SupCon', 'shared4_GNCL_SupCon', 'ordinary_M1_SupCon', 'factorized_M1_SupCon'),
   'i4': {'genuine_factorized_I4_SupCon': 'factorized_M1_SupCon'}},
 'nonlocal_label_retrieval': {'backbones': ('GCN', 'GAT', 'SAGE'), 'arms': ('live_shared4', 'detached_shared4', 'ordinary_M1_live'),
   'i4': {'genuine_ordinary_I4_live': 'ordinary_M1_live'}},
 'shared_recurrent_class_decoder': {'backbones': ('GCN', 'GAT', 'SAGE'), 'arms': ('ordinary_M1_graphP_commonB', 'factorized_allmap_M1_graphP_commonB', 'joint_untied4_graphP_commonB', 'shared4_graphP_commonB', 'shared4_graphP_privateB4', 'shared4_Pidentity_commonB'),
   'i4': {'factorized_allmap_genuine_I4_graphP_ownB': 'factorized_allmap_M1_graphP_commonB'}},
}
CONTROLS = ('ordinary_M1', 'factorized_allmap_M1', 'shared4_unchanged', 'original_gnnm')
CONTROL_I4 = {'ordinary_I4': 'ordinary_M1', 'equipped_I4': 'factorized_allmap_M1'}
CONTRACT = {
 'schema': 'native-loss-retrieval-decoder-prospective-port-v1', 'source_exact_recipe': False,
 'source_anchor': common.SOURCE_ANCHOR,
 'gncl': 'literal .5 mean own CE + .5 probability-pool CE; SupCon .05 with temperature .2 within each route on all TRAIN, no diagonal, same-label positives; no global/cross-route contrastive replacement',
 'retrieval': 'literal all-TRAIN native CE + common-half-query retrieval NLL; class-balanced CPU query RNG seed+5000011 reset per fit/member and restored with checkpoint; remove all Q labels from every support; cosine .1, TRAIN one-hot values; .5 native/.5 retrieval logaddexp serving outside TRAIN',
 'retrieval_detached': 'only attention log probabilities detached; ordinary/I4 controls are ordinary native, not factorized',
 'decoder': 'literal five q_next=softmax(Z+(Pq)B) steps; zero B before optimizer, live states; .5 native CE+.5 decoded CE; incoming D^-1(A_off+I) preserves factual offdiagonal multiplicity, adds identity once; native graph unchanged',
 'decoder_ownership': 'joint_untied4 is four separately constructed factorized bodies, common B, one AdamW; shared4 common/private B variants use one physical native body; genuine I4 has four separate optimizers and own B selected independently',
 'controls': common.CONTRACT,
 'new_roles': 'full WikiCS20/Photo10 GRAPH/TRAIN/SELECT/CAL/evaluate authority; original five datasets excluded for every arm',
 'controller': 'full5000 replaces historical1000/patience300; every-update strict first SELECT accuracy, checkpoint0 false, no early termination',
 'HPO': 'nine paired LR/dropout points around .001/.2; two search seeds and three comparative seeds; I4 uses component M1 source HPO and then four fresh comparative fits; no member0 artifact reuse',
 'serving': 'source probability mean except literal original R/S/B GNNM mean raw logits; common after-pool scalar CAL500, retain raw/calibrated outcomes',
 'historical_eligibility': 'unchanged; this is disabled source preparation only',
 'decoder_backbone_scope': 'base source explicitly supports GCN/GAT/SAGE; GCN/GAT decoder cells are prospective source-compatible extensions beyond historical representative SAGE roster; scientific expansion admission remains pending',
 'resource_status': 'unmeasured placeholders, no ETA/runtime pin/CUDA admission',
}


def bindings_common(bindings):
    return {key: bindings[key] for key in common.SOURCES}


def render_sources(root, bindings, family, features, classes, train_count):
    if set(bindings) != set(common.SOURCES) | set(EXTRA):
        raise ValueError('Exact twelve-source binding roster required')
    for key, sha in EXTRA.items():
        if bindings[key]['sha256'] != sha:
            raise ValueError('Frozen extra source changed: ' + key)
    common.render_sources(root, bindings_common(bindings), features, classes)
    paths = {key: common.bound(root, row) for key, row in bindings.items()}
    trees = {key: ast.parse(path.read_text()) for key, path in paths.items()}
    def methods(tree, nested=False):
        if nested:
            fc = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'family_class')
            cls = next(n for n in fc.body if isinstance(n, ast.ClassDef) and n.name == 'Family')
        else:
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Family')
        return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    changes = []
    class Bind(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'self.Model':
                assert len(node.args) == 9 and node.args[2].value == 300 and node.args[4].value == 10
                node.args[2], node.args[4] = ast.Constant(features), ast.Constant(classes)
                changes.extend(['Model.input_dim:300->features', 'Model.output_dim:10->classes'])
            if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'torch.randint':
                assert ast.unparse(node.args[2]) == '(1, 300)'
                node.args[2] = ast.Tuple(elts=[ast.Constant(1), ast.Constant(features)], ctx=ast.Load())
                changes.append('stem_sign_shape:(1,300)->(1,features)')
            if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'self.memory.common_half_query':
                assert node.args[3].value == 10
                node.args[3] = ast.Constant(classes)
                changes.append('query_class_domain:10->classes')
            return node
        def visit_Compare(self, node):
            self.generic_visit(node)
            if family == 'gncl_supcon' and ast.unparse(node) == 'len(targets) != 580':
                node.comparators[0] = ast.Constant(train_count)
                changes.append('SupCon_complete_TRAIN_count:580->train_count')
            return node
        def visit_Tuple(self, node):
            self.generic_visit(node)
            if family == 'shared_recurrent_class_decoder' and ast.unparse(node) in ('(members, 10, 10)', '(10, 10)'):
                for n in node.elts:
                    if isinstance(n, ast.Constant) and n.value == 10: n.value = classes
                changes.append('B_class_shape:10->classes')
            return node
    selected = []
    if family == 'gncl_supcon':
        m = methods(trees['gncl'])
        selected = [m[k] for k in ('seeds', 'make', 'scope', 'logits', 'training_outputs', 'within_route_supcon', 'probability_pool_ce')]
        loop = next(n for n in ast.walk(m['fit']) if isinstance(n, ast.For) and ast.unparse(n.target) == 'step')
        stop = next(i for i,n in enumerate(loop.body) if ast.unparse(n) == 'model.eval()')
        train = ast.FunctionDef(name='source_train', args=ast.parse('def f(self, model, optimizer, streams, bank, members, objective, use_supcon): pass').body[0].args,
          body=[copy.deepcopy(m['fit'].body[0])] + copy.deepcopy(loop.body[:stop]) + ast.parse("return {'total_ce':train_loss, 'native_ce':train_own_ce, 'pool_ce':train_pool_ce, 'supcon':train_supcon, 'supcon_routes':train_supcon_routes}").body, decorator_list=[])
        selected.append(train)
    elif family == 'nonlocal_label_retrieval':
        m = methods(trees['retrieval'])
        selected = [m[k] for k in ('seeds', 'make', 'scope', 'full_native', 'memory_read', 'logits', 'retrieval_train_step')]
    elif family == 'shared_recurrent_class_decoder':
        base = methods(trees['family'])
        # Use the exact extension's construct_native derivation, before optimizer/device.
        construction = copy.deepcopy(base['make']); construction.name = 'construct_native'
        stop = next(i for i,n in enumerate(construction.body) if ast.unparse(n) == 'model = wrapper if members > 1 else body')
        construction.body = construction.body[:stop] + ast.parse('return body, wrapper').body
        ext = methods(trees['extension'], nested=True)
        selected = [base['seeds'], base['scope'], base['synchronize'], construction] + [ext[k] for k in ('make', 'predictions', 'train_update', 'logits')]
        init = ext['__init__']
        start = next(i for i,n in enumerate(init.body) if ast.unparse(n) == 'torch = self.torch')
        stop = next(i for i,n in enumerate(init.body) if ast.unparse(n).startswith('self.source_hashes.update'))
        support = ast.FunctionDef(name='prepare_support', args=ast.parse('def f(self): pass').body[0].args,
            body=copy.deepcopy(init.body[start:stop]), decorator_list=[])
        selected.append(support)
    else: raise ValueError('Unknown source family')
    originals = copy.deepcopy(selected)
    selected = [Bind().visit(copy.deepcopy(n)) for n in selected]
    expected = ['Model.input_dim:300->features', 'Model.output_dim:10->classes', 'stem_sign_shape:(1,300)->(1,features)']
    expected += ['SupCon_complete_TRAIN_count:580->train_count'] if family == 'gncl_supcon' else ['query_class_domain:10->classes'] if family == 'nonlocal_label_retrieval' else ['B_class_shape:10->classes'] * 2
    if changes != expected: raise ValueError('Unexpected provider AST edits: ' + repr(changes))
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='SourceBuild', bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[]))
    return {'code': ast.unparse(tree)+'\n', 'family':family, 'features':features, 'classes':classes, 'train_count':train_count,
        'binding_changes':changes, 'source_AST_hashes':{n.name:common.ast_hash(n) for n in originals},
        'bound_AST_hashes':{n.name:common.ast_hash(n) for n in selected}, 'learned_law_edits':0}


def arm_shape(family, arm):
    if arm in CONTROLS:
        return (4 if arm in ('shared4_unchanged', 'original_gnnm') else 1), 1
    bank = arm.startswith('shared4_') or arm in ('live_shared4', 'detached_shared4', 'joint_untied4_graphP_commonB')
    return (4 if bank else 1), (4 if arm == 'joint_untied4_graphP_commonB' else 1)


def construct_source(torch, root, bindings, *, family, arm, features, classes, train_count, seed, body, backbone, dropout, learning_rate, device):
    if arm not in FAMILIES[family]['arms'] or backbone not in FAMILIES[family]['backbones']:
        raise ValueError('Unsupported exact source family cell')
    rendered = render_sources(root, bindings, family, features, classes, train_count)
    namespace = {'math':math, 'time':__import__('time')}
    if family == 'shared_recurrent_class_decoder': namespace['decoder'] = common.load_module(root, bindings['decoder'])
    exec(compile(rendered['code'], '<literal_family_provider_binding>', 'exec'), namespace)
    source = namespace['SourceBuild']()
    source.torch, source.F = torch, torch.nn.functional
    source.device, source.cuda = torch.device(device), torch.device(device).index or 0
    source.cfg = {'backbone':backbone, 'depth':2, 'hidden':128, 'rank':16, 'dropout':dropout, 'learning_rate':learning_rate}
    source.Model = common.load_module(root, bindings['native']).Model
    source.set_seed = common.load_module(root, bindings['driver']).set_seed
    source.routes, source.factors = (common.load_module(root, bindings[k]) for k in ('routes','factors'))
    source.current = {'arm':arm}
    members, bodies = arm_shape(family,arm)
    if family == 'nonlocal_label_retrieval':
        source.memory = common.load_module(root, bindings['memory'])
        source.process_counts = dict.fromkeys(('fullgraph_native_member_trajectories', 'native_graph_block_calls', 'retrieval_member_paths', 'similarity_MACs', 'query_draws', 'backwards', 'optimizer_steps'),0)
        source.query_generator = torch.Generator(device='cpu').manual_seed(seed+5000011)
        source.latest_native_logits = None
    factorized = not arm.startswith('ordinary_M1')
    model, optimizer, streams = source.make(seed, body, 'baseline', factorized, members)
    return source, model, optimizer, streams, rendered


def attach_adapter(torch, source, model, optimizer, streams, *, family, arm, x, edges, train_ids, train_y, select_ids, select_y, provenance):
    members, bodies = arm_shape(family, arm)
    labels = torch.zeros(len(x),dtype=torch.long,device=x.device); labels[train_ids] = train_y
    mask = torch.zeros(len(x),dtype=torch.bool,device=x.device); mask[train_ids] = True
    source.data = SimpleNamespace(x=x, edge_index=edges, y=labels, train_mask=mask)
    source.train_ids = train_ids
    source.batch = {'x':x, 'edge_index':edges, 'graph':SimpleNamespace(edge_index=edges)}
    bank = members == 4
    if family == 'shared_recurrent_class_decoder': source.prepare_support()
    if family in ('gncl_supcon','nonlocal_label_retrieval') and torch.bincount(train_y,minlength=provenance['classes']).min().item() < 2:
        raise ValueError('Source requires at least two TRAIN examples per class')
    def predictions(current,ids,step):
        if torch.isin(ids,train_ids).any() and family == 'nonlocal_label_retrieval':
            raise ValueError('Retrieval serve queries must be disjoint from TRAIN')
        scores = source.logits(current,streams,bank,ids)
        if family == 'nonlocal_label_retrieval': adapter.last_native_rows = source.latest_native_logits
        if family == 'shared_recurrent_class_decoder': adapter.last_native_rows = source._last_native_logits
        return scores
    def unused_full(current,training,step):
        if family == 'nonlocal_label_retrieval':
            raise ValueError('Retrieval requires explicit heldout query rows')
        return predictions(current,torch.arange(len(x),device=x.device),step)
    def train(adapter,step):
        if family == 'gncl_supcon':
            objective = 'gncl_half' if arm in ('shared4_GNCL','shared4_GNCL_SupCon') else 'own_only'
            components = source.source_train(model,optimizer,streams,bank,members,objective,arm!='shared4_GNCL')
            total = components['total_ce'];counts={'optimizer_updates':1,'backward_calls':1,'fullgraph_native_member_trajectories':members,
                'native_graph_block_calls':2*members,'supervised_examples':len(train_ids),
                'supcon_route_grams':members if arm!='shared4_GNCL' else 0,
                'supcon_gram_MACs':members*len(train_ids)**2*128 if arm!='shared4_GNCL' else 0}
        elif family == 'nonlocal_label_retrieval':
            before = dict(source.process_counts)
            components = source.retrieval_train_step(model,optimizer,streams,bank)
            total = components['total_ce']; counts={k:source.process_counts[k]-before[k] for k in before}
            counts.update(optimizer_updates=1,backward_calls=1,supervised_examples=len(train_ids),query_examples=len(train_ids)//2)
        else:
            total = source.train_update(model,optimizer,streams,bank,members)
            components={'total_ce':total}
            graph = arm!='shared4_Pidentity_commonB'
            counts={'optimizer_updates':1,'backward_calls':1,'physical_slow_bodies':bodies,'fullgraph_native_member_trajectories':members,
                'native_graph_block_calls':2*members,'supervised_examples':len(train_ids),'decoder_route_steps':5*members,
                'decoder_edge_class_values':5*members*len(source.decoder_support[0])*provenance['classes'] if graph else 0,
                'decoder_dense_class_MACs':5*members*len(x)*provenance['classes']**2}
        if not math.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite exact source update')
        return {'objective':total,'ce':components.get('native_ce',total),'regularization':0.,'objective_components':components,'operation_counts':counts}
    def capture():
        value={'member_streams':copy.deepcopy(streams)}
        if family=='nonlocal_label_retrieval':value['query_rng']=source.query_generator.get_state().clone()
        return value
    def restore(value):
        if set(value)!={'member_streams'} | ({'query_rng'} if family=='nonlocal_label_retrieval' else set()):raise ValueError('Incomplete source RNG checkpoint')
        streams[:]=copy.deepcopy(value['member_streams'])
        if family=='nonlocal_label_retrieval':source.query_generator.set_state(value['query_rng'])
    adapter = ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=unused_full,
        train_ids=train_ids,train_y=train_y,select_ids=select_ids,select_y=select_y,regularizer=lambda m:x.new_zeros(()),
        fusion='mean_probabilities',selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5000,'save_every':1},
        provenance=provenance,external_capture=capture,external_restore=restore,custom_train=train)
    adapter.predict_rows=lambda ids,step:predictions(adapter.model,ids.to(x.device),step)
    adapter.source, adapter.streams = source, streams
    return adapter


def make_adapter(*,torch,project_root,study,run,device):
    c=run['config'];family,arm=c['combination_family'],c['combination_arm']
    if study['dataset'] not in common.DATASETS or c['backbone'] not in FAMILIES[family]['backbones']:
        raise ValueError('Only declared new homogeneous provider cells')
    if arm in CONTROLS:
        copied=copy.deepcopy(run);copied['config']['wrapper_arm']=arm;copied['config']['wrapper_sources']=bindings_common(c['wrapper_sources'])
        return common.make_adapter(torch=torch,project_root=project_root,study=study,run=copied,device=device)
    members,bodies=arm_shape(family,arm)
    if (c['members']!=members or c['physical_slow_bodies']!=bodies or c['source_optimizer_count']!=1 or c['width']!=128 or c['depth']!=2
        or c['weight_decay']!=0 or c['learning_rate'] not in (.0005,.001,.002) or c['dropout'] not in (.1,.2,.4)):
        raise ValueError('Source ownership/capacity/domain differs')
    def load(row):return torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True)
    graph,train,select=(load(c['node_inputs'][k]) for k in ('graph','train','select'))
    if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'}:raise ValueError('Exact GRAPH/TRAIN/SELECT authorities required')
    x,edges=graph['x'].to(device),graph['edge_index'].to(device);metadata=c['provider_graph']
    if (x.dtype!=torch.float32 or tuple(x.shape)!=(metadata['nodes'],metadata['features']) or edges.dtype!=torch.long
        or tuple(edges.shape)!=(2,metadata['edges']) or int(edges[0].eq(edges[1]).sum())!=len(x)):
        raise ValueError('Custodied factual graph differs')
    for role in (train,select):
        ids,y=role['ids'],role['y']
        if (ids.dtype!=torch.long or y.dtype!=torch.long or ids.ndim!=1 or y.shape!=ids.shape or not len(ids)
            or ids.min()<0 or ids.max()>=len(x) or y.min()<0 or y.max()>=c['classes']
            or not torch.equal(ids,ids.sort().values) or len(torch.unique(ids))!=len(ids)):
            raise ValueError('Sorted unique complete role rows required')
    if torch.isin(train['ids'],select['ids']).any():raise ValueError('TRAIN/SELECT overlap')
    train_ids,train_y,select_ids,select_y=(row.to(device) for row in (train['ids'],train['y'],select['ids'],select['y']))
    if len(train_ids)!=c['train_count']:raise ValueError('Complete source TRAIN count binding differs')
    source,model,optimizer,streams,rendered=construct_source(torch,project_root,c['wrapper_sources'],family=family,arm=arm,
        features=x.shape[1],classes=c['classes'],train_count=len(train_ids),seed=run['seed'],body=run['body'],backbone=c['backbone'],
        dropout=c['dropout'],learning_rate=c['learning_rate'],device=device)
    return attach_adapter(torch,source,model,optimizer,streams,family=family,arm=arm,x=x,edges=edges,train_ids=train_ids,train_y=train_y,
        select_ids=select_ids,select_y=select_y,provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,
            'actual_model_seed':source.seeds(run['seed'],run['body'])[0],'classes':c['classes'],'source_contract':CONTRACT,'provider_AST_bindings':rendered})
