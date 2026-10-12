"""Direct random PT/GRU/landmark engineering builders; never paid ancestors.

Preparation reads stdlib metadata only. Execution is a separate reviewed-case
operation and is not authorized by preparing this manifest. No score, selector,
source checkpoint, registry, fit controller or paid-parent API is used.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import statistics
import sys
import time
import types

CAP_HELPER='practical_resource_qualifier_source_20261012_v2/runtime/qa/practical_zero_update_resources.py'
CAP_SHA='effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0'
PHYSICS={'owner':{'pid':640426,'start_ticks':6055039154,'pgid':640426},'child':{'pid':640428,'start_ticks':6055039165,'pgid':640428}}
PHYSICS_DIR='private_GRU_acquisition_root_20261011_v1/owner_v1'
READOUT_DIR='private_GRU_readout_after_whole_acquisition_root_20261011_v1'
READOUT_WATCHER={'pid':641081,'start_ticks':6055370992,'pgid':641081}
PACKETS={'pt':'retained_pt_packet_20261012_v2','gru':'gru_memory_packet_20261012_v3','landmark':'landmark_memory_packet_20261012_v3'}
ANCHORS={'pt':.4,'gru':.4,'landmark':.5}; FRONTIERS={'pt':.6,'gru':.6,'landmark':.7}


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def local(root,relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Confined project-relative source required')
    result=root/path
    if result.is_symlink() or not result.resolve().is_relative_to(root): raise ValueError('Source may not escape project')
    return result
def receipt(root,path):
    raw=path.read_bytes(); return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
    path=local(root,row['path'])
    if receipt(root,path)!=row: raise ValueError('Source/data bytes changed')
    return path
def module(root,row,name):
    path=bound(root,row); spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
def role_paths(value):
    if isinstance(value,str): return {value}
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=set(value): return {value['path']}
        return set().union(*(role_paths(v) for v in value.values())) if value else set()
    return set()


def process_identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists(): return None
    a=path.read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':int(a[19]),'pgid':int(a[2]),'state':a[0]}


def same_birth(current,expected):
    return current is not None and all(current.get(k)==v for k,v in expected.items())


def readout_watcher_custody(root,watcher_binding,acquisition_start,*,process_reader=process_identity):
    if not isinstance(watcher_binding,dict) or set(watcher_binding)!={'launch','watch_start','expected_identity'} or watcher_binding['expected_identity']!=READOUT_WATCHER: raise ValueError('Actual original readout watcher launch/start custody required')
    launch_path=bound(root,watcher_binding['launch']); start_path=bound(root,watcher_binding['watch_start'])
    if launch_path!=local(root,READOUT_DIR+'/LAUNCH.json') or start_path!=local(root,READOUT_DIR+'/watch_v1/WATCH_START.json'): raise ValueError('Original readout watcher metadata paths changed')
    launch=json.loads(launch_path.read_text()); start=json.loads(start_path.read_text())
    if not same_birth(launch.get('watcher_observed'),READOUT_WATCHER) or launch.get('watcher_pid')!=READOUT_WATCHER['pid'] or launch.get('watch_start')!=start or start.get('pid')!=READOUT_WATCHER['pid'] or start.get('stage')!='waiting_for_whole_acquisition': raise ValueError('Original readout watcher birth/start linkage changed')
    if launch.get('no_model_retry') is not True or launch.get('no_readout_retry') is not True or launch.get('partial_predictions_opened') is not False or launch.get('launch_exit') is not None: raise ValueError('Original admitted readout watcher launch policy changed')
    if not isinstance(start.get('config'),dict) or start['config'].get('sha256')!=launch.get('config_sha256'): raise ValueError('Original watcher CONFIG launch/start linkage changed')
    command=launch.get('command')
    if not isinstance(command,list) or len(command)!=5 or command[1:3]!=['-B','-u'] or not str(command[-2]).endswith('/private_GRU_whole_readout_watcher_source_20261011_v1/watcher.py') or command[-1]!=start['config'].get('path') or not str(command[-1]).endswith('/'+READOUT_DIR+'/CONFIG.json'): raise ValueError('Original readout watcher command/CONFIG linkage changed')
    linked=start.get('acquisition_owner_start',{})
    if any(linked.get(k)!=acquisition_start[k] for k in ('bytes','sha256')) or not str(linked.get('path','')).endswith('/'+PHYSICS_DIR+'/OWNER_START.json'): raise ValueError('Readout watcher is bound to another acquisition owner')
    negative=('partial_metrics_opened','partial_owner_logs_opened','scientific_gate_used_for_handoff','training_or_qualification_launched','model_retry','readout_retry')
    if any(start.get(k) is not False for k in negative) or start.get('complete') is not False: raise ValueError('Original watcher start lifecycle schema changed')
    current=process_reader(READOUT_WATCHER['pid']); end_path=local(root,READOUT_DIR+'/watch_v1/WATCH_END.json')
    evidence={'launch':watcher_binding['launch'],'watch_start':watcher_binding['watch_start'],'expected_identity':READOUT_WATCHER,'scientific_outcome_read':False}
    if end_path.exists():
        end=json.loads(end_path.read_text())
        stable=('config','root_decision','watcher_manifest','reader_source_manifest','acquisition_source_manifest','acquisition_config','acquisition_owner_start','acquisition_launch','confidence_clarification','finite_wall_seconds','cleanup_reserve_seconds')+negative
        if any(k not in start or end.get(k)!=start[k] for k in stable) or type(end.get('complete')) is not bool or not isinstance(end.get('last_stage'),str) or not end['last_stage'] or type(end.get('seconds')) not in (int,float) or not math.isfinite(end['seconds']) or end['seconds']<0: raise ValueError('Original watcher terminal lifecycle metadata differs from admitted start')
        if same_birth(current,READOUT_WATCHER): raise ValueError('Old readout watcher identity remains present after terminal receipt')
        return {**evidence,'mode':'original_readout_watcher_terminal_metadata_verified','watch_end':receipt(root,end_path),'old_identity_present':False,'current_pid_observation':current}
    if not same_birth(current,READOUT_WATCHER) or current.get('state')=='Z': raise ValueError('Exact original readout watcher missing without terminal metadata')
    return {**evidence,'mode':'original_readout_watcher_exact_process_live','observed':current}


def physics_custody(root,start_binding,watcher_binding=None,*,process_reader=process_identity):
    start=json.loads(bound(root,start_binding).read_text())
    if bound(root,start_binding)!=local(root,PHYSICS_DIR+'/OWNER_START.json') or any(start.get(k)!=v for k,v in PHYSICS.items()): raise ValueError('Exact Physics scientific-owner identities changed')
    if start.get('stage')!='acquire' or start.get('direct_child_wait') is not True or start.get('retry') is not False or not isinstance(start.get('command'),list) or not start['command']: raise ValueError('Exact Physics owner-start lifecycle schema required')
    watcher=readout_watcher_custody(root,watcher_binding,start_binding,process_reader=process_reader)
    end=local(root,PHYSICS_DIR+'/OWNER_END.json')
    if end.exists():
        terminal=json.loads(end.read_text()); terminal_fields={'status','exit_code','timed_out','seconds','child_pid_absent','owned_cuda_pid_absent'}
        if set(terminal)!=set(start)|terminal_fields or any(terminal.get(k)!=v for k,v in start.items()): raise ValueError('Physics terminal owner/child/command or start contract changed')
        if terminal.get('status') not in ('COMPLETE_EXIT_0','FAILED_RETAINED') or type(terminal.get('exit_code')) is not int or type(terminal.get('timed_out')) is not bool or type(terminal.get('seconds')) not in (int,float) or not math.isfinite(terminal['seconds']) or terminal['seconds']<0: raise ValueError('Physics terminal status/exit/deadline schema invalid')
        success=terminal['exit_code']==0 and terminal['timed_out'] is False
        if (terminal['status']=='COMPLETE_EXIT_0')!=success or terminal.get('child_pid_absent') is not True or terminal.get('owned_cuda_pid_absent') is not True: raise ValueError('Physics terminal reaping flags/status inconsistent')
        observed={name:process_reader(expected['pid']) for name,expected in PHYSICS.items()}
        if any(same_birth(observed[name],expected) for name,expected in PHYSICS.items()): raise ValueError('Old Physics owner/child identity remains present after terminal receipt')
        return {'mode':'Physics_terminal_owner_schema_and_original_readout_watcher_verified','start':start_binding,'end':receipt(root,end),'watcher':watcher,'old_identities_present':False,'current_pid_observations':observed,'scientific_outcome_read':False}
    observed={}
    for name,expected in PHYSICS.items():
        row=process_reader(expected['pid'])
        if not same_birth(row,expected) or row['state']=='Z': raise ValueError('Physics PID/start/PGID identity changed without terminal closure')
        observed[name]=row
    return {'mode':'both_exact_Physics_processes_live','start':start_binding,'identities':observed,'watcher':watcher,'scientific_outcome_read':False}


def prepare_manifest(root,helper_binding,physics_start,physics_watcher,cap_binding):
    """Finite source anchor/frontier plus actual four-prototype bank roster."""
    cases=[]; packets=[]
    for family,name in PACKETS.items():
        entry=json.loads((root/name/'MASTER_INPUT_ENTRY.json').read_text()); plan=json.loads(bound(root,entry['plan']).read_text())
        if plan['scientific_launch_enabled'] is not False: raise ValueError('Only exact disabled packets admitted')
        packets.append({'family':family,'entry':receipt(root,root/name/'MASTER_INPUT_ENTRY.json'),'plan':entry['plan'],'plan_hash':digest(plan)})
        for spec_row in plan['studies']:
            path=local(root,spec_row['path']); study=json.loads(gzip.decompress(path.read_bytes()))
            if digest(study)!=spec_row['spec_hash'] or study['dataset'] not in ('wikics','amazon-photo'): raise ValueError('Exact complete new-role packet required')
            custody_rows=[r for r in study['source']['files'] if r['path'].endswith('/'+study['dataset']+'/CUSTODY.json') and '/prepared_data/' in r['path']]
            if len(custody_rows)!=1: raise ValueError('One provider metadata custody required')
            custody=json.loads(bound(root,custody_rows[0]).read_text()); counts={r['split_id']:r['role_counts']['train'] for r in custody['splits']}; mask=min(counts,key=lambda k:(-counts[k],k))
            prohibited=set().union(*(role_paths(study['data']['roles'][role]) for role in ('select','cal','evaluate')))
            for run in study['runs']: prohibited.update(role_paths(run['config'].get('endpoint_inputs',{})))
            fits=[r for r in study['runs'] if r['kind']=='model_fit']; arms=sorted({r['config'][family+'_arm'] for r in fits})
            for arm in arms:
                for dropout in (ANCHORS[family],FRONTIERS[family]):
                    candidates=[r for r in fits if r['config'][family+'_arm']==arm and r['config']['dropout']==dropout and r['config']['stage']=='search']
                    original=min(candidates,key=lambda r:(r['seed'],r['run_id'])); cohort=min((r for r in fits if r['config'][family+'_arm']==arm and r['split']==mask and r['config']['stage']=='comparative'),key=lambda r:(r['body'],r['seed']))
                    recipe=copy.deepcopy(original); recipe['config']['node_inputs']={k:cohort['config']['node_inputs'][k] for k in ('graph','train')}
                    add_case(cases,root,helper_binding,family,study,path,spec_row,recipe,prohibited,custody_rows[0],counts[mask],1,None)
            for bank in study['runs']:
                if bank['kind']!='ensemble_assembly' or bank['split']!=mask or len(bank['config'].get('fit_parents',[]))!=4: continue
                # One independent multiplicity case per named complete bank;
                # pick one fixed comparative seed, never select by predictions.
                if bank['seed']!=min(r['seed'] for r in study['runs'] if r['kind']=='ensemble_assembly'): continue
                original=next(r for r in fits if r['run_id']==bank['config']['fit_parents'][0]); arm=original['config'][family+'_arm']
                search=min((r for r in fits if r['config'][family+'_arm']==arm and r['config']['dropout']==FRONTIERS[family] and r['config']['stage']=='search'),key=lambda r:(r['seed'],r['run_id']))
                recipe=copy.deepcopy(search); recipe['config']['node_inputs']={k:original['config']['node_inputs'][k] for k in ('graph','train')}
                add_case(cases,root,helper_binding,family,study,path,spec_row,recipe,prohibited,custody_rows[0],counts[mask],4,bank['method'])
    return {'schema':'staged-random-direct-resource-manifest-v1','helper':helper_binding,'cap_helper':cap_binding,'packets':packets,'cases':cases,
        'Physics_owner_start':physics_start,'Physics_original_watcher':physics_watcher,'execution_authorized':False,'scientific_launch_enabled':False,
        'actual_optimizer_updates_allowed':0,'case_timeout_seconds':600,'warm_iterations':3,'no_retry':True,'no_score':True,
        'qualification_limits':'random exact-source GRAPH/TRAIN storage/FWD/backward/serializer proxy only; paid/trained cache and restore, scientific source acquisition, SELECT/CAL/evaluation and full-study ETA unqualified'}


def add_case(cases,root,helper,family,study,path,spec_row,recipe,prohibited,custody,train_rows,bodies,bank):
    c=recipe['config']; arm=c[family+'_arm']; pins=c[family+'_sources']; prepared=c.get('gru_tours') if family=='gru' else c.get('landmark_graph') if family=='landmark' else None
    module_name={'pt':'retained_pt_routes','gru':'gru_memory','landmark':'landmark_memory'}[family]
    runtime=Path(recipe['adapter'].partition(':')[0]).parent.parent
    row={'schema':'engineering-only-resource-case-v1','case_id':'','family':family+'_random_direct_builder','architecture_family':family,'arm':arm,'dropout':c['dropout'],
        'source_spec_or_source_recipe_binding':receipt(root,path),'source_spec_hash':spec_row['spec_hash'],'source_run_id':recipe['run_id'],
        'builder_code_binding':helper,'source_adapter_binding':receipt(root,root/runtime/'adapters'/(module_name+'.py')),'runtime_package':str(runtime),
        'source_operator_pins':pins,'GRAPH_TRAIN_bindings':c['node_inputs'],'role_authority':'existing immutable prepared GRAPH/TRAIN only; no original heldout or raw labels',
        'prohibited_original_role_paths':sorted(prohibited),'provider_custody':custody,'maximum_TRAIN_rows':train_rows,'declared_dimensions':{k:c[k] for k in ('width','depth','classes')},
        'random_seed':c['initialization_seed'],'paired_seed':c['paired_seed'],'prepared_GRAPH_capsule':prepared,'prototype_bodies':bodies,'independent_bank_method':bank,
        'probe_unit':'one cold plus three warm complete source TRAIN objective cycles at fixed random parameters; pure gradients/denied step; source regularizer retained',
        'actual_parameter_and_optimizer_ownership':'source builder inspection; one fresh AdamW per own prototype; separable4 one joint optimizer; manual zero moments',
        'source_operator_counts':'body/view-specific native/PT/packed-GRU/write-read source cycles; counters emitted without objective/metric values',
        'random_or_paid_ancestry_status':'fresh random source/direct random H0 or raw-H/U; untrained/unselected; no registry or paid-parent call; scientific_parent_receipt=null',
        'state_equality_proofs':'exact parameter and manual shadow-moment bytes before/after; immutable graph/cache/capsule version guards',
        'allocator_cap_and_external_usage':'normal current-child Torch allocator min(fresh_free-4GiB,floor(.4*total)); external process usage retained',
        'owner_PID_start_PGID':'before numerical import; exact Physics owner custody before and after',
        'timeout_and_terminal_cost':{'maximum_seconds':600,'terminal_failures_retained':True},'no_retry':True,'no_score':True,
        'qualification_limits':'random-state storage and source-operator proxy; no actual source acquisition/selected-state/cache/optimizer-kernel or heldout qualification','scientific_parent_receipt':None}
    row['case_id']=digest(row)[:24]; cases.append(row)


def build_proxy(torch,root,case):
    """Constructor-only direct source paths; no SELECT stub or adapter factory."""
    from torch.nn import functional as F
    sys.path.insert(0,str(root/case['runtime_package']))
    from adapters.retained_pt_routes import compile_source,Streams,tensor_hash
    family=case['architecture_family']; sources=case['source_operator_pins']; arm=case['arm']; device=torch.device('cuda:0') if torch.cuda.is_initialized() else torch.device('cpu')
    graph=torch.load(bound(root,case['GRAPH_TRAIN_bindings']['graph']),map_location='cpu',weights_only=True); train=torch.load(bound(root,case['GRAPH_TRAIN_bindings']['train']),map_location='cpu',weights_only=True)
    if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'}: raise ValueError('Exact GRAPH/TRAIN schemas required')
    x,edge=graph['x'].to(device),graph['edge_index'].to(device); ids,y=train['ids'].to(device),train['y'].to(device); classes=case['declared_dimensions']['classes']; seed=case['random_seed']; dropout=case['dropout']
    fixed=[x,edge,ids,y]; data=None; source_models=[]; caches=[]; packed=None; backend=None; geometry=None
    if family=='landmark':
        from adapters.landmark_memory import isolated_helpers,NATIVE,compile_core,tree_hash
        capsule=torch.load(bound(root,case['prepared_GRAPH_capsule']['capsule']),map_location='cpu',weights_only=True); stored=capsule.pop('payload_hash')
        if stored!=case['prepared_GRAPH_capsule']['payload_hash'] or stored!=digest(tree_hash(torch,capsule)) or capsule['graph']!=case['GRAPH_TRAIN_bindings']['graph']: raise ValueError('Actual GRAPH-only landmark capsule changed')
        tokens=[v.to(device) for v in capsule['tokens']]; geometry={k:v.to(device) if torch.is_tensor(v) else v for k,v in capsule['geometry'].items()}; fixed+=tokens+[v for v in geometry.values() if torch.is_tensor(v)]
        p,b=isolated_helpers(sources); native={**NATIVE,'dropout':dropout}; factors=p.factor_definitions(torch,root)
    else:
        A=torch.sparse_coo_tensor(edge.flip(0),torch.ones(edge.shape[1],device=device),(len(x),len(x))).coalesce(); degree=torch.bincount(A.indices()[0],minlength=len(x)).float()[:,None]; fixed += [A,degree]
        if family=='gru':
            from adapters.gru_memory import compile_gru,load_source,graph_check,tree_hash
            graph_check(torch,x,edge,A,degree); backend=load_source(root,sources['backend']); backend.require_runtime(torch)
            if arm in ('B4','G1','GF1','GM4','GN4'):
                capsule=torch.load(bound(root,case['prepared_GRAPH_capsule']['capsule']),map_location='cpu',weights_only=True); stored=capsule.pop('payload_hash')
                if stored!=case['prepared_GRAPH_capsule']['payload_hash'] or stored!=digest(tree_hash(torch,capsule)) or capsule['graph']!=case['GRAPH_TRAIN_bindings']['graph']: raise ValueError('Actual GRAPH-only tour capsule changed')
                packed=tuple(backend.to_device(v,device) for v in capsule['packed']); fixed += [v for pack in packed for v in pack.values() if torch.is_tensor(v)]
    prototypes=[]; random_cache_hashes=[]
    for index in range(case['prototype_bodies']):
        actual_seed=seed+1000003*index; members=4 if arm in ('shared_factor4','factor_s4','original_boundary','original_gnnm') else 1
        memory_arm=arm in (('B1','B4','T16','S64','FS64','LM4') if family=='landmark' else ('B4','G1','GF1','GM4','GN4') if family=='gru' else ('head_only','mixed_single','bias_coordinate_single','private4','middle_state_mean','separable4'))
        if family=='landmark':
            factory=b.build_bank if arm=='original_boundary' else p.build_bank
            extra={} if arm=='original_boundary' else {'factorized':arm!='ordinary_m1' and arm!='continued_m1','factor_initialization':'unit'}
            bank=factory(torch,root,tokens,native=native,classes=classes,dataset='engineering_random',members=members,initialization_seed=actual_seed,dropout_seeds=[actual_seed+491005001+1009*v for v in range(max(4,members))] if arm!='original_boundary' else [actual_seed+491005001+1009*v for v in range(4)],**extra)
            source_models.append(bank)
            if memory_arm:
                bank.eval()
                with torch.no_grad(): _,H=bank.forward_route(0)
                H=H.detach().clone(); U=H/torch.linalg.vector_norm(H,dim=1,keepdim=True).clamp_min(1); fixed += [H,U]; caches.append({'random_H':H,'random_U':U})
                core=compile_core(root,sources['core'],nodes=len(x),features=x.shape[1],classes=classes,seed=actual_seed,dropout=dropout)
                result=core.build_arm(torch,F,arm,actual_seed,[bank.body],[{'H':H,'U':U}],geometry,FactorLinear=factors['FactorLinear']); core.inspect_parameter_ownership(result); model=result['models'][0].to(device)
                views=model.views
            else: model=bank; views=4 if arm in ('continued_m1','continued_f1') else members
        else:
            native=compile_source(root,sources['native'],dropout=dropout,dimensions=(len(x),x.shape[1],case['declared_dimensions']['width'],classes),seed=actual_seed)
            kind='F1' if arm in ('factor_f1','continued_f1') else 'S4' if arm=='factor_s4' else 'P1'
            if arm=='original_gnnm':
                namespace={}; path=bound(root,sources['original']); exec(compile(path.read_text(),str(path),'exec'),namespace)
                with torch.random.fork_rng(devices=[]): torch.manual_seed(actual_seed); model=namespace['TABMModel']('SAGE',2,x.shape[1],128,classes,1,8,'LayerNorm',dropout,tabm_inits=4,device='cpu').to(device)
                views=4; bank=None
            else:
                result=native.build_arm(torch,F,kind,actual_seed); native.inspect_parameter_ownership(result); bank=result['models'][0].to(device); source_models.append(bank)
                if memory_arm:
                    if family=='pt':
                        core=compile_source(root,sources['pt'],dropout=dropout); H=core.cache_stem(torch,F,bank,x,A,degree); caches.append({'random_H0':H}); fixed.append(H)
                        models=[]
                        for j in range(4 if arm=='separable4' else 1): models.append(core.build_arm(torch,F,'mixed_single' if arm=='separable4' else arm,[bank],[H],A,degree,initialization_seeds=[actual_seed+1000003*j])['models'][0])
                        if arm=='separable4':
                            stem=models[0].native.bodies[0].layers[0]
                            for value in models[1:]: value.native.bodies[0].layers[0]=stem
                        model=torch.nn.ModuleList(models).to(device); views=4*len(models)
                    else:
                        H=None
                        if arm!='GN4':
                            first=bank.bodies[0].layers[0]
                            with torch.no_grad():
                                z=torch.sparse.mm(A,first._transform(x,first.aggregate_weight,None,None))/degree+first._transform(x,first.skip_weight,None,None)+first.bias[0]; H=F.relu(z/torch.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1)).detach()
                            fixed.append(H); caches.append({'random_H0':H})
                        core=compile_gru(root,sources['gru'],nodes=len(x),features=x.shape[1],classes=classes,seed=actual_seed,dropout=dropout)
                        result=core.build_arm(torch,F,arm,actual_seed,[bank],[H],gru_backend=backend); core.inspect_parameter_ownership(result); model=result['models'][0].to(device); views=4
                else: model=bank; views=4 if arm in ('continued_p1','continued_f1') else members
        streams=Streams(torch,device,[actual_seed+3000007+1000033*v for v in range(views)]); prototypes.append((model,streams,memory_arm,views))
        random_cache_hashes += [{k:tensor_hash(v) for k,v in cache.items()} for cache in caches[-1:]] if memory_arm and caches else []
    class Proxy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.models=torch.nn.ModuleList([row[0] for row in prototypes]); self.versions=tuple(v._version for v in fixed); self.calls={}
        def guard(self):
            if tuple(v._version for v in fixed)!=self.versions: raise ValueError('Random GRAPH/TRAIN/cache/capsule tensors mutated')
        def objective_backward(self):
            self.guard(); self.train(); calls={'model_forwards':0,'backward_calls':0,'route_predictions':0,'common_projection_banks':0,
                'PT_bank_calls':0,'PT_T_map_calls':0,'GRU_direction_API_calls':0,'landmark_memory_map_products':0,'native_graph_propagations':0,'tail_dropout_draws':0,'supervised_examples':0}
            for model,streams,memory,views in prototypes:
                if family=='landmark' and memory:
                    constants=model.bank_constants(); calls['common_projection_banks']+=int(model.memory is not None)
                    for v in range(views):
                        z=model.view_forward(v,constants,streams.scope); loss=F.cross_entropy(z[ids],y)/views; loss.backward(retain_graph=model.memory is not None and v+1<views); calls['backward_calls']+=1; calls['model_forwards']+=1
                    del constants
                elif family=='gru' and memory:
                    for v in range(4):
                        z=model.view_forward(x,A,degree,packed,v,streams.scope); (F.cross_entropy(z[ids],y)/4).backward(); calls['backward_calls']+=1; calls['model_forwards']+=1
                    penalty=model.regularizer()
                    if penalty.requires_grad: penalty.backward(); calls['backward_calls']+=1
                elif family=='pt' and memory:
                    values=[m(route_rng_context=lambda view,j=j:streams.scope(4*j+view)) for j,m in enumerate(model)]; logits=torch.cat(values); penalty=torch.stack([m.regularizer() for m in model]).mean()
                    (torch.stack([F.cross_entropy(z[ids],y) for z in logits]).mean()+penalty).backward(); calls['backward_calls']+=1; calls['model_forwards']+=len(model)
                elif family!='landmark':
                    if arm=='original_gnnm':
                        values=[]
                        for v in range(4):
                            with streams.scope(v): values.append(model(types.SimpleNamespace(edge_index=edge),x,tabm_seed=v))
                        logits=torch.stack(values); penalty=x.new_zeros(()); actual_forwards=4
                    elif arm in ('continued_p1','continued_f1'):
                        logits=torch.cat([model(x,A,degree,route_rng_context=lambda route,v=v:streams.scope(v)) for v in range(4)]); penalty=model.regularizer(); actual_forwards=4
                    else:
                        logits=model(x,A,degree,route_rng_context=streams.scope); penalty=model.regularizer(); actual_forwards=1
                    (torch.stack([F.cross_entropy(z[ids],y) for z in logits]).mean()+penalty).backward(); calls['backward_calls']+=1; calls['model_forwards']+=actual_forwards
                else:
                    for v in range(views):
                        z=model.forward_route(v if members>1 else 0,slot=v)[0]
                        (F.cross_entropy(z[ids],y)/views).backward(); calls['backward_calls']+=1; calls['model_forwards']+=1
                calls['route_predictions']+=views
                calls['supervised_examples']+=views*len(ids)
                if memory:
                    calls['tail_dropout_draws']+=views
                    if family=='pt':
                        banks=(5 if arm=='private4' else 0 if arm=='head_only' else 2)*len(model)
                        calls['PT_bank_calls']+=banks; calls['PT_T_map_calls']+=4*banks; calls['native_graph_propagations']+=3*banks+views
                    elif family=='gru':
                        calls['GRU_direction_API_calls']+=0 if arm=='B4' else 8; calls['native_graph_propagations']+=8 if arm=='GN4' else 4
                    elif model.memory is not None: calls['landmark_memory_map_products']+=3+4*views
                elif family!='landmark': calls['native_graph_propagations']+=2*views
            self.guard(); self.calls=calls
    proxy=Proxy().to(device)
    optimizers=[torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.005 if family=='landmark' else .001,weight_decay=0) for model,_,_,_ in prototypes]
    flat=[id(p) for opt in optimizers for group in opt.param_groups for p in group['params']]
    if len(flat)!=len(set(flat)) or set(flat)!={id(p) for p in proxy.parameters() if p.requires_grad}: raise ValueError('Actual disjoint live-only random prototype optimizer ownership differs')
    def stored_bytes(value):
        if value.layout==torch.sparse_coo:
            value=value.coalesce(); return value.indices().numel()*value.indices().element_size()+value.values().numel()*value.values().element_size()
        return value.numel()*value.element_size()
    return types.SimpleNamespace(model=proxy,optimizer=types.SimpleNamespace(optimizers=optimizers),streams=[row[1] for row in prototypes],random_cache_hashes=random_cache_hashes,
        fixed_bytes=sum(stored_bytes(v) for v in fixed),source_random_model_count=len(source_models),source_random_parameter_bytes=sum(p.numel()*p.element_size() for model in source_models for p in model.parameters()))


def measure(torch,h,adapter,denial,output,warm=3):
    h.allocate_adam_moments(torch,adapter); p0=h.tensor_digest(torch,adapter.model.named_parameters()); m0=h.tensor_digest(torch,h.optimizer_tensors(adapter)); times=[]; calls=[]; maximum_grad=0
    for _ in range(warm+1):
        for optimizer in h.optimizers(adapter): optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_initialized(): torch.cuda.synchronize()
        tick=time.perf_counter(); adapter.model.objective_backward()
        for optimizer in h.optimizers(adapter): optimizer.step() # All concrete actions intercepted before any builder.
        if torch.cuda.is_initialized(): torch.cuda.synchronize()
        times.append(time.perf_counter()-tick); calls.append(copy.deepcopy(adapter.model.calls)); maximum_grad=max(maximum_grad,sum(p.grad.numel()*p.grad.element_size() for p in adapter.model.parameters() if p.grad is not None))
    p1=h.tensor_digest(torch,adapter.model.named_parameters()); m1=h.tensor_digest(torch,h.optimizer_tensors(adapter))
    if p0!=p1 or m0!=m1 or denial.denied_functional_calls or denial.actual_optimizer_kernels: raise ValueError('Random parameters/manual moments changed or forbidden optimizer kernel attempted')
    # Plain engineering state only; never engine.capture_checkpoint/selected.pt.
    tick=time.perf_counter(); payload={'engineering_random_untrained_unselected':True,'scientific_parent_receipt':None,'parameters':{k:v.detach().cpu().clone() for k,v in adapter.model.state_dict().items()},
        'shadow_moments':[(k,v.detach().cpu().clone()) for k,v in h.optimizer_tensors(adapter)],'private_streams':[copy.deepcopy(v.states) for v in adapter.streams],'random_cache_hashes':adapter.random_cache_hashes,
        'native_runtime_states':[m.capture_runtime_state() if hasattr(m,'capture_runtime_state') else {} for m in adapter.model.models],
        'unregistered_PolyAttn_bias':{k:m.bias.detach().cpu().clone() for k,m in adapter.model.named_modules() if type(m).__name__=='PolyAttn'},
        'torch_cpu_rng':torch.get_rng_state().clone(),'torch_cuda_rng':[v.cpu().clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}
    clone_seconds=time.perf_counter()-tick; path=output/'DISPOSABLE_RANDOM_ENGINEERING_STATE.pt'; tick=time.perf_counter(); torch.save(payload,path); saved=time.perf_counter()-tick; serialized=receipt(output,path); path.unlink(); del payload
    return {'scope':'RANDOM_UNTRAINED_SOURCE_OPERATOR_PROXY_ONLY','parameter_bytes':sum(p.numel()*p.element_size() for p in adapter.model.parameters()),'gradient_bytes':maximum_grad,
        'shadow_moment_bytes':sum(v.numel()*v.element_size() for _,v in h.optimizer_tensors(adapter)),'immutable_graph_TRAIN_cache_bytes':adapter.fixed_bytes,
        'random_source_model_count':adapter.source_random_model_count,'random_source_parameter_bytes':adapter.source_random_parameter_bytes,'random_cache_hashes':adapter.random_cache_hashes,
        'optimizer_objects':len(h.optimizers(adapter)),'parameters_sha256_before':p0,'parameters_sha256_after':p1,'moments_sha256_before':m0,'moments_sha256_after':m1,
        'cold_source_cycle_seconds':times[0],'warm_source_cycle_seconds':times[1:],'warm_median_seconds':statistics.median(times[1:]),'source_operator_counts_by_cycle':calls,
        'denied_optimizer_step_calls':denial.denied_step_calls,'actual_optimizer_kernel_calls':0,'actual_optimizer_updates':0,'scores_emitted':0,'accuracy_calls':0,'scientific_fits':0,
        'disposable_serialization':{**serialized,'clone_seconds':clone_seconds,'save_seconds':saved,'removed':True,'never_selected_checkpoint':True},'trained_or_paid_ancestry_qualified':False,'SELECT_CAL_evaluate_resource_qualified':False,'all_fit_ETA':None}


def execute_case(root,manifest,identifier,output):
    h=module(root,manifest['cap_helper'],'reviewed_zero_update_denial_cap'); physics0=physics_custody(root,manifest['Physics_owner_start'],manifest['Physics_original_watcher'])
    cases=[v for v in manifest['cases'] if v['case_id']==identifier]
    if len(cases)!=1: raise ValueError('One exact reviewed source case required')
    case=cases[0]; bound(root,case['source_spec_or_source_recipe_binding']); bound(root,case['source_adapter_binding'])
    for row in case['source_operator_pins'].values(): bound(root,row)
    import torch
    torch.set_num_threads(2); cap,usage0,usage1=h.configure_cuda_allocator(torch,output); torch.cuda.reset_peak_memory_stats(0)
    allowed={root/v for v in role_paths(case['GRAPH_TRAIN_bindings'])}
    if case['prepared_GRAPH_capsule']: allowed.add(root/case['prepared_GRAPH_capsule']['capsule']['path'])
    forbidden={root/v for v in case['prohibited_original_role_paths']}; io_guard=h.RoleIO(torch,root,allowed,forbidden,output)
    with h.OptimizerDenial(torch) as denial,io_guard:
        tick=time.perf_counter(); adapter=build_proxy(torch,root,case); torch.cuda.synchronize(); constructor=time.perf_counter()-tick
        if denial.denied_step_calls or denial.denied_functional_calls: raise ValueError('Builder attempted optimizer action')
        result=measure(torch,h,adapter,denial,output,manifest['warm_iterations']); result.update(random_constructor_cache_seconds=constructor,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            heldout_original_file_accesses=io_guard.denied,tensor_load_paths=io_guard.tensor_load_paths)
    result.update(allocator_cap=cap,GPU_usage_before_init=usage0,GPU_usage_after_init=usage1,GPU_usage_at_completion=h.GPU_process_usage(),
        physics_custody_before=physics0,physics_custody_after=physics_custody(root,manifest['Physics_owner_start'],manifest['Physics_original_watcher']),case_id=identifier,helper_sha256=manifest['helper']['sha256'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--project-root',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=('prepare-manifest','execute-case'),default='prepare-manifest'); parser.add_argument('--manifest'); parser.add_argument('--case-id'); parser.add_argument('--reviewed-helper-sha256'); parser.add_argument('--reviewed-manifest-sha256')
    parser.add_argument('--case-timeout-seconds',type=int,default=600); args=parser.parse_args()
    from prepare_node_data import verify_allocation
    verify_allocation(); root=Path(args.project_root).resolve(); output=local(root,args.output); output.mkdir(parents=True,exist_ok=False)
    helper=receipt(root,Path(__file__).resolve()); cap=receipt(root,local(root,CAP_HELPER))
    if cap['sha256']!=CAP_SHA: raise ValueError('Reviewed denial/cap helper changed')
    start=receipt(root,local(root,PHYSICS_DIR+'/OWNER_START.json'))
    watcher={'launch':receipt(root,local(root,READOUT_DIR+'/LAUNCH.json')),'watch_start':receipt(root,local(root,READOUT_DIR+'/watch_v1/WATCH_START.json')),'expected_identity':READOUT_WATCHER}
    if args.mode=='prepare-manifest':
        physics=physics_custody(root,start,watcher); value=prepare_manifest(root,helper,start,watcher,cap); value['Physics_ownership_at_preparation']=physics
        (output/'MANIFEST.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n'); print(canonical({'cases':len(value['cases']),'numerical_imports':0,'CUDA_executions':0,'scientific_fits':0,'GO':False})); return
    # Preparing a file never dispatches CUDA. A reviewed exact hash plus one
    # case identifier is required by the separately authorized caller.
    h=module(root,cap,'reviewed_resource_cap'); owner=h.own_identity(); tick=time.monotonic(); result=None
    (output/'OWNER_START.json').write_text(json.dumps({'owner':owner,'case_id':args.case_id,'helper':helper},indent=2)+'\n')
    try:
        if args.reviewed_helper_sha256!=helper['sha256'] or not args.manifest or not args.case_id: raise ValueError('Separate exact-helper reviewed one-case execution required')
        if not 1<=args.case_timeout_seconds<=600: raise ValueError('Case deadline must be1..600 seconds')
        def expired(signum,frame): raise h.ResourceTimeout('Owned engineering-case deadline/termination')
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired); signal.alarm(args.case_timeout_seconds)
        manifest_path=local(root,args.manifest)
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=args.reviewed_manifest_sha256: raise ValueError('Separate exact finite-manifest review required')
        manifest=json.loads(manifest_path.read_text())
        if manifest['helper']!=helper or manifest['cap_helper']!=cap: raise ValueError('Exact reviewed helper/manifest custody changed')
        match=[v for v in manifest['cases'] if v['case_id']==args.case_id]
        if len(match)!=1 or digest({**match[0],'case_id':''})[:24]!=args.case_id: raise ValueError('Finite source-case content hash changed')
        result=execute_case(root,manifest,args.case_id,output); result.update(status='RANDOM_SOURCE_PROXY_COMPLETE',owner=owner,elapsed_wall_seconds=time.monotonic()-tick,no_retry=True)
        (output/'RESULT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    except BaseException as error:
        result=h.terminal_failure(error,owner,tick); (output/'TERMINAL_FAILURE.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n'); raise SystemExit(2)
    finally: signal.alarm(0)


if __name__=='__main__': main()
