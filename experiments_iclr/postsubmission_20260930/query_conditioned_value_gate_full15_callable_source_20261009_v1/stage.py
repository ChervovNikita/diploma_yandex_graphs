"""Disabled query-gated full15 WikiCS banks; unchanged fixed-native staging."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101,6203,6307)
ARMS = ('C4_gate','S_joint4head_gate','U4_sameB_full_untied_gate','S_one_path_gate','C4_gate_identity_erased')
CONDITIONS = dict(C4_gate='label_only4_detached',S_joint4head_gate='multihead_single4',
    U4_sameB_full_untied_gate='shared_backbone_untied_correctors4',S_one_path_gate='one_path',
    C4_gate_identity_erased='label_only4_detached')
EXPECTED = dict(C4_gate=(4,4,4,1,109160),S_joint4head_gate=(1,4,1,1,382016),
    U4_sameB_full_untied_gate=(4,4,4,4,414976),S_one_path_gate=(1,1,1,1,103744),
    C4_gate_identity_erased=(4,4,4,1,109160))
PAYLOADS = frozenset(('stage.py', 'core.py', 'controls.py', 'query_value_gate_hook.py', 'posterior.py', 'collect.py', 'SOURCE_BINDINGS.json', 'INPUT_FILES.json', 'PROTOCOL.json', 'LAUNCH_TEMPLATE_DISABLED.json', 'README.md', 'STATIC_CHECKS.json', 'IMPLEMENTATION_DIFFS.json', 'diffs/core.py.diff', 'diffs/controls.py.diff', 'diffs/stage.py.diff', 'diffs/collect.py.diff'))


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): result.update(block)
    return result.hexdigest()


def write(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'); os.replace(temp,path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value; spec.loader.exec_module(value); return value


def verify_packet():
    """Verify the local disabled seal and every payload before code or scores load."""
    seal=json.loads((HERE/'SEAL.json').read_text()); manifest=HERE/'MANIFEST.json'
    require(seal['schema']=='query-value-gated-full15-disabled-source-seal-v1'
            and seal['source_only'] is True and seal['execution_enabled'] is False
            and seal['runtime_qualified'] is False and sha(manifest)==seal['manifest_sha256'],
            'Exact disabled staged source seal')
    rows=json.loads(manifest.read_text())['files']
    require(len(rows)==len(PAYLOADS) and {row['path'] for row in rows}==PAYLOADS,'Complete staged payload inventory')
    for row in rows:
        path=(HERE/row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],
                'Changed staged payload: '+row['path'])
    return seal['manifest_sha256']


def verify_bindings():
    verify_packet()
    pins=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in pins['files']:
        path=(PHASE/row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size==row['bytes']
                and sha(path)==row['sha256'],'Changed staged source binding')
        if path.name=='MANIFEST.json':
            for payload in json.loads(path.read_text())['files']:
                item=(path.parent/payload['path']).resolve(strict=True)
                require(item.is_relative_to(path.parent) and item.stat().st_size==payload['bytes']
                        and sha(item)==payload['sha256'],'Changed sealed dependency payload')
    return pins


def dependencies():
    pins=verify_bindings()
    root=PHASE/pins['integration_directory']
    engine=module(root/'train.py','_staged_posterior_original_integration')
    common,capture_module=engine._helpers()
    public,data,driver,original_core,public_root=common.dependencies()
    core=module(HERE/'core.py','_query_value_local_core_v1')
    controls=module(HERE/'controls.py','_query_value_local_controls_v1')
    posterior=module(HERE/'posterior.py','_staged_bounded_posterior')
    metadata=json.loads((PHASE/pins['native_metadata_path']).read_text())
    runtime=json.loads((PHASE/pins['runtime_path']).read_text())
    return engine,common,capture_module,public,data,core,public_root,controls,posterior,metadata,runtime


def verify_closed_family(family_root):
    """Mechanical all-15 barrier only: hashes are checked without loading metrics."""
    pins=verify_bindings(); source=source_identity()
    metadata=json.loads((PHASE/pins['native_metadata_path']).read_text())
    native_rows={row['seed']:row for row in metadata['native_states']}
    family_root=Path(family_root).resolve(strict=True)
    require(family_root.is_relative_to(PHASE),'Phase-owned complete family')
    required=[family_root/('seed'+str(seed))/'COMPLETE.json' for seed in SEEDS]
    required += [family_root/('seed'+str(seed))/arm/'COMPLETE.json' for seed in SEEDS for arm in ARMS]
    require(all(path.is_file() for path in required),'All 3 seed blocks and all 15 bank closures before score access')
    records={}
    for seed in SEEDS:
        block=family_root/('seed'+str(seed)); complete=json.loads((block/'COMPLETE.json').read_text())
        work=complete['work']
        require(complete['complete'] is True and complete['seed']==seed and complete['source']==source
                and complete['label_updates']==1100 and complete['arms']==list(ARMS)
                and work['all_bank_update_attempts']==work['all_bank_update_completions']==work['common_Q_checks']==1100
                and work['complete_VALID_events']==1100 and work['native_capture_calls']==1
                and work['learned_restores']==complete['new_native_training']==work['native_updates']==0
                and complete['run']['source']==source and complete['run']['seed']==seed
                and complete['run']['purpose']=='scientific_complete'
                and complete['run']['native_binding']==native_rows[seed]
                and sha(block/'RUN.json')==complete['run_sha256']
                and sha(block/'PRIVATE_VALID_TRACE.json')==complete['private_VALID_trace_sha256']
                and sha(block/'FIXED_CAPTURE.pt')==complete['fixed_capture_sha256'],
                'Complete exact-source seed block and one fixed native capture')
        for arm in ARMS:
            path=block/arm
            require(sha(path/'COMPLETE.json')==complete['bank_complete_sha256'][arm],'Exact bank work-receipt hash')
            record=json.loads((path/'COMPLETE.json').read_text()); counts=record['bank_counts']
            _,heads,backwards,adam,_=EXPECTED[arm]
            require(record['complete'] is True and record['seed']==seed and record['arm']==arm
                    and record['source']==source and record['label_updates']==1100
                    and counts['completed_training_steps']==counts['training_step_attempts']==counts['mask_draws']==1100
                    and counts['old_corrector_parameter_checks']==counts['serving_calls']==1100
                    and counts['common_label_fields_built']==2200
                    and counts['training_route_forwards']==counts['training_route_forward_attempts']==heads*1100
                    and counts['serving_route_forwards']==counts['serving_route_forward_attempts']==heads*1100
                    and counts['route_backwards']==counts['route_backward_attempts']==backwards*1100
                    and counts['corrector_Adam_steps']==counts['corrector_Adam_attempts']==adam*1100
                    and 1<=record['selected_label_epoch']<=1100 and record['native_binding']==native_rows[seed]
                    and record['fixed_capture_sha256']==complete['fixed_capture_sha256']
                    and sha(path/'selected.pt')==record['selected_sha256']
                    and sha(path/'PRIVATE_SELECTED_METRICS.json')==record['private_metrics_sha256'],
                    'All required bank/optimizer/selector source and exact endpoint custody')
            records[(seed,arm)]=(path,record)
    return records


def source_identity():
    return dict(manifest_sha256=verify_packet(),stage_sha256=sha(__file__),
                posterior_sha256=sha(HERE/'posterior.py'),collect_sha256=sha(HERE/'collect.py'),
                gated_core_sha256=sha(HERE/'core.py'),gated_controls_sha256=sha(HERE/'controls.py'),
                gate_helper_sha256=sha(HERE/'query_value_gate_hook.py'),
                protocol_sha256=sha(HERE/'PROTOCOL.json'),bindings_sha256=sha(HERE/'SOURCE_BINDINGS.json'))


def route(runtime):
    require(PHASE==Path(runtime['phase']).resolve(strict=True)
            and socket.gethostname()==runtime['hostname']
            and Path(sys.executable).resolve()==Path(runtime['python']).resolve()
            and os.environ.get('PYTHONPATH','').split(os.pathsep)==runtime['PYTHONPATH'],
            'Only the declared normal-repository runtime route')
    uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    require(uuid==runtime['GPU_uuid'],'Exact current allocated GPU UUID')


def load_roles(*,later_execution_authorized=False):
    require(later_execution_authorized is True,'Disabled role loading')
    deps=dependencies(); route(deps[-1]); binding=json.loads((HERE/'INPUT_FILES.json').read_text())
    for row in binding.values():
        path=(PHASE/row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size==row['bytes']
                and sha(path)==row['sha256'],'Exact root role/native-source file')
    train,valid,origin=deps[4].load_train_valid('wikics',PHASE/binding['train']['path'],PHASE/binding['valid']['path'])
    return train,valid,origin


def make_stage(*,train_data,origin,seed,device='cuda:0',purpose='engineering_qualification',
               capture_cache=None,only_arm=None,later_execution_authorized=False,
               scientific_execution_authorized=False):
    require(later_execution_authorized is True,'Disabled staged source; separate root adoption required')
    require(seed in SEEDS and type(seed) is int and device=='cuda:0'
            and purpose in ('engineering_qualification','scientific_complete','selected_serving'), 'Fixed block/purpose/device')
    require(purpose!='scientific_complete' or scientific_execution_authorized is True,'No scientific admission in this source packet')
    require(only_arm is None or purpose=='selected_serving','No dropping required banks during training')
    require(purpose!='scientific_complete' or capture_cache is None,'Scientific block makes its one actual native capture')
    require(purpose!='selected_serving' or (capture_cache is not None and only_arm in ARMS),
            'Selected reconstruction requires the original fixed cache and one declared arm')
    setup_started=time.monotonic(); timings={}
    engine,common,captures,public,data,core,public_root,controls,posterior,metadata,runtime=dependencies()
    route(runtime); timings['source_verification_imports_route_seconds']=time.monotonic()-setup_started
    row=next(x for x in metadata['native_states'] if x['seed']==seed)
    checkpoint_started=time.monotonic(); path=(PHASE/row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and path.stat().st_size==row['bytes']
            and sha(path)==row['sha256'],'Exact authentic own-selected native state')
    timings['native_checkpoint_hash_seconds']=time.monotonic()-checkpoint_started
    constructor_started=time.monotonic()
    native=public.Session('wikics','single',seed,device,PHASE/json.loads((HERE/'INPUT_FILES.json').read_text())['polynormer']['path'])
    torch=native.torch
    require(str(torch.__version__)==runtime['torch'] and native.np.__version__==runtime['numpy']
            and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)==runtime['GPU_name'],'Pinned current one-GPU runtime')
    torch.cuda.synchronize(0); timings['native_session_constructor_seconds']=time.monotonic()-constructor_started
    load_started=time.monotonic()
    loaded=torch.load(path,map_location='cpu',weights_only=False)  # Root-authenticated own state only.
    timings['trusted_native_checkpoint_deserialization_seconds']=time.monotonic()-load_started
    restore_started=time.monotonic(); context=common.context_identity(native,train_data)
    require(all(loaded[key]==row[key] for key in ('schema','kind','arm','epoch','global_mode','snapshot_purpose','selector_performed'))
            and loaded['run']==row['run_identity'] and loaded['native']['checkpoint_kind']==row['native_checkpoint_kind']
            and loaded['native']['epoch']==row['epoch'] and loaded['native']['global']==row['global_mode']
            and loaded['run']['data']==origin and loaded['run']['data_context']==context
            and loaded['run']['native_recipe']==native.config and loaded['run']['native_core']==native.core_provenance
            and loaded['run']['native_source']==native.native_provenance,'Exact authentic native source/role/selector custody')
    native.model.load_state_dict(loaded['native']['model'],strict=True); native.model.set_global(row['global_mode'])
    require(len(loaded['native']['optimizers'])==1,'Original own native Adam bank')
    native.optimizers[0].load_state_dict(loaded['native']['optimizers'][0])
    native.streams=common.clone_streams(loaded['native']['streams']); native.steps=loaded['native_logical_steps']
    native.model.eval()
    for parameter in native.model.parameters(): parameter.requires_grad_(False)
    del loaded  # Old correction weights/metrics are not fitted, selected or imported.
    torch.cuda.synchronize(0); timings['native_custody_checks_and_restore_seconds']=time.monotonic()-restore_started
    start_capture=time.monotonic(); before=common.native_rng_state(native); native_capture_calls=0
    if capture_cache is None:
        capture=captures.NativeSingleCapture(native,11701,512,10)
        try:
            with torch.no_grad():
                batch=dict(x=train_data['x'].to(native.device),edge_index=train_data['edge_index'].to(native.device),
                           ids=torch.arange(11701,device=native.device))
                capture.begin('SERVE',1); native.forward(batch); H,base=capture.finish(); native_capture_calls=1
        finally: capture.close()
        torch.cuda.synchronize(0)
    else:
        require(capture_cache['schema']=='staged-posterior-fixed-native-capture-v1'
                and capture_cache['native_binding']==row and capture_cache['context']==context,'Trusted same-source fixed capture')
        H=capture_cache['H'].to(native.device); base=capture_cache['native_logits'].to(native.device)
        torch.cuda.synchronize(0)
    common.require_same_native_rng(native,before,'one eval capture or fixed-cache loading')
    require(H.shape==(11701,512) and base.shape==(11701,10) and H.dtype==base.dtype==torch.float32
            and not H.requires_grad and not base.requires_grad and bool(torch.isfinite(H).all())
            and bool(torch.isfinite(base).all()),'Full frozen native float32 capture')
    capture_seconds=time.monotonic()-start_capture
    timings['fixed_capture_forward_or_cache_transfer_seconds']=capture_seconds
    digest_started=time.monotonic()
    capture_identity=dict(H_sha256=common.tensor_digest(H),native_logits_sha256=common.tensor_digest(base))
    if capture_cache is not None: require(capture_cache['identity']==capture_identity,'Exact saved capture bytes')
    timings['fixed_capture_digest_seconds']=time.monotonic()-digest_started
    banks_started=time.monotonic()
    zero=torch.zeros_like(base); banks={}; selected=(only_arm,) if only_arm else ARMS
    before=common.native_rng_state(native)
    args=dict(nodes=11701,feature_width=512,classes=10,edge_index=train_data['edge_index'],train_ids=train_data['ids'],
              train_labels=train_data['y'],device=str(native.device),later_execution_authorized=True)
    for arm in selected:
        if arm in ('C4_gate','C4_gate_identity_erased'):
            bank=core.make_core(**args,initializer_seed=seed,mask_seed=seed+1900001,public_root=public_root,
                erase_value_class_identity=(arm=='C4_gate_identity_erased'))
        else:
            bank=controls.make_shared_controls(**args,mode=CONDITIONS[arm],
                initializer_seeds=tuple(seed+1009*h for h in range(EXPECTED[arm][1])),mask_seed=seed+1900001,
                context_identity=context,backbone_ids=(common.backbone_id(native),),
                selection_policy=common.selection_policy(CONDITIONS[arm]),core_root=HERE)
            if arm=='S_joint4head_gate':
                bank=posterior.nonlinear_single(bank,torch=torch,core_module=core,controls_module=controls,
                                                initializer_seed=seed,device=native.device)
            # The factory checks its original capture protocol. This new outer
            # adapter owns the actual staged selector; it never invokes the old
            # local-transition or per-route selected-final APIs.
            policy=dict(common.selection_policy(CONDITIONS[arm]))
            policy.update(native_local_restore='completed_native_acquisition_only',
                          corrector_local_restore='none_in_staged_label_training',
                          final_selector='coherent_complete_bank_bounded_mixture_strict_first_VALID')
            policy.pop('policy_sha256'); policy['policy_sha256']=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
            bank.selection_policy=policy
        require(sum(p.numel() for p in bank.parameters())==EXPECTED[arm][4],'Declared learned-parameter inventory')
        optimizers=[bank.optimizer] if arm in ('C4_gate','C4_gate_identity_erased') else bank.optimizers
        owner_lists=[[id(p) for group in opt.param_groups for p in group['params']] for opt in optimizers]
        flat=[pid for group in owner_lists for pid in group]
        require(len(flat)==len(set(flat)) and set(flat)=={id(p) for p in bank.parameters()},
                'Each learned map/gate has exactly one fresh optimizer owner')
        if arm=='U4_sameB_full_untied_gate':
            require(len(optimizers)==4 and all(set(owner_lists[i])=={id(p) for p in bank.routes[i].parameters()}
                    for i in range(4)), 'Every fully untied route owns its private gate and Adam')
        banks[arm]=bank
    common.require_same_native_rng(native,before,'private label-posterior constructors')
    owned={id(p) for p in native.model.parameters()}
    for bank in banks.values():
        ids={id(p) for p in bank.parameters()}; require(not ids&owned,'No native or inter-bank learned sharing'); owned|=ids
    reach=posterior.anchor_reach(torch,next(iter(banks.values())))
    require(all(torch.equal(posterior.anchor_reach(torch,bank),reach) for bank in banks.values()),'Same structural anchor support')
    def forbidden(*args,**kwargs): raise RuntimeError('Staged native state is frozen; no native training or further forward')
    native.train_step=forbidden; native.forward=forbidden
    torch.cuda.synchronize(0); timings['fresh_label_banks_and_support_seconds']=time.monotonic()-banks_started
    timings['make_stage_total_seconds']=time.monotonic()-setup_started

    def memory():
        return dict(allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                    process_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    process_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                    peak_scope='This process since its last external CUDA peak reset; source does not reset peaks')

    class Stage:
        def __init__(self):
            self.native,self.banks,self.H,self.base,self.zero,self.reach=native,banks,H,base,zero,reach
            self.common,self.engine,self.posterior,self.purpose=common,engine,posterior,purpose
            self.label_epoch=self.parameter_label_epoch=0
            self._versions=(H._version,base._version,zero._version,tuple(p._version for p in native.model.parameters()))
            self.capture_identity=capture_identity
            self.run=dict(source=source_identity(),seed=seed,purpose=purpose,native_binding=row,data=origin,context=context,
                capture_identity=capture_identity,arms=list(selected),native_training_executed=0,
                capture_calls_executed=native_capture_calls,capture_seconds=capture_seconds,
                new_setup_timings_seconds=timings,setup_cuda_memory=memory(),
                historical_acquisition=dict(root_attested_complete_native_epochs=1100,selected_parameter_epoch=row['epoch'],
                    snapshot_logical_steps=native.steps,checkpoint_bytes=row['bytes'],elapsed_memory_cost_join_pending=True,
                    no_zero_or_free_acquisition_claim=True),readout='Each route:1/5 native+4/5 own label posterior; exact no-anchor fallback',
                label_stage_native_transitions=0,native_or_paper_scores_replaced=False,ordinary_independent4_available=False,
                exact_resume_supported=False,TEST_scoring=False)
            self.work=dict(all_bank_update_attempts=0,all_bank_update_completions=0,common_Q_checks=0,
                native_RNG_checks=0,complete_VALID_events=0,learned_restores=0,native_updates=0,
                native_capture_calls=native_capture_calls,bank_parameter_copies=0,bank_optimizer_copies=0)

        def fixed(self):
            require((H._version,base._version,zero._version,tuple(p._version for p in native.model.parameters()))==self._versions
                    and not H.requires_grad and not base.requires_grad and not zero.requires_grad
                    and not native.model.training and all(p.grad is None and not p.requires_grad for p in native.model.parameters()),
                    'Native/capture/zero-base state remains frozen')

        def metadata(self): return engine._capture_metadata(native,row['epoch'],common)

        def draw_common_masks(self):
            require(self.purpose!='selected_serving' and tuple(banks)==ARMS,'Complete training banks only')
            self.fixed(); before=common.native_rng_state(native)
            masks={arm:bank.draw_common_query_mask() for arm,bank in banks.items()}; first=masks['C4_gate']
            require(all((x.query_positions,x.query_ids,x.draw_id,x.training_value_scale)==
                    (first.query_positions,first.query_ids,first.draw_id,first.training_value_scale) for x in masks.values()),'Common Q across every arm')
            common.require_same_native_rng(native,before,'private common masks'); self.work['common_Q_checks']+=1
            self.work['native_RNG_checks']+=1; return masks

        def train_step(self,*,label_epoch,masks=None):
            require(self.purpose!='selected_serving' and tuple(banks)==ARMS and type(label_epoch) is int
                    and label_epoch==self.label_epoch+1
                    and label_epoch<=1100,'Fresh sequential full label-stage update; no resume')
            masks=self.draw_common_masks() if masks is None else masks; self.fixed()
            require(tuple(masks)==ARMS and all(bank._pending_mask is masks[arm] for arm,bank in banks.items()),
                    'Exactly the live complete bank mask set')
            first=masks['C4_gate']
            require(all((x.query_positions,x.query_ids,x.draw_id,x.training_value_scale)==
                        (first.query_positions,first.query_ids,first.draw_id,first.training_value_scale)
                        for x in masks.values()),'Every actual bank update uses the same common Q')
            before=common.native_rng_state(native); self.work['all_bank_update_attempts']+=1
            result={arm:engine._train_core(bank,H,zero,masks[arm],CONDITIONS[arm],self.metadata()) for arm,bank in banks.items()}
            common.require_same_native_rng(native,before,'all own-label CE updates on zero base'); self.fixed()
            self.label_epoch=self.parameter_label_epoch=label_epoch
            self.work['all_bank_update_completions']+=1; self.work['native_RNG_checks']+=1
            return {arm:dict(own_masked_label_CE=value['own_corrected_CE_mean'],queries=290) for arm,value in result.items()}

        def serve_ids(self,ids):
            self.fixed(); ids=ids.to(native.device); before=common.native_rng_state(native)
            result={arm:posterior.bounded(torch,engine._serve_core(bank,H,zero,ids,CONDITIONS[arm],self.metadata())['member_logits'],
                                         base,reach,ids) for arm,bank in banks.items()}
            common.require_same_native_rng(native,before,'fixed-context posterior serving'); self.fixed()
            self.work['native_RNG_checks']+=1; return result

        def evaluate(self,valid):
            require(self.purpose=='scientific_complete','Engineering/serving purpose cannot score VALID')
            predictions=self.serve_ids(valid['ids']); truth=valid['y'].to(native.device)
            stats={arm:posterior.readout(torch,prediction,truth) for arm,prediction in predictions.items()}
            self.work['complete_VALID_events']+=1; return stats

        def capture_state(self):
            self.fixed()
            return native._cpu_tree(dict(schema='staged-posterior-fixed-native-capture-v1',native_binding=row,
                context=context,identity=capture_identity,H=H,native_logits=base))

        def snapshot(self,*,kind='engineering_snapshot',arm=None,metrics=None):
            self.fixed()
            require(kind in ('engineering_snapshot','label_posterior_selected') and self.label_epoch>0
                    and all(bank._pending_mask is None and not bank._failed_update for bank in banks.values()),'Completed coherent label stage')
            require((kind=='engineering_snapshot' and arm is None and metrics is None) or
                    (kind=='label_posterior_selected' and self.purpose=='scientific_complete' and arm in banks), 'Unscored engineering or declared VALID selector')
            chosen=(arm,) if arm else tuple(banks)
            self.work['bank_parameter_copies']+=len(chosen); self.work['bank_optimizer_copies']+=len(chosen)
            return native._cpu_tree(dict(schema='staged-label-posterior-coherent-state-v1',kind=kind,arm=arm,
                selector_performed=kind=='label_posterior_selected',label_epoch=self.label_epoch,
                parameter_label_epoch=self.parameter_label_epoch,
                run=self.run,metrics=metrics,banks={key:dict(parameters=engine._learned_corrector_state(banks[key]),
                    optimizer=banks[key].optimizer.state_dict(),mask_rng_state=banks[key].mask_generator.get_state(),
                    logical_steps=banks[key].steps) for key in chosen},exact_resume_supported=False))

        def restore_learned(self,state):
            self.fixed()
            require(((self.purpose=='engineering_qualification' and state['kind']=='engineering_snapshot'
                      and state['selector_performed'] is False and state['metrics'] is None)
                     or (self.purpose=='selected_serving' and state['kind']=='label_posterior_selected'
                         and state['selector_performed'] is True and state['arm']==only_arm))
                    and state['schema']=='staged-label-posterior-coherent-state-v1'
                    and state['run']['native_binding']==row and state['run']['context']==context
                    and state['run']['capture_identity']==capture_identity and state['run']['source']==source_identity()
                    and tuple(state['banks'])==tuple(banks)
                    and 1<=state['parameter_label_epoch']<=state['label_epoch']<=1100
                    and all(bank._pending_mask is None and not bank._failed_update for bank in banks.values()),
                    'Engineering learned restore or selected serving only; no scientific resume')
            live={arm:(bank.mask_generator.get_state().clone(),bank.steps,copy.deepcopy(bank.counters)) for arm,bank in banks.items()}
            before=common.native_rng_state(native)
            for arm,bank in banks.items():
                engine._restore_learned_corrector(bank,state['banks'][arm],native)
                require(torch.equal(bank.mask_generator.get_state(),live[arm][0]) and bank.steps==live[arm][1]
                        and bank.counters==live[arm][2],'Restore learned/Adam only; masks/work remain live')
            common.require_same_native_rng(native,before,'staged learned/Adam restore'); self.fixed()
            self.parameter_label_epoch=state['parameter_label_epoch']
            self.work['learned_restores']+=1

        def memory(self): return memory()

        def close(self): pass  # Native hooks were removed immediately after the one capture.

    return Stage()


def run_complete(*,seed,output,later_execution_authorized=False,scientific_execution_authorized=False):
    require(later_execution_authorized is True and scientific_execution_authorized is True,'Disabled scientific driver')
    require(type(seed) is int and seed in SEEDS,'Declared seed block')
    output=Path(output); require(output.resolve().is_relative_to(PHASE) and not output.exists()
                                and output.name=='seed'+str(seed),'Fresh named phase-owned block; no retry/resume')
    started=time.monotonic(); output.mkdir(parents=True,exist_ok=False); stage=None; trace=[]
    timing=dict(role_hash_validation_loading_seconds=0.,capture_export_save_hash_seconds=0.,
                label_train_seconds=0.,complete_VALID_seconds=0.,selected_snapshot_save_seconds=0.,
                progress_private_trace_io_seconds=0.,endpoint_receipt_hash_io_seconds=0.)
    try:
        roles_started=time.monotonic(); train,valid,origin=load_roles(later_execution_authorized=True)
        timing['role_hash_validation_loading_seconds']=time.monotonic()-roles_started
        stage=make_stage(train_data=train,origin=origin,seed=seed,purpose='scientific_complete',
                         later_execution_authorized=True,scientific_execution_authorized=True)
        capture_save_started=time.monotonic()
        stage.native.torch.save(stage.capture_state(),output/'FIXED_CAPTURE.pt')
        capture_sha=sha(output/'FIXED_CAPTURE.pt'); best={arm:-1 for arm in ARMS}; selected_epochs={}; selected_stats={}
        timing['capture_export_save_hash_seconds']=time.monotonic()-capture_save_started
        stage.run['driver_timings_seconds']=timing; write(output/'RUN.json',stage.run)
        for arm in ARMS: (output/arm).mkdir()
        for epoch in range(1,1101):
            epoch_started=time.monotonic(); update=stage.train_step(label_epoch=epoch)
            timing['label_train_seconds']+=time.monotonic()-epoch_started
            valid_started=time.monotonic(); stats=stage.evaluate(valid)
            timing['complete_VALID_seconds']+=time.monotonic()-valid_started
            for arm in ARMS:
                if stats[arm]['correctcount']>best[arm]:
                    snapshot_started=time.monotonic()
                    best[arm]=stats[arm]['correctcount']; selected_epochs[arm]=epoch; selected_stats[arm]=stats[arm]
                    state=stage.snapshot(kind='label_posterior_selected',arm=arm,metrics=stats[arm])
                    state['fixed_capture_sha256']=capture_sha
                    stage.native.torch.save(state,output/arm/'selected.pt')
                    timing['selected_snapshot_save_seconds']+=time.monotonic()-snapshot_started
            trace.append(dict(label_epoch=epoch,TRAIN=update,VALID=stats,seconds=time.monotonic()-started))
            io_started=time.monotonic()
            write(output/'PRIVATE_VALID_TRACE.json',trace)
            write(output/'PROGRESS.json',dict(complete=False,label_epoch=epoch,work=stage.work,
                bank_counts={arm:dict(bank.counters) for arm,bank in stage.banks.items()},timings_seconds=timing,
                cuda_memory=stage.memory()))
            timing['progress_private_trace_io_seconds']+=time.monotonic()-io_started
        require(stage.work['all_bank_update_completions']==stage.work['common_Q_checks']==stage.work['complete_VALID_events']==1100
                and stage.work['native_updates']==0 and stage.work['native_capture_calls']==1,'Fixed full staged work')
        endpoint_started=time.monotonic()
        for arm,bank in stage.banks.items():
            _,heads,backwards,adam,_=EXPECTED[arm]
            require(bank.steps==bank.counters['mask_draws']==1100 and bank.counters['training_route_forwards']==heads*1100
                    and bank.counters['serving_route_forwards']==heads*1100 and bank.counters['route_backwards']==backwards*1100
                    and bank.counters['corrector_Adam_steps']==adam*1100,'Complete required bank work')
            write(output/arm/'PRIVATE_SELECTED_METRICS.json',selected_stats[arm])
            write(output/arm/'COMPLETE.json',dict(complete=True,seed=seed,arm=arm,label_updates=1100,
                source=source_identity(),selected_label_epoch=selected_epochs[arm],selected_sha256=sha(output/arm/'selected.pt'),
                private_metrics_sha256=sha(output/arm/'PRIVATE_SELECTED_METRICS.json'),fixed_capture_sha256=capture_sha,
                bank_counts=dict(bank.counters),native_binding=stage.run['native_binding'],ordinary_independent4=False))
        bank_receipt_hashes={arm:sha(output/arm/'COMPLETE.json') for arm in ARMS}
        private_trace_sha=sha(output/'PRIVATE_VALID_TRACE.json')
        timing['endpoint_receipt_hash_io_seconds']=time.monotonic()-endpoint_started
        stage.native.torch.cuda.synchronize(0)
        stage.run['final_cuda_memory']=stage.memory(); write(output/'RUN.json',stage.run)
        write(output/'COMPLETE.json',dict(complete=True,seed=seed,label_updates=1100,arms=list(ARMS),source=source_identity(),
            run=stage.run,work=stage.work,fixed_capture_sha256=capture_sha,seconds=time.monotonic()-started,
            bank_complete_sha256=bank_receipt_hashes,run_sha256=sha(output/'RUN.json'),
            private_VALID_trace_sha256=private_trace_sha,
            timings_seconds=timing,cuda_memory=stage.memory(),
            elapsed_scope='From fresh block creation through role loading, setup, all label work and endpoint receipts; excludes final COMPLETE write',
            comparative_opening_performed=False,new_native_training=0,ordinary_independent4_available=False))
        return dict(output=str(output),complete=True,comparative_opening_performed=False)
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            complete_label_epochs=len(trace),work=stage.work if stage else None,
            bank_counts={arm:dict(bank.counters) for arm,bank in stage.banks.items()} if stage else None,
            seconds=time.monotonic()-started,timings_seconds=timing,cuda_memory=stage.memory() if stage else None,
            automatic_retry=False,comparative_opening_performed=False))
        raise
    finally:
        if stage is not None: stage.close()


def reconstruct_selected(*,state,capture_cache,train_data,origin,later_execution_authorized=False):
    """Trusted in-memory trees only; use the file adapter for checkpoint byte custody."""
    require(later_execution_authorized is True and state['kind']=='label_posterior_selected','Disabled selected serving adapter')
    stage=make_stage(train_data=train_data,origin=origin,seed=state['run']['seed'],purpose='selected_serving',
                     capture_cache=capture_cache,only_arm=state['arm'],later_execution_authorized=True)
    stage.restore_learned(state); return stage


def reconstruct_selected_files(*,bank_directory,train_data,origin,later_execution_authorized=False):
    """Authenticate exact receipt-pinned selected/cache files before deserialization."""
    require(later_execution_authorized is True,'Disabled exact-file selected serving adapter')
    pins=verify_bindings(); runtime=json.loads((PHASE/pins['runtime_path']).read_text()); route(runtime)
    bank_directory=Path(bank_directory).resolve(strict=True); block=bank_directory.parent
    require(bank_directory.is_relative_to(PHASE) and bank_directory.name in ARMS,'Phase-owned selected bank endpoint')
    receipt=json.loads((bank_directory/'COMPLETE.json').read_text())
    require(receipt['complete'] is True and receipt['source']==source_identity() and receipt['label_updates']==1100
            and receipt['arm']==bank_directory.name and receipt['seed'] in SEEDS
            and block.name=='seed'+str(receipt['seed']),'Complete exact-source selected endpoint receipt')
    records=verify_closed_family(block.parent)
    require(records[(receipt['seed'],receipt['arm'])][0]==bank_directory,'All-15 closure before selected-state loading')
    selected=(bank_directory/'selected.pt').resolve(strict=True); capture=(block/'FIXED_CAPTURE.pt').resolve(strict=True)
    require(selected.is_relative_to(bank_directory) and capture.is_relative_to(block)
            and sha(selected)==receipt['selected_sha256'] and sha(capture)==receipt['fixed_capture_sha256'],
            'Exact selected and original fixed-cache bytes before torch.load')
    import torch  # Guarded later execution only; never imported in source preparation.
    require(str(torch.__version__)==runtime['torch'],'Pinned deserialization runtime')
    state=torch.load(selected,map_location='cpu',weights_only=False)
    cache=torch.load(capture,map_location='cpu',weights_only=False)
    require(state['fixed_capture_sha256']==receipt['fixed_capture_sha256']
            and state['label_epoch']==receipt['selected_label_epoch']
            and state['arm']==receipt['arm'] and state['run']['seed']==receipt['seed'],
            'Exact state/cache/selector receipt custody')
    return reconstruct_selected(state=state,capture_cache=cache,train_data=train_data,origin=origin,
                                later_execution_authorized=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--later-execution-authorized',action='store_true')
    parser.add_argument('--scientific-execution-authorized',action='store_true')
    args=parser.parse_args(); run_complete(seed=args.seed,output=args.output,later_execution_authorized=args.later_execution_authorized,
                                         scientific_execution_authorized=args.scientific_execution_authorized)


if __name__=='__main__': main()
