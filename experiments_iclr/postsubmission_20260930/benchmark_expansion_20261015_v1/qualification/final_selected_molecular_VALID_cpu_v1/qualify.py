"""One genuine saved molecular selected state, one CPU whole VALID replay."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time
import traceback
from support import HERE, EXPANSION, admission, digest, inside, receipt, verify, write_json


def tree_identity(torch, value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        return {'tensor_shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    if type(value).__module__.startswith('numpy') and hasattr(value,'tobytes'):
        return {'numpy_shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,dict): return {str(k):tree_identity(torch,v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [tree_identity(torch,v) for v in value]
    if hasattr(value,'item'): return value.item()
    return value


def restore_cpu_rng(torch, saved):
    import numpy as np
    random.setstate(saved['python']); np.random.set_state(copy.deepcopy(saved['numpy']))
    torch.set_rng_state(saved['torch_cpu'].cpu())
    # Saved CUDA bytes stay unchanged and are recorded, never applied on CPU.


@contextmanager
def no_training(adapter, work, no_updates):
    protected = []
    def blocked(*args,**kwargs):
        work['TRAIN_callback_attempts'] += 1
        raise RuntimeError('VALID qualification forbids every TRAIN callback')
    for owner,name in [(adapter,'train_epoch'),(adapter,'train_diagnostic'),(adapter.session,'train_update'),(adapter.session,'_next_train_batch')]:
        if hasattr(owner,name):
            protected.append((owner,name,getattr(owner,name))); setattr(owner,name,blocked)
    try:
        with no_updates(adapter):
            steps = [(opt,opt.step) for opt in adapter.optimizers]
            def counted(function):
                def step(*args,**kwargs):
                    work['optimizer_step_attempts'] += 1
                    return function(*args,**kwargs)  # exact source no_updates guard always raises
                return step
            for opt,function in steps: opt.step = counted(function)
            try: yield
            finally:
                for opt,function in steps: opt.step = function
    finally:
        for owner,name,function in protected: setattr(owner,name,function)


class CountedVALID:
    def __init__(self, loader, work, events): self.loader,self.work,self.events = loader,work,events
    def __iter__(self):
        for batch in self.loader:
            self.work['raw_VALID_batches'] += 1
            if self.work['raw_VALID_batches'] > 5: raise RuntimeError('Whole VALID batch bound exceeded')
            with self.events.open('a') as stream:
                stream.write(json.dumps({'event':'VALID_batch','batch':self.work['raw_VALID_batches'],'UTC_unix':time.time()})+'\n')
            yield batch


def execute(case_id, root, admission_path):
    request, inputs, output_root, admission_pin = admission(admission_path,root)
    case = next((row for row in inputs['cases'] if row['id']==case_id),None)
    if case is None: raise ValueError('Only the three exact frozen cases are allowed')
    output = inside(output_root/case_id,root)
    if output.exists(): raise ValueError('Preserve every prior case attempt; no overwrite/retry')
    output.mkdir(parents=True)
    started = time.perf_counter(); adapter = None; hooks = []; torch = None; outside = None
    work = {'native_constructions_started':0,'native_constructions_completed':0,'selected_state_loads':0,'whole_VALID_calls':0,'raw_VALID_batches':0,
            'native_forward_attempts':0,'native_forward_completed':0,'optimizer_step_attempts':0,'optimizer_steps':0,'TRAIN_callback_attempts':0}
    result = {'schema':'genuine-selected-molecular-CPU-VALID-qualification-v1','case_id':case_id,'status':'running',
        'role':'VALID','TEST_accessed':False,'execution_device':'cpu','original_training_device':'cuda:0',
        'scope':'genuine original producer plus current serving components on CPU; not exact-device or exact portable restoration qualification',
        'admission':admission_pin,'inputs':receipt(HERE/'INPUTS.json'),'harness':receipt(HERE/'MANIFEST.json'),
        'job_id':case['job_id'],'job_hash':case['job_hash'],'selected_epoch':case['selected_epoch'],
        'selected_checkpoint':case['checkpoint'],'selected_VALID_export':case['selected_VALID_export'],
        'rtol':1e-4,'atol':1e-5,'automatic_retries':0,'selection_changed':False,'actual_work':work}
    write_json(output/'START.json',result)
    def timeout(signum,frame): raise TimeoutError('Fixed CPU qualification wall bound reached')
    signal.signal(signal.SIGALRM,timeout); signal.signal(signal.SIGTERM,timeout); signal.alarm(request['wall_seconds_per_case'])
    try:
        if 'torch' in sys.modules: raise ValueError('Fresh worker required before CPU environment binding')
        os.environ['CUDA_VISIBLE_DEVICES']=''
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='2'
        resource.setrlimit(resource.RLIMIT_AS,(request['address_space_limit_bytes'],request['address_space_limit_bytes']))
        import torch as tensor_library
        torch = tensor_library
        import numpy as np
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized(): raise RuntimeError('CPU qualification must not initialize CUDA')
        outside = {'python':random.getstate(),'numpy':copy.deepcopy(np.random.get_state()),'torch_cpu':torch.get_rng_state().clone()}
        job = json.loads(verify(case['JOB'],root).read_text())
        fit = json.loads(verify(case['fit_RESULT'],root).read_text())
        if digest(job)!=case['job_hash'] or fit['job_hash']!=case['job_hash'] or job['id']!=case['job_id'] or fit['status']!='complete' or job['family']!='ogb' or job['procedure']!=case['procedure'] or job['TEST_access'] is not False or fit['TEST_accessed'] is not False:
            raise ValueError('Exact genuine completed own MolBACE producer required')
        if job['recipe']['dataset']!='ogbg-molbace' or job.get('device','cuda:0')!='cuda:0': raise ValueError('Frozen native MolBACE CUDA producer descriptor differs')
        runtime_paths = [verify(pin,root) for pin in case['genuine_runtime_sources'].values()]
        runtime = runtime_paths[0].parent
        if any(path.parent!=runtime for path in runtime_paths): raise ValueError('One coherent genuine original runtime required')
        if fit['runtime_sources'] != case['genuine_runtime_sources']: raise ValueError('Recorded producer runtime authority differs')
        checkpoint = torch.load(verify(case['checkpoint'],root),map_location='cpu',weights_only=False)
        saved = torch.load(verify(case['selected_VALID_export'],root),map_location='cpu',weights_only=False)
        epoch = checkpoint['selector']['selected_epoch']
        if checkpoint['schema']!='native-full-epoch-recovery-v1' or checkpoint['job_hash']!=case['job_hash'] or checkpoint['TEST_accessed'] is not False or epoch!=case['selected_epoch'] or epoch!=checkpoint['selector']['completed_epoch'] or fit['selected_epoch']!=epoch:
            raise ValueError('Exact own positive selected checkpoint required')
        if saved['schema']!='own-selected-VALID-logits-v1' or saved['job_hash']!=case['job_hash'] or saved['role']!='VALID' or saved['TEST_accessed'] is not False or saved['selected_checkpoint']!=case['checkpoint'] or saved['selected_epoch']!=epoch:
            raise ValueError('Exact frozen own selected VALID export required')
        binding = checkpoint['runtime_binding']
        actual_runtime = {name:pin['sha256'] for name,pin in case['genuine_runtime_sources'].items()}
        if binding!=fit['runtime_binding'] or binding['runtime_sha256']!=actual_runtime: raise ValueError('Original runtime binding differs')
        for path,sha in binding['dependency_sha256'].items(): verify({'path':path,'sha256':sha},root)
        verify({'path':job['callback_module'],'sha256':binding['callback_sha256']},root)
        extension = job.get('session_transform',job.get('extension_factory',job.get('session_extension')))
        if extension:
            extension_path = extension['path'] if isinstance(extension,dict) else extension.split(':',1)[0]
            verify({'path':extension_path,'sha256':binding['extension_sha256']},root)
        role_pins=case['original_source_metadata']['native_session_metadata']['scaffold_role_pins']
        for role,pin in role_pins.items():
            verify({'path':pin['path'],'sha256':pin['sha256']},root)
            if Path(pin['path']).name.endswith('.csv.gz') and Path(pin['path']).with_name(role+'.pt').exists():
                raise ValueError('Original native role-loader precedence changed')
        result['original_scaffold_role_pins']=role_pins
        encoders=case['original_source_metadata']['native_session_metadata'].get('encoding_dependencies',{})
        for pin in encoders.values():
            verify({'path':pin['path'],'sha256':pin['sha256']},root)
        saved_state_identity = digest(tree_identity(torch,checkpoint['adapter']))
        saved_rng_identity = digest(tree_identity(torch,checkpoint['global_rng']))
        for name in ('adapters','bookkeeping','common'):
            if name in sys.modules: raise ValueError('Fresh original-runtime module namespace required')
        sys.path.insert(0,str(runtime))
        factory = importlib.import_module('adapters'); policy = importlib.import_module('bookkeeping').Policy.from_dict(job['policy'],'ogb')
        sys.path.insert(0,str(EXPANSION))
        serving = importlib.import_module('final_selected_scoring_v4.role_serving')
        final_backend = importlib.import_module('final_selected_scoring_v4.backend')
        construction_job = copy.deepcopy(job); construction_job['device']='cpu'
        result['construction_override']={'device':'cpu','original_JOB_unchanged':digest(job)==case['job_hash'],
            'only_descriptor_field_changed_for_factory':'device','scope':'engineering CPU construction, not a new fit or rewritten saved descriptor'}
        work['native_constructions_started'] += 1
        adapter = factory.create_adapter(torch,construction_job,policy,Path(root).resolve())
        work['native_constructions_completed'] += 1
        if adapter.callback_receipt['sha256']!=binding['callback_sha256'] or (None if adapter.extension_receipt is None else adapter.extension_receipt['sha256'])!=binding['extension_sha256']:
            raise ValueError('Genuine callback/extension factory differs')
        if {pin['path']:pin['sha256'] for pin in adapter.dependency_receipts}!=binding['dependency_sha256']:
            raise ValueError('Constructed original source dependencies differ')
        adapter.load_state(copy.deepcopy(checkpoint['adapter'])); work['selected_state_loads'] += 1
        if adapter.session.split_pins!=role_pins: raise ValueError('Native role provider authority changed')
        if adapter.completed_epochs!=epoch or str(adapter.session.device)!='cpu': raise ValueError('Original selected clock or explicit CPU construction differs')
        if adapter.session.valid_loader.dataset is not adapter.session.valid_view or adapter.session.valid_view.role!='VALID' or len(adapter.session.valid_ids)!=151:
            raise ValueError('Only original complete official VALID view may be iterated')
        if any(value.device.type!='cpu' for value in adapter.model.parameters()): raise ValueError('Non-CPU model parameter found')
        before = final_backend.model_fingerprint(adapter.model)
        object_identity = lambda: [(scope,name,id(value),value.data_ptr(),str(value.device),str(value.dtype),list(value.shape))
            for scope,iterator in [('parameter',adapter.model.named_parameters()),('buffer',adapter.model.named_buffers())]
            for name,value in iterator]
        before_objects=object_identity()
        optimizer_before = digest(tree_identity(torch,[opt.state_dict() for opt in adapter.optimizers]))
        native = adapter.model if case['procedure']=='native_single' else adapter.model.native
        def forward_event(event):
            with (output/'EVENTS.jsonl').open('a') as stream:
                stream.write(json.dumps({'event':event,'native_forward_attempts':work['native_forward_attempts'],
                    'native_forward_completed':work['native_forward_completed'],'UTC_unix':time.time()})+'\n')
        def attempted(module,args):
            if work['native_forward_attempts'] >= case['expected_VALID']['native_forwards']:
                raise RuntimeError('Fixed native forward bound exhausted before another forward')
            work['native_forward_attempts'] += 1; forward_event('native_forward_attempt')
        def completed(module,args,output_value):
            work['native_forward_completed'] += 1; forward_event('native_forward_complete')
        hooks.extend([native.register_forward_pre_hook(attempted),native.register_forward_hook(completed)])
        adapter.session.valid_loader = CountedVALID(adapter.session.valid_loader,work,output/'EVENTS.jsonl')
        restore_cpu_rng(torch,checkpoint['global_rng'])
        result['saved_RNG']={'complete_saved_RNG_identity':saved_rng_identity,'CPU_Python_NumPy_restored_exactly':True,
            'CUDA_components_preserved_unapplied':tree_identity(torch,checkpoint['global_rng']['torch_cuda']),
            'no_CUDA_initialization':not torch.cuda.is_initialized(),'private_loader_state_restored_by_genuine_session':True}
        result['construction_and_restore_seconds']=time.perf_counter()-started
        with no_training(adapter,work,final_backend.no_updates):
            forward_start=time.perf_counter(); work['whole_VALID_calls'] += 1
            replay=serving.predict(torch,adapter,job,'VALID',None,Path(root).resolve())
            result['VALID_forward_seconds']=time.perf_counter()-forward_start
        if not torch.equal(replay['ids'],saved['ids']) or replay['labels'].shape!=saved['labels'].shape or replay['labels'].dtype!=saved['labels'].dtype or not torch.allclose(replay['labels'].double(),saved['labels'].double(),rtol=0,atol=0,equal_nan=True):
            raise ValueError('Exact selected VALID IDs/order/labels differ')
        if len(replay['ids'])!=151 or len(torch.unique(replay['ids']))!=151 or not torch.equal(replay['ids'],adapter.session.valid_ids.cpu()): raise ValueError('Whole original ordered official VALID population required')
        if replay['work']['native_single_node_skipped_graph_ids']!=[] or work['raw_VALID_batches']!=5 or replay['work']['native_batches']!=5 or work['native_forward_attempts']!=case['expected_VALID']['native_forwards'] or work['native_forward_completed']!=work['native_forward_attempts']:
            raise ValueError('Native complete VALID/skip/forward counts differ')
        if replay['logits'].shape!=saved['raw_logits'].shape or replay['logits'].dtype!=saved['raw_logits'].dtype or not torch.isfinite(replay['logits']).all().item() or not torch.isfinite(replay['members']).all().item(): raise ValueError('Finite native selected logit schema differs')
        if object_identity()!=before_objects or final_backend.model_fingerprint(adapter.model)!=before or digest(tree_identity(torch,[opt.state_dict() for opt in adapter.optimizers]))!=optimizer_before:
            raise ValueError('CPU inference changed parameters/buffers/private banks or optimizer state')
        if saved_state_identity!=digest(tree_identity(torch,checkpoint['adapter'])) or saved_rng_identity!=digest(tree_identity(torch,checkpoint['global_rng'])):
            raise ValueError('Saved checkpoint/private/RNG value was modified')
        delta=(replay['logits'].double()-saved['raw_logits'].double()).abs(); threshold=1e-5+1e-4*saved['raw_logits'].double().abs()
        passed=bool(torch.allclose(replay['logits'],saved['raw_logits'],rtol=1e-4,atol=1e-5))
        result.update(status='passed' if passed else 'failed_numerical_reconstruction',
            parity={'passed':passed,'max_absolute_difference':float(delta.max()),'coordinates_outside_tolerance':int((delta>threshold).sum())},
            exact_IDs_labels_order=True,parameters_buffers_private_banks_and_optimizer_unchanged=True,
            saved_checkpoint_and_RNG_unchanged=True,serving_work=replay['work'],readout=replay['readout'],
            current_serving_source=receipt(EXPANSION/'final_selected_scoring_v4/MANIFEST.json'))
        payload={'schema':'genuine-selected-molecular-CPU-VALID-replay-v1','role':'VALID','TEST_accessed':False,
            'case_id':case_id,'job_hash':case['job_hash'],'selected_checkpoint':case['checkpoint'],'selected_epoch':epoch,
            'serving_source':result['current_serving_source'],'logits':replay['logits'],'mean':replay['mean'],
            'members':replay['members'],'ids':replay['ids'],'labels':replay['labels'],'parity':result['parity']}
        torch.save(payload,output/'CPU_VALID.pt'); result['CPU_VALID_export']=receipt(output/'CPU_VALID.pt')
    except BaseException as error:
        result.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)})
    finally:
        signal.alarm(0)
        for handle in hooks: handle.remove()
        if adapter is not None:
            try: adapter.close()
            except Exception as error: result['close_error']=str(error); result['status']='failed'
        if torch is not None:
            result['CUDA_context_initialized']=bool(torch.cuda.is_initialized())
            if result['CUDA_context_initialized']: result['status']='failed'; result['CUDA_violation']=True
            if outside is not None: restore_cpu_rng(torch,outside)
        result.update(wall_seconds=time.perf_counter()-started,own_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            process_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,process_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime)
        write_json(output/'RESULT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True); parser.add_argument('--admission',required=True); parser.add_argument('--case',required=True)
    args=parser.parse_args(); result=execute(args.case,args.root,args.admission)
    print(json.dumps({'case_id':result['case_id'],'status':result['status'],'wall_seconds':result['wall_seconds']}))
    raise SystemExit(0 if result['status']=='passed' else 2)
