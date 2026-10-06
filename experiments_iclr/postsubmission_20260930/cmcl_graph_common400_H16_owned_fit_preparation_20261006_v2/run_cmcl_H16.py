"""Disabled once-only owned CMCL H16 fit/context owner; held labels stay closed."""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import traceback

SOURCE_RELEASED = False
CAPS = {'max_elapsed_seconds':900,'max_process_rss_bytes':8589934592,
        'max_cuda_allocated_bytes':77309411328,'max_cuda_reserved_bytes':79456894976}
WATCHDOG, FREE, GRACE, REAP = 1020, 83751862272, 5, 10
OUTPUT = 'cmcl_graph_common400_H16_owned_fit_execution_root_20261006_v2'
PINS = {'accessor': {'bytes': 19074,
              'path': 'amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py',
              'sha256': '9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85'},
 'boundary': {'bytes': 6797,
              'path': 'cmcl_graph_common400_native_first_order_parity_activation_root_20261006_v1/readonly_sources/boundary.py',
              'sha256': '699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac'},
 'cmcl': {'bytes': 4330,
          'path': 'cmcl_graph_objective_transplant_source_preparation_20261006_v1/cmcl_loss.py',
          'sha256': '4f04eb87381512c9f7506b6458917b83fd12228f3c7651c6291e902c99fe6ac9'},
 'compare': {'bytes': 28390,
             'path': 'amazon_ordinary_shared_bank_first_order_qualification_preparation_20261006_v1/qualify.py',
             'sha256': 'd175dc6bf44886d874959cecd9833dc9f192f30b764bc7e9b938be83f2ad8af2'},
 'custody': {'bytes': 29855,
             'path': 'learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py',
             'sha256': 'baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688'},
 'h16': {'bytes': 14261,
         'path': 'cmcl_graph_common400_H16_owned_fit_preparation_20261006_v2/cmcl_H16_released.py',
         'sha256': '239ee8048808a8d53263d384829dbb9569418702ad6b6d8e39bd1aa71510b774'},
 'helpers': {'bytes': 41361,
             'path': 'amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v3/six_arm_worker.py',
             'sha256': 'f19a94be4102e74f30d2b779ca602e288cf40c446ae367ce9d9ab44091569ad1'},
 'native': {'bytes': 7129,
            'path': 'cmcl_graph_common400_native_first_order_parity_activation_root_20261006_v1/readonly_sources/native.py',
            'sha256': '9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8'},
 'operator': {'bytes': 14428,
              'path': 'cmcl_graph_common400_native_first_order_parity_activation_root_20261006_v1/readonly_sources/operator.py',
              'sha256': 'fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3'},
 'ordinary': {'bytes': 34961,
              'path': 'amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py',
              'sha256': '5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b'},
 'owned': {'bytes': 23700,
           'path': 'public_path_responsibility_two_hop_paired_qualification_owned_execution_preparation_20261006_v2/control.py',
           'sha256': '5f8d214660dceaa55bb2393bc3b7a2f7c840ce7508da0e630ee30a694dc6a458'},
 'port': {'bytes': 11837,
          'path': 'cmcl_graph_common400_native_first_order_parity_activation_root_20261006_v1/readonly_sources/port.py',
          'sha256': 'a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86'},
 'process': {'bytes': 25151,
             'path': 'amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3/run_scientific.py',
             'sha256': '65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb'}}
EVIDENCE = {'CPU_owned_terminal_PASS': {'bytes': 2970,
                             'path': 'cmcl_CPU_fixture_terminal_handoff_root_20261006_v2/TERMINAL.json',
                             'sha256': 'b65805fcaa585de3cd46bc58b4a7c19b0293c18604de97aee37b5c76e9d7d0d4'},
 'CPU_worker_PASS': {'bytes': 8530,
                     'path': 'cmcl_CPU_fixture_terminal_handoff_root_20261006_v2/WORKER_RESULT.json',
                     'sha256': 'd73ac3939f3857d6f96d42b5dc079721e912a8b6392486a76a44b931b88d18dc'},
 'native_PARITY_terminal': {'bytes': 5610,
                            'path': 'cmcl_graph_common400_native_first_order_parity_activation_root_20261006_v3/RESULTS/TERMINAL.json',
                            'sha256': '1ee25c02f81946b3a55fc98a3293feb50f07db4335f27a8574ede4006520cc76'},
 'native_PARITY_worker': {'bytes': 8669,
                          'path': 'cmcl_graph_common400_native_first_order_parity_activation_root_20261006_v3/RESULTS/WORKER_RESULT.json',
                          'sha256': '386bdd06193b1692fb82606648e01d3f82f1bbccbaac85eb97a98e05705fb17e'}}
EXPECTED = {'native_value_callbacks':64,'native_replay_callbacks':64,'small_logit_grad_APIs':16,
    'native_parameter_VJP_APIs':64,'simultaneous_SGD_attempts':16,'shared_SGD_maps':16,
    'private_SGD_row_maps':64,'completed_updates':16,'serving_callbacks':8}
OWNER_BILL = {'callable_native_callbacks':136,'serialization_replay_callbacks':4,'owner_native_callbacks':140}


def require(value,message):
    if not value: raise RuntimeError(message)


def load(root,row,key):
    require(row['sha256'] == PINS[key]['sha256'] and row['bytes'] == PINS[key]['bytes'], 'Exact source required: '+key)
    path=root/row['path']; require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
        and path.stat().st_mode & 0o222 == 0 and path.stat().st_size == row['bytes']
        and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'],'Immutable source differs: '+key)
    name='_owned_CMCL_H16_'+key; require(name not in sys.modules,'Fresh namespace required')
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module


def authenticate(args):
    root=Path(args.source_root).absolute(); require(root.resolve() == root and sys.dont_write_bytecode
        and not any(n == 'torch' or n.startswith('torch.') for n in sys.modules),'Resolved fresh phase/-B/pre-Torch required')
    owned=load(root,PINS['owned'],'owned')
    scope_path=Path(args.scope).absolute(); scope=owned.read(owned.bound(root,{
        'path':str(scope_path.relative_to(root)),'bytes':scope_path.stat().st_size,'sha256':args.scope_sha256}))
    require(scope['schema'] == 'root_owned_CMCL_H16_fit_scope_v1' and all(scope[k] is True for k in
        ('root_owned_fit_authorized','owner_source_review_approved','fixed_before_fit','exclusive_GPU_window'))
        and scope['A_VALID_TEST_scoring'] is False and scope['automatic_retry'] is False,
        'Separate actual reviewed root fit scope required; parity scope grants no fit')
    require(scope['planned_native_callback_bill'] == OWNER_BILL,'Fixed 136 callable + 4 serialization replay bill required before fit')
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd().resolve() == owned.REPO and root == owned.PHASE
        and Path(sys.executable).absolute() == owned.PYTHON and str(Path(sys.executable).resolve()) == scope['python_resolved']
        and scope['GPU_UUID'] == owned.GPU and scope['resource_limits'] == CAPS and scope['external_watchdog_seconds'] == WATCHDOG
        and scope['minimum_initial_cuda_free_bytes'] == FREE,'Exact normal single-GPU runtime and prospective caps required')
    require(owned.bound(root,scope['executor']).resolve() == Path(__file__).resolve(),'Exact fit owner source required')
    owned.bound(root,scope['owner_review'])
    require(set(scope['sources']) == set(PINS)-{'owned'},'Complete frozen source map required')
    for key,row in scope['sources'].items():
        require(row['sha256'] == PINS[key]['sha256'] and row['bytes'] == PINS[key]['bytes'],'Source pin differs')
        owned.bound(root,row)
    for key,row in EVIDENCE.items():
        require(scope[key]['sha256'] == row['sha256'] and scope[key]['bytes'] == row['bytes'],'Exact existing evidence required: '+key)
    worker=owned.read(owned.bound(root,scope['native_PARITY_worker']))
    terminal=owned.read(owned.bound(root,scope['native_PARITY_terminal']))
    require(worker['status'] == 'PASS_CMCL_NATIVE_COMMON400_FIRST_ORDER_PARITY_ENGINEERING_ONLY'
        and worker['numeric_PARITY_supported'] is True and worker['bill']['native_callbacks'] == 20
        and not worker['restoration_errors'] and terminal['status'] == 'PASS_OWNED_CMCL_ENGINEERING_CLOSED'
        and terminal['owned_resource_closure_PASS'] is True and not terminal['cleanup_errors']
        and terminal['original_worker_result']['sha256'] == scope['native_PARITY_worker']['sha256']
        and len(terminal['children']) == 1 and terminal['children'][0]['wait4_closed'] is True
        and terminal['children'][0]['child_exit_code'] == 0 and terminal['children'][0]['external_timeout'] is False,
        'Actual parity worker/whole-child join required')
    cpu=owned.read(owned.bound(root,scope['CPU_worker_PASS'])); cpu_terminal=owned.read(owned.bound(root,scope['CPU_owned_terminal_PASS']))
    require(cpu['status'] == 'PASS_CPU_FLOAT64_CMCL_ANALYTIC_FIXTURES_ENGINEERING_ONLY' and cpu['completed_frozen_checks'] == 19
        and not cpu['restoration_errors'] and cpu_terminal['status'] == 'PASS_CPU_CMCL_OWNED_PUBLICATION_EXIT_RESOURCE_CLOSURE_ENGINEERING_ONLY', 'Existing actual CPU fixture evidence required')
    return root,owned,scope


def available(owned,scope):
    return owned.availability({'prior_job_closures':scope['prior_job_closures'],'minimum_free_GPU_bytes':FREE})


def launch(args,root,owned,scope):
    try: sample=available(owned,scope)
    except BaseException as error:
        print(json.dumps({'status':'WAIT_NO_LAUNCH','error':str(error),'automatic_launch_after_WAIT':False})); return 2
    base=root/OUTPUT; require(not base.exists() and not base.is_symlink(),'One fixed fit output already claimed')
    base.mkdir(mode=0o700); record={'token':os.urandom(16).hex(),'device_inode':[base.stat().st_dev,base.stat().st_ino]}; owner=owned.Owner(base,record)
    argv=[str(owned.PYTHON),'-B',str(Path(__file__).resolve()),'--execute-authorized','--mode','supervise','--source-root',str(root),
          '--scope',str(Path(args.scope).absolute()),'--scope-sha256',args.scope_sha256]
    owned.write(owner,'LAUNCH_SENTINEL.json',{'owner':record,'scope_sha256':args.scope_sha256,'supervisor_argv':argv,'availability':sample,'one_fit_no_retry':True},True)
    with (base/'supervisor.stdout.log').open('xb') as out,(base/'supervisor.stderr.log').open('xb') as err:
        child=subprocess.Popen(argv,cwd=owned.REPO,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
    observed=owned.physical(child.pid,argv,parent=os.getpid())
    owned.write(owner,'LAUNCH_HANDLE.json',{'owner':record,'scope_sha256':args.scope_sha256,'supervisor':observed},True)
    print(json.dumps({'status':'DETACHED_CMCL_H16_FIT_HANDLE','pid':child.pid,'start_ticks':observed['starttime_ticks'],'wait_in_tool_connection':False})); return 0


def supervise(args,root,owned,scope):
    base=root/OUTPUT; claim=owned.read(base/'LAUNCH_SENTINEL.json'); owner=owned.Owner(base,claim['owner']); owner.verify(claim['owner']['token'])
    receipt={'status':'FAIL_OWNED_CMCL_H16_FIT','scope_sha256':args.scope_sha256,'launch_attempts':0,'children':[],'cleanup_errors':[],
             'A_VALID_TEST_scoring':False,'automatic_retry':False}; process=None; code=1
    old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(): owned.write(owner,'SUPERVISOR_STATUS.json',receipt)
    def interrupted(number,frame): raise InterruptedError('Owned CMCL H16 supervisor interrupted')
    try:
        deadline=time.monotonic()+5
        while not (base/'LAUNCH_HANDLE.json').exists():
            require(time.monotonic()<deadline,'No verified launch handle; no fit child'); time.sleep(0.05)
        handle=owned.read(base/'LAUNCH_HANDLE.json')
        require(handle['supervisor']['pid'] == os.getpid() and handle['scope_sha256'] == args.scope_sha256,'Fit supervisor handle differs')
        owned.physical(os.getpid(),claim['supervisor_argv'],start=handle['supervisor']['starttime_ticks'])
        available(owned,scope); process=owned.helper(); process.validate_caps(CAPS,WATCHDOG,GRACE,REAP)
        for s in old: signal.signal(s,interrupted)
        argv=['--execute-authorized','--mode','fit','--source-root',str(root),'--scope',str(Path(args.scope).absolute()),
              '--scope-sha256',args.scope_sha256,'--output',str(base/'fit')]
        receipt['launch_attempts']=1; save()
        child=process.launch(owned.REPO,owner,claim['owner']['token'],'fit',str(owned.PYTHON),Path(__file__).resolve(),argv,
            dict(os.environ,CUDA_VISIBLE_DEVICES=scope['GPU_UUID'],PYTHONDONTWRITEBYTECODE='1'),receipt['children'])
        child['verified_physical_identity']=owned.physical(child['pid'],[str(owned.PYTHON),'-B',str(Path(__file__).resolve()),*argv],os.getpid(),child['starttime_ticks']); save()
        while not process.watch(child,WATCHDOG,GRACE,REAP): time.sleep(0.25)
        result=owned.read(base/'fit/RESULT.json'); process.resource_closure(child,result,CAPS)
        require(result['status'] == 'CMCL_H16_COMPLETE_A_CLOSED_RESOURCE_ONLY' and result['counts'] == EXPECTED
            and result['native_callback_attempts'] == 136 and result['serialization_replay_callback_attempts'] == 4
            and result['owner_native_callback_attempts'] == 140 and result['planned_native_callback_bill'] == OWNER_BILL
            and result['initial_family_state_exact'] is True and result['serialization_checks']['all_four_complete_native_logits_match'] is True
            and result['serialization_checks']['mean_softmax_probabilities_match'] is True
            and set(result['custody_checks']) == {'original_functional_state_and_inputs','original_native_custody','original_common_file'}
            and all(result['custody_checks'].values()) and not result['custody_errors']
            and result['scope_sha256'] == args.scope_sha256 and result['A_VALID_TEST_scoring'] is False
            and not result['restoration_errors'],'Complete fixed fit result required')
        receipt['status']='CMCL_H16_FIT_COMPLETE_OWNED_WHOLE_CHILD_CLOSED'; code=0
    except BaseException as error: receipt.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        for s in old: signal.signal(s,signal.SIG_IGN)
        if process is not None:
            try: receipt['cleanup_errors'].extend(process.cleanup(receipt['children'],GRACE,REAP))
            except BaseException as error: receipt['cleanup_errors'].append(str(error))
        for s,h in old.items(): signal.signal(s,h)
        if receipt['cleanup_errors'] or any(not c['wait4_closed'] for c in receipt['children']): code=1
        if process is not None and (base/'fit/RESULT.json').is_file():
            try: receipt['worker_result']=process.immutable_descriptor(root,base/'fit/RESULT.json')
            except BaseException as error: receipt.setdefault('receipt_errors',[]).append(str(error)); code=1
        if code: receipt['status']='FAIL_OWNED_CMCL_H16_FIT'
        receipt['owned_fit_resource_closure_complete']=code == 0; owned.write(owner,'TERMINAL.json',receipt,True)
    return code


def fit(args,root,owned,scope):
    base=root/OUTPUT; handle=owned.read(base/'LAUNCH_HANDLE.json'); claim=owned.read(base/'LAUNCH_SENTINEL.json')
    require(handle['scope_sha256'] == claim['scope_sha256'] == args.scope_sha256 and os.getppid() == handle['supervisor']['pid'], 'Fit must be direct child of this verified owner')
    owned.physical(handle['supervisor']['pid'],claim['supervisor_argv'],start=handle['supervisor']['starttime_ticks'])
    output=Path(args.output).absolute(); require(output == base/'fit' and not output.exists() and not output.is_symlink(),'Fixed fresh fit output required')
    output.mkdir(mode=0o700); owner=owned.Owner(output,{'token':os.urandom(16).hex(),'device_inode':[output.stat().st_dev,output.stat().st_ino]})
    progress={}; modules={}; torch=None; backend=None; old_threads=None; rng=None; body_error=None
    before=None; original=None; common_path=None
    receipt={'status':'RUNNING_CMCL_H16','scope_sha256':args.scope_sha256,'native_callback_attempts':0,'counts':{},'artifacts':{},
        'serialization_replay_callback_attempts':0,'owner_native_callback_attempts':0,'planned_native_callback_bill':dict(OWNER_BILL),
        'initial_family_state_exact':False,'serialization_checks':{},'custody_checks':{},'custody_errors':[],
        'model_fits':0,'fit_invocations':0,'A_VALID_TEST_scoring':False,'optional_feature_sharing':False,'restoration_errors':[],'callback_finalization_errors':[]}
    def save(): owned.write(owner,'RESULT.json',receipt)
    old_path=list(sys.path); old_env=('CUBLAS_WORKSPACE_CONFIG' in os.environ,os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    old_signal=signal.getsignal(signal.SIGALRM); old_timer=signal.getitimer(signal.ITIMER_REAL); require(old_timer == (0.0,0.0),'Fresh fit child timer required')
    def resources():
        import resource
        row={'elapsed_seconds':time.monotonic()-STARTED,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cuda_peak_allocated_bytes':0,'cuda_peak_reserved_bytes':0}
        if torch is not None and torch.cuda.is_initialized(): row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated('cuda:0'),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved('cuda:0'))
        return row
    def limits():
        row=resources(); receipt['whole_process_resources']=row
        require(all(row[k] <= CAPS[c] for k,c in [('elapsed_seconds','max_elapsed_seconds'),('process_peak_rss_bytes','max_process_rss_bytes'),('cuda_peak_allocated_bytes','max_cuda_allocated_bytes'),('cuda_peak_reserved_bytes','max_cuda_reserved_bytes')]),'Fixed prospective fit cap exceeded; no expansion')
    try:
        def expired(number,frame): raise TimeoutError('CMCL H16 whole-child cap exceeded')
        signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,max(0.001,900-(time.monotonic()-STARTED)))
        modules={k:load(root,row,k) for k,row in scope['sources'].items() if k not in ('native','boundary')}
        h16=modules['h16']; require(h16.SOURCE_RELEASED is True,'Disclosed exact one-flag H16 release required')
        ordinary,helpers,port,process,accessor,custody=(modules[k] for k in ('ordinary','helpers','port','process','accessor','custody'))
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; sys.path.insert(0,scope['site_packages'])
        import torch
        import numpy as np
        require(str(Path(torch.__file__).resolve()) == scope['torch_module_path'] and not torch.cuda.is_initialized(),'Exact pre-CUDA normal Torch required')
        process.torch=torch; backend=process.backend_snapshot(); old_threads=torch.get_num_threads()
        torch.use_deterministic_algorithms(True,warn_only=False); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(process.backend_snapshot() == scope['strict_backend_expected'],'Existing strict backend differs')
        require(process.runtime_identity() == scope['runtime_identity_expected'] and torch.cuda.device_count() == 1,'Existing qualified native runtime differs')
        require(torch.cuda.mem_get_info('cuda:0')[0] >= FREE,'Exclusive GPU free floor failed')
        rng=helpers._rng('cuda:0'); modules.update({k:load(root,scope['sources'][k],k) for k in ('native','boundary')})
        common_path=owned.bound(root,h16.COMMON); origin=owned.read(owned.bound(root,h16.ORIGIN)); expected=origin['recipe']
        require(origin['schema'] == 'amazon_G0_six_arm_run_v1' and origin['A_labels_received'] is False and origin['A_scoring_performed'] is False,'Original W-only common origin required')
        preflight=helpers._public_identity_before_w(accessor,root,root/ordinary.PUBLIC_B_RELATIVE,ordinary.INPUT_IDENTITY)
        data=accessor.load_public_b(root,root/ordinary.PUBLIC_B_RELATIVE,device='cuda:0'); provenance=data['provenance']
        require(all(provenance[k]['sha256'] == expected[v] for k,v in [('public_b_manifest','public_b_manifest_sha256'),('public_graph','public_graph_sha256'),('roles','roles_sha256')])
            and provenance['preprocessing'] == preflight['preprocessing'] and provenance['preprocessing']['edge_logical_sha256'] == expected['native_edge_logical_sha256'], 'Original fixed role/graph custody differs')
        S,yS,R,yR=(data[k] for k in ('inner_indices','inner_labels','query_indices','query_labels'))
        require(data['features'].shape == (24492,300) and data['features'].dtype == torch.float32
            and S.numel() == 2449 and R.numel() == 2450 and not bool(torch.isin(S,R).any())
            and not bool(torch.isin(torch.cat((S,R)),torch.cat((data['W_ids'],data['A_ids']))).any()),'Only fixed S/R targets may train; W/A excluded')
        image=torch.load(common_path,map_location='cpu',weights_only=True)
        require(image['schema'] == 'amazon_G0_frozen_state_v1' and image['id'] == 'initial' and image['episodes'] == 0
            and image['warm_updates'] == 400 and image['warm_role'] == 'W' and image['construction']['seed'] == 17
            and image['global_stage'] is True and image['eval_mode'] is True and image['recipe'] == expected and len(image['W_own_CE_trace']) == 400,'Exact last commonW400 image required')
        family,_=helpers._fresh_family(modules['native'],modules['boundary'],'cuda:0'); family.load_state_dict(image['family_state'],strict=True); family.set_global_stage(True); family.eval()
        ordinary.restore_rng(image['rng'],'cuda:0'); forward,theta,phis,_=port._native_callback_and_state(family,data['features'],data['edge_index'],expected_nodes=24492,global_stage=True)
        custody.torch=torch; custody.np=np; before=custody.snapshot(family,data['features'],data['edge_index'],torch.device('cuda:0'))
        original=helpers._cpu_tree((theta,phis,S,yS,R,yR,image)); trace_written=0
        def checkpoint(name,t,p,completed):
            owner.verify(owner.record['token']); values=helpers._cpu_tree(family.state_dict())
            for key,value in t.items(): values[key]=value.detach().cpu().clone()
            for key in port.PRIVATE_NAMES: values[key]=torch.stack([phi[key] for phi in p]).detach().cpu().clone()
            payload={'schema':'common400_CMCL_H16_frozen_state_v1','arm':h16.ARM,'completed_updates':completed,'H':16,'M_K_beta':[4,3,0.75],
                'eta_core_private':[0.001,0.01],'common400':h16.COMMON,'origin_run':h16.ORIGIN,'recipe':expected,'seed':17,'split':0,
                'global_stage':True,'eval_mode':True,'family_state':values,'rng':helpers._cpu_tree(helpers._rng('cuda:0')),
                'A_VALID_TEST_scoring':False,'W_loss_during_continuation':False,'fit_scope_sha256':args.scope_sha256}
            receipt['artifacts'][name]=helpers._save_state(output/(name+'.pt'),payload)
            return values
        checkpoint('initial',theta,phis,0)
        initial_saved=torch.load(output/'initial.pt',map_location='cpu',weights_only=True)
        modules['compare'].compare(initial_saved['family_state'],image['family_state'],exact=True)
        receipt['initial_family_state_exact']=True; del initial_saved
        def persist():
            nonlocal trace_written
            owner.verify(owner.record['token']); receipt['counts']=dict(progress.get('counts',{})); receipt['last_completed_update']=receipt['counts'].get('completed_updates',0)
            history=progress.get('history',[])
            if len(history)>trace_written:
                with (output/'TRACE.jsonl').open('a') as stream:
                    for row in history[trace_written:]: stream.write(json.dumps(row,allow_nan=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                trace_written=len(history)
            for key in ('common400_serving','endpoint_serving'):
                if key in progress and key not in receipt['artifacts']:
                    receipt['artifacts'][key]=helpers._save_state(output/(key+'.pt'),helpers._cpu_tree(progress[key]))
            save()
        def checked(t,p):
            persist(); limits(); require(os.getppid() == handle['supervisor']['pid'],'Owned supervisor disappeared; no continuation')
            receipt['native_callback_attempts']+=1; receipt['owner_native_callback_attempts']+=1; save(); active_error=None
            try: return forward(t,p)
            except BaseException as error: active_error=error; raise
            finally:
                errors=[]
                for label,call in [('resources',limits),('progress',persist)]:
                    try: call()
                    except BaseException: errors.append({'stage':label,'traceback':traceback.format_exc()})
                receipt['callback_finalization_errors'].extend(errors)
                if active_error is None: require(not errors,'Native callback resource/publication failure')
        def replay_endpoint(values):
            saved=torch.load(output/'endpoint.pt',map_location='cpu',weights_only=True)
            modules['compare'].compare(saved['family_state'],values,exact=True)
            receipt['serialization_checks']['endpoint_family_state_exact']=True
            logical_rng=helpers._rng('cuda:0'); replay_error=None
            try:
                replay_family,_=helpers._fresh_family(modules['native'],modules['boundary'],'cuda:0')
                require(type(replay_family) is type(family) and replay_family is not family and replay_family.core is not family.core
                    and not ({id(p) for p in replay_family.parameters()} & {id(p) for p in family.parameters()}),
                    'Genuinely fresh identical native boundary family required')
                replay_family.load_state_dict(saved['family_state'],strict=True); replay_family.set_global_stage(True); replay_family.eval()
                modules['compare'].compare(helpers._cpu_tree(replay_family.state_dict()),saved['family_state'],exact=True)
                receipt['serialization_checks'].update(fresh_native_family=True,installed_family_state_exact=True)
                with torch.no_grad():
                    rows=[]
                    for member in range(4):
                        persist(); limits(); require(os.getppid() == handle['supervisor']['pid'],'Owned supervisor disappeared; no replay')
                        receipt['serialization_replay_callback_attempts']+=1; receipt['owner_native_callback_attempts']+=1; save(); active_error=None
                        try:
                            z=replay_family.forward_member(data['features'],data['edge_index'],member)
                            require(z.shape == (24492,5) and z.dtype == torch.float32,'Complete native FP32 serialization replay required')
                            rows.append(z)
                        except BaseException as error: active_error=error; raise
                        finally:
                            errors=[]
                            for label,call in [('serialization_resources',limits),('serialization_progress',persist)]:
                                try: call()
                                except BaseException: errors.append({'stage':label,'traceback':traceback.format_exc()})
                            receipt['callback_finalization_errors'].extend(errors)
                            if active_error is None: require(not errors,'Serialization replay resource/publication failure')
                    bank=torch.stack(rows)
                    pool=modules['cmcl']._engineering_mean_probabilities(bank,engineering_authorized=True)
                served=progress['endpoint_serving']
                receipt['serialization_checks']['member_logits_max_abs_difference']=modules['compare'].compare(bank,served['member_logits'])
                receipt['serialization_checks']['mean_softmax_probabilities_max_abs_difference']=modules['compare'].compare(pool,served['mean_softmax_probabilities'])
                receipt['serialization_checks'].update(all_four_complete_native_logits_match=True,mean_softmax_probabilities_match=True)
            except BaseException as error: replay_error=error; raise
            finally:
                try: ordinary.restore_rng(logical_rng,'cuda:0')
                except BaseException:
                    receipt['restoration_errors'].append({'stage':'serialization_replay_numeric_RNG','traceback':traceback.format_exc()})
                    if replay_error is None: raise
        pure={k:modules[k] for k in h16.PINS}
        fit_body_error=None; receipt['fit_invocations']=1; receipt['model_fits']=1; save()
        try:
            result=h16.run_H16(pure,checked,theta,phis,S,yS,R,yR,admission=scope['context_admission'],progress=progress)
            persist(); require(result['counts'] == EXPECTED and receipt['native_callback_attempts'] == 136,'Complete fixed CMCL H16 bill required')
            endpoint_values=checkpoint('endpoint',result['theta'],result['phis'],16)
            replay_endpoint(endpoint_values)
            require(receipt['serialization_replay_callback_attempts'] == 4 and receipt['owner_native_callback_attempts'] == 140,
                'Complete fixed serialization replay and owner bill required')
        except BaseException as error:
            fit_body_error=error; raise
        finally:
            if (fit_body_error is not None or progress.get('status') == 'FAIL_CMCL_H16') and 'last_completed_theta' in progress:
                try: checkpoint('partial',progress['last_completed_theta'],progress['last_completed_phis'],progress['counts']['completed_updates'])
                except BaseException: receipt.setdefault('partial_publication_errors',[]).append(traceback.format_exc())
            try: persist()
            except BaseException:
                receipt.setdefault('fit_finalization_errors',[]).append(traceback.format_exc())
                if fit_body_error is None: raise
        limits(); receipt['status']='CMCL_H16_COMPLETE_A_CLOSED_RESOURCE_ONLY'
    except BaseException as error:
        body_error=error; receipt.update(status='FAIL_CMCL_H16_FIT',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        errors=receipt['restoration_errors']
        def restore(label,call):
            try: call()
            except BaseException: errors.append({'stage':label,'traceback':traceback.format_exc()})
        restore('cancel_alarm',lambda:signal.setitimer(signal.ITIMER_REAL,0))
        def check_custody(label,call):
            try: call(); receipt['custody_checks'][label]=True
            except BaseException:
                receipt['custody_checks'][label]=False
                receipt['custody_errors'].append({'stage':label,'traceback':traceback.format_exc()})
        if original is not None:
            check_custody('original_functional_state_and_inputs',lambda:modules['compare'].compare(
                helpers._cpu_tree((theta,phis,S,yS,R,yR,image)),original,exact=True))
        if before is not None:
            check_custody('original_native_custody',lambda:custody.unchanged(before,family,data['features'],data['edge_index'],torch.device('cuda:0')))
        if common_path is not None:
            check_custody('original_common_file',lambda:require(ordinary.sha(common_path) == h16.COMMON['sha256'],'Original common image changed'))
        if rng is not None: restore('caller_numeric_RNG',lambda:modules['ordinary'].restore_rng(rng,'cuda:0'))
        if backend is not None: restore('backend',lambda:modules['process'].backend_restore(backend))
        if old_threads is not None: restore('intra_op_threads',lambda:torch.set_num_threads(old_threads))
        receipt['interop_restoration']='Fresh-child one-time initialization; ends with child termination.'
        sys.path[:]=old_path
        if old_env[0]: os.environ['CUBLAS_WORKSPACE_CONFIG']=old_env[1]
        else: os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
        restore('signal_timer',lambda:(signal.signal(signal.SIGALRM,old_signal),signal.setitimer(signal.ITIMER_REAL,*old_timer)))
        restore('terminal_resources',limits)
        if errors or receipt['custody_errors'] or receipt['callback_finalization_errors']: receipt['status']='FAIL_CMCL_H16_FIT'
        save(); (output/'RESULT.json').chmod(0o444)
        if (output/'TRACE.jsonl').is_file(): (output/'TRACE.jsonl').chmod(0o444)
    return 0 if receipt['status'] == 'CMCL_H16_COMPLETE_A_CLOSED_RESOURCE_ONLY' else 1


def poll(args,root,owned,scope):
    base=root/OUTPUT
    if (base/'TERMINAL.json').exists():
        print(json.dumps({'status':'TERMINAL','path':str(base/'TERMINAL.json'),'automatic_retry':False})); return 0
    h=owned.read(base/'LAUNCH_HANDLE.json')
    try: owned.physical(h['supervisor']['pid'],h['supervisor']['argv'],start=h['supervisor']['starttime_ticks'])
    except BaseException as error:
        print(json.dumps({'status':'OBSERVATION_FAILURE_NO_RELAUNCH','error':str(error),'automatic_retry':False})); return 1
    print(json.dumps({'status':'RUNNING','automatic_retry':False})); return 0


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--execute-authorized',action='store_true'); parser.add_argument('--mode',choices=('launch','supervise','fit','poll'))
    for name in ('source-root','scope','scope-sha256','output'): parser.add_argument('--'+name)
    args=parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({'status':'DISABLED_CMCL_H16_OWNED_FIT','SOURCE_RELEASED':False})); return 0
    require(SOURCE_RELEASED is False and all((args.mode,args.source_root,args.scope,args.scope_sha256)),'Explicit new root fit scope required')
    root,owned,scope=authenticate(args)
    return {'launch':launch,'supervise':supervise,'fit':fit,'poll':poll}[args.mode](args,root,owned,scope)


if __name__ == '__main__': raise SystemExit(main())
