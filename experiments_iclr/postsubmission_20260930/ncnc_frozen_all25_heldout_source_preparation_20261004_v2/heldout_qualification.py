"""One separately released fabricated wrapper QA; never a heldout numerical invocation."""
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from heldout_gate import require, manifest, pin_value, sha, PROFILE, POLICY, AUDIT_SOURCE, AUDIT_SOURCE_SHA
from heldout_accounting import ARMS, atomic_json, descriptor, new_cells, new_work, progress, supervisor_closure

HERE = Path(__file__).resolve().parent
STAGE = 'fabricated_heldout_wrapper_qualification'


def admission(path, expected_sha):
    path = Path(path); require(path.is_absolute() and path.resolve() == path and sha(path) == expected_sha, 'Qualification release bytes differ')
    release = json.loads(path.read_text())
    require(release['schema'] == 'ncnc-heldout-wrapper-fabricated-root-release-v2' and release['execution_enabled'] is True and release['root_authorization_reference'], 'Separate fabricated-only root release required')
    require(release['fabricated_inputs_only'] is True and release['TEST_access_authorized'] is False and release['study_checkpoint_access_authorized'] is False, 'Fabricated-only scope required')
    require(not any(k in release for k in ('family_lock','unit_custody','data_authority','dataset_root','TEST_data_authority')), 'Study inputs forbidden in qualification release')
    output = Path(release['output_directory'])
    invocation = dict(stage=STAGE, output_directory=str(output), cuda_visible_devices=release['cuda_visible_devices'])
    require(release['authorized_stages'] == [STAGE] and release['authorized_invocations'] == [invocation] and os.environ.get('CUDA_VISIBLE_DEVICES') == release['cuda_visible_devices'], 'Exact qualification invocation required')
    require(output.is_absolute() and output.resolve() == output and not output.exists() and not any((p/'MANIFEST.json').exists() for p in (output,*output.parents)), 'Fresh qualification output outside sealed packets required')
    manifest(HERE, release['heldout_source_manifest_sha256']); manifest(AUDIT_SOURCE, AUDIT_SOURCE_SHA)
    sys.path.insert(0,str(AUDIT_SOURCE))
    import replay_gate
    paths = {k:Path(release[k]) for k in ('driver_root','design_root','prototype_root','resource_root')}
    for key, expected in (('driver_root',replay_gate.DRIVER_SHA),('design_root',replay_gate.DESIGN_SHA),('prototype_root',replay_gate.PROTOTYPE_SHA),('resource_root',replay_gate.RESOURCE_SHA)):
        require(release[key.replace('_root','_manifest_sha256')] == expected, 'Original source identity differs'); manifest(paths[key],expected)
    runtime = pin_value(release['runtime_authority'],decode=True)
    require(release['runtime_authority']['sha256'] == replay_gate.RUNTIME_SHA and runtime['ordinary_host_execution'] is True, 'Original ordinary runtime required')
    review = pin_value(release['independent_source_review'],decode=True)
    require(review['status'] == 'PASS' and review['heldout_source_manifest_sha256'] == release['heldout_source_manifest_sha256'] and review['policy'] == POLICY and review['qualification_invocation'] == invocation, 'Successor source review and exact QA invocation required before QA')
    for pin in (release['supervisor_source'],release['dispatcher_source']): pin_value(pin)
    authority = json.loads((HERE/'FABRICATED_AUTHORITY.json').read_text())
    require(authority['fabricated_inputs_only'] is True and authority['files'] == {} and authority['test_file_opened'] is False, 'No study data authority admitted')
    gpu = subprocess.run(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15).stdout.strip().split(',')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    require(gpu[0].strip() == release['cuda_visible_devices'] and int(gpu[1]) >= release['minimum_GPU_free_MiB'] and available >= release['minimum_host_MemAvailable_bytes'], 'Qualification resource dispatch recheck failed')
    return dict(release=release, release_path=path, release_sha256=expected_sha, output=output, paths=paths,
                runtime=runtime, authority=authority, plan=json.loads((paths['design_root']/'PILOT_PLAN.json').read_text()),
                identity=dict(family_id='FABRICATED_HELDOUT_WRAPPER_QA_ONLY',synthetic_only=True), synthetic_only=True)


def load_module(pin, name):
    path = pin_value(pin)
    spec = importlib.util.spec_from_file_location(name,path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module



def begin_case(context, name):
    result=context['qualification_result']; next(c for c in result['cases'] if c['case']==name)['status']='IN_PROGRESS'
    atomic_json(context['output']/'QUALIFICATION.json',result)


def record_case(context, row):
    result=context['qualification_result']; next(c for c in result['cases'] if c['case']==row['case']).update(row)
    atomic_json(context['output']/'QUALIFICATION.json',result)

def rejected(function, fragment=None):
    try: function()
    except Exception as error:
        require(fragment is None or fragment in str(error), 'Wrong rejection endpoint: '+str(error))
        return dict(exception_type=type(error).__name__,condition=str(error))
    raise RuntimeError('Fabricated invalid case was accepted')


def assert_failed(result):
    require(result['status'] == 'FAILED_HELDOUT_CONFIRMATION' and result['summary'] is None and len(result['cells']) == 25 and {(c['arm'],c['base_seed']) for c in result['cells']} == {(a,s) for a in ARMS for s in range(5)} and all(c['TEST_hits50'] is None for c in result['cells']), 'Failed public all25 endpoint differs')


def loader_cases(context, torch, data_api, device):
    output=context['output']
    import numpy as np
    import heldout_run
    positive = np.array([[0,1],[4,5],[1,3]],dtype=np.int64)
    negative = np.tile(np.array([[3,0]],dtype=np.int64),(100000,1))
    stored = dict(edge=positive,edge_neg=negative,weight=np.ones(3,dtype=np.int64),year=np.full(3,2019,dtype=np.int64))
    cases = []
    for name in ('loader_complete_same_open_CPU','loader_wrong_dtype','loader_wrong_year','loader_wrong_file_hash','loader_wrong_typed_order','loader_incomplete_negative_pool'):
        begin_case(context,name)
        value = copy.deepcopy(stored)
        if name == 'loader_wrong_dtype': value['edge'] = value['edge'].astype(np.int32)
        if name == 'loader_wrong_year': value['year'][1] = 2018
        if name == 'loader_incomplete_negative_pool': value['edge_neg'] = value['edge_neg'][:-1]
        path = output/(name+'.pt'); torch.save(value,path)
        authority = dict(official_TEST_file=descriptor(path),typed_query_digests=dict(positive_sha256=data_api.tensor_sha(torch.from_numpy(positive)),negative_sha256=data_api.tensor_sha(torch.from_numpy(negative)),pair_order_sha256=data_api.tensor_sha(torch.from_numpy(np.concatenate([positive,negative])))))
        if name == 'loader_wrong_file_hash': authority['official_TEST_file']['sha256'] = '0'*64
        if name == 'loader_wrong_typed_order': authority['typed_query_digests']['positive_sha256'] = '0'*64
        ctx = dict(TEST_authority=authority); loads = []
        original_load = torch.load
        def observed_load(stream, **kw):
            require(stream.name == str(path) and stream.tell() == 0 and ctx.get('TEST_file_opened') is True and kw == dict(map_location='cpu',weights_only=True), 'Loader did not reuse authenticated same-open stream safely')
            loads.append(dict(stream=stream.name,options=kw)); return original_load(stream,**kw)
        with patch.object(torch,'load',side_effect=observed_load):
            if name == 'loader_complete_same_open_CPU':
                p,n,receipt = heldout_run.load_TEST(ctx,torch,data_api,device)
                require(torch.equal(p.cpu(),torch.from_numpy(positive)) and torch.equal(n.cpu(),torch.from_numpy(negative)) and receipt['positive_rows'] == 3 and receipt['negative_rows'] == 100000 and len(loads) == 1, 'Complete fabricated pool/order differs')
                evidence = dict(receipt=receipt)
            else:
                fragment = {'loader_wrong_year':'year/record','loader_wrong_file_hash':'bytes changed','loader_wrong_typed_order':'rows/order','loader_incomplete_negative_pool':'Complete official'}.get(name)
                evidence = rejected(lambda:heldout_run.load_TEST(ctx,torch,data_api,device),fragment)
                if name == 'loader_wrong_file_hash': require(not loads,'Wrong file hash reached deserialization')
        cases.append(dict(case=name,status='PASS',load_count=len(loads),fabricated_file=descriptor(path),**evidence)); record_case(context,cases[-1])
    return cases, torch.from_numpy(positive).to(device),torch.from_numpy(negative).to(device)


def sentinel_factories(torch, expected_graph, positive, negative, model_api):
    """Tiny models observe exact original scorer adapter paths; original model kernels are already qualified."""
    observed = []
    def graph_keys(graph):
        if hasattr(graph,'coo'): row,col,_ = graph.coo()
        else: row,col = graph.row,graph.col
        return set(zip(row.detach().cpu().tolist(),col.detach().cpu().tolist()))
    class Encoder(torch.nn.Module):
        def __init__(self, stamp):
            super().__init__(); self.stamp=torch.nn.Parameter(torch.tensor(float(stamp))); self.last_graph=None
        def forward(self,x,graph):
            require(graph_keys(graph) == expected_graph,'Encoder topology changed / targets removed / TEST added')
            self.last_graph=graph; return x
    class Decoder(torch.nn.Module):
        def __init__(self,stamp,encoder,factor):
            super().__init__(); self.stamp=torch.nn.Parameter(torch.tensor(float(stamp))); self.encoder_ref=lambda:encoder; self.factor=factor; self.query_count=0
        def forward(self,h,graph,queries,mode=None):
            require(graph is self.encoder_ref().last_graph and graph_keys(graph) == expected_graph,'Encoder/completion topology differs')
            q=queries if self.factor else queries.T
            expected=positive if self.query_count==0 else negative
            require(self.query_count in (0,1) and torch.equal(q,expected),'Original positive/negative query order or tail differs')
            self.query_count+=1
            observed.append(dict(stamp=float(self.stamp.detach().cpu()),factor=self.factor,mode=mode,pool='positive' if self.query_count==1 else 'negative',rows=len(q)))
            score=q[:,0].float()*6+q[:,1].float()+self.stamp
            return score[:,None]+torch.arange(4,device=q.device,dtype=torch.float32)[None,:] if self.factor else score[:,None]
    class Twin(torch.nn.Module):
        def __init__(self,stamp):
            super().__init__(); self.encoder=Encoder(stamp); self.decoder=Decoder(stamp,self.encoder,True)
    def native(mods,seed,width,device):
        model_api.seed_all(seed); e=Encoder(seed+width/1000).to(device); d=Decoder(seed+width/1000,e,False).to(device)
        return (e,d),torch.optim.Adam([*e.parameters(),*d.parameters()])
    def factor(mods,seed,device):
        model_api.seed_all(seed); m=Twin(seed+100).to(device); return m,torch.optim.Adam(m.parameters())
    return native,factor,observed


def wrapper_cases(context,api,mods,torch,device,positive,negative):
    import heldout_run, replay_gate, replay_numeric, replay_contract
    configure_actual=replay_numeric.configure_environment
    data_api, model_api, state_api, evaluate=(api[k] for k in ('pilot_data','pilot_model','pilot_state','pilot_evaluate'))
    train=torch.tensor([[0,1],[1,2],[0,1],[2,3]],dtype=torch.int64,device=device)
    valid=torch.tensor([[3,4],[1,3]],dtype=torch.int64,device=device)
    original=dict(x=torch.arange(6*128,device=device,dtype=torch.float32).reshape(6,128)/1000,pairs=train,raw_edge_index=train.T,valid_positive=valid,valid_negative=negative)
    expected=set(map(tuple,torch.cat([train,valid]).cpu().tolist())); expected |= {(b,a) for a,b in list(expected)}
    native,factor,observed=sentinel_factories(torch,expected,positive,negative,model_api)
    family=dict(units={},cells=[]); snapshots={}
    for arm in ARMS:
        for seed in range(5):
            unit=arm; row=dict(arm=arm,base_seed=seed,unit=unit,selection=dict(fabricated=True),checkpoint=dict(path=arm+'_'+str(seed),fabricated_in_memory=True))
            family['cells'].append(row); family['units'][(unit,seed)]=dict(root=Path('/FABRICATED_IN_MEMORY_NO_FILE'),journal=dict(state_file=dict(path='journal')),complete={})
            bank=[]
            for member in range(4 if arm==ARMS[1] else 1):
                m,o=native(mods,seed+5*member if arm==ARMS[1] else seed,70 if arm==ARMS[4] else 64,device) if arm in (ARMS[0],ARMS[1],ARMS[4]) else factor(mods,seed,device)
                bank.append(state_api.snapshot(m,o)); del m,o
            snapshots[row['checkpoint']['path']]=bank
    tensor_path=context['output']/'FABRICATED_INPUTS.pt'; torch.save(dict(original=original,positive=positive,negative=negative),tensor_path)
    context['synthetic_tensor_custody']=descriptor(tensor_path)
    actual_adapter=heldout_run.adapt_TEST_queries
    def fabricated_adapter(original,p,n,t,d): return actual_adapter(original,p,n,t,d,expected_records=(4,2))
    def fake_scores(instance,data,mods,mode=None):
        model_api.train_flag(instance,False)
        p=torch.tensor([0.,2.,0.],dtype=torch.float32); n=torch.ones(100000,dtype=torch.float32)
        return p,n,dict(positive_queries=3,negative_queries=100000,query_batches=[1,1],encoder_calls=1,all_query_rows_complete=True,serving_pool='mean_raw_logits',graph='complete_TRAIN_only',score_digests=dict(positive=data_api.tensor_sha(p),negative=data_api.tensor_sha(n)),wall_seconds=0.,fabricated_stub=True)
    cases=[]
    names=('baseline_exact_adapter_all25','failure_pre_TEST_admission','failure_scorer_entry','failure_pooling','failure_metric_entry','failure_metric_return_guard','failure_final_custody')
    for name in names:
        begin_case(context,name)
        case_metric=evaluate.evaluator(context)
        ctx={**context,'output':context['output']/name,'heldout_release':dict(family_lock=dict(fabricated=True),v4_audit_result=dict(fabricated=True)),'heldout_release_sha256':context['release_sha256'],'heldout_source_sha256':context['release']['heldout_source_manifest_sha256']}
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluate,'evaluator',return_value=case_metric))
            stack.enter_context(patch.object(replay_numeric,'configure_environment',side_effect=lambda:configure_actual(already_imported=True)))
            stack.enter_context(patch.object(model_api,'runtime',return_value=(device,None)))
            stack.enter_context(patch.object(model_api,'modules',return_value=mods))
            stack.enter_context(patch.object(model_api,'make_native',side_effect=native)); stack.enter_context(patch.object(model_api,'make_factorized',side_effect=factor))
            stack.enter_context(patch.object(data_api,'load_data',return_value=original))
            def fabricated_TEST(ctx,t,d,dev):
                ctx['TEST_file_opened']=False; return positive,negative,dict(fabricated_inputs_only=True,positive_rows=3,negative_rows=100000)
            stack.enter_context(patch.object(heldout_run,'load_TEST',side_effect=fabricated_TEST))
            stack.enter_context(patch.object(heldout_run,'adapt_TEST_queries',side_effect=fabricated_adapter))
            stack.enter_context(patch.object(replay_numeric,'trusted_load',side_effect=lambda t,r,p:dict(snapshots=snapshots.get(p['path']))))
            stack.enter_context(patch.object(replay_contract,'validate_journal_payload',return_value={}))
            stack.enter_context(patch.object(replay_contract,'validate_selected_payload',side_effect=lambda payload,*a:(payload['snapshots'],None)))
            stack.enter_context(patch.object(heldout_run,'final_custody',side_effect=lambda c,f:(c['loaded_tensor_guard']() or dict(fabricated_input_guard='PASS'))))
            if name=='failure_pre_TEST_admission': stack.enter_context(patch.object(replay_gate,'runtime_and_data_custody',side_effect=RuntimeError('FABRICATED_PRE_TEST_ADMISSION')))
            if name!='baseline_exact_adapter_all25': stack.enter_context(patch.object(evaluate,'score_valid',side_effect=fake_scores))
            if name=='failure_scorer_entry':
                def fail_entry(*a,**kw):
                    status=json.loads((ctx['output']/'STATUS.json').read_text()); require(status['work']['entered_original_scorer']==1 and status['phase']=='original_scorer_entry','Entry event not persisted before scorer'); raise RuntimeError('FABRICATED_SCORER_ENTRY')
                stack.enter_context(patch.object(evaluate,'score_valid',side_effect=fail_entry))
            if name=='failure_pooling': stack.enter_context(patch.object(evaluate,'mean_native_scores',side_effect=RuntimeError('FABRICATED_POOLING')))
            if name=='failure_metric_entry': stack.enter_context(patch.object(case_metric,'eval',side_effect=RuntimeError('FABRICATED_METRIC_ENTRY')))
            if name=='failure_metric_return_guard': stack.enter_context(patch.object(case_metric,'eval',return_value={'hits@50':float('nan')}))
            if name=='failure_final_custody': stack.enter_context(patch.object(heldout_run,'final_custody',side_effect=RuntimeError('FABRICATED_FINAL_CUSTODY')))
            code=heldout_run.execute(ctx,family)
        result=json.loads((ctx['output']/'HELDOUT_RESULT.json').read_text()); work=result['work']
        if name=='baseline_exact_adapter_all25':
            require(code==0 and work['entered_original_scorer']==work['returned']==40 and work['official_metric_attempted']==work['official_metric_returned']==work['official_metric_validated']==25 and all(c['TEST_hits50']==1/3 for c in result['cells']),'Wrapper baseline/actual-positive-length normalization failed')
            expected_order=[(seed+5*m if arm==ARMS[1] else seed)+(70 if arm==ARMS[4] else 64)/1000 if arm in (ARMS[0],ARMS[1],ARMS[4]) else seed+100 for arm in ARMS for seed in range(5) for m in range(4 if arm==ARMS[1] else 1)]
            require(len(observed)==80 and all(abs(observed[2*i]['stamp']-stamp)<1e-5 and observed[2*i]['pool']=='positive' and observed[2*i+1]['pool']=='negative' for i,stamp in enumerate(expected_order)),'Selected member/order/query pool route differs')
            factor_modes=[r['mode'] for r in observed if r['factor']]; require(factor_modes==['private']*10+['pooled_after_clamp']*10,'Factorized completion mode route differs')
            details=json.loads((ctx['output']/'PRIVATE_CELL_RECEIPTS.json').read_text())
            i4=next(c for c in details['cells'] if (c['arm'],c['base_seed'])==(ARMS[1],0))
            raw=torch.load(i4['raw_served_score_evidence']['path'],map_location='cpu',weights_only=True)['values']
            expected_scores=torch.stack([positive.cpu()[:,0].float()*6+positive.cpu()[:,1].float()+torch.tensor(m*5+.064) for m in range(4)],dim=1).mean(1)
            require(torch.equal(raw['positive'],expected_scores),'I4 original member-order mean raw logits differs')
        else:
            require(code!=0,'Fault succeeded'); assert_failed(result)
            expected_work=json.loads((HERE/'FABRICATED_QUALIFICATION_PLAN.json').read_text())['failure_work'][name]
            require([work[k] for k in ('attempted','entered_original_scorer','returned','completed_validated')]==expected_work,'Literal failed work tuple differs')
            expected_phase={'failure_pooling':'pooling_entry','failure_metric_entry':'official_evaluator_entry','failure_metric_return_guard':'official_metric_internal_return_guard','failure_final_custody':'final_custody_before_publication','failure_scorer_entry':'original_scorer_entry','failure_pre_TEST_admission':'original_runtime_custody'}[name]
            require(result['failed_phase']==expected_phase,'Cell failure phase differs')
            if name in ('failure_pooling','failure_metric_entry','failure_metric_return_guard'):
                cell=next(c for c in result['cells'] if c['arm']==(ARMS[1] if name=='failure_pooling' else ARMS[0]) and c['base_seed']==0); require(cell['status']=='FAILED','Failed metric/pooling cell reported not attempted')
            if name=='failure_metric_entry': require((work['official_metric_attempted'],work['official_metric_returned'],work['official_metric_validated'])==(1,0,0),'Metric attempt/return conflated')
            if name=='failure_metric_return_guard': require((work['official_metric_attempted'],work['official_metric_returned'],work['official_metric_validated'])==(1,1,0) and any('metric.pt' in r['path'] for r in result['private_evidence']),'Raw invalid returned metric not retained')
            if name=='failure_final_custody':
                private=json.loads((ctx['output']/'PRIVATE_CELL_RECEIPTS.json').read_text()); require(all(c['TEST_hits50']==1/3 for c in private['cells']) and len([r for r in result['private_evidence'] if 'returned.pt' in r['path']])==40,'Completed numerical evidence lost on final failure')
        cases.append(dict(case=name,status='PASS',child_status=result['status'],work=work,original_scorer_calls=40 if name=='baseline_exact_adapter_all25' else 0,stub_scorer_entries=0 if name=='baseline_exact_adapter_all25' else work['entered_original_scorer'],result=descriptor(ctx['output']/'HELDOUT_RESULT.json'))); record_case(context,cases[-1])
    return cases


def supervisor_cases(context):
    supervisor=load_module(context['release']['supervisor_source'],'fabricated_heldout_supervisor')
    dispatcher=load_module(context['release']['dispatcher_source'],'fabricated_heldout_dispatcher')
    cases=[]
    for name in ('failure_child_import_before_progress','failure_owned_bounded_stop','failure_physical_after_child_success','single_use_claim_and_dispatch_rejection'):
        begin_case(context,name)
        root=context['output']/name; root.mkdir(mode=0o700); out=root/'child_output'
        with patch.object(dispatcher,'HERE',root): dispatcher.require_fresh()
        claim=root/'ONE_TIME_TEST_CLAIM.json'; supervisor.claim_once(claim,dict(fabricated_inputs_only=True,no_retry=True))
        rejected(lambda:supervisor.claim_once(claim,{}));
        with patch.object(dispatcher,'HERE',root): rejected(dispatcher.require_fresh,'no retry')
        if name=='single_use_claim_and_dispatch_rejection': cases.append(dict(case=name,status='PASS',claim=descriptor(claim))); record_case(context,cases[-1]); continue
        fake_release=dict(identity=context['identity'],family_lock=dict(fabricated=True),v4_audit_result=dict(fabricated=True),policy=POLICY)
        if name=='failure_physical_after_child_success':
            out.mkdir(mode=0o700); cells=new_cells()
            for c in cells: c.update(status='PASS',phase='cell_complete',TEST_hits50=.5,official_metric_attempted=True,official_metric_returned=True,official_metric_validated=True)
            work=new_work()
            for k in ('attempted','entered_original_scorer','returned','completed_validated'): work[k]=40
            for k in ('cells_completed','official_metric_calls','official_metric_attempted','official_metric_returned','official_metric_validated','metric_helper_attempted','metric_helper_returned'): work[k]=25
            logical=dict(schema='ncnc-frozen-all25-heldout-result-v2',identity=context['identity'],root_release_sha256=context['release_sha256'],heldout_source_manifest_sha256=context['release']['heldout_source_manifest_sha256'],status='ALL25_FROZEN_HELDOUT_CONFIRMATION_COMPLETE',cells=cells,work=work,summary=dict(fabricated_metric=.5),TEST_opened=False)
            atomic_json(out/'HELDOUT_RESULT.json',logical); before=descriptor(out/'HELDOUT_RESULT.json')
            physical=dict(status='PHYSICALLY_FAILED_OR_BOUNDED_STOP',exit_code=0,bounded_stop_reason=None,error=dict(type='FABRICATED_PHYSICAL_FAILURE'))
            receipt=supervisor_closure(out,fake_release,context['release_sha256'],context['release']['heldout_source_manifest_sha256'],physical,True)
            closure=json.loads((out/'HELDOUT_RESULT.json').read_text()); assert_failed(closure)
            private=descriptor(out/'PRIVATE_PRE_SUPERVISOR_HELDOUT_RESULT.json')
            require(private['bytes']==before['bytes'] and private['sha256']==before['sha256'] and closure['work']==work,'Physical failure lost exact raw completed child metrics/counts')
            cases.append(dict(case=name,status='PASS',physical=physical,logical_closure=receipt,preserved_exact_child_result=private)); record_case(context,cases[-1]); continue
        if name=='failure_owned_bounded_stop':
            out.mkdir(mode=0o700); cells=new_cells(); cells[0].update(status='ATTEMPTED',phase='original_scorer_entry'); work=new_work(); work['attempted']=work['entered_original_scorer']=1
            result=dict(identity=context['identity'],root_release_sha256=context['release_sha256'],heldout_source_manifest_sha256=context['release']['heldout_source_manifest_sha256'])
            progress(out,result,work,cells,'original_scorer_entry',dict(arm=ARMS[0],base_seed=0),dict(arm=ARMS[0],base_seed=0,member=0,status='SCORER_ENTRY_RECORDED',returned=False,validated=False))
            partial=out/'PRIVATE_NUMERICAL_EVIDENCE'; partial.mkdir(); (partial/'slot.partial.fabricated').write_bytes(b'FABRICATED_PARTIAL_EVIDENCE_ONLY')
            code='import time; time.sleep(60)'
        else: code='import FABRICATED_NONEXISTENT_MODULE_FOR_IMPORT_FAILURE'
        with (root/'child.stdout').open('xb') as stdout,(root/'child.stderr').open('xb') as stderr:
            child=subprocess.Popen([sys.executable,'-I','-S','-B','-c',code],stdout=stdout,stderr=stderr,start_new_session=True)
            handle=supervisor.identity(child.pid); events=[]
            try:
                wait_status,usage,exit_code,reason,peak=supervisor.monitor_owned_child(child,handle,time.monotonic(),events,max_seconds=.35,max_rss_bytes=32*1024**3)
            finally:
                if child.returncode is None: supervisor.kill_owned_session(handle,'FABRICATED_CLEANUP',events); child.wait(timeout=5)
        physical=dict(status='PHYSICALLY_FAILED_OR_BOUNDED_STOP',exit_code=exit_code,bounded_stop_reason=reason,error=None)
        receipt=supervisor_closure(out,fake_release,context['release_sha256'],context['release']['heldout_source_manifest_sha256'],physical,True)
        result=json.loads((out/'HELDOUT_RESULT.json').read_text()); assert_failed(result)
        if name=='failure_owned_bounded_stop': require(exit_code<0 and len(events)==1 and events[0]['own_pgid']==child.pid and result['work']['entered_original_scorer']==1 and 'lower_bounds' in result['count_semantics'] and result['in_flight_work']['actual_call_completion']=='UNKNOWN' and any('partial.fabricated' in r['path'] for r in result['private_evidence']),'Bounded owned stop/partial evidence/unknown work differs')
        else: require(exit_code!=0 and b'ModuleNotFoundError' in (root/'child.stderr').read_bytes() and all(result['work'][k] is None for k in ('attempted','entered_original_scorer','returned')),'Absent progress invented zero work')
        require(claim.exists(),'Claim removed after failure')
        cases.append(dict(case=name,status='PASS',physical=physical,signal_events=events,logical_closure=receipt,claim=descriptor(claim),wait4_reaped=True)); record_case(context,cases[-1])
    return cases


def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument('--root-release',required=True); parser.add_argument('--release-sha256',required=True); args=parser.parse_args()
    context=admission(args.root_release,args.release_sha256); output=context['output']; output.mkdir(parents=True,mode=0o700)
    plan=json.loads((HERE/'FABRICATED_QUALIFICATION_PLAN.json').read_text())
    cases=[dict(case=n,status='NOT_ATTEMPTED') for n in plan['cases']]
    result=dict(schema='ncnc-heldout-wrapper-fabricated-qualification-v2',status='IN_PROGRESS',source_entry=descriptor(Path(__file__).resolve()),heldout_source_manifest_sha256=context['release']['heldout_source_manifest_sha256'],root_release_sha256=context['release_sha256'],original_family_identity=context['release']['original_family_identity'],invocation=context['release']['authorized_invocations'][0],policy=POLICY,case_count=len(cases),runtime_authority=context['release']['runtime_authority'],cases=cases,fabricated_inputs_only=True,TEST_opened=False,study_data_or_selected_checkpoint_accessed=False,training_updates=0,automatic_retry=False)
    context['qualification_result']=result
    atomic_json(output/'QUALIFICATION.json',result)
    try:
        import replay_numeric,replay_gate
        replay_numeric.configure_environment(); api=replay_numeric.original_modules(context); api['pilot_common'].runtime_stdlib(context)
        device,_=api['pilot_model'].runtime(context)
        import torch
        mods=api['pilot_model'].modules(context); replay_numeric.set_profile(torch,PROFILE); replay_numeric.profile_receipt(torch,PROFILE)
        # The actual ordinary runtime is admitted once. Later cases use that same device/profile.
        metric=api['pilot_evaluate'].evaluator(context)
        begin_case(context,'official_ties_actual_positive_normalization_and_query_tail')
        negative=torch.arange(100.,50.,-1.,dtype=torch.float32); positive=torch.tensor([50.,51.,52.],dtype=torch.float32)
        require(api['pilot_evaluate'].strict_hits50(metric,positive,negative)==1/3,'Official tie/below/above50th geometry or positive-length normalization differs')
        iterator=mods['native_utils'].PermIterator(device,131073,131072,False); batches=list(iterator)
        require([len(b) for b in batches]==[131072,1] and torch.equal(torch.cat(batches),torch.arange(131073,device=device)),'Sequential evaluation tail/order differs')
        rows=[dict(case='official_ties_actual_positive_normalization_and_query_tail',status='PASS',positive_rows=3,expected_strict_hits50=1/3,query_tail_rows=1)]
        record_case(context,rows[0])
        loader,p,n=loader_cases(context,torch,api['pilot_data'],device); rows+=loader
        rows+=wrapper_cases(context,api,mods,torch,device,p,n)
        rows+=supervisor_cases(context)
        for row in rows: next(c for c in cases if c['case']==row['case']).update(row)
        require(len(rows)==len(cases) and all(c['status']=='PASS' for c in cases),'Every declared case must actually pass once')
        manifest(HERE,context['release']['heldout_source_manifest_sha256']); pin_value(context['release']['runtime_authority']); replay_gate.runtime_and_data_custody(context)
        for pin in (context['release']['supervisor_source'],context['release']['dispatcher_source']): pin_value(pin)
        require(sha(context['release_path'])==context['release_sha256'],'Qualification release changed')
        result.update(status='PASS',actual_original_scorer_calls=sum(c.get('original_scorer_calls',0) for c in cases),actual_stub_scorer_entries=sum(c.get('stub_scorer_entries',0) for c in cases),profile_receipt=replay_numeric.profile_receipt(torch,PROFILE),scope='Adapter/scorer topology/query/metric composition and all25 failure endpoints using sentinel models; original model kernels and study checkpoint selection are not requalified.')
    except Exception as error:
        for c in cases:
            if c['status']=='IN_PROGRESS': c.update(status='FAILED',error=dict(type=type(error).__name__,condition=str(error)))
        result.update(status='FAILED',error=dict(type=type(error).__name__,condition=str(error)))
    atomic_json(output/'QUALIFICATION.json',result)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
