"""Two seed1 native input traversals, with zero model forwards or updates."""
import argparse,copy,gc,hashlib,importlib.util,json,os,resource,socket,subprocess,sys,time,traceback
from contextlib import nullcontext
from pathlib import Path

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.output).resolve();Q=Path(__file__).resolve().parent;E=Q.parents[1]
    assert Q.is_relative_to(root) and out.is_relative_to(root) and not out.exists();out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(600,610))
    started=time.perf_counter();report=dict(schema='BUDDY-seed1-native-input-parity-v2',status='running',stage='before_import',seed=1,device='cpu',CPU_threads=2,
        native_constructions_attempted=0,native_constructions=0,sampler_calls=0,TRAIN_hydration_calls=0,VALID_role_hydration_calls=0,
        model_forward_attempts=0,model_forwards=0,optimizer_attempts=0,optimizer_steps=0,gradient_collections=0,
        query_traversal_not_trained_epoch=True,TEST_accessed=False,scientific_fits=0,memory_events_before=Path('/sys/fs/cgroup/memory.events').read_text())
    def write():
        u=resource.getrusage(resource.RUSAGE_SELF);report.update(elapsed_seconds=time.perf_counter()-started,process_user_seconds=u.ru_utime,process_system_seconds=u.ru_stime,process_peak_RSS_KiB=u.ru_maxrss)
        temp=out/'RESULT.json.part';temp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');temp.replace(out/'RESULT.json')
    write();adapter=None
    try:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        provider=load(E/'buddy_native_provider_v1/provider.py','_parity_provider');callback=load(E/'buddy_native_provider_v1/callback.py','_parity_callback')
        job=json.loads((Q/'JOB.json').read_text());assert job['seed']==1 and job['recipe']['_repository_root']==str(root)
        policy=load(E/'link_policy_v1/policy.py','_parity_policy').create_policy(job,root)
        native_path=E/'link_native_integration_v1/session.py';native=provider.load(native_path)
        base_job=copy.deepcopy(job);base_job.update(callback_module=str(native_path),callback_sha256=provider.receipt(native_path)['sha256'],method_settings={})
        report['job']=job;report['source_manifest']=provider.receipt(E/'buddy_native_provider_v1/MANIFEST.json')
        seed_path=E/'runtime_v4/common.py'
        if provider.receipt(seed_path)['sha256']!='f9cd69b52386742fdd06b007829d5e94439f278a975c4be6d45621bb2e5b915d':raise AssertionError('Production seed helper changed')
        common=load(seed_path,'_parity_production_common');report['production_seed_helper']=provider.receipt(seed_path)
        originals=[];input_records=[];tensor_reads=[];original_load=torch.load
        report.update(initial_records=originals,input_records=input_records,tensor_file_reads=tensor_reads,constructor_rng_receipts=[],production_seed_calls=0)
        def observe_load(*args,**kwargs):
            target=args[0] if args else kwargs.get('f');label=str(Path(target).resolve()) if isinstance(target,(str,Path)) else type(target).__name__
            if 'CONTEXT' in label or 'context_attempt01' in label:raise AssertionError('Full seed0 tensor oracle read attempted')
            tensor_reads.append(label);return original_load(*args,**kwargs)
        torch.load=observe_load
        for index in range(2):
            common.seed_all(torch,job['seed'],device='cpu');report['production_seed_calls']+=1
            report['constructor_rng_receipts'].append({'construction':index,'requested_seed':job['seed'],'before_constructor_sha256':provider.rng(torch)})
            report['stage']='construct_original' if index==0 else 'construct_bridge';report['native_constructions_attempted']+=1;write()
            adapter=native.create_adapter(torch,copy.deepcopy(base_job),policy,root) if index==0 else callback.create_adapter(torch,copy.deepcopy(job),policy,root)
            report['native_constructions']+=1;n=adapter.session.native
            report['constructor_rng_receipts'][-1]['after_constructor_sha256']=provider.rng(torch);write()
            if any(p.device.type!='cpu' for p in n.model.parameters()):raise AssertionError('Non-CPU source model')
            def forbid_forward(module,inputs):report['model_forward_attempts']+=1;write();raise AssertionError('No native forwards authorized')
            hook=n.model.register_forward_pre_hook(forbid_forward)
            def forbid_step(*args,**kwargs):report['optimizer_attempts']+=1;write();raise AssertionError('No optimizer steps authorized')
            n.optimizer.step=forbid_step
            initial={'seed':n.seed,'source_parameters':{name:provider.tensor(p) for name,p in n.source_parameter_references.items()},
                'wrapper_parameters':{name:provider.tensor(p) for name,p in n.core.named_parameters()},'negative':provider.tensor(n.train_negative),
                'static':{name:provider.tensor(x) for name,x in [('features',n.features),('degrees',n.degrees),('cards',n.cards),('source_positive_queries',n.positive)]},
                'hash_tables':{str(hop):{name:provider.tensor(x) for name,x in table.items()} for hop,table in n.hash_tables.items()},
                'post_constructor_rng':provider.rng(torch),'layout':copy.deepcopy(adapter.session.layout)}
            if initial['layout']['optimizer_maps_per_epoch']!=1585 or initial['layout']['included_partial_tail']!=794 or len(n.links)!=1622810:raise AssertionError('Native query count/tail changed')
            originals.append(initial);write();observations=[];chain=provider.digest([])
            report['stage']='original_query_traversal' if index==0 else 'bridge_query_traversal';write()
            with nullcontext(n) if index==0 else adapter.session.operation('QUERY_TRAIN'):
                for batch in range(1,1586):
                    before=provider.rng(torch);ids=n._next_batch();after=provider.rng(torch);report['sampler_calls']+=1
                    row={'epoch':int(n.epoch),'batch_cursor':int(n.batch_cursor),'source_sampler_ids':provider.tensor(ids),'source_links':provider.tensor(n.links[ids]),'source_labels':provider.tensor(n.labels[ids]),'records':int(ids.numel()),'native_rng_before':before,'native_rng_after':after}
                    chain=provider.digest([chain,row])
                    if batch in (1,2,1585):
                        pre=provider.rng(torch);values=n._hydrate(n.links[ids]);post=provider.rng(torch);report['TRAIN_hydration_calls']+=1
                        row.update(inputs=[provider.tensor(x) for x in values],hydration_rng_before=pre,hydration_rng_after=post);observations.append(row);write();del values
                    if batch%500==0:write()
            if observations[-1]['records']!=794 or n.updates!=0:raise AssertionError('Tail omitted or query traversal counted as training')
            report['stage']='original_VALID_inputs' if index==0 else 'bridge_VALID_inputs';write();valid=[]
            with nullcontext(n) if index==0 else adapter.session.operation('QUERY_VALID'):
                for role in ('edge','edge_neg'):
                    pre=provider.rng(torch);values=n._hydrate(n.valid[role]);post=provider.rng(torch);report['VALID_role_hydration_calls']+=1
                    valid.append({'role':role,'source_links':provider.tensor(n.valid[role]),'inputs':[provider.tensor(x) for x in values],'hydration_rng_before':pre,'hydration_rng_after':post});del values;write()
            record={'query_chain_sha256':chain,'TRAIN':observations,'VALID':valid,'post_traversal_rng':provider.rng(torch)};input_records.append(record);write()
            if index:
                ledger=adapter.session.ledger
                if (ledger['sampler_calls'],ledger['TRAIN_hydration_calls'],ledger['VALID_hydration_calls'])!=(1585,3,2) or ledger['query_chain_sha256']!=chain or ledger['TRAIN_observations']!=observations or ledger['VALID_observations']!=valid:raise AssertionError('Provider ledger differs from independently observed native returned inputs')
                try:adapter.state()
                except RuntimeError:report['query_traversal_fit_checkpoint_rejected']=True
                else:raise AssertionError('Query-only traversal serialized as training')
                hook.remove();hook=None
                try:n.model()
                except RuntimeError:report['query_traversal_model_forward_rejected']=True
                else:raise AssertionError('Query-only source forward allowed')
                report['provider_metadata']=adapter.metadata()['native_BUDDY_provider']
            if hook is not None:hook.remove()
            adapter.close();adapter=None;del n,ids,initial,record;gc.collect()
        torch.load=original_load
        def component_equality(left,right):
            if isinstance(left,dict) and isinstance(right,dict):return {k:component_equality(left.get(k),right.get(k)) for k in sorted(set(left)|set(right))}
            if isinstance(left,list) and isinstance(right,list):return {'length_equal':len(left)==len(right),'items':[component_equality(x,y) for x,y in zip(left,right)]}
            return left==right
        report['component_comparisons']={'initial':component_equality(originals[0],originals[1]),'inputs':component_equality(input_records[0],input_records[1])}
        report['initial_state_equal']=originals[0]==originals[1];report['all_input_records_equal']=input_records[0]==input_records[1]
        report['constructor_pre_rng_equal']=report['constructor_rng_receipts'][0]['before_constructor_sha256']==report['constructor_rng_receipts'][1]['before_constructor_sha256']
        write()
        if not all(report[k] for k in ('initial_state_equal','all_input_records_equal','constructor_pre_rng_equal')):raise AssertionError('Original seed1 source and bridge inputs/initialization/RNG differ')
        if input_records[1]['TRAIN'][0]['source_sampler_ids']['sha256']=='b6fe6c89275923a0d87c311078260229fa4da15683b48ee7cd91357240938d0f':raise AssertionError('Seed1 first native batch equals seed0 receipt unexpectedly')
        if report['production_seed_calls']!=2:raise AssertionError('Exact runner seeding protocol omitted')
        if (report['native_constructions'],report['sampler_calls'],report['TRAIN_hydration_calls'],report['VALID_role_hydration_calls'],report['model_forward_attempts'],report['optimizer_attempts'])!=(2,3170,6,4,0,0):raise AssertionError('Finite work bound changed')
        report.update(status='passed',stage='complete',original_and_bridge_initial_state_exact=True,all_native_query_sequence_and_selected_inputs_exact=True,seed1_sampler_distinct_from_seed0=True,static_seed1_inputs_match_compact_authority=True,full_tensor_oracle_reads=0,tensor_file_reads=tensor_reads,source_seed1_initialization=originals[0],input_parity_records=input_records[0])
    except BaseException as exc:report.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(limit=18)})
    finally:
        if adapter is not None:
            try:adapter.close()
            except Exception as exc:report['close_error']=str(exc)
        report['memory_events_after']=Path('/sys/fs/cgroup/memory.events').read_text();write()
    print(json.dumps({k:report[k] for k in ('status','stage','native_constructions','sampler_calls','TRAIN_hydration_calls','VALID_role_hydration_calls','model_forwards','optimizer_steps','elapsed_seconds')}))
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
