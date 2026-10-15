"""One root-admitted CUDA producer: full native TRAIN epoch + two VALID calls.

Cost engineering only. No selector, checkpoint, scientific fit, queue or owner.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def require(ok,message):
    if not ok:raise ValueError(message)


def pin(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path,name):
    definition=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(definition)
    sys.modules[name]=module;definition.loader.exec_module(module);return module


def guard():
    require(socket.gethostname()=='anogena-2-0' and subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU],
        'Exact dedicated allocation required before project metadata')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')in ('0',GPU),'One authorized visible CUDA device required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--arm',choices=('structured','native_common'),required=True)
    parser.add_argument('--admission',required=True);parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();guard();started=time.perf_counter()
    root,output=Path(args.project_root).resolve(),Path(args.output).resolve()
    require(HERE.is_relative_to(root) and output.is_relative_to(root) and not output.exists() and
        not output.is_relative_to(HERE) and not HERE.is_relative_to(output),'Fresh repository-contained output separate from source required')
    require(Path(sys.prefix).resolve()==(root/'.venv').resolve(),'Original repository virtualenv required')
    admission_path=Path(args.admission).resolve()
    require(admission_path.is_relative_to(root) and pin(admission_path)['sha256']==args.admission_sha256,'Actual root admission differs')
    admission=json.loads(admission_path.read_text());manifest=pin(HERE/'MANIFEST.json')
    require(admission.get('schema')=='root-reviewed-molbace-pair-CUDA-cost-v1' and admission.get('approved')is True and
        admission.get('source_reviewed')is True and admission.get('resource_admission_approved')is True and
        admission.get('manifest_sha256')==manifest['sha256'] and admission.get('arm')==args.arm and
        admission.get('maximum_new_workers')==1 and admission.get('maximum_native_constructions')==1 and
        admission.get('maximum_TRAIN_epochs')==1 and admission.get('maximum_optimizer_transitions')==38 and
        admission.get('maximum_full_VALID_calls')==2 and admission.get('new_scientific_fits')==0 and
        admission.get('scientific_checkpoints')==0 and admission.get('TEST_access')is False and
        admission.get('allocation_GPU_UUID')==GPU,'Exact bounded root source/resource admission required')
    require(all(type(admission.get(k))is int and admission[k]>0 for k in
        ('maximum_wall_seconds','reserved_GPU_bytes','reserved_host_bytes')),'Finite outer watchdog and resource reservation required')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path=(HERE/row['path']).resolve();require(path.is_relative_to(HERE) and pin(path)['sha256']==row['sha256'],'Own source changed')
    contract=json.loads((HERE/'CONTRACT.json').read_text())
    for row in contract['source_pins']:
        path=Path(row['path']).resolve();require(path.is_relative_to(root) and pin(path)['sha256']==row['sha256'],'Frozen source/CPU receipt changed')
    producer=load(E/'research/molbace_pairwise_competence_v1/extension.py','_pair_cost_frozen_producer')
    producer.verify_sources(root)
    for row in json.loads((producer.HERE/'MANIFEST.json').read_text())['files']:
        require(pin(producer.HERE/row['path'])['sha256']==row['sha256'],'Frozen producer source changed')
    job=producer.descriptor(str(root),101,args.arm,0 if args.arm=='native_common' else None)
    cpu_pin=next(r['receipt'] for r in contract['completed_CPU_lifecycles'] if r['arm']==args.arm)
    cpu=json.loads(Path(cpu_pin['path']).read_text())['result']
    require(cpu['status']=='passed' and cpu['planned_job']==job and cpu['optimizer_calls']=={'attempts':[2],'completed':[2]} and
        cpu['VALID_completed']==3 and cpu['fresh_restore_VALID_bit_exact']is True and cpu['TEST_accessed']is False,
        'Already-completed exact CPU correctness qualification differs')
    sys.path.insert(0,str(E/'runtime'))
    from run_fit import validate_job,OptimizerObserver
    from adapters import create_adapter
    from common import seed_all
    policy=validate_job(job);output.mkdir(parents=True)
    report={'schema':'molbace-pairwise-full-epoch-CUDA-production-cost-v1','status':'running','arm':args.arm,
        'job':job,'manifest':manifest,'admission':pin(admission_path),'CPU_correctness_receipt':cpu_pin,
        'CPU_receipt_scope':'Correctness only; its timings are not CUDA production cost or resource approval.',
        'scientific_fit':False,'selector_calls':0,'scientific_checkpoints_saved':0,'TEST_accessed':False,
        'automatic_retries':0,'construction_attempts':0,'constructions_completed':0,
        'optimizer_calls':{'attempts':[0],'completed':[0]},'VALID_attempts':0,'VALID_completed':0,
        'native_forwards':{k:{'attempts':0,'completed':0} for k in ('TRAIN','VALID_1','VALID_2')},
        'phase_times':{},'TRAIN_batches':[],'update_pair_support':[],'outer_wall_limit_seconds':admission['maximum_wall_seconds'],
        'outer_watchdog_owner':'Root must enforce this timeout; this source starts no owner, watchdog, queue or child process.'}
    adapter=observer=None;hooks=[];original_update=original_next=capped_original=None;phase=None
    def write():
        if observer is not None:report['optimizer_calls']=observer.state()
        report.update(wall_seconds=time.perf_counter()-started,
            own_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        temporary=output/'RESULT.tmp';temporary.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(output/'RESULT.json')
    def interrupted(signum,frame):raise KeyboardInterrupt('Root interrupted bounded cost probe')
    signal.signal(signal.SIGTERM,interrupted)
    write()
    try:
        import torch
        torch.set_num_threads(2);require(torch.cuda.device_count()==1,'Exactly one visible CUDA device required')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();seed_all(torch,job['seed'])
        before=time.perf_counter();report['construction_attempts']+=1
        adapter=create_adapter(torch,job,policy,root);report['constructions_completed']+=1
        torch.cuda.synchronize();report['phase_times']['construction_synchronized_wall_seconds']=time.perf_counter()-before
        model=adapter.model if args.arm=='native_common' else adapter.model.native
        routes=1 if args.arm=='native_common' else 4
        def before_forward(module,arguments):
            require(phase in report['native_forwards'],'No uncharged native forward outside TRAIN/VALID phases')
            row=report['native_forwards'][phase];row['attempts']+=1
            require(row['attempts']<=(38 if phase=='TRAIN' else 5)*routes,'Bounded native-forward limit exceeded')
        def after_forward(module,arguments,value):report['native_forwards'][phase]['completed']+=1
        hooks=[model.register_forward_pre_hook(before_forward),model.register_forward_hook(after_forward)]
        observer=OptimizerObserver(torch,adapter.optimizers)
        capped_original=adapter.session.optimizer.step
        def capped_step(*arguments,**keywords):
            require(observer.state()['attempts'][0]<38,'No39th optimizer transition allowed')
            return capped_original(*arguments,**keywords)
        adapter.session.optimizer.step=capped_step
        original_next=adapter.session._next_train_batch
        def next_batch():
            batch,positions,skipped=original_next()
            report['TRAIN_batches'].append({'graphs':len(positions),'atoms':int(batch.x.shape[0]),
                'directed_bond_records':int(batch.edge_index.shape[1]),'source_skipped_before':skipped})
            return batch,positions,skipped
        adapter.session._next_train_batch=next_batch
        original_update=adapter.session.train_update
        def update():
            value=original_update();report['update_pair_support'].append(dict(value['pairwise']))
            return value
        adapter.session.train_update=update
        parameters=tuple(adapter.model.parameters());owners=tuple(p for opt in adapter.optimizers for group in opt.param_groups for p in group['params'])
        require(len(owners)==len({id(p) for p in owners})==len(parameters) and {id(p) for p in owners}=={id(p) for p in parameters},
            'Every native/private parameter must retain one optimizer owner')
        report['parameters']={'native':sum(p.numel() for p in model.parameters()),'total':sum(p.numel() for p in parameters),
            'private':sum(p.numel() for p in parameters)-sum(p.numel() for p in model.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in parameters),
            'private_BN_buffer_bytes':0 if routes==1 else adapter.model.route_bn_state.metadata()['bank_bytes']}
        write();phase='TRAIN';torch.cuda.synchronize();before=time.perf_counter()
        epoch=adapter.train_epoch()
        torch.cuda.synchronize();report['phase_times']['TRAIN_epoch_synchronized_wall_seconds']=time.perf_counter()-before
        report['TRAIN_epoch']={k:v for k,v in epoch.items() if k!='mean_update_loss'}
        require(epoch['epoch']==1 and epoch['complete_native_TRAIN']is True and epoch['TRAIN_records_traversed']==1210 and
            epoch['raw_batches']==epoch['optimizer_maps']==38 and epoch['source_skipped_batches']==0 and
            observer.state()=={'attempts':[38],'completed':[38]} and adapter.session.updates==38 and
            [r['graphs'] for r in report['TRAIN_batches']]==[32]*37+[26],
            'Exactly one complete original TRAIN1210 epoch /38 ordinary maps required')
        require(report['native_forwards']['TRAIN']=={'attempts':38*routes,'completed':38*routes},'Exact native TRAIN forward accounting required')
        write()
        for index in (1,2):
            phase='VALID_'+str(index);report['VALID_attempts']+=1;torch.cuda.synchronize();before=time.perf_counter()
            valid=adapter.validate()
            torch.cuda.synchronize();report['phase_times'][phase+'_synchronized_wall_seconds']=time.perf_counter()-before
            require(valid['role']=='VALID' and valid['complete_source_VALID']is True and valid['TEST_accessed']is False and
                len(valid['ids'])==151 and torch.equal(valid['ids'],adapter.session.valid_ids) and
                torch.isfinite(valid['outputs']).all().item(),'Whole finite original VALID151 required')
            require(report['native_forwards'][phase]=={'attempts':5*routes,'completed':5*routes},'Exactly one native whole VALID call required')
            report['VALID_completed']+=1;write()
        pair=report['update_pair_support'];histogram={}
        for row in pair:
            key=str(row['observed_pairs']);histogram[key]=histogram.get(key,0)+1
        report['pair_support']={'updates':len(pair),'empty_pair_batches':sum(row['empty_pair_batch'] for row in pair),
            'observed_pairs_total':sum(row['observed_pairs'] for row in pair),'pair_count_histogram':histogram,
            'positive_graph_visits':sum(row['positive_graphs'] for row in pair),
            'negative_graph_visits':sum(row['negative_graphs'] for row in pair),'missing_label_visits':sum(row['missing_labels'] for row in pair)}
        require(len(pair)==38 and observer.state()=={'attempts':[38],'completed':[38]} and
            report['VALID_completed']==2 and adapter.completed_epochs==1,'Bounded final work accounting differs')
        report['actual_Adam_state_tensor_bytes']=sum(v.numel()*v.element_size() for opt in adapter.optimizers for fields in opt.state.values()
            for v in fields.values() if hasattr(v,'numel') and hasattr(v,'element_size'))
        report['producer_metadata']=adapter.metadata()
        report.update(status='passed',torch_version=str(torch.__version__),CUDA_device_name=torch.cuda.get_device_name(0),
            limitations='One cold complete epoch and two whole VALID calls per producer/seed101; no convergence, selector, fit or scientific score claim. Root separately budgets21fits using actual variation and current envelopes.')
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)})
    finally:
        if 'torch'in locals() and torch.cuda.is_initialized():
            report['GPU_resources']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'current_allocated_bytes':torch.cuda.memory_allocated(),'current_reserved_bytes':torch.cuda.memory_reserved()}
        write()
        if adapter is not None:
            if original_update is not None:adapter.session.train_update=original_update
            if original_next is not None:adapter.session._next_train_batch=original_next
            if capped_original is not None:adapter.session.optimizer.step=capped_original
        if observer is not None:observer.close()
        for hook in hooks:hook.remove()
        if adapter is not None:
            try:adapter.close()
            except Exception as error:report.update(close_error=str(error),status='failed')
        write()
    print(json.dumps({k:report[k] for k in ('status','arm','constructions_completed','VALID_completed','optimizer_calls','phase_times','own_peak_rss_bytes')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
