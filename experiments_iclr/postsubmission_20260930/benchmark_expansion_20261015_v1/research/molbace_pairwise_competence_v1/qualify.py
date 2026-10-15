"""Root-admitted CPU TRAIN support inspection or two-transition lifecycle.

No fit, selector, TEST indexing, extra diagnostic forward or automatic retry.
Default inspect mode constructs no model. Source preparation executes neither.
"""
import argparse
import copy
import csv
import gzip
import hashlib
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


def guard():
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[
            'GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Dedicated allocation required before project metadata reads')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('This bounded qualifier is CPU only')


def pin(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect_train(torch,root,extension,variant='structured',seed=101):
    """Only full native TRAIN1210 records/labels; zero model constructors.

    Called after the CLI's allocation/admission/source gates. The actual
    frozen native provider is reused; VALID and TEST records are not indexed.
    """
    base=json.loads((extension.HERE/'BASE_CASE.json').read_text())
    native=extension.load(base['callback_module'],'_pair_support_native_provider')
    dataset=native._load_dataset(base['recipe'],base['data_root'],'ogbg-molbace')
    authority=json.loads((extension.HERE/'SOURCES.json').read_text())
    train_pin=next(p for p in authority['data_role_pins'] if p['path'].endswith('/train.csv.gz'))
    with gzip.open(train_pin['path'],'rt') as handle:ids=torch.tensor([int(row[0]) for row in csv.reader(handle)],dtype=torch.long)
    extension.require(len(ids)==1210 and len(torch.unique(ids))==1210 and int(ids.min())>=0 and int(ids.max())<len(dataset),
        'Complete original TRAIN1210 role required')
    from ogb.graphproppred import mol_encoder
    extension.require(pin(mol_encoder.__file__)['sha256']==next(p['sha256'] for p in authority['source_pins'] if p['path'].endswith('/mol_encoder.py')),
        'Actual native categorical vocabulary source differs')
    atom,bond=mol_encoder.get_atom_feature_dims(),mol_encoder.get_bond_feature_dims()
    frozen=json.loads((extension.HERE/'METHOD.json').read_text())['native_vocabularies']
    extension.require(atom==frozen['atom'] and bond==frozen['bond'],'Actual native categorical vocabularies differ')
    view=native._RoleView(dataset,ids,torch,'TRAIN')
    table,receipt=extension.table_from_view(torch,view,ids,atom,bond,variant,seed)
    return {'TRAIN_role':train_pin,'native_encoder_source':pin(mol_encoder.__file__),'role_receipt':receipt,
        'new_model_constructions':0,'new_native_forwards':0,'TRAIN_graphs_indexed':1210,
        'VALID_graphs_indexed':0,'TEST_graphs_indexed':0,'TRAIN_labels_only':True,
        'retained_pair_code_bytes':len(table.codes)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--mode',choices=('inspect','lifecycle'),default='inspect')
    parser.add_argument('--arm',choices=('common','structured','permuted','native_common'),default='structured')
    parser.add_argument('--output',required=True);parser.add_argument('--admission',required=True);parser.add_argument('--admission-sha256',required=True)
    args=parser.parse_args();guard()
    import extension as ex
    root,here,output=Path(args.project_root).resolve(),Path(__file__).resolve().parent,Path(args.output).resolve()
    ex.require(here.is_relative_to(root) and output.is_relative_to(root) and not output.exists() and
        Path(sys.prefix).resolve()==(root/'.venv').resolve(),'Original virtualenv and fresh contained output required')
    a_path=Path(args.admission).resolve();ex.require(a_path.is_relative_to(root) and pin(a_path)['sha256']==args.admission_sha256,'Actual root admission differs')
    a=json.loads(a_path.read_text());manifest=pin(here/'MANIFEST.json')
    ex.require(a.get('schema')=='root-reviewed-molbace-pair-CPU-v1' and a.get('approved')is True and
        a.get('source_reviewed')is True and a.get('mode')==args.mode and a.get('arm')==args.arm and
        a.get('manifest_sha256')==manifest['sha256'] and a.get('new_scientific_fits')==0 and a.get('TEST_access')is False and
        a.get('maximum_native_constructions')==(0 if args.mode=='inspect' else 2) and
        a.get('maximum_optimizer_transitions')==(0 if args.mode=='inspect' else 2) and
        a.get('maximum_full_VALID_calls')==(0 if args.mode=='inspect' else 3),'Exact bounded root CPU admission required')
    for row in json.loads((here/'MANIFEST.json').read_text())['files']:
        ex.require(pin(here/row['path'])['sha256']==row['sha256'],'Own frozen source changed')
    ex.verify_sources(root)
    ex.require(args.mode!='inspect' or args.arm!='native_common','Inspect uses common/structured/permuted TRAIN roles')
    started=time.perf_counter();output.mkdir(parents=True)
    report={'schema':'molbace-pairwise-bounded-CPU-qualification-v1','status':'running','mode':args.mode,'arm':args.arm,
        'manifest':manifest,'admission':pin(a_path),'source':pin(__file__),'device':'cpu','threads':2,
        'scientific_fit':False,'scientific_selection':False,'TEST_accessed':False,'automatic_retries':0,
        'construction_attempts':0,'constructions_completed':0,'native_forward_attempts':0,'native_forwards_completed':0,
        'VALID_attempts':0,'VALID_completed':0,'optimizer_calls':{'attempts':[0],'completed':[0]},
        'full_epoch_or_CUDA_qualification':False,'repeated_original_producer_qualification':False}
    adapters=[];hooks=[];observer=None
    def write():
        if observer is not None:report['optimizer_calls']=observer.state()
        report.update(wall_seconds=time.perf_counter()-started,
            own_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (output/'RESULT.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    def interrupted(signum,frame):raise KeyboardInterrupt('Bounded CPU qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        import torch
        torch.set_num_threads(2)
        if args.mode=='inspect':report.update(TRAIN_support=inspect_train(torch,root,ex,args.arm,101),status='passed')
        else:
            sys.path.insert(0,str(ex.E/'runtime'))
            from adapters import create_adapter
            from run_fit import validate_job,OptimizerObserver
            from common import seed_all,global_rng,restore_rng,cpu_clone
            planned=ex.descriptor(str(root),101,args.arm,0 if args.arm=='native_common' else None)
            policy=validate_job(planned);job=dict(planned,device='cpu')
            report['planned_job']=planned;report['execution_override']={'device':'cpu'}
            def construct():
                report['construction_attempts']+=1
                seed_all(torch,job['seed']);adapter=create_adapter(torch,job,policy,root);adapters.append(adapter)
                report['constructions_completed']+=1
                model=adapter.model if args.arm=='native_common' else adapter.model.native
                def before(module,arguments):report['native_forward_attempts']+=1
                def after(module,arguments,result):report['native_forwards_completed']+=1
                hooks.extend([model.register_forward_pre_hook(before),model.register_forward_hook(after)])
                return adapter
            def validate(adapter):
                report['VALID_attempts']+=1;value=adapter.validate()
                ex.require(len(value['ids'])==151 and torch.equal(value['ids'],adapter.session.valid_ids) and
                    value['complete_source_VALID']is True and value['TEST_accessed']is False,'Whole original VALID151 required')
                report['VALID_completed']+=1;return value
            source=construct()
            # One genuine native shuffled minibatch is fixed for this bounded
            # two-transition lifecycle. No label-based search or resampling.
            batch=next(iter(source.train_loader));positions=torch.tensor([source._position[int(i)] for i in batch.smoke_graph_id.reshape(-1)],dtype=torch.long)
            ex.require(batch.x.shape[0]!=1 and int(batch.batch[-1])!=0,'First native minibatch must be source eligible; no search')
            source.session._next_train_batch=lambda:(batch,positions,0)
            observer=OptimizerObserver(torch,source.optimizers)
            first=source.session.train_update();report['first_update']=first
            ex.require(first['pairwise']['observed_pairs']>0 and observer.state()=={'attempts':[1],'completed':[1]},
                'First genuine native minibatch must exercise observed pairs; preserve failure without resampling')
            valid=validate(source)
            payload={'adapter':cpu_clone(torch,source.state()),'global_rng':cpu_clone(torch,global_rng(torch)),
                'planned_job':planned,'optimizer_calls':observer.state(),'engineering_only':True}
            checkpoint=output/'ENGINEERING_after_one.pt';torch.save(payload,checkpoint);report['checkpoint']=pin(checkpoint)
            calls=observer.state();observer.close();observer=None;report['optimizer_calls']=calls
            fresh=construct();fresh.session._next_train_batch=lambda:(batch,positions,0)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            ex.require(saved['planned_job']==planned,'Engineering own checkpoint differs')
            fresh.load_state(saved['adapter']);replay=validate(fresh)
            ex.require(torch.equal(valid['outputs'],replay['outputs']) and torch.equal(valid['ids'],replay['ids']),
                'Fresh-restored full VALID must be bit exact')
            fresh.load_state(saved['adapter']);restore_rng(torch,saved['global_rng'])
            observer=OptimizerObserver(torch,fresh.optimizers,previous=saved['optimizer_calls'])
            second=fresh.session.train_update();report['second_update_after_fresh_restore']=second;validate(fresh)
            ex.require(observer.state()=={'attempts':[2],'completed':[2]} and fresh.session.updates==2,'Exactly two total native optimizer transitions required')
            expected_forwards=17 if args.arm=='native_common' else 68
            ex.require(report['native_forward_attempts']==report['native_forwards_completed']==expected_forwards,
                'Same native logits only: unexpected extra model forward')
            report.update(status='passed',fresh_restore_VALID_bit_exact=True,candidate_metadata=fresh.metadata(),
                engineering_completed_epochs=0,no_selector_called=True,expected_native_forwards=expected_forwards,
                limits='Same first eligible native batch reused twice; no full epoch, scientific fit, CUDA or cost admission.')
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)})
    finally:
        write()
        if observer is not None:observer.close()
        for hook in hooks:hook.remove()
        for adapter in adapters:
            try:adapter.close()
            except Exception as error:report.setdefault('close_errors',[]).append(str(error));report['status']='failed'
        write()
    print(json.dumps({k:report[k] for k in ('status','mode','arm','constructions_completed','native_forwards_completed','VALID_completed','optimizer_calls')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
