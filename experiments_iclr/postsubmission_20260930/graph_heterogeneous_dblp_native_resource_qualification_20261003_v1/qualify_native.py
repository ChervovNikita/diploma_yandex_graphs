"""Bounded native full-DBLP CPU resource qualification; no VAL/TEST scores."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ARMS=('native_GAT','native_Simple_HGN','native_SeHGNN')
SEEDS=(131,137,139,149,151)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):digest.update(data)
    return digest.hexdigest()


def verify(record):
    path=Path(record['path'])
    require(sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],'Source/input fingerprint differs: '+str(path))
    return path


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def preservation(manifest_sha,binding,extra=()):
    require(sha(HERE/'MANIFEST.json')==manifest_sha,'Qualification manifest changed')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['payload']:verify(dict(row,path=str(HERE/row['path'])))
    for row in binding['source_records']+[binding['freeze'],binding['paired_HGT_freeze']]+list(extra):verify(row)


def admission(binding,frozen,release,manifest_sha,run_name):
    require(release['qualification_authorized'] is True and release['execution_authorized'] is False
        and release['CPU_fixture_passed'] is True and release['device']=='cpu'
        and release['mode']=='native_full_graph_resource_only' and release['run_name']==run_name
        and release['qualification_manifest_sha256']==manifest_sha
        and release['prepared_native_manifest_sha256']==binding['native_manifest_sha256']
        and release['native_study_freeze_sha256']==binding['freeze']['sha256']
        and release['paired_HGT_freeze_sha256']==binding['paired_HGT_freeze']['sha256']
        and release['native_CPU_original_receipt_sha256']==binding['CPU_original_receipt']['sha256'],
        'Exact root source-only native CPU qualification release required')
    for key,value in binding['limits'].items():require(release[key]==value,'Root qualification limit differs: '+key)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU qualification requires CUDA hidden')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key)=='1','One CPU thread per native qualification process required: '+key)
    require(frozen['study_adopted_by_root'] is True and frozen['test_labels_closed'] is True
        and tuple(frozen['seeds'])==SEEDS and tuple(frozen['arms'])==ARMS,'Exact adopted fifteen-case native freeze required')
    template=json.loads(verify(binding['native_template']).read_text())
    for key in ('dataset','scope','seeds','arms','recipes','evaluation_batch_protocol','expected_node_counts',
        'expected_feature_widths','expected_relations','member_sha256','members','source_label_member_sha256'):
        require(frozen[key]==template[key],'Root native protocol differs: '+key)
    paired=json.loads(verify(binding['paired_HGT_freeze']).read_text())
    for key in ('dataset','scope','seeds','archive','development_labels','splits','source_label_member_sha256'):
        require(paired[key]==frozen[key],'Native/HGT paired input binding differs: '+key)
    require(paired['study_adopted_by_root'] is True and paired['test_labels_closed'] is True,'Paired adopted HGT freeze required')
    cpu=json.loads(verify(binding['CPU_original_receipt']).read_text())
    require(cpu['exit_code']==0 and cpu['prepared_manifest_sha256']==binding['native_manifest_sha256']
        and cpu['original_source_hashes_preserved'] is True and cpu['scope']['real_dataset_labels_opened'] is False
        and cpu['scope']['GPU_model_execution'] is False and cpu['scope']['training_driver_main_called'] is False
        and json.loads(cpu['stdout'])['status']=='PASS','Actual native CPU original correspondence receipt required')


def rss():
    current=None
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):current=int(line.split()[1])*1024
    return dict(current_RSS_bytes=current,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def measured(callback):
    tick=time.perf_counter();value=callback();return value,time.perf_counter()-tick


def unscored_whole_target(torch,model,arm,graph,features,label_features,eval_ids):
    """Literal native inference forwards, retaining all target outputs without scoring."""
    model.eval()
    with torch.no_grad():
        if arm!='native_SeHGNN':return model(graph)[:len(features['A'])]
        batch=torch.tensor(eval_ids,dtype=torch.long)
        return model(batch,{k:v[batch] for k,v in features.items()},{k:v[batch] for k,v in label_features.items()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',required=True);parser.add_argument('--run-name',required=True)
    args=parser.parse_args(argv);require(Path(args.run_name).name==args.run_name and args.run_name not in ('','.','..'),'Fresh simple qualification name')
    manifest_sha=sha(HERE/'MANIFEST.json')
    require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Qualification seal differs')
    binding=json.loads((HERE/'BINDINGS.json').read_text());preservation(manifest_sha,binding)
    frozen=json.loads(verify(binding['freeze']).read_text());release_bytes=Path(args.admission).read_bytes();release=json.loads(release_bytes)
    admission(binding,frozen,release,manifest_sha,args.run_name)
    require(resource.getrlimit(resource.RLIMIT_AS)==(binding['limits']['address_space_limit_bytes'],)*2
        and resource.getrlimit(resource.RLIMIT_CPU)==(binding['limits']['CPU_soft_seconds'],binding['limits']['CPU_hard_seconds']),
        'Qualification must run under admitted preimport address-space and CPU limits')
    native=Path(binding['canonical_phase'])/'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'
    require(not list((native/'runs').glob('*/STUDY_STARTED.json')),'Native study already started')
    inputs=[frozen['archive'],frozen['development_labels']]+[r['descriptor'] for r in frozen['splits']]+[
        dict(path=str(Path(args.admission).resolve()),sha256=hashlib.sha256(release_bytes).hexdigest(),bytes=len(release_bytes))]
    for row in inputs:verify(row)
    out=HERE/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=False)
    write(out/'QUALIFICATION_STARTED.json',dict(qualification_manifest_sha256=manifest_sha,root_release_sha256=sha(args.admission),
        native_study_freeze_sha256=binding['freeze']['sha256'],study_training_authorized=False))
    result=dict(status='preparing',rows=[],device='cpu',qualification_only=True,training_driver_main_called=False,
        validation_or_test_scored=False,model_selection=False,labels_saved=False,scientific_study_outcomes=False,
        native_study_freeze_sha256=binding['freeze']['sha256'],qualification_manifest_sha256=manifest_sha,
        native_prepared_manifest_sha256=binding['native_manifest_sha256'],root_release_sha256=sha(args.admission),limits=binding['limits'])
    started=time.perf_counter()
    try:
        import torch
        import numpy as np
        import scipy
        require(torch.__version__=='2.1.2+cu118' and sys.version_info[:3]==(3,11,14),'Exact root-qualified CPU runtime required')
        torch.set_num_threads(1);torch.set_num_interop_threads(1);device=torch.device('cpu')
        require('dgl' not in sys.modules and 'torch_geometric' not in sys.modules,'Native prepared runtime is pure Torch/NumPy/SciPy')
        result['runtime']=dict(torch=torch.__version__,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
            threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),CUDA_VISIBLE_DEVICES='')
        driver=load('resource_exact_native_driver',native/'train_native.py')
        loader=load('resource_exact_DBLP_loader',verify(binding['loader_source']))
        data=load('resource_exact_native_inputs',native/'native_inputs.py')
        models=load('resource_exact_native_models',native/'native_models.py')
        preparation=time.perf_counter();timings={}
        (schema,attributes,edges),timings['stream_all_attributes_seconds']=measured(lambda:data.stream_all_attributes(loader,frozen['archive'],frozen['members']))
        require(schema['node_counts']==frozen['expected_node_counts'] and schema['provided_attribute_widths']==frozen['expected_feature_widths']
            and schema['member_sha256']==frozen['member_sha256'],'Exact full native DBLP feature/schema geometry required')
        require({str(r['raw_id']):(r['source'],r['target']) for r in schema['relations']}=={k:tuple(v) for k,v in frozen['expected_relations'].items()},'Native relation geometry differs')
        records,timings['homogeneous_records_seconds']=measured(lambda:data.homogeneous_records(schema,edges))
        graph,timings['homogeneous_tensor_materialization_seconds']=measured(lambda:data.materialize_homogeneous(records,models,device))
        adjs,timings['normalized_adjacencies_seconds']=measured(lambda:data.normalized_native_adjacencies(schema,edges))
        features,timings['feature_channels_seconds']=measured(lambda:data.feature_channels(schema,attributes,adjs))
        products,timings['label_products_seconds']=measured(lambda:data.label_products(adjs))
        del attributes,edges
        timings['shared_full_graph_preprocessing_seconds']=time.perf_counter()-preparation
        result['preprocessing']=timings;result['preprocessing_memory']=rss()
        development=loader.read_development_labels(frozen['development_labels'],frozen['archive']['sha256'])
        require(development['source_label_member_sha256']==frozen['source_label_member_sha256']
            and development['train_class_schema']==[0,1,2,3] and len(development['node_ids'])==1217,'Exact frozen development pool/class schema required')
        split_record=frozen['splits'][0];split=json.loads(verify(split_record['descriptor']).read_text());train=split['train_ids'];validation=split['validation_ids'];seed=131
        require(split_record['seed']==split['seed']==seed and train==sorted(set(train)) and validation==sorted(set(validation))
            and len(train)==974 and len(validation)==243 and not set(train)&set(validation)
            and sorted(train+validation)==development['node_ids'],'Exact first frozen TRAIN/VAL membership required')
        selected=set(train);train_lookup={i:y for i,y in zip(development['node_ids'],development['labels']) if i in selected}
        train_labels=[train_lookup[i] for i in train]
        require(set(train_labels)=={0,1,2,3},'Class4 must be established by TRAIN')
        del development,selected,train_lookup
        ids=torch.tensor(train,dtype=torch.long);labels=torch.tensor(train_labels,dtype=torch.long)
        label_features,label_seconds=measured(lambda:data.train_only_label_channels(products,4057,train,train_labels))
        eval_ids=data.native_evaluation_ids(4057,split)
        require(eval_ids==train+validation+sorted(set(range(4057))-set(train+validation)),'Exact sorted TRAIN/VAL/topological complement evaluation composition')
        result['seed_preprocessing']=dict(seed=seed,TRAIN_only_label_channels_seconds=label_seconds,
            evaluation_ids_sha256=hashlib.sha256((json.dumps(eval_ids,separators=(',',':'))+'\n').encode()).hexdigest(),
            sorted_TRAIN=974,sorted_VAL_membership_only=243,topological_complement=2840,evaluation_batch_nodes=4057)
        result['label_scope']=dict(source_pool='explicit frozen TRAIN_VAL_ONLY source',loss_targets='974 frozen seed131 TRAIN only',
            validation_targets_constructed=False,ZIP_label_members_opened=[],TEST_labels_opened=False,TRAIN_only_propagated_labels=True)
        result['graph_geometry']=dict(node_counts=schema['node_counts'],attribute_widths=schema['provided_attribute_widths'],
            homogeneous_audit=records['audit'],feature_shapes={k:list(v.shape) for k,v in features.items()},
            label_shapes={k:list(v.shape) for k,v in label_features.items()},product_nnz={k:int(v.nnz) for k,v in products.items()},
            product_shapes={k:list(v.shape) for k,v in products.items()},adjacency_convention='native destination-first raw file row',
            label_diagonal_rule='remove diagonal from complete normalized product; no renormalization')
        for arm in ARMS:
            row=dict(arm=arm,seed=seed,status='preparing',attempted=True);case=out/arm;case.mkdir();model=optimizer=logits=state=None
            try:
                driver.seed_all(torch,seed,device)
                build=lambda:models.SeHGNN({k:v.shape[1] for k,v in features.items()},label_features.keys()).to(device) if arm=='native_SeHGNN' else models.HGBGAT([schema['node_counts'][str(i)] for i in range(4)],simple=arm=='native_Simple_HGN').to(device)
                model,row['model_build_seconds']=measured(build);sehgnn=arm=='native_SeHGNN'
                optimizer=torch.optim.Adam(model.parameters(),lr=.001 if sehgnn else .0005,weight_decay=0. if sehgnn else .0001)
                training_loader=torch.utils.data.DataLoader(train,batch_size=10000,shuffle=True,drop_last=False) if sehgnn else None
                row['before_update']=rss();row['parameters']=sum(p.numel() for p in model.parameters())
                _,row['TRAIN_epoch_seconds']=measured(lambda:driver.train_epoch(torch,model,optimizer,arm,graph,features,label_features,ids,labels,training_loader,None))
                logits,row['unscored_whole_target_evaluation_seconds']=measured(lambda:unscored_whole_target(torch,model,arm,graph,features,label_features,eval_ids))
                require(tuple(logits.shape)==(4057,4) and bool(torch.isfinite(logits).all()),'Finite whole-target class4 output required')
                state=dict(schema='native_DBLP_disposable_resource_state_v1',qualification_only=True,arm=arm,seed=seed,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),AMP=None,global_RNG=driver.global_state(torch,device),
                    optimizer_parameter_names=[name for name,_ in model.named_parameters()],fixed_evaluation_ids=eval_ids if sehgnn else None,
                    bindings=dict(native_study_freeze_sha256=binding['freeze']['sha256'],native_prepared_manifest_sha256=binding['native_manifest_sha256']))
                def save():
                    torch.save(state,case/'DISPOSABLE_STATE.pt');torch.save(logits,case/'DISPOSABLE_UNSCORED_TARGET_LOGITS.pt')
                _,row['disposable_state_and_full_target_logits_write_seconds']=measured(save)
                row['disposable_artifacts']=[dict(path=str(case/name),sha256=sha(case/name),bytes=(case/name).stat().st_size)
                    for name in ('DISPOSABLE_STATE.pt','DISPOSABLE_UNSCORED_TARGET_LOGITS.pt')]
                if sehgnn:
                    row['BatchNorm']=[dict(name=name,features=module.num_features,affine=module.affine,
                        track_running_stats=module.track_running_stats,num_batches_tracked=None if module.num_batches_tracked is None else int(module.num_batches_tracked))
                        for name,module in model.named_modules() if isinstance(module,torch.nn.BatchNorm1d)]
                    require(row['BatchNorm'][-1]['features']==4 and row['BatchNorm'][-1]['track_running_stats'] is False,'Native target BatchNorm must use whole evaluation batch')
                row.update(status='qualified',optimizer_updates=1,TRAIN_batches=1,native_AMP_used=False,maximum_study_epochs=200 if sehgnn else 300,
                    validation_or_test_scored=False,model_selection=False,after_update_and_eval=rss())
                row['observed_epoch_cost_seconds']=sum(row[k] for k in ('TRAIN_epoch_seconds','unscored_whole_target_evaluation_seconds','disposable_state_and_full_target_logits_write_seconds'))
            except Exception as error:
                row.update(status='qualification_failed',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),memory=rss())
            finally:
                write(case/'RECEIPT.json',row);result['rows'].append(row)
                with (out/'ARM_TERMINALS.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                del model,optimizer,logits,state
        complete=all(r['status']=='qualified' for r in result['rows']) and len(result['rows'])==3
        result['status']='qualified' if complete else 'qualification_incomplete'
        if complete:
            per_seed=label_seconds+sum(r['model_build_seconds']+r['maximum_study_epochs']*r['observed_epoch_cost_seconds']+r['unscored_whole_target_evaluation_seconds'] for r in result['rows'])
            serial=timings['shared_full_graph_preprocessing_seconds']+5*per_seed
            parallel=timings['shared_full_graph_preprocessing_seconds']+per_seed
            result['forecast']=dict(cases=15,maximum_epochs_by_arm={r['arm']:r['maximum_study_epochs'] for r in result['rows']},
                exact_native_v2_serial_CPU_nominal_hours=serial/3600,exact_native_v2_serial_CPU_factor2_planning_hours=2*serial/3600,
                prospective_five_seed_worker_nominal_hours=parallel/3600,prospective_five_seed_worker_factor2_planning_hours=2*parallel/3600,
                measured_process_peak_RSS_bytes=rss()['process_peak_RSS_bytes'],projected_five_worker_peak_RSS_bytes=5*rss()['process_peak_RSS_bytes'],
                scope='one first native TRAIN epoch plus whole-target unscored eval and worst-case checkpoint write per epoch; explicit 300/300/200 caps; preprocessing shared once for serial v2, repeated per seed if future parallel wrapper admitted; source-validation metric overhead unmeasured; factor2 is a planning allowance, not a guaranteed bound',
                future_parallel_scheduler_prepared=False,CPU_forecast_not_GPU_measurement=True,qualification_states_adopted_as_study_fits=False)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='qualification_failed',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
    finally:
        done={r['arm'] for r in result['rows']}
        for arm in ARMS:
            if arm not in done:result['rows'].append(dict(arm=arm,seed=131,status='resource_deferred',attempted=False,reason='qualification_prerequisite_or_process_failure',validation_or_test_scored=False))
        try:preservation(manifest_sha,binding,inputs);result['originals_preserved']=True
        except Exception as error:result.update(status='original_preservation_failed',originals_preserved=False,preservation_error=str(error),forecast=None)
        result.update(wall_seconds=time.perf_counter()-started,memory=rss(),DGL_imported='dgl' in sys.modules)
        if result['memory']['process_peak_RSS_bytes']>binding['limits']['RSS_limit_bytes']:
            result.update(status='resource_deferred',forecast=None,reason='process_peak_RSS_budget_exceeded')
        write(out/'RESOURCE_QUALIFICATION.json',result)
    print(json.dumps(dict(status=result['status'],output=str(out),memory=result['memory'],forecast=result.get('forecast'),
        rows=[{k:r.get(k) for k in ('arm','status','observed_epoch_cost_seconds')} for r in result['rows']]),sort_keys=True))
    return 0 if result['status']=='qualified' else 1


if __name__=='__main__':raise SystemExit(main())
