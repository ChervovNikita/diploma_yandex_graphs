"""Disabled complete-input native/C&S qualification or fixed evaluator worker."""
import argparse
import importlib.util
from pathlib import Path
import signal
import sys
import time
from owned import PHASE,SEEDS,bound,read,require,sha,validate,write

QUALIFICATION_IDS=('CS01','CS07','CS19','CS23')


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def run(*,release_path,release_sha256,later_execution_authorized=False):
    cfg,pins,owner,runtime,output=validate(release_path,release_sha256,later_execution_authorized)
    require(Path(sys.executable).resolve()==Path(runtime['python']).resolve(),'Exact existing owner runtime')
    began=time.perf_counter();cpu=time.process_time();costs=[]
    result=dict(complete=False,operation=cfg['operation'],wrapper_manifest_sha256=cfg['wrapper_manifest_sha256'],
        protocol_sha256=pins['CS_protocol']['sha256'],release_sha256=release_sha256,
        TEST_access=False,development_scoring=cfg['operation']=='evaluate',costs=costs)
    if cfg['operation']=='qualify':
        result.update(scheduled_propagation_passes=302,native_forward_calls_attempted=0,
            native_forward_calls_completed=0,development_truths_loaded=False,metrics_computed=False)
        result['qualification_configurations']=[dict(configuration_id=key,complete=False,status='pending',
            scheduled_propagation_passes=100 if key in ('CS01','CS07') else 51,completed_propagation_passes=0)
                                               for key in QUALIFICATION_IDS]
    def interrupted(number,frame):raise KeyboardInterrupt('Owned worker interrupted '+str(number))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def timed(name,operation):
        started=time.perf_counter();started_cpu=time.process_time();row=dict(operation=name,complete=False);costs.append(row)
        try:value=operation();row['complete']=True;return value
        finally:row.update(wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-started_cpu)
    try:
        protocol=read(bound(pins['CS_protocol']))
        require(protocol['native_seeds']==list(SEEDS) and protocol['configuration_count']==24
            and protocol['complete_reference_records']==72,'Frozen reference family')
        states={seed:bound(dict(path=cfg['native_state_paths'][str(seed)],sha256=cfg['native_state_sha256'][str(seed)]))
            for seed in SEEDS}
        train_path=bound(owner['roles']['train']);valid_path=bound(owner['roles']['valid'])
        polynormer=bound(owner['polynormer'])
        import torch
        import numpy as np
        screen=module(bound(owner['screen_program']),'_CS_runtime_frozen_screen')
        _,_,common,_,_,data,_,_,_,_,_=screen._dependencies()
        role_source=data._data()
        train=timed('safe_complete_TRAIN_role_load',lambda:role_source.load_npz(train_path,('x','edge_index','ids','y')))
        role_source.check_projection('wikics',train,None,{'split_index':0})
        role_source.inspect_npz(valid_path,('ids','y'))
        with np.load(valid_path,allow_pickle=False) as archive:ids=torch.from_numpy(archive['ids'].copy())
        tensors=dict(TRAIN_x=train['x'],prepared_edge_index=train['edge_index'],TRAIN_ids=train['ids'],
                     TRAIN_y=train['y'],development_ids=ids)
        for key,value in tensors.items():
            require(common.tensor_digest(value)==cfg['numeric_role_sha256'][key],'Exact numeric role join: '+key)
        if cfg['operation']=='qualify':
            adapter=module(PHASE/protocol['source_directory']/'native_adapter.py','_CS_runtime_native_adapter')
            cs=module(PHASE/protocol['source_directory']/'cs_reference.py','_CS_runtime_TRAIN_only_CS')
            state=timed('trusted_native_own_best_checkpoint_load',lambda:torch.load(states[6101],map_location='cpu',weights_only=False))
            require(state['run']['seed']==6101,'Authentic qualification seed6101')
            reference=timed('complete_input_authentic_native_construct_restore',lambda:adapter.reconstruct_native_own_best(
                state=state,train_data=train,polynormer=polynormer,development_ids=ids,
                development_ids_sha256=cfg['numeric_role_sha256']['development_ids'],
                device='cuda:0',later_execution_authorized=True))
            def forward():
                result['native_forward_calls_attempted']+=1
                served=reference.probabilities();torch.cuda.synchronize();result['native_forward_calls_completed']+=1
                return served['full_probabilities'].detach().cpu()
            full=timed('one_full_node_native_forward_no_readout',forward)
            edges=train['edge_index'];self_mask=edges[0]==edges[1]
            require(int(self_mask.sum())==11701 and torch.equal(edges[0,self_mask].sort().values,torch.arange(11701)),
                'One known prepared self record per node')
            nonself=edges[:,~self_mask].clone();require(tuple(nonself.shape)==(2,431206),'Fixed full nonself graph')
            records=result['qualification_configurations']
            for config_id in QUALIFICATION_IDS:
                row=next(row for row in protocol['configurations'] if row['id']==config_id)
                record=next(record for record in records if record['configuration_id']==config_id)
                record['status']='running'
                record['completed_propagation_passes']=None
                try:
                    scores=timed(config_id+'_full_TRAIN_only_CPU_sparse_operations',lambda:cs.correct_and_smooth_train_only(
                        probabilities=full,edge_index=nonself,train_ids=train['ids'],train_labels=train['y'],classes=10,
                        configuration=row['configuration'],device='cpu',later_execution_authorized=True))
                    require(tuple(scores['smoothed_scores'].shape)==(11701,10)
                        and bool((scores['smoothed_scores']>=0).all()),'Declared nonnegative score-map domain')
                    record.update(complete=True,status='complete',propagation_passes=scores['propagation_passes'],
                        completed_propagation_passes=scores['propagation_passes'],full_shape=[11701,10],metrics_computed=False)
                except Exception as error:
                    record.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
            result.update(complete=all(row['complete'] for row in records),native_forward_calls=1,
                development_truths_loaded=False,metrics_computed=False,probabilities_saved=False,
                qualification_native_seed=6101,qualification_native_state_sha256=cfg['native_state_sha256']['6101'],
                supplied_nonself_edge_records=431206,scheduled_propagation_passes=302,
                numeric_role_sha256={key:cfg['numeric_role_sha256'][key] for key in tensors})
        else:
            qualification=read(bound(cfg['qualification_receipt']))
            require(qualification['complete'] is True and qualification['metrics_computed'] is False
                and qualification['operation']=='qualify' and qualification['development_truths_loaded'] is False
                and qualification['protocol_sha256']==pins['CS_protocol']['sha256']
                and qualification['wrapper_manifest_sha256']==cfg['wrapper_manifest_sha256']
                and qualification['qualification_native_state_sha256']==cfg['native_state_sha256']['6101'],
                'Exact successful unscored native/sparse qualification')
            require(qualification['numeric_role_sha256']=={key:cfg['numeric_role_sha256'][key] for key in tensors},
                'Same qualified complete TRAIN/graph/development-ID roles')
            evaluator_release_path=bound(cfg['evaluator_release']);evaluator_release=read(evaluator_release_path)
            require(evaluator_release['native_state_sha256']==cfg['native_state_sha256']
                and evaluator_release['numeric_role_sha256']==cfg['numeric_role_sha256']
                and evaluator_release['whole_family_opening']==cfg['whole_family_opening'],
                'Evaluator uses the same authentic states, numeric roles and whole-family opening')
            with np.load(valid_path,allow_pickle=False) as archive:truth=torch.from_numpy(archive['y'].copy())
            require(common.tensor_digest(truth)==cfg['numeric_role_sha256']['development_y'],'Exact evaluator-only truths')
            evaluator=module(PHASE/pins['evaluator_directory']/'evaluate.py','_CS_fixed72_evaluator')
            evaluated=timed('unchanged_fixed72_reference_evaluator',lambda:evaluator.evaluate_fixed_reference(
                native_state_paths=states,train_data=train,development_ids=ids,development_truths=truth,
                polynormer=polynormer,release_path=evaluator_release_path,output=output/'reference72',
                native_device='cuda:0',later_execution_authorized=True))
            result.update(complete=evaluated['complete'],reference_output='reference72',
                selected_configuration_id=evaluated['selected_configuration_id'])
    except BaseException as error:
        result['error']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        for record in result.get('qualification_configurations',[]):
            if record['status']=='pending':record.update(status='blocked',error=dict(type='QualificationInterrupted',
                message='Native/setup failure or interruption before this operation; no substitute or retry'))
            elif record['status']=='running':record.update(status='failed',error=dict(type='QualificationInterrupted',
                message='Interrupted during this fixed operation; partial costs retained'))
        result.update(wall_seconds=time.perf_counter()-began,process_cpu_seconds=time.process_time()-cpu,
            automatic_retry=False,state_kind_rewritten=False,native_refit=False)
        write(output/('QUALIFICATION.json' if cfg['operation']=='qualify' else 'EVALUATION_WORKER.json'),result)
    return 0 if result['complete'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True);parser.add_argument('--authorized',action='store_true');args=parser.parse_args()
    sys.exit(run(release_path=args.release,release_sha256=args.release_sha256,later_execution_authorized=args.authorized))


if __name__=='__main__':main()
