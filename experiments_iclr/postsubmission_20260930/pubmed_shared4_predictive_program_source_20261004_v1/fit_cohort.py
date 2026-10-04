#!/usr/bin/env python3
"""Six fresh shared4 Pubmed fits; fixed native cadence/selector/stop policy."""
import argparse
import json
import os
from pathlib import Path
import time
import common as native
from common import (HERE, EXECUTION, require, sha, write, utc, gate, capture, validation,
                    setup_runtime, save_tensor, clone, inventory, start_monitor, extension,
                    fresh_unit, equal, Progress, rng, stream_identity)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, cohort = gate(args.root_release, args.release_sha256, 'fit_cohort')
    output = EXECUTION / 'fit_cohort/run01'
    require(not output.exists(), 'Fresh cohort output required; no retry/resume')
    output.mkdir(parents=True)
    progress = Progress(output,'fit_cohort')
    stopped = observer = sample = None
    fit_freezes, paired = [], []
    try:
        torch, np, bodies, evaluate_mrr, eval_mrr, inspection, x, train, valid, negative, identities = setup_runtime(cohort)
        bridge, prototype, graph_ops, observed = extension(bodies,torch)
        stopped, observer, peaks, sample = start_monitor(output,cohort['stages']['fit_cohort']['caps'],progress,torch)
        write(output/'AVAILABLE_INSPECTION.json',inspection)
        config = {'schema':'pubmed-shared4-predictive-config-v1','cohort':cohort,'input_files':identities,
                  'author_commit':'c447cbff4c493b60d14b6544c3c39d3b9c5ddff0','root_release_sha256':args.release_sha256,
                  'source_manifest_sha256':release['source_manifest_sha256'],'feature_authority':release['feature_authority'],
                  'negative_pool_authority':release['negative_pool_authority'],'prerequisites':release['prerequisites'],
                  'feature_equivalence_receipt_sha256':release['feature_equivalence_receipt_sha256'],
                  'qualified_source_binding_sha256':sha(HERE/'SOURCE_BINDING.json'),
                  'initialized_from':'Fresh exact sealed shared4 factory; no baseline/engineering/selected state donor.',
                  'serving_pool':'mean_raw_NCNC_logits','capacity':bridge.expected_parameter_counts(),
                  'native_baseline_capacity_comparison':'Descriptive; factorization/member serving/capacity also differ.'}
        write(output/'CONFIG.json',config)
        config_sha = sha(output/'CONFIG.json')
        for fit in cohort['fits']:
            model_name, seed, seed_id, mode = fit['model'],fit['seed'],fit['literal_seed_id'],fit['mode']
            directory = output/fit['fit_id']
            directory.mkdir()
            progress.update(current_fit=fit,current_epoch=0,current_phase='fresh_shared4_factory')
            # Sole initialization of this scientific fit. No restore, reseed/reset
            # after this point, selected-state splice, engineering or baseline donor.
            unit = fresh_unit(bridge,prototype,graph_ops,bodies,fit,x,train)
            model,predictor,optimizer,ux,data,train_device = unit
            initial = capture(unit,torch,np)
            initial_identity = {'schema':'pubmed-shared4-fresh-initialization-v1',**fit,'config_sha256':config_sha,
                'source_manifest_sha256':release['source_manifest_sha256'],'input_files':identities,
                'initial_full_state_sha256':observed.state_identity(initial,torch),
                'initial_RNG_sha256':observed.state_identity(initial['RNG'],torch),
                'scientific_state_donor':False}
            equal(initial['RNG'],rng(torch,np,True),torch,fit['fit_id']+'/initial_digest_RNG_neutral')
            write(directory/'INITIALIZATION.json',initial_identity)
            del initial
            private_reader = None
            private_freeze = private_init = None
            compared_epochs = 0
            if mode == 'pooled_after_clamp':
                private_dir = output/('shared4_private_seed'+str(seed))
                private_freeze = json.loads((private_dir/'FIT_FREEZE.json').read_text())
                private_init = json.loads((private_dir/'INITIALIZATION.json').read_text())
                require(private_freeze==fit_freezes[-1] and private_freeze['schema']=='pubmed-shared4-fit-freeze-v1'
                        and private_freeze['status']=='COMPLETE' and private_freeze['seed']==seed
                        and private_freeze['mode']=='private' and private_freeze['config_sha256']==config_sha
                        and private_freeze['source_manifest_sha256']==release['source_manifest_sha256']
                        and private_freeze['input_files']==identities,'Own completed same-seed private fit absent')
                require(sha(private_dir/'INITIALIZATION.json')==private_freeze['initialization_sha256']
                        and sha(private_dir/'TRAIN_STREAMS.jsonl')==private_freeze['native_train_streams_sha256'],'Own private initialization/stream custody differs')
                require(private_init['schema']=='pubmed-shared4-fresh-initialization-v1'
                        and private_init['fit_id']==private_freeze['fit_id']
                        and private_init['seed']==seed and private_init['mode']=='private'
                        and private_init['config_sha256']==config_sha and private_init['input_files']==identities
                        and private_init['source_manifest_sha256']==release['source_manifest_sha256']
                        and private_init['initial_full_state_sha256']==initial_identity['initial_full_state_sha256']
                        and private_init['initial_RNG_sha256']==initial_identity['initial_RNG_sha256'],'Paired fresh initial state/RNG differs')
                private_reader = (private_dir/'TRAIN_STREAMS.jsonl').open('r')
            fit_started_updates = progress.snapshot()['Adam_completed']
            optimizer.register_step_pre_hook(lambda optimizer,args,kwargs:progress.add(Adam_started=1))
            optimizer.register_step_post_hook(lambda optimizer,args,kwargs:progress.add(Adam_completed=1))
            best_valid, selected_score, selected_epoch, kill = 0.0,None,None,0
            epoch, stop_reason = 0,'native_max_epochs'
            selected_state = directory/'selected_state.pt'
            selected_scores = directory/'selected_VALID_scores.pt'
            neutrality = {'checks':0}
            def exact_observation(a,b,label,reports):
                equal(a,b,torch,label)
                neutrality['checks'] += 1
            try:
                with (directory/'VALID_HISTORY.jsonl').open('x') as history, (directory/'TRAIN_STREAMS.jsonl').open('x') as train_history:
                    for epoch in range(1,10000):
                        progress.update(current_epoch=epoch,current_phase='native_train')
                        before_neutrality = neutrality['checks']
                        loss,stream = observed.observed_epoch(native,bodies,unit,train,torch,np,progress,exact_observation,None)
                        expected_updates = epoch*36
                        require(progress.snapshot()['Adam_completed']-fit_started_updates==expected_updates,'Native full epoch/tail update coverage differs')
                        require(neutrality['checks']-before_neutrality==38,'Original sampler/iterator/batch observation coverage differs')
                        stream_row = {'fit_id':fit['fit_id'],'seed':seed,'mode':mode,'epoch':epoch,
                                      'config_sha256':config_sha,'stream':stream,'RNG_neutral_observations':38,
                                      'completed_Adam_updates':expected_updates}
                        train_history.write(json.dumps(stream_row,sort_keys=True,allow_nan=False)+'\n')
                        train_history.flush()
                        os.fsync(train_history.fileno())
                        if private_reader is not None and epoch<=private_freeze['last_training_epoch']:
                            line = private_reader.readline()
                            require(bool(line),'Own private stream ended before declared overlap')
                            reference = json.loads(line)
                            require(reference['fit_id']==private_freeze['fit_id'] and reference['epoch']==epoch
                                    and reference['seed']==seed and reference['mode']=='private'
                                    and reference['completed_Adam_updates']==expected_updates
                                    and reference['RNG_neutral_observations']==38
                                    and reference['config_sha256']==config_sha,'Own private stream identity/order differs')
                            require(stream_identity(reference['stream'])==stream_identity(stream),'Paired actual native stream/RNG differs at epoch'+str(epoch))
                            compared_epochs += 1
                            if epoch==private_freeze['last_training_epoch']:
                                require(private_reader.readline()=='','Own private stream has unreported extra epochs')
                        progress.update(current_phase='native_epoch_complete')
                        if epoch%5:
                            continue
                        progress.update(current_phase='complete_native_VALID')
                        before = capture(unit,torch,np)
                        pos,neg = validation(bodies,'NCNC',unit,valid,negative,torch)
                        result = evaluate_mrr(None,pos,neg)
                        per_query = clone(eval_mrr(pos,neg),torch)
                        score = result['MRR']  # native evaluator already rounds to4 decimals
                        after = capture(unit,torch,np)
                        equal(before['RNG'],after['RNG'],torch,fit['fit_id']+'/complete_VALID_TRAIN_RNG_neutral')
                        progress.add(complete_VALID_serves=1)
                        # Exact original six-baseline first-maximum/kill policy.
                        improves_selection = selected_score is None or score>selected_score
                        if score>best_valid:
                            best_valid,kill = score,0
                        else:
                            kill += 1
                        row = {'fit_id':fit['fit_id'],'model':model_name,'mode':mode,'seed':seed,'literal_seed_id':seed_id,
                               'epoch':epoch,'loss_last_native_epoch':float(loss),'metrics':result,
                               'raw_VALID_MRR':float(per_query['mrr_list'].mean()),'selection_improved':improves_selection,
                               'kill_count':kill,'completed_Adam_updates':expected_updates}
                        history.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                        history.flush()
                        if improves_selection:
                            selected_score,selected_epoch = score,epoch
                            identity = {**fit,'selected_epoch':epoch,'selected_VALID_MRR':score,'config_sha256':config_sha,
                                        'source_manifest_sha256':release['source_manifest_sha256']}
                            save_tensor(selected_scores,{'schema':'pubmed-shared4-selected-VALID-scores-v1',**identity,
                                'positive_scores':pos,'negative_scores':neg,'score_semantics':'Mean of four native unbounded raw logits',
                                'native_metrics_rounded4':result,'native_per_query_metrics':per_query,'input_files':identities,
                                'row_order':'Exact released VALID positive/pool row and candidate order',
                                'native_evaluator_sha256':cohort['metric']['evaluator_sha256']},torch)
                            save_tensor(selected_state,{'schema':'pubmed-shared4-selected-full-state-v1',**identity,
                                'state_before_VALID':before,'state_after_VALID':after,'selected_VALID_scores_sha256':sha(selected_scores),
                                'input_files':identities,'completed_Adam_updates':expected_updates,
                                'selector_state':{'best_valid':best_valid,'kill_count':kill},
                                'initialization_source':'Fresh shared4 factory; no prior baseline/engineering/selected state donor.',
                                'initialization_sha256':sha(directory/'INITIALIZATION.json')},torch)
                        equal(after['RNG'],rng(torch,np,True),torch,fit['fit_id']+'/metric_checkpoint_IO_RNG_neutral')
                        progress.update(current_phase='VALID_and_selection_complete')
                        del before,after,pos,neg,per_query
                        if kill>10:
                            stop_reason = 'native_kill_count_gt10'
                            break
            finally:
                if private_reader is not None:
                    private_reader.close()
            require(selected_epoch is not None and progress.snapshot()['Adam_started']==progress.snapshot()['Adam_completed'],'Selected state or complete update accounting absent')
            freeze = {'schema':'pubmed-shared4-fit-freeze-v1','status':'COMPLETE',**fit,
                'selected_epoch':selected_epoch,'selected_VALID_MRR':selected_score,'last_training_epoch':epoch,
                'stop_reason':stop_reason,'last_kill_count':kill,'native_Adam_updates':epoch*36,
                'selected_state_sha256':sha(selected_state),'selected_VALID_scores_sha256':sha(selected_scores),
                'VALID_history_sha256':sha(directory/'VALID_HISTORY.jsonl'),
                'native_train_streams_sha256':sha(directory/'TRAIN_STREAMS.jsonl'),
                'initialization_sha256':sha(directory/'INITIALIZATION.json'),'config_sha256':config_sha,
                'source_manifest_sha256':release['source_manifest_sha256'],
                'input_files':identities,'model_reset_after_fit':False,'observer_RNG_neutral_checks':neutrality['checks'],
                'selected_state_replay_status':'PENDING_SEPARATE_ROOT_RELEASE'}
            write(directory/'FIT_FREEZE.json',freeze)
            fit_freezes.append(freeze)
            if mode=='pooled_after_clamp':
                overlap = min(private_freeze['last_training_epoch'],epoch)
                require(compared_epochs==overlap,'Incomplete paired executed-overlap stream evidence')
                pair = {'schema':'pubmed-shared4-paired-native-stream-evidence-v1','status':'PASS','seed':seed,
                    'private_fit_id':private_freeze['fit_id'],'pooled_fit_id':fit['fit_id'],
                    'initial_full_state_sha256':initial_identity['initial_full_state_sha256'],
                    'initial_RNG_sha256':initial_identity['initial_RNG_sha256'],'paired_initial_equality':'Exact scalar tensor/state/RNG digest identity; fixed mode identity excluded.',
                    'private_final_epoch':private_freeze['last_training_epoch'],'pooled_final_epoch':epoch,
                    'compared_epochs':compared_epochs,'executed_overlap':[1,overlap],'all_actual_native_streams_and_RNG_exact':True,
                    'private_FIT_FREEZE_sha256':sha(output/private_freeze['fit_id']/'FIT_FREEZE.json'),
                    'pooled_FIT_FREEZE_sha256':sha(directory/'FIT_FREEZE.json'),
                    'private_streams_sha256':private_freeze['native_train_streams_sha256'],
                    'pooled_streams_sha256':freeze['native_train_streams_sha256'],
                    'source_manifest_sha256':release['source_manifest_sha256'],'config_sha256':config_sha,
                    'input_files':identities,'unshared_later_epochs':'Native survivor only; stopped arm not continued, no selected-state splice.'}
                write(output/('PAIRED_SEED'+str(seed)+'.json'),pair)
                paired.append(pair)
            progress.completed('completed_fits',fit['fit_id'])
            del unit,model,predictor,optimizer,ux,data,train_device
        require(len(fit_freezes)==6 and [row['seed'] for row in paired]==[0,1,2],'Complete six-fit/three-pair cohort required')
        final = progress.snapshot()
        total_epochs = sum(row['last_training_epoch'] for row in fit_freezes)
        require(final['native_epochs_started']==final['native_epochs_completed']==total_epochs
                and final['Adam_started']==final['Adam_completed']==36*total_epochs
                and final['complete_VALID_serves']==sum(row['last_training_epoch']//5 for row in fit_freezes),
                'Complete cohort native epoch/update/VALID accounting differs')
        torch.cuda.synchronize(0)
        stopped.set()
        observer.join()
        sample()
        write(output/'COHORT_FREEZE.json',{'schema':'pubmed-shared4-cohort-freeze-v1','status':'COMPLETE','UTC':utc(),
            'fits':fit_freezes,'paired_train_stream_evidence':paired,'cohort_sha256':sha(HERE/'COHORT.json'),
            'source_manifest_sha256':release['source_manifest_sha256'],'root_release_sha256':args.release_sha256,
            'config_sha256':config_sha,'progress':progress.snapshot(),'inclusive_child_wall_seconds':time.monotonic()-started,
            'scientific_claim':'TRAIN/VALID shared4 comparison only; no TEST, published-score reproduction or novelty claim.',
            'native_capacity_comparisons':'Descriptive; factorization/member serving/capacity differ from native singles.',
            'selected_state_replay_status':'PENDING_SEPARATE_ROOT_RELEASE'})
    except BaseException as error:
        write(output/'FAILURE.json',{'status':'FAILED','type':type(error).__name__,'condition':str(error),
            'progress':progress.snapshot(),'inclusive_child_wall_seconds':time.monotonic()-started,'automatic_retry':False})
        raise
    finally:
        if stopped is not None:
            stopped.set()
            observer.join()
        write(output/'FINAL_CUSTODY.json',{'files':inventory(output),'stage':'fit_cohort','completed':(output/'COHORT_FREEZE.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
