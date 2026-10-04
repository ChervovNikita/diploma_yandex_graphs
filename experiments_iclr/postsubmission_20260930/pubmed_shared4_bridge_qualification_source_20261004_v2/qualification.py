#!/usr/bin/env python3
"""Bounded fresh-engineering Pubmed shared4 qualification; root-disabled."""
import argparse
import json
from pathlib import Path
import time
from common import HERE, PHASE, EXECUTION, require, sha, utc, write, gate, runtime, monitor, Progress, inventory


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, plan = gate(args.root_release, args.release_sha256, 'qualification')
    output = EXECUTION / 'qualification/run01'
    require(not output.exists(), 'Fresh engineering output required; no retry/resume')
    output.mkdir(parents=True)
    progress = Progress(output)
    comparisons, phases, streams, serialized = [], [], {}, []
    stopped = observer = sample = native = None
    try:
        native, loaded, bridge, prototype, graph_ops = runtime(plan)
        torch, np, bodies, evaluate_mrr, eval_mrr, inspection, x, train, valid, pool, identities = loaded
        stopped, observer, sample = monitor(output, plan['stages']['qualification']['caps'], progress, torch)
        write(output / 'AVAILABLE_INSPECTION.json', inspection)
        tolerance = plan['engineering_tolerance']
        reused_qualification = PHASE / 'pubmed_heart_native_numerical_qualification_source_20261004_v2/qualify.py'
        # Reuse only this reviewed numeric tree comparison, not the old runner.
        import ast
        tree = ast.parse(reused_qualification.read_text())
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compare']
        require(len(definitions) == 1, 'Qualified comparison body absent')
        scope = {'require': require}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reused_qualification), 'exec'), scope)
        numeric_compare = scope['compare']
        def compare(a, b, label, reports):
            numeric_compare(native.clone(a,torch), native.clone(b,torch), torch, tolerance['atol'], tolerance['rtol'], label, reports)
        def exact(a, b, label, reports):
            numeric_compare(native.clone(a,torch), native.clone(b,torch), torch, 0, 0, label, reports)
        def make(seed, mode):
            unit = bridge.make_shared4(PHASE, bodies, prototype, graph_ops, seed, mode, x, train)
            progress.add(factory_calls=1)
            return unit
        from geometry import run_geometry
        from native_observer import observed_epoch, observe_work, state_identity

        # Three paired factory identities only, no training/VALID at these points.
        progress.update(phase='all_seed_pair_initialization')
        initialization = []
        for seed in (0, 1, 2):
            private = make(seed, 'private')
            private_initial = native.capture(private, torch, np)
            pooled = make(seed, 'pooled_after_clamp')
            pooled_initial = native.capture(pooled, torch, np)
            exact(private_initial, pooled_initial, 'seed' + str(seed) + '/exact_private_pooled_initial_state_and_RNG', comparisons)
            initialization.append({'seed':seed,'initial_state_sha256':state_identity(private_initial,torch),
                                   'initial_RNG_sha256':state_identity(private_initial['RNG'],torch),'status':'PASS'})
            del private, pooled, private_initial, pooled_initial
        phases.append({'id':'all_seed_pair_initialization','status':'PASS','seeds':[0,1,2],'fresh_factories':6,'optimizer_updates':0})
        geometry = run_geometry(native,bodies,bridge,prototype,graph_ops,make,x,train,valid,pool,torch,np,progress,compare,exact,comparisons)
        phases.append({'id':'new_geometry_parity',**geometry})

        def epoch(unit, mode, epoch_id, label):
            progress.update(phase=label,current_arm=mode,current_epoch=epoch_id)
            before = progress.snapshot()
            torch.cuda.synchronize(0)
            point = time.monotonic()
            remove = observe_work(unit,prototype,progress)
            try:
                loss, stream = observed_epoch(native,bodies,unit,train,torch,np,progress,exact,comparisons)
            finally:
                remove()
            torch.cuda.synchronize(0)
            after = progress.snapshot()
            stream['wall_seconds'] = time.monotonic()-point
            stream['work'] = {key:after[key]-before[key] for key in (
                'Adam_started','Adam_completed','encoder_started','encoder_completed','root_decoder_started',
                'root_decoder_completed','root_query_rows_started','root_query_rows_completed',
                'neighbor_calls_started','neighbor_calls_completed','neighbor_query_rows','common_rows',
                'left_residual_rows','right_residual_rows')}
            require(stream['work']['Adam_started'] == stream['work']['Adam_completed'] == 36
                    and stream['work']['encoder_started'] == stream['work']['encoder_completed'] == 36
                    and stream['work']['root_decoder_started'] == stream['work']['root_decoder_completed'] == 72
                    and stream['work']['root_query_rows_started'] == stream['work']['root_query_rows_completed'] == 73728,
                    'Exact full native epoch outer work differs')
            return loss, stream

        def serve(unit, mode, label):
            progress.update(phase=label,current_arm=mode)
            before = native.capture(unit,torch,np)
            before_work = progress.snapshot()
            progress.add(VALID_started=1)
            torch.cuda.synchronize(0)
            point = time.monotonic()
            remove = observe_work(unit,prototype,progress)
            try:
                positive,negative = native.validation(bodies,'NCNC',unit,valid,pool,torch)
            finally:
                remove()
            torch.cuda.synchronize(0)
            elapsed = time.monotonic()-point
            after = native.capture(unit,torch,np)
            progress.add(VALID_completed=1)
            exact(before['RNG'],after['RNG'],label+'/VALID_full_TRAIN_RNG_neutral',comparisons)
            exact(before['encoder'],after['encoder'],label+'/VALID_encoder_state_neutral',comparisons)
            exact(before['predictor'],after['predictor'],label+'/VALID_predictor_state_neutral',comparisons)
            exact(before['Adam'],after['Adam'],label+'/VALID_Adam_neutral',comparisons)
            exact(before['gradients'],after['gradients'],label+'/VALID_gradients_neutral',comparisons)
            delta = {key:progress.snapshot()[key]-before_work[key] for key in ('encoder_started','encoder_completed','root_decoder_started','root_decoder_completed','root_query_rows_started','root_query_rows_completed')}
            require(delta['encoder_started']==delta['encoder_completed']==1 and delta['root_decoder_started']==delta['root_decoder_completed']==10
                    and delta['root_query_rows_started']==delta['root_query_rows_completed']==1110216,'Full native VALID batch/tail/candidate work differs')
            return before,after,positive,negative,{'wall_seconds':elapsed,'positive_queries':2216,'negative_queries':1108000,
                    'native_positive_batch_rows':[512,512,512,512,168],'native_negative_batch_rows':[256000,256000,256000,256000,84000],
                    'work':delta,'RNG_neutral':True}

        resource = []
        for mode in ('private','pooled_after_clamp'):
            unit = make(0,mode)  # New fresh factory; never a geometry or pairing donor.
            rows = []
            for number in range(1,6):
                _,stream = epoch(unit,mode,number,'paired_native_resource_epoch')
                rows.append(stream)
                write(output / (mode+'_STREAMS.json'),{'mode':mode,'seed':0,'epochs':rows})
            streams[mode] = rows
            # The first native-cadence candidate is fixed epoch5, not a score search.
            before,after,positive,negative,valid_receipt = serve(unit,mode,'first_native_cadence_VALID')
            metrics = evaluate_mrr(None,positive,negative)
            per_query = native.clone(eval_mrr(positive,negative),torch)
            score_path = output/(mode+'_SELECTED_ENGINEERING_SCORES_NEVER_DONOR.pt')
            state_path = output/(mode+'_SELECTED_ENGINEERING_STATE_NEVER_DONOR.pt')
            identity = {'schema':'pubmed-shared4-owned-engineering-selected-v1','mode':mode,'seed':0,'selected_epoch':5,
                        'source_manifest_sha256':release['source_manifest_sha256'],'root_release_sha256':args.release_sha256,
                        'input_files':identities,'engineering_only':True,'scientific_donor_allowed':False,
                        'factory_initialization':'Fresh shared4 factory; no prior state donor.'}
            native.save_tensor(score_path,{**identity,'positive':positive,'negative':negative,'rounded_metrics':metrics,'per_query':per_query},torch)
            native.save_tensor(state_path,{**identity,'before_VALID':before,'after_VALID':after,'scores_sha256':sha(score_path)},torch)
            progress.add(serializations=2)
            custody = {**identity,'state':{'path':state_path.name,'sha256':sha(state_path),'bytes':state_path.stat().st_size},
                       'scores':{'path':score_path.name,'sha256':sha(score_path),'bytes':score_path.stat().st_size}}
            write(output/(mode+'_SERIALIZED_IDENTITY.json'),custody)
            serialized.append(custody)

            # Read only just-owned files after scalar custody/identity checks.
            require(not state_path.is_symlink() and not score_path.is_symlink()
                    and sha(state_path)==custody['state']['sha256'] and sha(score_path)==custody['scores']['sha256'],'Owned serialized bytes differ')
            saved = torch.load(state_path,map_location='cpu',weights_only=True)
            scores = torch.load(score_path,map_location='cpu',weights_only=True)
            progress.add(weights_only_loads=2)
            for key,value in identity.items():
                require(saved[key]==scores[key]==value,'Owned engineering identity differs: '+key)
            require(saved['scores_sha256']==sha(score_path),'Owned selected score custody differs')
            exact(saved['before_VALID'],before,mode+'/exact_serialized_before_VALID',comparisons)
            exact(saved['after_VALID'],after,mode+'/exact_serialized_after_VALID',comparisons)
            restored = make(0,mode)
            native.restore(restored,saved['after_VALID'],torch,np)
            exact(native.capture(restored,torch,np),saved['after_VALID'],mode+'/exact_fresh_restore_after_VALID',comparisons)
            native.restore(restored,saved['before_VALID'],torch,np)
            exact(native.capture(restored,torch,np),saved['before_VALID'],mode+'/exact_fresh_restore_before_VALID',comparisons)
            _,replayed_after,replayed_positive,replayed_negative,replay_receipt = serve(restored,mode,'owned_selected_serialized_VALID_replay')
            compare(replayed_positive,scores['positive'],mode+'/owned_selected_positive_replay',comparisons)
            compare(replayed_negative,scores['negative'],mode+'/owned_selected_negative_replay',comparisons)
            exact(native.clone(eval_mrr(replayed_positive,replayed_negative),torch),scores['per_query'],mode+'/exact_all_native_per_query_replay',comparisons)
            require(evaluate_mrr(None,replayed_positive,replayed_negative)==scores['rounded_metrics'],'Exact rounded native evaluator replay differs')
            exact(replayed_after,saved['after_VALID'],mode+'/exact_full_postserve_state_RNG',comparisons)

            # Genuine new serialized continuation: unchanged complete next epoch
            # in the original copy and the fresh restored copy from the same state.
            native.restore(unit,saved['after_VALID'],torch,np)
            original_loss,original_stream = epoch(unit,mode,6,'original_selected_engineering_continuation')
            original_end = native.capture(unit,torch,np)
            native.restore(restored,saved['after_VALID'],torch,np)
            restored_loss,restored_stream = epoch(restored,mode,6,'serialized_selected_engineering_continuation')
            restored_end = native.capture(restored,torch,np)
            stream_keys = ('negative_calls','iterator_calls','batches','start_RNG_sha256','end_RNG_sha256')
            require({key:original_stream[key] for key in stream_keys}=={key:restored_stream[key] for key in stream_keys},'Serialized continuation original native streams differ')
            compare(restored_loss,original_loss,mode+'/serialized_complete_next_epoch_loss',comparisons)
            for key in ('encoder','predictor','Adam','gradients'):
                compare(restored_end[key],original_end[key],mode+'/serialized_next_epoch_'+key,comparisons)
            for key in ('RNG','flags','invest'):
                exact(restored_end[key],original_end[key],mode+'/exact_serialized_next_epoch_'+key,comparisons)
            for name,value in restored[0].named_parameters():
                require(value.grad is not None and bool(torch.isfinite(value.grad).all()),'Missing/nonfinite continued native encoder gradient: '+name)
            resource.append({'mode':mode,'seed':0,'full_resource_epochs':5,'resource_updates':180,
                             'complete_next_epochs_original_and_restored':2,'continuation_updates':72,
                             'complete_VALID':valid_receipt,'serialized_selected_VALID_replay':replay_receipt,
                             'selected_epoch':5,'serialized_continuation_status':'PASS','selected_VALID_replay_status':'PASS',
                             'continuation_streams':{'original':original_stream,'restored':restored_stream},
                             'quality_metric_values_published':False})
            write(output/'RESOURCE_PROGRESS.json',{'reports':resource,'progress':progress.snapshot()})
            del unit,restored,saved,scores,positive,negative,before,after,original_end,restored_end,per_query,replayed_after,replayed_positive,replayed_negative

        stream_keys = ('negative_calls','iterator_calls','batches','start_RNG_sha256','end_RNG_sha256')
        for number,(a,b) in enumerate(zip(streams['private'],streams['pooled_after_clamp']),1):
            require({key:a[key] for key in stream_keys}=={key:b[key] for key in stream_keys},'Private/pooled actual native stream/RNG differs at epoch'+str(number))
        phases.append({'id':'paired_full_native_resource_and_serialized_continuation','status':'PASS','private_pooled_shared_epochs':5,
                       'native_epochs':14,'Adam_updates':504,'complete_VALID_serves':4,'serialized_artifacts':4,'weights_only_loads':4})
        final = progress.snapshot()
        require(final['Adam_started']==final['Adam_completed']==504 and final['native_epochs_started']==final['native_epochs_completed']==14
                and final['VALID_started']==final['VALID_completed']==4 and final['factory_calls']==12
                and final['native_reference_factory_calls']==1 and final['gradient_backward_calls']==3
                and final['serializations']==final['weights_only_loads']==4,'Complete bounded qualification work differs')
        for prefix in ('encoder','root_decoder','neighbor_calls'):
            require(final[prefix+'_started']==final[prefix+'_completed'],'Begun/completed source work differs: '+prefix)
        require(final['encoder_completed']==508 and final['root_decoder_completed']==1048
                and final['neighbor_calls_completed']==9432
                and final['root_query_rows_started']==final['root_query_rows_completed']==5473056,'Complete native query/encoder/completion work differs')
        torch.cuda.synchronize(0)
        stopped.set()
        observer.join()
        sample()
        write(output/'QUALIFICATION.json',{'schema':'pubmed-shared4-bridge-engineering-qualification-v1','status':'PASS','UTC':utc(),
                'source_manifest_sha256':release['source_manifest_sha256'],'root_release_sha256':args.release_sha256,
                'input_files':identities,'phases':phases,'initialization':initialization,'comparisons':comparisons,
                'resource':resource,'owned_serialized_artifacts':serialized,'progress':final,
                'inclusive_child_wall_seconds':time.monotonic()-started,'engineering_tolerance':tolerance,
                'scientific_fit_admitted':False,'TEST_supported':False,'state_donor_allowed':False,
                'existing_native_six_baseline_qualification_repeated':False,'published_score_reproduction_claim':False,
                'quality_metric_values_published':False,'all_incurred_costs_required_in_future_reporting':True})
    except BaseException as error:
        write(output/'FAILURE.json',{'status':'FAILED','type':type(error).__name__,'condition':str(error),'progress':progress.snapshot(),
                'comparisons':comparisons,'completed_phases':phases,'inclusive_child_wall_seconds':time.monotonic()-started,'automatic_retry':False})
        raise
    finally:
        if stopped is not None:
            stopped.set()
            observer.join()
        write(output/'FINAL_CUSTODY.json',{'files':inventory(output),'stage':'qualification','completed':(output/'QUALIFICATION.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
