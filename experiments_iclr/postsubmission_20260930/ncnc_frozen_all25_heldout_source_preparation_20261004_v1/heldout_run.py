"""Exact original scorer on TRAIN+VALID and official TEST rows; no training or retry."""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import argparse
import codecs
import json
import math
import os
import statistics
from heldout_gate import metadata_admission, final_custody, require, sha, PROFILE, POLICY, EXECUTION, OUTPUT

def atomic_json(path, value):
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def load_TEST(context, torch, data_api, device):
    import numpy as np
    from hashlib import sha256
    pin = context['TEST_authority']['official_TEST_file']
    path = Path(pin['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'Noncanonical TEST file')
    # Authenticate the same open file before safe CPU deserialization.
    with path.open('rb') as stream:
        context['TEST_file_opened'] = True
        digest, size = sha256(), 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk); size += len(chunk)
        require(size == pin['bytes'] and digest.hexdigest() == pin['sha256'], 'Official TEST bytes changed')
        stream.seek(0)
        allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.int64)), codecs.encode]
        with torch.serialization.safe_globals(allowed):
            stored = torch.load(stream, map_location='cpu', weights_only=True)
    require(type(stored) is dict and set(stored) == {'edge', 'edge_neg', 'weight', 'year'}, 'Official TEST split keys differ')
    tensors = {}
    for key, value in stored.items():
        require(type(value) is np.ndarray and value.dtype == np.int64, 'Official TEST dtype differs')
        tensors[key] = torch.from_numpy(value)
    positive, negative = tensors['edge'], tensors['edge_neg']
    require(positive.ndim == negative.ndim == 2 and positive.shape[1] == negative.shape[1] == 2 and len(positive) > 0 and len(negative) == 100000, 'Complete official TEST pool required')
    require(tensors['year'].shape == tensors['weight'].shape == (len(positive),) and bool((tensors['year'] == 2019).all()), 'Official TEST year/record metadata differs')
    for query in (positive, negative): require(int(query.min()) >= 0 and int(query.max()) < 235868, 'TEST endpoints out of range')
    expected = context['TEST_authority']['typed_query_digests']
    actual = dict(positive_sha256=data_api.tensor_sha(positive), negative_sha256=data_api.tensor_sha(negative), pair_order_sha256=data_api.tensor_sha(torch.cat([positive, negative])))
    require(actual == expected, 'Complete original TEST rows/order changed')
    return positive.to(device), negative.to(device), dict(positive_rows=len(positive), negative_rows=len(negative), typed_query_digests=actual, official_TEST_file=pin, year=2019, complete_original_order=True)

def summarize(cells):
    from replay_gate import ARMS, EXPECTED_CELLS
    require(len(cells) == 25 and {(c['arm'], c['base_seed']) for c in cells} == EXPECTED_CELLS and all(c['status'] == 'PASS' for c in cells), 'Whole25 confirmation required')
    values = {(c['arm'], c['base_seed']): c['TEST_hits50'] for c in cells}
    require(all(type(v) is float and math.isfinite(v) and 0 <= v <= 1 for v in values.values()), 'Invalid official TEST metric')
    differences = [values[(ARMS[2], seed)] - values[(ARMS[3], seed)] for seed in range(5)]
    return dict(primary_frozen_private_minus_pooled=dict(candidate=ARMS[2], control=ARMS[3], base_seeds=list(range(5)), paired_TEST_hits50_differences=differences, mean=statistics.mean(differences), sample_sd=statistics.stdev(differences), range=[min(differences), max(differences)], sign_count=dict(positive=sum(d>0 for d in differences), zero=sum(d==0 for d in differences), negative=sum(d<0 for d in differences)), directional_hypothesis='private_greater_than_pooled', new_quality_threshold=None), arm_summaries=[dict(arm=arm, base_seeds=list(range(5)), TEST_hits50=[values[(arm,s)] for s in range(5)], mean=statistics.mean(values[(arm,s)] for s in range(5)), sample_sd=statistics.stdev(values[(arm,s)] for s in range(5))) for arm in ARMS], uncertainty_scope='five_training_seed_blocks_conditional_on_one_graph_time_split', broad_SOTA_or_novelty_claim=False)

def execute(context, family):
    from replay_gate import ARMS, descriptor, runtime_and_data_custody
    from replay_contract import validate_journal_payload, validate_selected_payload
    from replay_numeric import configure_environment, original_modules, trusted_load, set_profile, profile_receipt, expected_profile
    started = perf_counter(); output = context['output']
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    work = dict(scorer_calls_planned=40, attempted=0, entered_original_scorer=0, returned=0, completed_validated=0, cells_planned=25, cells_completed=0, official_metric_calls_planned=25, official_metric_calls=0, training_updates=0, automatic_retry=False)
    cells = [dict(arm=arm, base_seed=seed, status='NOT_ATTEMPTED', TEST_hits50=None) for arm in ARMS for seed in range(5)]
    failures, private_rows = [], []
    current_slot = None
    result = dict(schema='ncnc-frozen-all25-heldout-result-v1', status='IN_PROGRESS', identity=context['identity'], root_release_sha256=context['heldout_release_sha256'], heldout_source_manifest_sha256=context['heldout_source_sha256'], family_lock=context['heldout_release']['family_lock'], v4_audit_result=context['heldout_release']['v4_audit_result'], policy=POLICY, frozen_unique_training_fits=35, frozen_served_cells=25, original_optimizer_updates=59500, training_updates=0, TEST_opened=False, new_checkpoint_selection=False, new_calibration=False, no_success_only_subset_summary=True, old_v2_exact_replay_failure_repaired=False, summary=None)
    torch = data = prepared = None
    def flush(phase):
        atomic_json(output/'STATUS.json', dict(schema='ncnc-frozen-all25-heldout-count-status-v1', status='IN_PROGRESS', phase=phase, work=work, predictive_values_exposed=False, authoritative_terminal_file='HELDOUT_RESULT.json'))
    try:
        flush('original_runtime_custody'); configure_environment(); runtime_and_data_custody(context)
        api = original_modules(context)
        common, model, data_api, state_api, evaluate = (api[n] for n in ('pilot_common','pilot_model','pilot_data','pilot_state','pilot_evaluate'))
        common.runtime_stdlib(context); device, _ = model.runtime(context)
        import torch
        torch.cuda.reset_peak_memory_stats(0)
        mods = model.modules(context); mods['design'].validate_plan(context['plan']); metric = evaluate.evaluator(context)
        prepared = {}
        flush('all25_exact_selected_state_contracts_before_TEST_open')
        for (unit, seed), row in sorted(family['units'].items()):
            payload = trusted_load(torch, row['root'], row['journal']['state_file'])
            state = validate_journal_payload(payload, row['journal'], row['complete'], context['identity'], unit, seed, mods['design'].select_validation_candidate)
            for cell in [c for c in family['cells'] if (c['unit'],c['base_seed']) == (unit,seed)]:
                payload = trusted_load(torch, row['root'], cell['checkpoint'])
                snapshots, _ = validate_selected_payload(payload, cell, state, context['identity'], state_api.state_digest)
                prepared[(cell['arm'],seed)] = dict(snapshots=snapshots, state_digests=[state_api.state_digest(s) for s in snapshots], cell=cell)
        require(len(prepared) == 25 and sum(len(v['snapshots']) for v in prepared.values()) == 40, 'All25/40 selected snapshots required before TEST open')
        del payload, state
        flush('complete_TRAIN_VALID_load_then_one_official_TEST_deserialization')
        original = data_api.load_data(context, device)
        positive, negative, test_receipt = load_TEST(context, torch, data_api, device); result['TEST_opened'] = True
        # Preserve the exact original score_valid body. Its input keys are aliases:
        # pairs is TRAIN+VALID, valid_positive/negative are official TEST arrays.
        data = {**original, 'pairs':torch.cat([original['pairs'],original['valid_positive']]), 'valid_positive':positive, 'valid_negative':negative}
        guard_keys = ('x','pairs','raw_edge_index','valid_positive','valid_negative')
        custody = {k:data_api.tensor_sha(data[k]) for k in guard_keys}
        graph_receipt = dict(actual_graph='native_TRAIN_plus_VALID', train_records=1179052, VALID_positive_records=60084, graph_records=len(data['pairs']), canonical_pairs_sha256=custody['pairs'], TEST_rows_added=False, weights_used=False, query_target_removal=False, alias_map={'pairs':'original_TRAIN_concat_original_VALID_positive', 'valid_positive':'official_TEST_positive', 'valid_negative':'official_TEST_shared_negative'}, original_returned_graph_label='complete_TRAIN_only', original_label_is_not_actual_topology_authority=True)
        require(len(data['pairs']) == 1239136 and torch.equal(data['pairs'][:1179052], original['pairs']) and torch.equal(data['pairs'][1179052:], original['valid_positive']), 'Exact TRAIN+VALID topology required')
        context['loaded_tensor_guard'] = lambda: require({k:data_api.tensor_sha(data[k]) for k in guard_keys} == custody, 'Loaded graph/TEST rows mutated')
        result.update(official_TEST_query_receipt=test_receipt, actual_graph_receipt=graph_receipt)
        del original
        set_profile(torch, PROFILE); profile_receipt(torch, PROFILE)
        scorer_sha = sha(evaluate.__file__)
        flush('fixed40_original_scorer_calls_no_retry')
        for cell in cells:
            arm, seed = cell['arm'], cell['base_seed']; item = prepared[(arm,seed)]; rows = []
            for member, saved in enumerate(item['snapshots']):
                work['attempted'] += 1; flush('fixed40_original_scorer_calls_no_retry')
                current_slot = dict(arm=arm,base_seed=seed,member=member,status='ATTEMPTED',selected_state_digest=item['state_digests'][member],returned=False,validated=False)
                private_rows.append(current_slot)
                atomic_json(output/'PRIVATE_SCORER_RECEIPTS.json',dict(rows=private_rows,predictive_metric_values_exposed=False))
                instance, optimizer = model.make_native(mods, seed+5*member if arm==ARMS[1] else seed, 70 if arm==ARMS[4] else 64, device) if arm in (ARMS[0],ARMS[1],ARMS[4]) else model.make_factorized(mods,seed,device)
                state_api.restore_snapshot(instance,optimizer,saved,restore_random=True)
                require(state_api.state_digest(saved) == item['state_digests'][member] and state_api.state_digest(state_api.snapshot(instance,optimizer)) == item['state_digests'][member], 'Exact restored selected model/Adam/flags/RNG differs')
                fixed = state_api.state_digest({'models':saved['models'],'optimizer':saved['optimizer']}); rng = state_api.rng_digest(state_api.rng_state())
                before = profile_receipt(torch,PROFILE); context['loaded_tensor_guard'](); require(sha(evaluate.__file__) == scorer_sha, 'Original scorer source changed')
                mode = 'private' if arm==ARMS[2] else 'pooled_after_clamp' if arm==ARMS[3] else None
                work['entered_original_scorer'] += 1
                pos, neg, receipt = evaluate.score_valid(instance,data,mods,mode=mode)
                work['returned'] += 1
                current_slot.update(status='RETURNED_PENDING_GUARDS',returned=True,original_scorer_receipt=receipt,actual_graph_receipt=graph_receipt)
                atomic_json(output/'PRIVATE_SCORER_RECEIPTS.json',dict(rows=private_rows,predictive_metric_values_exposed=False))
                require(pos.shape == (len(positive),) and neg.shape == (100000,) and pos.dtype == neg.dtype == torch.float32 and bool(torch.isfinite(pos).all()) and bool(torch.isfinite(neg).all()), 'Incomplete/nonfinite full TEST score pools')
                require(receipt['positive_queries']==len(positive) and receipt['negative_queries']==100000 and receipt['query_batches']==[1,1] and receipt['encoder_calls']==1 and receipt['all_query_rows_complete'] is True and receipt['serving_pool']=='mean_raw_logits' and receipt['graph']=='complete_TRAIN_only', 'Original aliased scorer receipt differs')
                require(receipt['score_digests']=={'positive':data_api.tensor_sha(pos),'negative':data_api.tensor_sha(neg)}, 'Original full TEST score digests differ')
                context['loaded_tensor_guard'](); require(state_api.rng_digest(state_api.rng_state())==rng, 'Scoring changed restored RNG')
                after=state_api.snapshot(instance,optimizer,rng=saved['rng'])
                require(state_api.state_digest({'models':after['models'],'optimizer':after['optimizer']})==fixed and all(flag is False for tree in after['flags'] for flag in tree.values()), 'Scoring mutated selected state or eval flags differ')
                require(profile_receipt(torch,PROFILE)==before==expected_profile(PROFILE), 'Fixed serving profile changed')
                rows.append((pos,neg,receipt)); current_slot.update(status='VALIDATED',validated=True,before_profile=before,after_profile=before)
                work['completed_validated'] += 1
                atomic_json(output/'PRIVATE_SCORER_RECEIPTS.json',dict(rows=private_rows,predictive_metric_values_exposed=False)); flush('fixed40_original_scorer_calls_no_retry')
                del instance, optimizer, after
            require(len(rows)==(4 if arm==ARMS[1] else 1), 'Complete served member bank required')
            pos,neg = evaluate.mean_native_scores(rows) if arm==ARMS[1] else rows[0][:2]
            require(pos.shape==(len(positive),) and neg.shape==(100000,), 'Complete served TEST pools required')
            work['official_metric_calls'] += 1
            value=evaluate.strict_hits50(metric,pos,neg)
            cell.update(status='PASS',TEST_hits50=value,selection=item['cell']['selection'],checkpoint=item['cell']['checkpoint'],selected_state_digests=item['state_digests'],served_score_digests={'positive':data_api.tensor_sha(pos),'negative':data_api.tensor_sha(neg)})
            work['cells_completed'] += 1; flush('fixed25_official_metric_calls')
        require(all(work[k]==40 for k in ('attempted','entered_original_scorer','returned','completed_validated')) and work['cells_completed']==work['official_metric_calls']==25, 'Whole cohort work incomplete')
        candidate_summary=summarize(cells)
        torch.cuda.synchronize(0); result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        flush('final_custody_before_publication')
        result['private_scorer_receipts']=descriptor(output/'PRIVATE_SCORER_RECEIPTS.json')
        result['inclusive_wall_seconds_through_accounting']=perf_counter()-started
        result['final_input_custody']=final_custody(context,family)
        result.update(status='ALL25_FROZEN_HELDOUT_CONFIRMATION_COMPLETE',summary=candidate_summary,cells=cells)
    except Exception as error:
        failure=dict(category='FAILED_HELDOUT_OR_FINAL_CUSTODY',exception_type=type(error).__name__,condition=str(error),last_slot=None if current_slot is None else {k:current_slot[k] for k in ('arm','base_seed','member','status','returned','validated')})
        failures.append(failure)
        if current_slot is not None and not current_slot['validated']:
            current_slot.update(status='FAILED',failure=failure)
            cell['status']='FAILED'
            atomic_json(output/'PRIVATE_SCORER_RECEIPTS.json',dict(rows=private_rows,predictive_metric_values_exposed=False))
        result.update(status='FAILED_HELDOUT_CONFIRMATION',summary=None,cells=[dict(arm=c['arm'],base_seed=c['base_seed'],status=c['status'],TEST_hits50=None) for c in cells],observed_wall_seconds=perf_counter()-started)
    result.update(work=work,failures=failures,TEST_opened=context.get('TEST_file_opened',False),UTC=datetime.now(timezone.utc).isoformat(),automatic_retry=False,terminal_write_tail_measured=False)
    atomic_json(output/'HELDOUT_RESULT.json',result)
    atomic_json(output/'STATUS.json',dict(schema='ncnc-frozen-all25-heldout-count-status-v1',status=result['status'],work=work,predictive_values_exposed=False,record_kind='terminal_mirror_of_authoritative_result',authoritative_result_receipt=descriptor(output/'HELDOUT_RESULT.json')))
    print('FROZEN_ALL25_HELDOUT_TERMINAL status='+result['status'],flush=True)
    return 0 if result['status']=='ALL25_FROZEN_HELDOUT_CONFIRMATION_COMPLETE' else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root-release',required=True); parser.add_argument('--release-sha256',required=True); args=parser.parse_args()
    context,family=metadata_admission(args.root_release,OUTPUT,expected_sha=args.release_sha256,require_fresh=False)
    require(not OUTPUT.exists(), 'Heldout output exists; no retry')
    # The supervisor makes the exclusive claim before the numerical child.
    claim=json.loads((EXECUTION/'ONE_TIME_TEST_CLAIM.json').read_text())
    require(claim['root_release_sha256']==args.release_sha256 and claim['output_directory']==str(OUTPUT), 'Missing separate one-time supervisor claim')
    return execute(context,family)

if __name__=='__main__': raise SystemExit(main())
