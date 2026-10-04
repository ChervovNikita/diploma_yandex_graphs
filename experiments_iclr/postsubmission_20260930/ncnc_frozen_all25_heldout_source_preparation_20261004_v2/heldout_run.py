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

from heldout_accounting import atomic_json, new_work, new_cells, progress, public_cells, private_json, private_inventory


def retain_numerical(output, torch, label, binding, **values):
    """Persist returned/pooled raw numerical evidence before guards; never selected states."""
    import tempfile
    folder = output / 'PRIVATE_NUMERICAL_EVIDENCE'
    folder.mkdir(mode=0o700, exist_ok=True)
    path = folder / (label + '.pt')
    atomic_json(folder/(label+'.IDENTITY.json'),dict(binding=binding,validation_status='EVIDENCE_ONLY_PENDING_GUARDS',expected_final_file=path.name,partial_temporary_prefix=label+'.partial.'))
    fd, temporary = tempfile.mkstemp(prefix=label + '.partial.', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(dict(schema='ncnc-heldout-private-returned-numerical-evidence-v2',
                            binding=binding, values=values, validation_status='EVIDENCE_ONLY_PENDING_GUARDS'), stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        from heldout_accounting import descriptor
        return descriptor(path)
    except BaseException:
        # Preserve a failed/partially written numerical file with its separate exact binding.
        # A future supervisor inventory hashes these literal bytes; no save retry occurs.
        raise


def adapt_TEST_queries(original, positive, negative, torch, data_api, *, expected_records=(1179052, 60084)):
    """Exact TRAIN-then-VALID composition; QA uses fabricated row counts through this helper."""
    train_count, valid_count = expected_records
    require(len(original['pairs']) == train_count and len(original['valid_positive']) == valid_count, 'Original TRAIN/VALID record counts differ')
    data = {**original, 'pairs':torch.cat([original['pairs'],original['valid_positive']]), 'valid_positive':positive, 'valid_negative':negative}
    guard_keys = ('x','pairs','raw_edge_index','valid_positive','valid_negative')
    custody = {k:data_api.tensor_sha(data[k]) for k in guard_keys}
    require(len(data['pairs']) == train_count + valid_count and torch.equal(data['pairs'][:train_count], original['pairs']) and torch.equal(data['pairs'][train_count:], original['valid_positive']), 'Exact TRAIN+VALID topology required')
    receipt = dict(actual_graph='native_TRAIN_plus_VALID', train_records=train_count, VALID_positive_records=valid_count, graph_records=len(data['pairs']), canonical_pairs_sha256=custody['pairs'], TEST_rows_added=False, weights_used=False, query_target_removal=False, alias_map={'pairs':'original_TRAIN_concat_original_VALID_positive', 'valid_positive':'official_TEST_positive', 'valid_negative':'official_TEST_shared_negative'}, original_returned_graph_label='complete_TRAIN_only', original_label_is_not_actual_topology_authority=True)
    return data, custody, receipt

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
    work, cells = new_work(), new_cells()
    failures, private_rows = [], []
    current_slot = current_cell = None
    phase = 'initialization'
    result = dict(schema='ncnc-frozen-all25-heldout-result-v2', status='IN_PROGRESS', identity=context['identity'], root_release_sha256=context['heldout_release_sha256'], heldout_source_manifest_sha256=context['heldout_source_sha256'], family_lock=context['heldout_release']['family_lock'], v4_audit_result=context['heldout_release']['v4_audit_result'], policy=POLICY, frozen_unique_training_fits=35, frozen_served_cells=25, original_optimizer_updates=59500, training_updates=0, TEST_opened=False, new_checkpoint_selection=False, new_calibration=False, no_success_only_subset_summary=True, old_v2_exact_replay_failure_repaired=False, summary=None)
    torch = data = prepared = None
    def flush(next_phase):
        nonlocal phase
        phase = next_phase
        if current_cell is not None and next_phase != 'failed_terminal_preparation': cell['phase'] = phase
        progress(output, result, work, cells, phase, current_cell, None if current_slot is None else
                 {k:current_slot[k] for k in ('arm','base_seed','member','status','returned','validated','selected_state_digest')})
    def private_flush():
        atomic_json(output/'PRIVATE_SCORER_RECEIPTS.json', private_json(dict(identity=context['identity'],
                    root_release_sha256=context['heldout_release_sha256'], heldout_source_manifest_sha256=context['heldout_source_sha256'],
                    rows=private_rows, predictive_metric_values_exposed=False)))
    def binding(item, member=None):
        return dict(identity=context['identity'], root_release_sha256=context['heldout_release_sha256'],
                    heldout_source_manifest_sha256=context['heldout_source_sha256'], arm=arm, base_seed=seed,
                    member=member, selected_checkpoint=item['cell']['checkpoint'], selected_state_digests=item['state_digests'],
                    canonical_graph_and_query_digests=custody)
    class RecordedMetric:
        # strict_hits50 remains unchanged and calls the authenticated original evaluator once.
        def eval(self, payload):
            work['official_metric_calls'] += 1; work['official_metric_attempted'] += 1
            cell['official_metric_attempted'] = True; flush('official_evaluator_entry')
            returned = metric.eval(payload)
            work['official_metric_returned'] += 1; cell['official_metric_returned'] = True
            flush('official_evaluator_returned_evidence_pending')
            cell['raw_official_metric_evidence'] = retain_numerical(output,torch,'cell_%s_%s_official_metric' % (arm,seed),binding(item),returned_official_evaluator=returned)
            flush('official_metric_internal_return_guard')
            return returned
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
        data, custody, graph_receipt = adapt_TEST_queries(original, positive, negative, torch, data_api)
        guard_keys = ('x','pairs','raw_edge_index','valid_positive','valid_negative')
        context['loaded_tensor_guard'] = lambda: require({k:data_api.tensor_sha(data[k]) for k in guard_keys} == custody, 'Loaded graph/TEST rows mutated')
        result.update(official_TEST_query_receipt=test_receipt, actual_graph_receipt=graph_receipt)
        del original
        set_profile(torch, PROFILE); profile_receipt(torch, PROFILE)
        scorer_sha = sha(evaluate.__file__)
        flush('fixed40_original_scorer_calls_no_retry')
        for cell in cells:
            arm, seed = cell['arm'], cell['base_seed']; item = prepared[(arm,seed)]; rows = []
            current_cell = dict(arm=arm, base_seed=seed); current_slot = None
            cell.update(status='ATTEMPTED', selection=item['cell']['selection'], checkpoint=item['cell']['checkpoint'], selected_state_digests=item['state_digests'])
            flush('cell_started')
            for member, saved in enumerate(item['snapshots']):
                work['attempted'] += 1
                current_slot = dict(arm=arm,base_seed=seed,member=member,status='ATTEMPTED',selected_state_digest=item['state_digests'][member],returned=False,validated=False)
                private_rows.append(current_slot)
                private_flush()
                flush('selected_state_construction_and_restore')
                instance, optimizer = model.make_native(mods, seed+5*member if arm==ARMS[1] else seed, 70 if arm==ARMS[4] else 64, device) if arm in (ARMS[0],ARMS[1],ARMS[4]) else model.make_factorized(mods,seed,device)
                flush('selected_state_restore')
                state_api.restore_snapshot(instance,optimizer,saved,restore_random=True)
                require(state_api.state_digest(saved) == item['state_digests'][member] and state_api.state_digest(state_api.snapshot(instance,optimizer)) == item['state_digests'][member], 'Exact restored selected model/Adam/flags/RNG differs')
                fixed = state_api.state_digest({'models':saved['models'],'optimizer':saved['optimizer']}); rng = state_api.rng_digest(state_api.rng_state())
                before = profile_receipt(torch,PROFILE); context['loaded_tensor_guard'](); require(sha(evaluate.__file__) == scorer_sha, 'Original scorer source changed')
                mode = 'private' if arm==ARMS[2] else 'pooled_after_clamp' if arm==ARMS[3] else None
                work['entered_original_scorer'] += 1
                current_slot['status'] = 'SCORER_ENTRY_RECORDED'
                flush('original_scorer_entry')
                pos, neg, receipt = evaluate.score_valid(instance,data,mods,mode=mode)
                work['returned'] += 1
                current_slot.update(status='RETURNED_PENDING_EVIDENCE',returned=True,original_scorer_receipt=receipt,actual_graph_receipt=graph_receipt)
                flush('original_scorer_returned_evidence_pending')
                raw_pin = retain_numerical(output,torch,'slot_%02d_returned' % (work['returned']-1),binding(item,member), positive=pos,negative=neg,receipt=receipt)
                current_slot.update(status='RETURNED_PENDING_GUARDS',returned=True,raw_returned_evidence=raw_pin,original_scorer_receipt=receipt,actual_graph_receipt=graph_receipt)
                private_flush()
                flush('scorer_return_guards')
                require(pos.shape == (len(positive),) and neg.shape == (100000,) and pos.dtype == neg.dtype == torch.float32 and bool(torch.isfinite(pos).all()) and bool(torch.isfinite(neg).all()), 'Incomplete/nonfinite full TEST score pools')
                require(receipt['positive_queries']==len(positive) and receipt['negative_queries']==100000 and receipt['query_batches']==[1,1] and receipt['encoder_calls']==1 and receipt['all_query_rows_complete'] is True and receipt['serving_pool']=='mean_raw_logits' and receipt['graph']=='complete_TRAIN_only', 'Original aliased scorer receipt differs')
                require(receipt['score_digests']=={'positive':data_api.tensor_sha(pos),'negative':data_api.tensor_sha(neg)}, 'Original full TEST score digests differ')
                context['loaded_tensor_guard'](); require(state_api.rng_digest(state_api.rng_state())==rng, 'Scoring changed restored RNG')
                after=state_api.snapshot(instance,optimizer,rng=saved['rng'])
                require(state_api.state_digest({'models':after['models'],'optimizer':after['optimizer']})==fixed and all(flag is False for tree in after['flags'] for flag in tree.values()), 'Scoring mutated selected state or eval flags differ')
                require(profile_receipt(torch,PROFILE)==before==expected_profile(PROFILE), 'Fixed serving profile changed')
                rows.append((pos,neg,receipt)); current_slot.update(status='VALIDATED',validated=True,before_profile=before,after_profile=before)
                work['completed_validated'] += 1
                private_flush(); flush('scorer_validated')
                del instance, optimizer, after
            current_slot = None
            flush('pooling_entry')
            require(len(rows)==(4 if arm==ARMS[1] else 1), 'Complete served member bank required')
            pos,neg = evaluate.mean_native_scores(rows) if arm==ARMS[1] else rows[0][:2]
            flush('pooling_returned_evidence_pending')
            cell['raw_served_score_evidence'] = retain_numerical(output,torch,'cell_%s_%s_served' % (arm,seed),binding(item),positive=pos,negative=neg)
            flush('pooling_guards')
            require(pos.shape==(len(positive),) and neg.shape==(100000,), 'Complete served TEST pools required')
            work['metric_helper_attempted'] += 1; flush('official_metric_helper_entry')
            value=evaluate.strict_hits50(RecordedMetric(),pos,neg)
            work['metric_helper_returned'] += 1
            flush('official_metric_returned_evidence_pending')
            cell['raw_metric_evidence'] = retain_numerical(output,torch,'cell_%s_%s_metric' % (arm,seed),binding(item),returned_metric=value)
            flush('official_metric_return_guards')
            require(type(value) is float and math.isfinite(value) and 0 <= value <= 1, 'Invalid returned official metric')
            work['official_metric_validated'] += 1; cell['official_metric_validated'] = True
            cell.update(status='PASS',TEST_hits50=value,selection=item['cell']['selection'],checkpoint=item['cell']['checkpoint'],selected_state_digests=item['state_digests'],served_score_digests={'positive':data_api.tensor_sha(pos),'negative':data_api.tensor_sha(neg)})
            work['cells_completed'] += 1; flush('cell_complete')
        current_cell = current_slot = None
        flush('whole25_summary_entry')
        require(all(work[k]==40 for k in ('attempted','entered_original_scorer','returned','completed_validated')) and work['cells_completed']==work['official_metric_calls']==work['official_metric_attempted']==work['official_metric_returned']==work['official_metric_validated']==work['metric_helper_attempted']==work['metric_helper_returned']==25, 'Whole cohort work incomplete')
        candidate_summary=summarize(cells)
        atomic_json(output/'PRIVATE_CANDIDATE_SUMMARY.json',dict(identity=context['identity'],root_release_sha256=context['heldout_release_sha256'],heldout_source_manifest_sha256=context['heldout_source_sha256'],summary=candidate_summary,adoption_status='PENDING_FINAL_CUSTODY_AND_PHYSICAL_TERMINAL'))
        torch.cuda.synchronize(0); result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        flush('final_custody_before_publication')
        result['private_cell_receipts']=descriptor(output/'PRIVATE_CELL_RECEIPTS.json')
        result['private_numerical_evidence']=private_inventory(output)
        result['private_scorer_receipts']=descriptor(output/'PRIVATE_SCORER_RECEIPTS.json')
        result['inclusive_wall_seconds_through_accounting']=perf_counter()-started
        result['final_input_custody']=final_custody(context,family)
        result.update(status='ALL25_FROZEN_HELDOUT_CONFIRMATION_COMPLETE',summary=candidate_summary,cells=cells)
    except Exception as error:
        failure=dict(category='FAILED_HELDOUT_OR_FINAL_CUSTODY',phase=phase,exception_type=type(error).__name__,condition=str(error),current_cell=current_cell,last_slot=None if current_slot is None else {k:current_slot[k] for k in ('arm','base_seed','member','status','returned','validated')})
        failures.append(failure)
        failed_phase = phase
        if current_cell is not None:
            cell.update(status='FAILED',failed_phase=phase,failure=failure)
        if current_slot is not None:
            current_slot.update(status='VALIDATED_WITH_LATER_FAILURE' if current_slot['validated'] else 'FAILED',failure=failure)
            private_flush()
        flush('failed_terminal_preparation')
        result.update(status='FAILED_HELDOUT_CONFIRMATION',summary=None,cells=public_cells(cells),observed_wall_seconds=perf_counter()-started,
                      failed_phase=failed_phase,private_evidence=private_inventory(output),
                      unavailable_partial_scorer_values='No caller-visible values exist for an unreturned scorer; unflushed work stays unknown on interruption.')
    result.update(work=work,count_semantics='literal_observed_child_events; metric_attempted_returned_validated_are_distinct',failures=failures,TEST_opened=context.get('TEST_file_opened',False),UTC=datetime.now(timezone.utc).isoformat(),automatic_retry=False,terminal_write_tail_measured=False)
    atomic_json(output/'HELDOUT_RESULT.json',result)
    atomic_json(output/'STATUS.json',dict(schema='ncnc-frozen-all25-heldout-progress-v2',status=result['status'],identity=result['identity'],root_release_sha256=result['root_release_sha256'],heldout_source_manifest_sha256=result['heldout_source_manifest_sha256'],cells=public_cells(cells),work=work,predictive_values_exposed=False,record_kind='terminal_mirror_of_authoritative_result',authoritative_result_receipt=descriptor(output/'HELDOUT_RESULT.json')))
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
