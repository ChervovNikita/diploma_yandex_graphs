"""Disabled CPU counts from saved factual probabilities; no model/forward/fit."""
import gc
import os
import resource
import sys
import time
from .common import (CLOSED, EDGES, METHODS, InputUnavailable, admitted, canonical,
                     digest, inside, require, write)

FIELDS = ('targets', 'full_member_observed_event_probabilities', 'full_pool_probabilities',
          'member_correct', 'full_pool_correct', 'common_wrong_events', 'any_correct_member_events')


def load_bank(torch, row, expected_identity, caps=CLOSED):
    caps.require()
    descriptor = row['diagnostic_binding']
    path = inside(descriptor['path'])
    require(path.name == 'SELECTED_VALID_DIAGNOSTICS.pt' and not path.is_symlink(), 'not_original_diagnostic')
    if not path.is_file():
        raise InputUnavailable(('original_diagnostic_archive',))
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        require(before.st_size == descriptor['bytes'], 'changed_archive_length')
        raw = handle.read()
        require(digest(raw) == descriptor['sha256'], 'changed_archive_digest')
        del raw
        handle.seek(0)
        payload = torch.load(handle, map_location='cpu', weights_only=True)
        after = os.fstat(handle.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'archive_changed_while_open')
    require(type(payload) is dict and payload.get('member_order') == [0, 1, 2, 3]
            and payload.get('paired_identity') == expected_identity, 'wrong_original_identity')
    ids = payload.get('valid_ids')
    require(type(ids) is list and len(ids) == len(set(ids)) == 274 and all(type(n) is int for n in ids), 'wrong_ordered_VALID_cohort')
    missing = [key for key in FIELDS if key not in payload]
    if missing:
        raise InputUnavailable(missing)
    values = {key: payload[key] for key in FIELDS}
    for key, value in values.items():
        shape = (4, 274, 5) if key in ('member_correct', 'full_member_observed_event_probabilities') else (274, 5)
        dtype = torch.bool if key in ('member_correct', 'full_pool_correct', 'common_wrong_events', 'any_correct_member_events') else torch.float32
        require(type(value) is torch.Tensor and value.device.type == 'cpu'
                and value.dtype == dtype and tuple(value.shape) == shape, 'wrong_saved_tensor_schema')
        if dtype == torch.float32:
            require(torch.isfinite(value).all().item() and ((value >= 0) & (value <= 1)).all().item(), 'invalid_saved_probability_or_truth')
    y, correct = values['targets'], values['member_correct']
    require(((y == 0) | (y == 1)).all().item(), 'nonbinary_saved_targets')
    require(torch.equal(values['common_wrong_events'], (~correct).all(dim=0))
            and torch.equal(values['any_correct_member_events'], correct.any(dim=0)), 'incoherent_original_masks')
    require(not (values['full_pool_correct'] & values['common_wrong_events']).any().item()
            and not ((~values['full_pool_correct']) & correct.all(dim=0)).any().item(), 'incoherent_native_unanimity')
    values['valid_ids'] = ids
    del payload
    gc.collect()
    return values


def hist(torch, values, caps=CLOSED):
    caps.require()
    counts = []
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        selected = (values >= lo) & ((values <= hi) if i == 3 else (values < hi))
        counts.append(int(selected.sum().item()))
    require(sum(counts) == values.numel(), 'bins_do_not_partition_complete_cohort')
    return dict(event_instances=values.numel(), bin_counts=counts, zero_distance_events=int((values == 0).sum().item()))


def bank_rows(torch, bank, anchor, method, role, caps=CLOSED):
    caps.require()
    # Stored q is probability of the observed Bernoulli event, not a decoded
    # native sigmoid array. Float64 distance uses its retained FP32 values.
    q = bank['full_member_observed_event_probabilities']
    distance = (q.double() - .5).abs()
    measures = {**{'member'+str(m): distance[m] for m in range(4)},
                'nearest_member_boundary': distance.min(dim=0).values,
                'farthest_member_boundary': distance.max(dim=0).values,
                'original_probability_pool_boundary': (bank['full_pool_probabilities'].double() - .5).abs()}
    rows = []
    for cohort_name, cohort in (('own_stored_common_wrong', bank['common_wrong_events']),
                                ('fixed_shared_own_only_common_wrong', anchor)):
        for label in (0, 1, 2, 3, 4, 'all_labels'):
            column = slice(None) if label == 'all_labels' else label
            for target in (0, 1):
                mask = cohort[:, column] & (bank['targets'][:, column] == target)
                for name, margin in measures.items():
                    row = dict(method=method, role=role, cohort=cohort_name, label=label,
                               observed_target=target, measure=name, **hist(torch, margin[:, column][mask], caps))
                    # Representation check only. Stored native decisions are
                    # never rebuilt from q, including q exactly equal to .5.
                    row['saved_q_above_half_in_own_wrong_cohort'] = (int((q[int(name[-1]), :, column][mask] > .5).sum().item())
                        if name.startswith('member') and cohort_name == 'own_stored_common_wrong' else None)
                    rows.append(row)
    return rows


def execute(release_path, caps=CLOSED):
    caps.require()
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    release, bindings, output, release_sha = admitted(release_path, caps)
    output.mkdir(parents=True, exist_ok=False)
    completed, status, unavailable, attempted = [], 'failed', [], None
    try:
        import torch  # After exact source/route/release admission; CPU only.
        require(str(torch.__version__) == release['expected_torch_version'], 'wrong_bound_Torch')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        banks, cohorts = {}, {}
        expected = {(r['pair'], r['method']): r for r in bindings['previous_all24_verified_origins']}
        baseline = {(r['role'], r['method'], r['label']): r for r in bindings['previous_mask_count_rows']}
        for row in bindings['banks']:
            require(time.perf_counter()-started < release['wall_budget_seconds'], 'finite_budget_exhausted')
            role, method = row['pair'], row['method']
            attempted = dict(role=role, method=method)
            bank = load_bank(torch, row, bindings['expected_paired_identity'][str(role)], caps)
            origin = expected[role, method]
            require(origin['diagnostic_sha256'] == row['diagnostic_binding']['sha256']
                    and origin['ordered_ids_sha256'] == digest(canonical(bank['valid_ids']))
                    and origin['paired_identity_sha256'] == digest(canonical(bindings['expected_paired_identity'][str(role)])), 'changed_completed_mask_origin')
            if role in cohorts:
                require(bank['valid_ids'] == cohorts[role]['valid_ids']
                        and torch.equal(bank['targets'], cohorts[role]['targets']), 'changed_matched_rows_or_truth')
            else:
                cohorts[role] = bank
            for label in (0, 1, 2, 3, 4, 'all_labels'):
                mask = bank['common_wrong_events'] if label == 'all_labels' else bank['common_wrong_events'][:, label]
                require(int(mask.sum().item()) == baseline[role, method, label]['all_members_wrong_events'], 'changed_frozen_common_wrong_count')
            banks[role, method] = bank
            completed.append(dict(role=role, method=method, diagnostic_sha256=origin['diagnostic_sha256'], required_probability_fields_present=True))
        require(len(banks) == len(completed) == 24, 'complete24_required_before_binning')
        rows = []
        for role in (1, 2, 3):
            anchor = banks[role, 'shared_own_only']['common_wrong_events']
            for method in METHODS:
                rows.extend(bank_rows(torch, banks[role, method], anchor, method, role, caps))
        require(len(rows) == 4032 and time.perf_counter()-started < release['wall_budget_seconds'], 'incomplete_bins_or_budget_exhausted')
        result = dict(schema='IMDB24-stored-common-wrong-confidence-bin-counts-v1', complete=True,
                      study_binding=bindings['study_binding'], release_sha256=release_sha,
                      all24_verified_origins=completed, distance_edges=list(EDGES), rows=rows,
                      stored_observed_event_q_used_directly=True, native_logits_or_member_p_reconstructed=False,
                      original_masks_define_cohorts=True, no_new_scores_threshold_search_or_predictions=True,
                      no_arrays_targets_IDs_or_probabilities_exported=True, selected_VALID_not_unused_confirmation=True,
                      roles_members_labels_not_independent_replicates=True)
        write(output/'COUNTS.json', result)
        status = 'complete'
        return result
    except InputUnavailable as error:
        status = 'input_unavailable'
        unavailable = list(error.fields)
        raise
    finally:
        end = resource.getrusage(resource.RUSAGE_SELF)
        write(output/'TERMINAL.json', dict(complete=status == 'complete', status=status, banks_verified=len(completed),
              exception_type=None if status == 'complete' else (sys.exc_info()[0].__name__ if sys.exc_info()[0] else None),
              missing_original_fields=unavailable, attempted_bank=attempted,
              all24_required=True, no_retry=True, no_re_inference=True,
              seconds_including_admission=time.perf_counter()-started,
              CPU_user_seconds=end.ru_utime-usage.ru_utime, CPU_system_seconds=end.ru_stime-usage.ru_stime,
              cumulative_process_RSS_peak_bytes=int(end.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
              RSS_is_lifetime_highwater=True, nested_wall_times_must_not_be_summed=True))
