"""One finite factual-mask aggregation. This source has not been executed."""
import time
START = time.perf_counter()
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import resource
import sys

from common import (MASKS, METHODS, PAIRED_REFERENCES, admitted, canonical,
                    digest, inside_phase, require, verify_custody, write_json)


def costs():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'seconds_from_before_remaining_imports': time.perf_counter() - START,
            'CPU_user_seconds': usage.ru_utime, 'CPU_system_seconds': usage.ru_stime,
            'process_lifetime_RSS_peak_bytes': usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
            'RSS_is_lifetime_highwater_not_incremental': True}


def load_masks(torch, row, expected_identity):
    descriptor = row['diagnostic_binding']
    path = inside_phase(descriptor['path'])
    require(path.is_file() and not path.is_symlink(), 'missing_or_symlinked_diagnostic')
    # Hash and deserialize the same open file; never open checkpoint/data paths.
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        require(before.st_size == descriptor['bytes'], 'diagnostic_length_changed')
        raw = handle.read()
        require(digest(raw) == descriptor['sha256'], 'diagnostic_digest_changed')
        del raw
        handle.seek(0)
        payload = torch.load(handle, map_location='cpu', weights_only=True)
        after = os.fstat(handle.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'diagnostic_changed_while_open')
    require(type(payload) is dict and payload.get('member_order') == [0, 1, 2, 3], 'wrong_member_order')
    require(payload.get('paired_identity') == expected_identity, 'wrong_paired_identity')
    ids = payload.get('valid_ids')
    require(type(ids) is list and len(ids) == 274 and all(type(x) is int for x in ids)
            and len(set(ids)) == 274, 'wrong_ordered_VALID_ids')
    masks = {name: payload[name] for name in MASKS}
    for name, value in masks.items():
        shape = (4, 274, 5) if name == 'member_correct' else (274, 5)
        require(type(value) is torch.Tensor and value.dtype == torch.bool and value.device.type == 'cpu'
                and tuple(value.shape) == shape, 'wrong_factual_mask_schema')
    members, pool = masks['member_correct'], masks['full_pool_correct']
    require(torch.equal(masks['any_correct_member_events'], members.any(dim=0)), 'any_correct_mask_incoherent')
    require(torch.equal(masks['common_wrong_events'], (~members).all(dim=0)), 'common_wrong_mask_incoherent')
    require(not (pool & masks['common_wrong_events']).any().item(), 'binary_unanimously_wrong_pool_incoherent')
    require(not ((~pool) & members.all(dim=0)).any().item(), 'binary_unanimously_correct_pool_incoherent')
    # Other tensors may be materialized by archive loading; none is accessed.
    del payload
    gc.collect()
    return {'masks': masks, 'valid_ids': ids,
            'ordered_ids_sha256': digest(canonical(ids)), 'paired_identity_sha256': digest(canonical(expected_identity))}


def count(mask):
    return int(mask.sum().item())


def bank_counts(bank, method, role, label):
    masks = bank['masks']
    m, p = masks['member_correct'], masks['full_pool_correct']
    a, w = masks['any_correct_member_events'], masks['common_wrong_events']
    if label != 'all_labels':
        m, p, a, w = m[:, :, label], p[:, label], a[:, label], w[:, label]
    k = m.sum(dim=0)
    loss = a & ~p
    result = {'method': method, 'role': role, 'label': label, 'event_instances': p.numel(),
              'pool_wrong_with_no_correct_member': count(w & ~p),
              'pool_wrong_despite_a_correct_member': count(loss),
              'available_correct_member_events': count(a), 'all_members_wrong_events': count(w)}
    for value in range(5):
        subgroup = k == value
        result['events_with_' + str(value) + '_correct_members'] = count(subgroup)
        result['pool_wrong_with_' + str(value) + '_correct_members'] = count(subgroup & ~p)
        result['pool_correct_with_' + str(value) + '_correct_members'] = count(subgroup & p)
    for member in range(4):
        result['member' + str(member) + '_correct_events'] = count(m[member])
        result['member' + str(member) + '_uniquely_correct_events'] = count(m[member] & (k == 1))
        result['pool_correct_member' + str(member) + '_wrong_events'] = count(p & ~m[member])
        result['member' + str(member) + '_correct_pool_wrong_events'] = count(m[member] & ~p)
    return result


def paired_counts(reference, candidate, method, role, label):
    r, c = reference['masks'], candidate['masks']
    rp, cp = r['full_pool_correct'], c['full_pool_correct']
    ra, ca = r['any_correct_member_events'], c['any_correct_member_events']
    rk = r['member_correct'].sum(dim=0)
    if label != 'all_labels':
        rp, cp, ra, ca, rk = rp[:, label], cp[:, label], ra[:, label], ca[:, label], rk[:, label]
    repair, harm = ~rp & cp, rp & ~cp
    result = {'candidate': 'assigned_source_supply', 'reference': method, 'role': role,
              'label': label, 'event_instances': rp.numel(), 'pool_repairs': count(repair), 'pool_harms': count(harm),
              'correct_member_coverage_gained': count(~ra & ca), 'correct_member_coverage_lost': count(ra & ~ca),
              'reference_lost_alternative_then_candidate_pool_correct': count(ra & ~rp & cp),
              'reference_pool_correct_then_candidate_pool_wrong_with_alternative': count(harm & ca),
              'reference_pool_correct_then_candidate_pool_wrong_without_alternative': count(harm & ~ca)}
    for value in range(5):
        subgroup = rk == value
        result['reference_' + str(value) + '_correct_members_event_instances'] = count(subgroup)
        result['pool_repairs_in_reference_' + str(value) + '_correct_members'] = count(subgroup & repair)
        result['pool_harms_in_reference_' + str(value) + '_correct_members'] = count(subgroup & harm)
    return result


def summed_rows(rows, keys):
    grouped = {}
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if identity not in grouped:
            grouped[identity] = {key: row[key] for key in keys}
            grouped[identity]['aggregation'] = 'sum_of_three_role_event_instances_not_unique_nodes'
        for key, value in row.items():
            if key not in keys and key != 'role':
                require(type(value) is int, 'noncount_in_role_sum')
                grouped[identity][key] = grouped[identity].get(key, 0) + value
    return list(grouped.values())


def run(release_path, release_sha256):
    output = None
    completed = []
    try:
        release, bindings, parent = admitted(release_path, release_sha256)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA_not_hidden')
        output = parent / 'readout'
        output.mkdir(exist_ok=False)
        custody = verify_custody(bindings)
        # Import only after complete source/release/route/custody admission.
        import torch
        require(str(torch.__version__) == release.get('expected_torch_version'), 'Torch_runtime_not_bound')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        banks = {}
        cohort = {}
        deadline = START + release['wall_budget_seconds']
        for row in bindings['banks']:
            require(time.perf_counter() < deadline, 'internal_budget_exhausted')
            pair, method = row['pair'], row['method']
            bank = load_masks(torch, row, bindings['expected_paired_identity'][str(pair)])
            if pair in cohort:
                require(bank['valid_ids'] == cohort[pair], 'paired_cohort_order_changed')
            else:
                cohort[pair] = bank['valid_ids']
            banks[(pair, method)] = bank
            completed.append({'pair': pair, 'method': method, 'diagnostic_sha256': row['diagnostic_binding']['sha256'],
                              'ordered_ids_sha256': bank['ordered_ids_sha256'],
                              'paired_identity_sha256': bank['paired_identity_sha256']})
        rows, paired = [], []
        for pair in (1, 2, 3):
            for method in METHODS:
                for label in [0, 1, 2, 3, 4, 'all_labels']:
                    rows.append(bank_counts(banks[(pair, method)], method, pair, label))
            for method in PAIRED_REFERENCES:
                for label in [0, 1, 2, 3, 4, 'all_labels']:
                    paired.append(paired_counts(banks[(pair, method)], banks[(pair, 'assigned_source_supply')], method, pair, label))
        require(time.perf_counter() < deadline, 'internal_budget_exhausted')
        result = {'schema': 'IMDB24-factual-alternative-counts-v1', 'complete': True,
                  'study_binding': bindings['study_binding'], 'source_manifest_sha256': release['source_manifest_sha256'],
                  'release_sha256': release_sha256, 'all24_verified_origins': completed,
                  'bank_role_label_counts': rows, 'three_role_sums': summed_rows(rows, ('method', 'label')),
                  'paired_role_label_counts': paired, 'paired_three_role_sums': summed_rows(paired, ('candidate', 'reference', 'label')),
                  'only_integer_event_counts': True, 'no_quality_scores_or_new_serving_rule': True,
                  'no_predictions_targets_probabilities_or_IDs_exported': True,
                  'roles_members_labels_not_independent_replicates': True,
                  'VALID_was_used_for_checkpoint_selection': True, 'no_unused_confirmation': True,
                  'verified_historical_custody_metadata': custody, 'costs': costs(),
                  'torch_version': str(torch.__version__), 'CPU_only': True,
                  'prohibited_probability_fields_not_accessed': True}
        write_json(output / 'COUNTS.json', result)
        with (output / 'BANK_ROLE_LABEL_COUNTS.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output / 'PAIRED_ROLE_LABEL_COUNTS.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        write_json(output / 'TERMINAL.json', {'complete': True, 'status': 'complete', 'banks_verified': 24,
                                             'costs': costs(), 'no_retry': True})
    except BaseException as error:
        if output is not None and output.is_dir() and not (output / 'TERMINAL.json').exists():
            # Error text is omitted to prevent tensor/pickle content in reports.
            write_json(output / 'TERMINAL.json', {'complete': False, 'status': 'failed',
                                                 'exception_type': type(error).__name__, 'banks_verified': len(completed),
                                                 'costs': costs(), 'no_retry': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disabled unless separately admitted for all24 original resident masks.')
    parser.add_argument('--release', required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    try:
        run(args.release, args.release_sha256)
    except BaseException as error:
        print(json.dumps({'complete': False, 'exception_type': type(error).__name__, 'no_retry': True}))
        sys.exit(1)
