"""Separate, charged, trusted CPU checkpoint deserialization after whole24 gate.

No model construction, state restoration, forwards, dataset loading or selection.
Native checkpoints contain NumPy RNG state, so weights_only=False is explicit.
Model/optimizer tensors are incidentally deserialized and immediately discarded.
"""
import argparse
import gc
import os
from pathlib import Path
import resource
import time
from gate import binding, bound, fresh_output, preflight, read, require, sha, write


def accuracy(value):
    require(type(value) in (int, float) and 0 <= value <= 1, 'Saved finite accuracy in [0,1]')
    return float(value)


def epoch(value):
    require(type(value) is int and 1 <= value <= 1100, 'Saved selected epoch in original horizon')
    return value


def load_checkpoint(torch, checkpoint, cost):
    path = bound(checkpoint)
    cost['checkpoint_deserialization_attempts'].append(checkpoint)
    cost['checkpoint_file_bytes_submitted_to_deserializer'] += checkpoint['bytes']
    return torch.load(path, map_location='cpu', weights_only=False)


def extract(torch, state, cost):
    result = {k: state[k] for k in ('arm', 'seed', 'cell', 'status')}
    if state['status'] != 'complete':
        result.update(selected_VALID=None, member_VALID=None, selected_epochs=None,
                      member_global=None, selection=None, unavailable_reason=state['closure_row'])
        return result
    selected_binding = state['selected_checkpoint']
    saved = load_checkpoint(torch, selected_binding, cost)
    require(type(saved) is dict and saved.get('job') == state['job']
            and saved.get('config') == state['job_config'], 'Trusted official selected metadata binds exact job/config')
    selected_VALID = accuracy(saved['selected_VALID']); member_count = 1 if state['arm'].startswith('single') else 4
    if state['arm'] == 'independent4':
        require(saved.get('evaluation_only') is True and saved.get('candidate') == 'individual_best_bank_only',
                'Ordinary independent4 evaluation-only own-selected bank')
        member_global = saved['body_global']
        require(type(member_global) is list and len(member_global) == 4
                and all(type(x) is bool for x in member_global), 'Own bank per-body serving modes')
        bank = read(bound(state['own_bank_metric_json']))
        require(accuracy(bank['VALID']) == selected_VALID and len(bank['members']) == 4, 'Saved own-bank metric custody')
        member_VALID = [accuracy(x) for x in bank['members']]
        del saved; gc.collect()
        selected_epochs = []
        for member, checkpoint in enumerate(state['own_checkpoints']):
            own = load_checkpoint(torch, checkpoint, cost)
            require(type(own) is dict and accuracy(own['metric']) == member_VALID[member]
                    and type(own.get('global')) is bool and own['global'] == member_global[member], 'Source own-selected member metadata agrees with final bank')
            selected_epochs.append(epoch(own['epoch']))
            del own; gc.collect()
        selection = 'independently_selected_evaluation_only_bank'
    else:
        require(saved.get('evaluation_only') is not True and len(saved['member_VALID']) == member_count
                and type(saved.get('global')) is bool, 'Source coherent joint selection metadata')
        member_VALID = [accuracy(x) for x in saved['member_VALID']]
        selected_epochs = [epoch(saved['epoch'])]; member_global = [saved['global']] * member_count
        selection = 'joint_strict_first_VALID_maximum'
        del saved; gc.collect()
    result.update(selected_VALID=selected_VALID, member_VALID=member_VALID, selected_epochs=selected_epochs,
                  member_global=member_global, selection=selection, selected_checkpoint=selected_binding)
    return result


def main():
    started = time.monotonic(); cpu_started = time.process_time(); os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation', type=Path, required=True); parser.add_argument('--activation-sha256', required=True)
    args = parser.parse_args()
    cfg, gate = preflight(args.activation, args.activation_sha256, 'extract_selected_metadata')
    require(cfg.get('trusted_checkpoint_metadata_deserialization_authorized') is True
            and cfg.get('metadata_extraction_cost_charged') is True, 'Explicit separate trusted deserialization and charge authority')
    output = fresh_output(cfg); write(output / 'SAVED_METADATA_GATE.json', gate)
    cost = dict(schema='internal-be-Wiki24-metadata-extraction-cost-v1', status='failed',
                activation=gate['activation'], closure=gate['closure'], checkpoint_file_bytes_submitted_to_deserializer=0,
                checkpoint_deserialization_attempts=[],
                completed_cells=0, model_constructed=False, model_state_loaded=False, forwards=0,
                dataset_payloads_opened=0, TEST_access=False,
                deserialization='trusted torch.load map_location=cpu weights_only=False; incidental tensor allocation is charged')
    try:
        # First framework import and first predictive access are below whole24 gate.
        os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'
        import torch
        require(torch.__version__ == '2.1.2+cu118', 'Original pinned torch runtime')
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        records = []
        for state in gate['cells']:
            state_for_extract = dict(state, job_config=gate['config'])
            record = extract(torch, state_for_extract, cost)
            records.append(record)
            cost['completed_cells'] += int(state['status'] == 'complete')
        # Rehash all sources of extracted values after deserialization, without reopening scores.
        for state in gate['cells']:
            if state['status'] == 'complete':
                bound(state['selected_checkpoint'])
                for row in state.get('own_checkpoints', []): bound(row)
                if 'own_bank_metric_json' in state: bound(state['own_bank_metric_json'])
        write(output / 'METADATA_EXPORT.json', dict(schema='internal-be-Wiki24-selected-metadata-export-v1',
              gate=gate, cells=records, whole24_accounted=True, TEST_access=False, reselected=False,
              metric='official WikiCS split0 full5274 VALID accuracy, source float32 mean',
              checkpoint_loading_scope='CPU deserialization only; no model state restoration or inference',
              own_member_hash_authority='Reader captures own_best and bank-JSON digests after closure; native FREEZE authenticates selected.pt only'))
        cost['status'] = 'complete'; cost['metadata_export'] = binding(output / 'METADATA_EXPORT.json')
    except Exception as error:
        cost['error_type'] = type(error).__name__; cost['error'] = str(error)
        raise
    finally:
        cost.update(inclusive_wall_seconds=time.monotonic() - started, process_CPU_seconds=time.process_time() - cpu_started,
                    process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    output_storage_bytes_before_cost_receipt=sum(p.stat().st_size for p in output.iterdir() if p.is_file()))
        write(output / 'EXTRACTION_COST.json', cost)
    print(str(output / 'METADATA_EXPORT.json'))


if __name__ == '__main__':
    main()
