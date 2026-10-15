"""Read only paid TRAIN/VALID role support; no TEST label indexing or model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT / 'experiments_iclr/postsubmission_20260930'
E = PHASE / 'benchmark_expansion_20261015_v1'
PINS = {
    'expanded_study_design_v1/NATIVE_SETTINGS.json': '1cadbc4e36a57d3740e4298a52c21d7f87c6e1397d13ff86ce44729f7335fe05',
    'node_all_split_provider_audit_v1/attempt01/RESULT.json': '036c3c9bd6148da2cf8a2220fcf2e4984ca81f2264fd48f68e9f5421f2b9559b',
}
UTILS = PHASE / 'reference_wrapper_redesign_20261013_v1/native_sources/polyformer/node_classification/utils.py'
UTILS_SHA = '2db2f0bbf42bcdf3c49715d4bcb5a197cc592521ce957395f2e7aea75268732b'
CONFIG = {'K': 6, 'attn_lr': .0005, 'attn_wd': .0001, 'd_ffn': 64, 'dprate': 0.,
          'dropout': .30000000000000004, 'hidden': 128, 'lr': .0005, 'multi': 2.,
          'n_head': 16, 'nlayer': 1, 'q': 2., 'weight_decay': .001}


def require(ok, message):
    if not ok: raise ValueError(message)


def pin(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def main():
    # The outer literal-route wrapper must also guard before reading this file.
    require(socket.gethostname() == 'anogena-2-0' and subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() ==
        ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Dedicated allocation required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'No CUDA visibility permitted')
    require(Path(sys.prefix).resolve() == (ROOT / '.venv').resolve(), 'Original repository virtualenv required')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output).resolve()
    require(output.is_relative_to(E / 'analysis_outputs') and not output.exists(), 'Fresh analysis output required')
    started = time.monotonic(); sources = []
    for path, expected in PINS.items():
        receipt = pin(E / path); require(receipt['sha256'] == expected, 'Frozen metadata changed'); sources.append(receipt)
    utils_pin = pin(UTILS); require(utils_pin['sha256'] == UTILS_SHA, 'Native splitter source changed')
    settings = json.loads((E / 'expanded_study_design_v1/NATIVE_SETTINGS.json').read_text())
    preset = [r for r in settings['rows'] if r['preset'] == 'polyformer-pubmed-cheb']
    require(len(preset) == 1 and preset[0]['native_config'] == CONFIG, 'Exact current PubMed-Cheb preset required')
    audit = json.loads((E / 'node_all_split_provider_audit_v1/attempt01/RESULT.json').read_text())
    require(audit['status'] == 'passed' and audit['native_models_constructed'] == audit['optimizer_steps'] == 0 and
            audit['CUDA_context_initialized'] is False, 'Actual paid provider authority required')
    rows = [r for r in audit['records'] if r['dataset'] == 'pubmed' and r['split'] == '0']
    require(len(rows) == 1 and rows[0]['source_split_seed'] == 1941488137 and
            rows[0]['repeat_materialization_bit_exact'] is True, 'Frozen native split0 required')
    row = rows[0]; data_path = Path(row['native_data_receipt']['path']).resolve()
    require(data_path.is_relative_to(PHASE), 'Existing provider container must stay in project')
    data_pin = pin(data_path); require(data_pin == row['native_data_receipt'], 'Native provider container changed')
    roles_path = E / 'node_all_split_provider_audit_v1/attempt01/pubmed_TRAIN_VALID_roles.pt'
    roles_pin = pin(roles_path)
    import torch
    torch.set_num_threads(2); require(not torch.cuda.is_initialized(), 'CUDA context must remain absent')
    packet = torch.load(roles_path, map_location='cpu', weights_only=False)
    require(packet['schema'] == 'native-node-TRAIN-VALID-roles-v1' and packet['dataset'] == 'pubmed' and
            packet['TEST_ids_or_labels_exported'] is False and packet['source_data'] == data_pin and
            packet['source_utils'] == utils_pin, 'Paid TRAIN/VALID-only role provenance differs')
    split = packet['splits']['0']; require(split['source_split_seed'] == 1941488137, 'Source split seed differs')
    ids = {role: split[role+'_ids'].detach().cpu() for role in ('TRAIN', 'VALID')}
    masks = {}
    for role in ('TRAIN', 'VALID'):
        selected = ids[role]
        require(selected.dtype == torch.long and selected.ndim == 1 and len(selected) == row[role+'_count'] and
                torch.unique(selected).numel() == len(selected) and selected.min().item() >= 0 and
                selected.max().item() < row['nodes'], 'Complete unique role IDs required')
        mask = torch.zeros(row['nodes'], dtype=torch.bool); mask[selected] = True; masks[role] = mask
        name = 'train_mask' if role == 'TRAIN' else 'val_mask'
        require(hashlib.sha256(mask.numpy().tobytes()).hexdigest() == row['mask_sha256'][name], 'Paid source role mask differs')
    require(not (masks['TRAIN'] & masks['VALID']).any().item(), 'TRAIN/VALID overlap')
    # Existing native container includes the provider's full graph. Access only
    # the already-proved TRAIN/VALID indices; no TEST-specific tensor is loaded
    # and no unselected/TEST label is inspected, reduced, scored or exported.
    native_packet = torch.load(data_path, map_location='cpu', weights_only=False)
    obj = native_packet[0]; labels = obj['y'] if isinstance(obj, dict) else obj.y
    counts, role_digests = {}, {}
    for role in ('TRAIN', 'VALID'):
        chosen = labels.reshape(-1)[ids[role]].detach().cpu().clone()
        require(chosen.dtype == torch.long and chosen.min().item() >= 0 and chosen.max().item() < 3,
                'Native role class range differs')
        counts[role] = torch.bincount(chosen, minlength=3).tolist()
        role_digests[role] = {'ids_sha256': hashlib.sha256(ids[role].numpy().tobytes()).hexdigest(),
                              'labels_sha256': hashlib.sha256(chosen.numpy().tobytes()).hexdigest()}
    require(counts['TRAIN'] == row['TRAIN_per_class'] == [3943,3943,3943], 'Native TRAIN support changed')
    require(not torch.cuda.is_initialized(), 'CUDA context unexpectedly initialized')
    result = {'schema': 'native-pubmed-cheb-split0-TRAIN-VALID-class-support-v1', 'status': 'complete',
        'preset': 'polyformer-pubmed-cheb', 'source_split_seed': 1941488137, 'split': 0,
        'native_config': CONFIG, 'counts': counts, 'role_digests': role_digests,
        'all_three_VALID_classes_supported': all(n > 0 for n in counts['VALID']),
        'provider_result': sources[1], 'source_utils': utils_pin, 'source_data': data_pin,
        'paid_TRAIN_VALID_roles': roles_pin, 'script': pin(__file__), 'sources': sources,
        'split_generation': 'Reuses paid twice-materialized native split0 proof; exact role mask hashes rechecked.',
        'native_container_deserialized': True, 'feature_tensors_used': False,
        'TEST_label_indexing': False, 'TEST_specific_tensor_loaded': False, 'TEST_scores': False,
        'models_constructed': 0, 'task_forwards': 0, 'optimizer_updates': 0, 'CUDA_context_initialized': False,
        'full_campaign_GO': False, 'scientific_fit': False, 'wall_seconds': time.monotonic()-started}
    output.mkdir(parents=True)
    with (output / 'RESULT.json').open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'status': result['status'], 'counts': counts, 'all_three_VALID_classes_supported':
        result['all_three_VALID_classes_supported'], 'RESULT': pin(output / 'RESULT.json')}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
