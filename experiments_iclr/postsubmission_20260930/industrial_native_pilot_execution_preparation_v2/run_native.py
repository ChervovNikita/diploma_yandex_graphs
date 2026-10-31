"""Future isolated entrypoint: qualification or the frozen nine native fits only.

No raw GraphDataset/GraphTask imports, extension mode, test metrics or final
labels. Parent must first provide a reviewed allowlist runtime tree and compact
custodian export; source sealing itself never authorizes scientific execution.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import resource
import time
import traceback
from closed_inputs import load, sha, verify_inputs, write_json
from namespace_guard import observe_namespace
from runtime_fingerprint import canonical_sha, fingerprint, verify_source_packet, verify_runtime_image


def mounted_gpu_uuid():
    """Bind kernel UUID to the one mounted physical GPU device minor.

    Torch2.4.0 source exposes no UUID property. Read-only NVIDIA proc metadata
    is public device information, never a project/archive/test-data mount.
    """
    import os
    import re
    devices = [path for path in Path('/dev').iterdir() if re.fullmatch(r'nvidia[0-9]+', path.name)]
    if len(devices) != 1:
        raise RuntimeError('Exactly one physical NVIDIA device node must be mounted')
    minor = os.minor(devices[0].stat().st_rdev)
    matches = []
    for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
        if int(fields.get('Device Minor', '-1').strip()) == minor:
            matches.append(fields.get('GPU UUID', '').strip())
    if len(matches) != 1 or not matches[0].startswith('GPU-'):
        raise RuntimeError('Mounted GPU UUID cannot be bound to read-only driver metadata')
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('isolation-probe', 'qualify', 'run-nine'))
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    parser.add_argument('--expected-gpu-uuid')
    args = parser.parse_args()
    source_root = Path('/sources')
    inputs = Path('/inputs')
    outputs = Path('/outputs')
    namespace = None
    phase = outputs / {'isolation-probe': 'isolation_probe', 'qualify': 'qualification',
                       'run-nine': 'native_grid'}[args.mode]
    phase.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(phase / 'phase.json', {'status': 'STARTED', 'mode': args.mode,
                                    'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'namespace_observation': namespace, 'test_labels': 'CLOSED'})
    try:
        namespace = observe_namespace(args.device)  # Actual OS observations before scientific imports.
        source_identity = verify_source_packet(source_root)
        image_identity = verify_runtime_image()
        if args.mode == 'isolation-probe':
            _, _, public_identity = verify_inputs(inputs, source_root / 'ROLE_POLICY.json', public_only=True)
            observed_uuid = mounted_gpu_uuid() if args.device == 'cuda:0' else None
            if observed_uuid != args.expected_gpu_uuid:
                raise RuntimeError('Probe physical GPU UUID differs from declared scope')
            receipt = {'schema': 'observed-public-only-worker-isolation-probe-v2', 'status': 'PASSED',
                       'source_identity': source_identity, 'runtime_image_identity': image_identity,
                       'public_input_identity': public_identity, 'namespace_observation': namespace,
                       'device_scope': args.device, 'mounted_gpu_uuid': observed_uuid,
                       'scientific_imports_or_updates': False, 'test_labels': 'CLOSED',
                       'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write_json(phase / 'receipt.json', receipt)
            write_json(phase / 'phase.json', {'status': 'SUCCEEDED', 'mode': args.mode,
                       'elapsed_seconds': time.perf_counter() - started,
                       'namespace_observation': namespace, 'test_labels': 'CLOSED',
                       'scientific_imports_or_updates': False})
            return
        matrix = load(source_root / 'CONFIG_MATRIX.json')
        if (matrix['seeds'] != [17, 29, 43] or matrix['configuration_order'] != ['c0', 'c1', 'c2']
                or matrix['maximum_native_fits'] != 9):
            raise ValueError('Frozen nine-cell configuration/seed matrix differs')
        roles, compact, input_identity = verify_inputs(inputs, source_root / 'ROLE_POLICY.json')
        runtime = fingerprint()
        runtime['allowlist_runtime_image'] = image_identity
        import torch
        import dgl
        import numpy as np
        import pandas as pd
        if torch.version.cuda != '11.8':
            raise RuntimeError('Candidate cu118 wheel runtime differs; no downgrade/substitution')
        parser_receipt = load(inputs / 'SOURCE_LABEL_PARITY.json')
        if parser_receipt['pandas_version'] != pd.__version__ or parser_receipt['numpy_version'] != np.__version__:
            raise RuntimeError('Source-target parser runtime differs from worker native preprocessing')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        accelerator = {'device': args.device, 'torch_cuda': torch.version.cuda,
                       'torch_version': torch.__version__, 'dgl_version': dgl.__version__}
        if device.type == 'cuda':
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not args.expected_gpu_uuid:
                raise RuntimeError('Exactly one qualified CUDA GPU/UUID is required')
            properties = torch.cuda.get_device_properties(0)
            observed_uuid = mounted_gpu_uuid()
            if observed_uuid != args.expected_gpu_uuid:
                raise RuntimeError('CUDA device UUID differs or unsupported; preserve blocker')
            accelerator.update(name=properties.name, total_memory=properties.total_memory,
                               uuid=observed_uuid, capability=[properties.major, properties.minor])
            torch.cuda.reset_peak_memory_stats(device)
        elif args.expected_gpu_uuid:
            raise RuntimeError('CPU scope has a GPU identity')
        runtime['accelerator'] = accelerator
        runtime['deterministic_algorithms'] = True
        runtime['float32_no_AMP_no_TF32'] = True
        runtime_sha = canonical_sha(runtime)
        write_json(phase / 'runtime.json', runtime)
        write_json(phase / 'input_identity.json', input_identity)
        if args.mode == 'qualify':
            from qualify_native import qualify
            result = qualify(inputs, roles, compact, matrix, device, phase)
            certificate = {'schema': 'native-complete-graph-cold-qualification-v2',
                           'status': 'PASSED', 'runtime_sha256': runtime_sha,
                           'input_identity_sha256': canonical_sha(input_identity),
                           'source_identity': source_identity, 'device_scope': args.device,
                           'qualified_configurations': ['c0', 'c1', 'c2'], 'tested_seed': 17,
                           'no_actual_warm_or_extension_certificate': True,
                           'records_sha256': {name: sha(phase / (name + '_qualification.json'))
                                              for name in matrix['configuration_order']},
                           'result': result, 'test_labels': 'CLOSED'}
            write_json(phase / 'certificate.json', certificate)
        else:
            qualification = outputs / 'qualification'
            certificate = load(qualification / 'certificate.json')
            if (certificate['status'] != 'PASSED' or certificate['runtime_sha256'] != runtime_sha
                    or certificate['input_identity_sha256'] != canonical_sha(input_identity)
                    or certificate['source_identity'] != source_identity
                    or certificate['device_scope'] != args.device
                    or certificate['qualified_configurations'] != ['c0', 'c1', 'c2']):
                raise RuntimeError('Runtime/input/source qualification binding differs')
            for name, digest in certificate['records_sha256'].items():
                if sha(qualification / (name + '_qualification.json')) != digest:
                    raise RuntimeError('Qualification record hash differs')
            from native_fit import run_nine
            result = run_nine(inputs, roles, compact, matrix, device, phase, input_identity,
                              {'sha256': runtime_sha, 'qualification_certificate_sha256': sha(qualification / 'certificate.json')},
                              source_identity)
            write_json(phase / 'summary.json', result)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        write_json(phase / 'phase.json', {'status': 'SUCCEEDED', 'mode': args.mode,
                    'elapsed_seconds': time.perf_counter() - started,
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                    'runtime_sha256': runtime_sha, 'test_labels': 'CLOSED',
                    'namespace_observation': namespace, 'source_identity': source_identity})
    except BaseException as error:
        write_json(phase / 'phase.json', {'status': 'FAILED', 'mode': args.mode,
                    'error_type': type(error).__name__, 'error': str(error),
                    'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter() - started,
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'namespace_observation': namespace, 'test_labels': 'CLOSED',
                    'failed_artifacts_preserved': True, 'dependent_fits_forbidden': True})
        raise


if __name__ == '__main__':
    main()
