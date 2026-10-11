"""One root-invoked full-fit binding after the physical acquisition queue gate."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from fullrun_owner_v1 import R, H, S, UUID, LIMITS, SEEDS, ARMS, SOURCE_MANIFEST_SHA256, route, item, read, write, stamp, cuda_processes


def bind(root_admission_path):
    route()
    root_admission_path = Path(root_admission_path).resolve()
    root = read(root_admission_path)
    assert root_admission_path == H / 'ROOT_PIPELINE_ADMISSION_V1.json'
    assert root['status'] == 'ROOT_FIXED_FiLM_PIPELINE_ADMITTED'
    gate = read(H / 'PHYSICAL_ACQUISITION_CLOSURE_V1.json')
    assert gate['physically_closed'] is True and gate['outcome_gate'] is False
    for row in gate['receipts']: assert item(row['path']) == row
    assert all(not Path('/proc', str(pid)).exists() and pid not in cuda_processes() for pid in gate['acquisition_PIDs'])
    assert item(S / 'MANIFEST.json')['sha256'] == SOURCE_MANIFEST_SHA256
    for row in read(S / 'MANIFEST.json')['files']:
        actual = item(S / row['path']); assert (actual['bytes'], actual['sha256']) == (row['bytes'], row['sha256'])
    cfg = read(H / 'READINESS_CONFIG_V1.json')
    assert cfg['seeds'] == SEEDS and cfg['study_kind'] == 'native_PyG_FiLM_WikiCS'
    owner = read(H / 'readiness_owner_v1/OWNER_END.json')
    readiness = read(H / 'readiness_v1/READINESS_RESULT.json')
    assert owner['successful_readiness'] and owner['direct_child_wait'] and owner['child_absent'] and owner['owned_cuda_pid_absent']
    assert not Path('/proc', str(owner['child_pid'])).exists() and owner['child_pid'] not in cuda_processes()
    assert readiness['complete'] and readiness['parameters'] == 354146 and readiness['native_film_conv_completed_calls'] == 24
    assert not readiness['scientific_fit_or_selection'] and not readiness['VALID_payload_read'] and not readiness['TEST_access']
    for row in cfg['source_receipts'].values(): assert item(row['path']) == row
    for row in cfg['safe_roles'].values(): assert item(row['path']) == row
    assert cfg['native_repo'] == str(S / 'deps/native') and cfg['device'] == 'cuda:0'
    fingerprint = read(H / 'RUNTIME_FINGERPRINT_V1.json')
    assert fingerprint['hostname'] == socket.gethostname() and fingerprint['gpu_uuid'] == UUID
    for row in fingerprint['python_sources'].values(): assert item(row['path']) == row
    assert fingerprint['driver'] == subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=15).strip()
    write(H / 'SCIENTIFIC_RUNTIME_FINGERPRINT_V1.json', dict(fingerprint,
        python_executable=str(R / '.venv/bin/python'), real_executable_path=str((R / '.venv/bin/python').resolve()), UTC=stamp()))
    resources = dict(UTC=stamp(), hostname=socket.gethostname(), sole_GPU_UUID=UUID,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader'], text=True, timeout=15).splitlines(),
        CUDA_process_bytes=cuda_processes(), system_MemAvailable_bytes=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024,
        remaining_cohort_overlap_possible=True, isolated_benchmark=False)
    write(H / 'RESOURCE_OBSERVATION_BEFORE_FULLFIT_V1.json', resources)
    cfg['family_limits'] = dict(LIMITS)
    canonical = hashlib.sha256(json.dumps({key: value for key, value in cfg.items() if key != 'root_admission'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    compatibility = dict(readiness_cuda_allocated_bytes=readiness['cuda_peak_allocated_bytes'],
        readiness_cuda_reserved_bytes=readiness['cuda_peak_reserved_bytes'], readiness_RSS_bytes=readiness['process_peak_rss_bytes'],
        within_16GiB_caps=all(readiness[key] < LIMITS[cap] for key, cap in (
            ('cuda_peak_allocated_bytes', 'cuda_bytes'), ('cuda_peak_reserved_bytes', 'cuda_bytes'), ('process_peak_rss_bytes', 'rss_bytes'))),
        full_family_time_or_memory_guaranteed=False)
    assert compatibility['within_16GiB_caps']
    admission = dict(status='HOST_GPU_SOURCE_DATA_ADMITTED', action='fixed_family_fit', study_kind=cfg['study_kind'],
        hostname=socket.gethostname(), gpu_uuid=UUID, UTC=stamp(), banks=9, optimizer_bundles=18, native_body_records=18,
        seeds=SEEDS, arms=ARMS, runtime_fingerprint_sha256=item(H / 'SCIENTIFIC_RUNTIME_FINGERPRINT_V1.json')['sha256'],
        source_sha256={str(Path(row['path']).resolve()): row['sha256'] for row in cfg['source_receipts'].values()},
        safe_role_sha256={key: row['sha256'] for key, row in cfg['safe_roles'].items()}, science_config_sha256=canonical,
        root_pipeline_admission=item(root_admission_path), source_manifest=item(S / 'MANIFEST.json'),
        physical_acquisition_closure=item(H / 'PHYSICAL_ACQUISITION_CLOSURE_V1.json'),
        resource_observation=item(H / 'RESOURCE_OBSERVATION_BEFORE_FULLFIT_V1.json'),
        readiness=item(H / 'readiness_v1/READINESS_RESULT.json'), readiness_owner=item(H / 'readiness_owner_v1/OWNER_END.json'),
        readiness_cap_compatibility=compatibility, finite_caps=LIMITS, partial_predictive_metrics_read=False, TEST_access=False,
        retry=False, seed_subset=False, utility_or_novelty_gate=False)
    write(H / 'SCIENTIFIC_ADMISSION_V1.json', admission)
    cfg['root_admission'] = item(H / 'SCIENTIFIC_ADMISSION_V1.json')
    write(H / 'SCIENTIFIC_CONFIG_V1.json', cfg)
    write(H / 'FULLFIT_BINDING_V1.json', dict(config=item(H / 'SCIENTIFIC_CONFIG_V1.json'), admission=item(H / 'SCIENTIFIC_ADMISSION_V1.json'),
        runtime=item(H / 'SCIENTIFIC_RUNTIME_FINGERPRINT_V1.json'), resource_observation=admission['resource_observation'],
        banks=9, optimizer_bundles=18, native_body_records=18, calibration_attempts=45, TEST_access=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--admission', type=Path, required=True)
    bind(parser.parse_args().admission)
