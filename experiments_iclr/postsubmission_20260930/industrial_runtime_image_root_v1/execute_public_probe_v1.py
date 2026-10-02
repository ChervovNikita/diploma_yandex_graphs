"""Root-only public extraction and observed isolation; no model/labels/GPU compute."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'protocols'))
from fetch_authorized_evidence_v2 import SSH, REMOTE

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'PUBLIC_PROBE_LAUNCH_v1.json'
if REPORT.exists():
    raise RuntimeError('Preserve the previous attempt; do not rerun this version')
manifest = HERE / 'runtime_image_v1/RUNTIME_IMAGE_MANIFEST.json'
image_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
image = json.loads(manifest.read_text())
if image['schema'] != 'candidate-allowlist-runtime-image-v2':
    raise RuntimeError('Unexpected assembled image')
names = [x['path'] for x in image['payload']]
if len(names) != len(set(names)):
    raise RuntimeError('Duplicate image payload path')
for name in names:
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise RuntimeError('Unnormalized library image member')

REMOTE_CODE = r'''
import datetime, hashlib, json, os, pathlib, subprocess, sys, time, traceback
phase = pathlib.Path(REMOTE_PHASE)
packet = phase / 'industrial_native_pilot_execution_preparation_v2'
runtime = phase / 'industrial_runtime_image_root_v1/runtime_image_v1'
root = phase / 'industrial_runtime_image_root_v1/public_probe_run_v1'
root.mkdir(mode=0o700, exist_ok=False)
record = {'schema': 'root-public-only-isolation-launch-v1',
          'start_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scientific_imports_or_updates': False, 'test_labels': 'CLOSED', 'steps': []}
def save():
    (root / 'ROOT_LAUNCH.json').write_text(json.dumps(record, indent=2) + '\n')
def step(name, argv, cap):
    begin = time.monotonic()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=cap,
                            stdin=subprocess.DEVNULL, close_fds=True)
    (root / (name + '.stdout')).write_text(result.stdout)
    (root / (name + '.stderr')).write_text(result.stderr)
    record['steps'].append({'name': name, 'argv': argv, 'exit_code': result.returncode,
                            'seconds': time.monotonic() - begin})
    save()
    if result.returncode:
        raise RuntimeError(name + ' exited ' + str(result.returncode))
    return result.stdout
save()
try:
    uuid = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    visible = step('gpu_route', ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], 20)
    if visible.strip().splitlines() != [uuid]:
        raise RuntimeError('Authorized one-GPU route mismatch')
    import xml.etree.ElementTree as ET
    xml = ET.fromstring(step('gpu_device', ['nvidia-smi', '-q', '-x'], 20))
    gpus = xml.findall('gpu')
    if len(gpus) != 1 or gpus[0].findtext('uuid') != uuid or gpus[0].findtext('minor_number') != '7':
        raise RuntimeError('Authorized physical device identity changed')
    if hashlib.sha256((runtime / 'RUNTIME_IMAGE_MANIFEST.json').read_bytes()).hexdigest() != EXPECTED_IMAGE_SHA:
        raise RuntimeError('Assembled runtime manifest changed')
    sys.path.insert(0, str(packet))
    from runtime_fingerprint import verify_source_packet
    record['source_identity'] = verify_source_packet(packet)
    record['runtime_manifest_sha256'] = EXPECTED_IMAGE_SHA
    record['library_payload_entries'] = 26061
    record['gpu_uuid'] = uuid
    record['gpu_device'] = '/dev/nvidia7'
    save()
    inputs = root / 'public_inputs'
    step('public_export', [sys.executable, str(packet / 'custodian_export.py'), 'public',
                          '--archive', str(phase / 'industrial_dataset_acquisition_root_v1/tolokers2_run01/tolokers-2.zip'),
                          '--output', str(inputs)], 60)
    outputs = root / 'probe_outputs'
    outputs.mkdir(mode=0o700)
    step('isolation_probe', [sys.executable, str(packet / 'launch_isolated.py'), 'isolation-probe',
                            '--runtime-image', str(runtime), '--inputs', str(inputs),
                            '--outputs', str(outputs), '--namespace-root', str(root / 'namespace_probe'),
                            '--device', 'cuda:0', '--gpu-device', '/dev/nvidia7', '--gpu-uuid', uuid], 600)
    receipt = json.loads((outputs / 'isolation_probe/receipt.json').read_text())
    if receipt.get('status') != 'PASSED' or receipt.get('scientific_imports_or_updates') is not False:
        raise RuntimeError('Actual probe receipt does not authorize the next stage')
    record['status'] = 'PUBLIC_ONLY_ISOLATION_PASSED'
    record['receipt_sha256'] = hashlib.sha256((outputs / 'isolation_probe/receipt.json').read_bytes()).hexdigest()
except BaseException:
    record['status'] = 'FAILED_NO_DEPENDENT_LABEL_OR_MODEL_PHASE'
    record['error'] = traceback.format_exc()
    raise
finally:
    record['terminal_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
'''
code = ('REMOTE_PHASE = ' + repr(REMOTE) + '\nEXPECTED_IMAGE_SHA = ' + repr(image_sha) + '\n' + REMOTE_CODE)
python = REMOTE + '/envs/industrial_native_py312_v1/bin/python'
command = 'cd ' + shlex.quote(REMOTE) + ' && . ' + shlex.quote(REMOTE + '/protocols/repo_env.sh') + ' && ' + shlex.quote(python) + ' -'
start = datetime.now(timezone.utc).isoformat()
(HERE / 'PUBLIC_PROBE_PLAN_v1.json').write_text(json.dumps({
    'start_UTC': start, 'ssh_destination': SSH[-1], 'remote_command': command,
    'remote_code_sha256': hashlib.sha256(code.encode()).hexdigest(), 'runtime_manifest_sha256': image_sha,
    'public_only': True, 'GPU_compute': False, 'labels_decoded': False,
    'GPU_science_serializes_after_live_study': True}, indent=2) + '\n')
(HERE / 'PUBLIC_PROBE_REMOTE_CODE_v1.txt').write_text(code)
result = subprocess.run(SSH + [command], input=code, text=True, capture_output=True)
REPORT.write_text(json.dumps({'start_UTC': start, 'terminal_UTC': datetime.now(timezone.utc).isoformat(),
                             'exit_code': result.returncode, 'stdout': result.stdout,
                             'stderr': result.stderr}, indent=2) + '\n')
print(json.dumps({'exit_code': result.returncode, 'report': str(REPORT)}))
raise SystemExit(result.returncode)
