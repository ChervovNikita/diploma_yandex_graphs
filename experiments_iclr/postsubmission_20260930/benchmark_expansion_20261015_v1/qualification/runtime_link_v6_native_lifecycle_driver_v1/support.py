"""Standard-library custody helpers for the one admitted NCNC lifecycle attempt."""
from pathlib import Path
import hashlib
import json
import os
import socket
import subprocess

RUNTIME_SHA = '73e4140da736f597572968ca32a50860e16541934515090c3f98c5409e5e354f'
PLAN_SHA = '9f3fd4a67cd25de540fc6c1d20f5842d837136dbbf1c2abc69cbbb2446281bd1'
METHOD_SHA = 'e7fb265ea3c3650d7ba4c24157adda04b20cdba88ad46ebc54a498b9b892b0c1'
ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
LIMITS = {'native_constructions_attempted':3, 'TRAIN_epochs_started':2,
          'updates_attempted':34, 'optimizer_attempts':34, 'reverse_attempts':34,
          'VALID_attempts':4, 'mean_exports_attempted':2}


def require(value, message):
    if not value:
        raise ValueError(message)


def guard():
    require(socket.gethostname() == 'anogena-2-0', 'Wrong allocation hostname')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
        text=True, timeout=10).splitlines() == [UUID], 'Wrong sole allocation GPU UUID')


def inside(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Path outside authorized repository')
    return path


def receipt(path):
    path = Path(path); raw = path.read_bytes()
    return {'path':str(path), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def package(folder, expected):
    folder = inside(folder); manifest = folder/'MANIFEST.json'
    require(receipt(manifest)['sha256'] == expected, 'Frozen manifest changed: '+str(folder))
    result = [receipt(manifest)]
    for pin in json.loads(manifest.read_text())['files']:
        path = inside(folder/pin['path'])
        require(path.is_relative_to(folder), 'Manifest member escaped source package')
        now = receipt(path)
        require(all(now[k] == pin[k] for k in ('bytes','sha256')), 'Frozen member changed: '+str(path))
        result.append(now)
    return result


def sources(here, driver_sha):
    expansion = here.parents[1]
    rows = package(here, driver_sha)
    for folder, expected in [('runtime_link_v6',RUNTIME_SHA),
          ('qualification/runtime_link_v6_native_lifecycle_v1',PLAN_SHA),
          ('interaction_link_ncnc_v1',METHOD_SHA)]:
        rows += package(expansion/folder, expected)
    return rows


def load_admission(path, here):
    admission_path = inside(path); value = json.loads(admission_path.read_text())
    require(value.get('schema') == 'runtime-link-v6-native-lifecycle-root-admission-v1', 'Explicit root admission required')
    require(value.get('published') is True and len(value.get('publication_commit','')) == 40
        and all(c in '0123456789abcdef' for c in value['publication_commit']), 'Publication receipt must precede admission')
    require(value.get('runtime_manifest_sha256') == RUNTIME_SHA and value.get('plan_manifest_sha256') == PLAN_SHA,
        'Admission does not bind frozen runtime/plan')
    require(value.get('device') in ('cpu','cuda:0') and value.get('maximum_attempts') == 1,
        'Choose one device attempt, no retries')
    require(value.get('limits') == LIMITS and value.get('TEST_access') is False,
        'Fixed work bounds and closed TEST required')
    require(value.get('automatic_retries') == 0 and value.get('full_campaign_GO') is False,
        'This admission is the finite engineering gate only')
    sources(here,value['driver_manifest_sha256'])
    proposal = here.parent/'runtime_link_v6_native_lifecycle_v1'
    job_path = proposal/('JOB_CPU.json' if value['device']=='cpu' else 'JOB_CUDA.json')
    require(receipt(job_path)['sha256'] == value['job_sha256'], 'Admission JOB bytes changed')
    return value, job_path
