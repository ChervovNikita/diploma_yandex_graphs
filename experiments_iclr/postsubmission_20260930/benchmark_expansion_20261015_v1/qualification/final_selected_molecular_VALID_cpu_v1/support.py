"""Finite CPU qualification authority; no model/data imports or execution."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parents[1]
PHASE = EXPANSION.parent
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root): raise ValueError('Qualification path escapes actual project')
    return path


def receipt(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':h.hexdigest()}


def verify(pin, root):
    path = inside(pin['path'], root)
    actual = receipt(path)
    if any(actual[k] != pin[k] for k in ('path','sha256') if k in pin) or ('bytes' in pin and actual['bytes'] != pin['bytes']):
        raise ValueError('Frozen source/evidence receipt differs: '+str(path))
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def allocation():
    if socket.gethostname() != 'anogena-2-0': raise ValueError('Dedicated allocation required before source/model/data reads')
    values = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
    if values != [UUID]: raise ValueError('Allocation physical GPU route differs; no model/data work admitted')


def admission(path, root):
    allocation()
    root = Path(root).resolve(); inside(HERE,root)
    source = json.loads((HERE/'MANIFEST.json').read_text())
    for row in source['files']:
        verify({'path':str(HERE/row['path']),**{k:row[k] for k in ('bytes','sha256')}},root)
    for row in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        verify({'path':str(PHASE/row['path_relative_to_phase']),**{k:row[k] for k in ('bytes','sha256')}},root)
    inputs = json.loads((HERE/'INPUTS.json').read_text())
    request_path = inside(path,root); request = json.loads(request_path.read_text())
    if request.get('schema') != 'genuine-selected-MolBACE-CPU-VALID-admission-v1' or request.get('status') != 'ADMITTED':
        raise PermissionError('Root-owned CPU engineering admission is required; template is not permission')
    if request.get('harness_manifest_sha256') != receipt(HERE/'MANIFEST.json')['sha256'] or request.get('INPUTS_sha256') != receipt(HERE/'INPUTS.json')['sha256']:
        raise ValueError('Admission does not bind this exact frozen harness and three states')
    if request.get('case_ids') != [r['id'] for r in inputs['cases']] or request.get('execution_device') != 'cpu' or request.get('TEST_calls') != 0 or request.get('optimizer_steps') != 0 or request.get('automatic_retries') != 0:
        raise ValueError('Exact finite CPU VALID scope required')
    if request.get('torch_threads') != 2 or request.get('torch_interop_threads') != 1:
        raise ValueError('Fixed bounded CPU thread settings required')
    for key,maximum in [('wall_seconds_per_case',600),('host_RSS_reservation_bytes',12*1024**3),('address_space_limit_bytes',32*1024**3)]:
        if type(request.get(key)) is not int or not 0 < request[key] <= maximum:
            raise ValueError('Root must bind finite reviewed resource field '+key)
    if request['address_space_limit_bytes'] < request['host_RSS_reservation_bytes']:
        raise ValueError('Address-space bound must cover root RSS reservation')
    verify(request['peer_resource_review'],root)
    verify(request['qualified_python_review'],root)
    if str(Path(sys.executable)) != request.get('python_executable'):
        raise ValueError('Root-qualified exact CPU interpreter required')
    output = inside(request['output_root'],root)
    if not output.is_relative_to(EXPANSION/'qualification'): raise ValueError('Own fresh qualification output required')
    return request, inputs, output, receipt(request_path)
