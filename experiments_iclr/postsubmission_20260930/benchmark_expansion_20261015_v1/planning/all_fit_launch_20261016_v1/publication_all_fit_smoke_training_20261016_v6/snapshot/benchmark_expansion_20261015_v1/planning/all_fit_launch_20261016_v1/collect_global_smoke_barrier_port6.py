"""Collect the frozen three-worker smoke proof and deliver its exact PASS marker.

Nothing remote happens on import. Invoke only with root's frozen binding file.
This coordinator never starts or restarts a queue owner or numerical job.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zlib

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
PHASE_REL = 'experiments_iclr/postsubmission_20260930'
EXPANSION_REL = PHASE_REL + '/benchmark_expansion_20261015_v1'
PACKAGES = {'port': 'server51_campaign_port_v6', 'execution': 'campaign_execution_20261016_v7',
            'smoke': 'qualification/all_fit_smoke_20261016_v6'}
ROOTS = {'allocation': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs',
         'server51': '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'}
WORKERS = ('allocation/gpu0', 'server51/gpu0', 'server51/gpu1')
CHUNK = 24576
UPLOAD_CHUNK = 1024
COMMAND_LIMIT = 5000
STAGED_ENGINES = set()


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def receipt(path):
    path = Path(path).resolve()
    return {'path': str(path), **identity(path.read_bytes())}


def exact_write(path, raw):
    """Never replace prior evidence, including after a partially successful retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, 'Existing proof differs: ' + str(path))
        return
    temp = path.with_name(path.name + '.part-' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    except FileExistsError:
        require(path.read_bytes() == raw, 'Concurrent proof differs: ' + str(path))
    finally:
        temp.unlink()


def check_pin(path, expected):
    actual = receipt(path)
    require(all(actual[key] == expected[key] for key in ('bytes', 'sha256')), 'Changed file: ' + str(path))
    return actual


def source_manifest(folder, expected):
    check_pin(folder / 'MANIFEST.json', expected)
    for row in json.loads((folder / 'MANIFEST.json').read_text())['files']:
        path = (folder / row['path']).resolve()
        require(path.is_relative_to(folder.resolve()), 'Source member escapes package')
        check_pin(path, row)


def load_bindings(path, expected_sha):
    path = Path(path).resolve()
    require(path.is_relative_to(HERE), 'Binding file must be in the launch planning folder')
    raw = path.read_bytes()
    require(identity(raw)['sha256'] == expected_sha, 'Binding configuration changed')
    config = json.loads(raw)
    require(config.get('schema') == 'all-fit-global-smoke-coordinator-bindings-v1', 'Wrong binding schema')
    require(set(config['sources']) == set(PACKAGES), 'Bind all three final source manifests')
    rows = config['workers']
    require(len(rows) == 3 and {row['worker'] for row in rows} == set(WORKERS), 'Bind exactly three workers')
    for kind, folder in PACKAGES.items():
        source_manifest(E / folder, config['sources'][kind])
    for row in rows:
        root = Path(ROOTS[row['worker'].split('/')[0]])
        expected = root / EXPANSION_REL / PACKAGES['execution'] / ('PLAN-' + row['worker'].replace('/', '_') + '.json')
        require(row['plan']['path'] == str(expected), 'Plan path is not the explicit execution-v4 worker plan')
        require(type(row['plan']['bytes']) is int and row['plan']['bytes'] > 0 and
                isinstance(row['plan']['sha256'], str) and len(row['plan']['sha256']) == 64, 'Exact plan receipt required')
    return config, raw


# All deliberate remote operations stay under the two literal authorized repos.
# The existing transport additionally checks hostname and the complete GPU UUID list.
REMOTE = r'''
import base64,hashlib,json,os,uuid,zlib
from pathlib import Path
root=Path(REQUEST['root']); E=root/REQUEST['expansion_relative']
assert root.resolve()==root
packages=REQUEST['packages']; cfg=REQUEST['config']
queue=E/packages['execution']; transfer=queue/'.global_smoke_barrier_transfer'
def need(ok,msg):
 if not ok: raise ValueError(msg)
def inside(value):
 p=Path(value).resolve();need(p.is_relative_to(root),'Path escapes authorized project');return p
def pin(p):
 p=inside(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify(p,expected):
 p=inside(p);a=pin(p);need(all(a[k]==expected[k] for k in ('bytes','sha256')),'Changed receipt: '+str(p));return p
def transport_file(name):
 need(not transfer.is_symlink(),'Transport directory may not be a symlink')
 folder=inside(transfer);folder.mkdir(exist_ok=True)
 target=folder/name;need(not target.is_symlink(),'Transport file may not be a symlink');return inside(target)
def probe():
 for kind,rel in packages.items():
  folder=E/rel;manifest=verify(folder/'MANIFEST.json',cfg['sources'][kind])
  for row in json.loads(manifest.read_text())['files']:
   member=inside(folder/row['path']);need(member.is_relative_to(folder),'Source member escapes package');verify(member,row)
 result=[]
 for entry in cfg['workers']:
  worker=entry['worker']
  if worker.split('/')[0]!=REQUEST['host']:continue
  pp=verify(entry['plan']['path'],entry['plan']);plan=json.loads(pp.read_text())
  need(plan['worker']==worker and plan['project_root']==str(root),'Wrong active worker/root')
  need(plan['source_manifest']==pin(queue/'MANIFEST.json'),'Wrong execution source')
  need(plan['smoke_source_manifest']==pin(E/packages['smoke']/'MANIFEST.json'),'Wrong smoke source')
  need(plan['reporting_entrypoint']==pin(E/packages['port']/'reporting/runtime_entry.py'),'Wrong port source')
  state=queue/'runs'/worker.replace('/','_')
  need(plan['state_root']==str(state) and plan['global_smoke_pass']==str(queue/'GLOBAL_SMOKE_PASS.json'),'Wrong state/barrier path')
  owner=state/'OWNER.json'; summary=state/'SMOKE_SUMMARY.json'
  if not owner.exists() or not summary.exists():
   result.append({'worker':worker,'status':'pending'});continue
  own=json.loads(owner.read_text())
  need(own['plan']==pin(pp) and own['source_manifest']==plan['source_manifest'] and own['inventory']==plan['inventory'],'Owner source/plan/inventory differs')
  summary_pin=pin(summary);value=json.loads(summary.read_text())
  need(value.get('schema') in ('allfit-worker-smoke-summary-v1','all-fit-worker-smoke-summary-v1'),'Wrong summary schema')
  need(value.get('status')=='PASS' and value.get('worker')==worker,'Existing summary is not worker PASS')
  need(value['inventory']==plan['inventory'],'Summary/plan inventory differs')
  inventory_pin=pin(verify(plan['inventory']['path'],plan['inventory']))
  result.append({'worker':worker,'status':'PASS','plan':pin(pp),'inventory':inventory_pin,'summary':summary_pin})
 return sorted(result,key=lambda row:row['worker'])
op=REQUEST['operation']
if op=='probe': output={'workers':probe()}
elif op=='snapshot':
 expected=REQUEST['file']; p=verify(expected['path'],expected);raw=p.read_bytes()
 packet=zlib.compress(raw,6)
 target=transport_file(expected['sha256']+'.zlib')
 if target.exists():need(target.read_bytes()==packet,'Existing transport snapshot differs')
 else:
  temp=target.with_name(target.name+'.part-'+uuid.uuid4().hex)
  with temp.open('xb') as f:f.write(packet);f.flush();os.fsync(f.fileno())
  try:os.link(temp,target)
  except FileExistsError:need(target.read_bytes()==packet,'Concurrent transport snapshot differs')
  finally:temp.unlink()
 output={'packet':pin(target),'uncompressed':expected}
elif op=='chunk':
 p=verify(REQUEST['packet']['path'],REQUEST['packet']);offset=REQUEST['offset'];limit=REQUEST['limit']
 need(type(offset)is int and offset>=0 and 0<limit<=24576,'Invalid transport range')
 with p.open('rb') as f:f.seek(offset);block=f.read(limit)
 output={'offset':offset,'data':base64.b64encode(block).decode()}
elif op=='upload':
 packet=REQUEST['packet'];target=transport_file(packet['sha256']+'.upload')
 block=base64.b64decode(REQUEST['data'],validate=True);offset=REQUEST['offset']
 need(len(block)<=24576 and type(offset)is int and offset>=0 and offset+len(block)<=packet['bytes'],'Invalid upload range')
 fd=os.open(target,os.O_RDWR|os.O_CREAT,0o600)
 with os.fdopen(fd,'r+b') as f:
  f.seek(0,2);size=f.tell();need(size>=offset,'Upload gap')
  if size>offset:
   f.seek(offset);need(f.read(len(block))==block,'Existing upload bytes differ')
  else:f.write(block);f.flush();os.fsync(f.fileno())
 output={'next_offset':offset+len(block)}
elif op=='commit':
 actual=probe();need(actual==REQUEST['ready_workers'],'Source, inventory or summary changed before delivery')
 marker=queue/'GLOBAL_SMOKE_PASS.json';expected=REQUEST['marker']
 if marker.exists():verify(marker,expected);output={'marker':pin(marker),'existing_identical':True}
 else:
  packet=REQUEST['packet'];p=verify(transfer/(packet['sha256']+'.upload'),packet)
  raw=zlib.decompress(p.read_bytes());need(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],'Marker packet identity differs')
  value=json.loads(raw);need(value.get('schema')=='all-fit-global-smoke-pass-v1' and value.get('status')=='PASS' and value.get('fit_count')==6460,'Wrong global PASS marker')
  need(value['source_manifest']['sha256']==cfg['sources']['smoke']['sha256'],'Wrong marker validator source')
  for row in actual:
   need(value['inventory_sha256s'][row['worker']]==row['inventory']['sha256'] and value['worker_summary_sha256s'][row['worker']]==row['summary']['sha256'],'Marker worker binding differs')
  temp=marker.with_name(marker.name+'.part-'+uuid.uuid4().hex)
  with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.chmod(temp,0o444)
  try:os.link(temp,marker)
  except FileExistsError:verify(marker,expected)
  finally:temp.unlink()
  output={'marker':pin(marker),'existing_identical':False}
else:raise ValueError('Unknown bounded operation')
print(json.dumps(output,sort_keys=True))
'''


def engine_path(host):
    return str(Path(ROOTS[host]) / EXPANSION_REL / PACKAGES['execution'] /
               '.global_smoke_barrier_transfer' / ('engine-' + identity(REMOTE.encode())['sha256'] + '.zlib'))


def stage_engine(transport, host):
    """Copy the exact small metadata engine once; no frozen source is replaced."""
    raw = zlib.compress(REMOTE.encode(), 9)
    target = engine_path(host)
    for offset in range(0, len(raw), 3000):
        block = raw[offset:offset + 3000]
        code = '''import base64,hashlib,json,os
from pathlib import Path
p=Path(TARGET);root=Path(ROOT)
assert p.resolve().is_relative_to(root) and not p.is_symlink() and not p.parent.is_symlink()
p.parent.mkdir(exist_ok=True)
b=base64.b64decode(DATA,validate=True);offset=OFFSET
with p.open('a+b') as f:
 f.seek(0,2);size=f.tell()
 if size>offset:
  f.seek(offset);assert f.read(len(b))==b,'Existing coordinator engine bytes differ'
 else:
  assert size==offset;f.write(b);f.flush();os.fsync(f.fileno())
 if offset+len(b)==TOTAL:
  f.seek(0);saved=f.read();assert len(saved)==TOTAL and hashlib.sha256(saved).hexdigest()==SHA
print(json.dumps({'next_offset':offset+len(b)}))
'''.replace('TARGET', repr(target)).replace('ROOT', repr(ROOTS[host])) \
            .replace('OFFSET', str(offset)).replace('TOTAL', str(len(raw))) \
            .replace('SHA', repr(identity(raw)['sha256'])).replace('DATA', repr(base64.b64encode(block).decode()))
        require(len(code.encode()) <= COMMAND_LIMIT, 'Engine staging command exceeds bounded transport limit')
        require(json.loads(transport.call(code, timeout=60))['next_offset'] == offset + len(block),
                'Wrong coordinator-engine staging acknowledgement')


def invocation_code(host, request):
    packet = zlib.compress(REMOTE.encode(), 9)
    payload = base64.b64encode(zlib.compress(json.dumps(request, separators=(',', ':')).encode(), 9)).decode()
    code = '''import base64,hashlib,json,zlib
from pathlib import Path
p=Path(TARGET);root=Path(ROOT)
assert p.resolve().is_relative_to(root) and not p.is_symlink()
b=p.read_bytes();assert len(b)==TOTAL and hashlib.sha256(b).hexdigest()==PACKET_SHA
source=zlib.decompress(b);assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
REQUEST=json.loads(zlib.decompress(base64.b64decode(PAYLOAD,validate=True)))
exec(compile(source,'<pinned-global-smoke-coordinator-remote>','exec'))
'''.replace('TARGET', repr(engine_path(host))).replace('ROOT', repr(ROOTS[host])) \
        .replace('TOTAL', str(len(packet))).replace('PACKET_SHA', repr(identity(packet)['sha256'])) \
        .replace('SOURCE_SHA', repr(identity(REMOTE.encode())['sha256'])).replace('PAYLOAD', repr(payload))
    require(len(code.encode()) <= COMMAND_LIMIT, 'Request exceeds bounded transport command limit')
    return code


def remote(host, config, operation, **extra):
    # These imports do not connect; .call is the sole remote entry point.
    folders = (str(E), str(E / 'planning/three_gpu_sync_20261015_v1'))
    for folder in folders:
        if folder not in sys.path:
            sys.path.insert(0, folder)
    import remote_control
    import remote51
    transport = remote_control if host == 'allocation' else remote51
    request = {'host': host, 'root': ROOTS[host], 'expansion_relative': EXPANSION_REL,
               'packages': PACKAGES, 'config': config, 'operation': operation, **extra}
    if host not in STAGED_ENGINES:
        stage_engine(transport, host)
        STAGED_ENGINES.add(host)
    code = invocation_code(host, request)
    return json.loads(transport.call(code, timeout=60))


def ready_workers(probes):
    rows = [row for host in ROOTS for row in probes[host]['workers']]
    require(len(rows) == 3 and {row['worker'] for row in rows} == set(WORKERS), 'Wrong probe worker set')
    if any(row['status'] == 'pending' for row in rows):
        return None
    require(all(row['status'] == 'PASS' for row in rows), 'Non-PASS worker')
    rows.sort(key=lambda row: WORKERS.index(row['worker']))
    require(rows[1]['inventory'] == rows[2]['inventory'], 'Server51 workers use different inventories')
    return rows


def fetch(host, config, source, destination):
    if destination.exists():
        check_pin(destination, source); return
    response = remote(host, config, 'snapshot', file=source)
    require(response['uncompressed'] == source, 'Snapshot input changed')
    packet = response['packet']; compressed = bytearray()
    for offset in range(0, packet['bytes'], CHUNK):
        value = remote(host, config, 'chunk', packet=packet, offset=offset, limit=CHUNK)
        require(value['offset'] == offset, 'Wrong returned chunk')
        block = base64.b64decode(value['data'], validate=True)
        require(len(block) == min(CHUNK, packet['bytes'] - offset), 'Short transport chunk')
        compressed.extend(block)
    require(identity(compressed) == {k: packet[k] for k in ('bytes', 'sha256')}, 'Compressed snapshot changed')
    raw = zlib.decompress(compressed)
    require(identity(raw) == {k: source[k] for k in ('bytes', 'sha256')}, 'Fetched file changed')
    exact_write(destination, raw)


def deliver(host, config, rows, raw):
    packet_raw = zlib.compress(raw, 6)
    packet = identity(packet_raw)
    for offset in range(0, len(packet_raw), UPLOAD_CHUNK):
        block = packet_raw[offset:offset + UPLOAD_CHUNK]
        result = remote(host, config, 'upload', packet=packet, offset=offset,
                        data=base64.b64encode(block).decode())
        require(result['next_offset'] == offset + len(block), 'Wrong upload acknowledgement')
    return remote(host, config, 'commit', packet=packet, marker=identity(raw),
                  ready_workers=[row for row in rows if row['worker'].split('/')[0] == host])


def collect(config, binding_raw, proof):
    exact_write(proof / 'BINDINGS.json', binding_raw)
    probes = {host: remote(host, config, 'probe') for host in ROOTS}
    rows = ready_workers(probes)
    if rows is None:
        return {'status': 'pending', 'workers': probes, 'large_files_transferred': 0, 'markers_deployed': 0}
    exact_write(proof / 'READY.json', encoded(rows))
    bundle_rows = []
    for row in rows:
        host = row['worker'].split('/')[0]
        inventory = proof / ('INVENTORY-' + host + '.json')
        summary = proof / ('SMOKE_SUMMARY-' + row['worker'].replace('/', '_') + '.json')
        fetch(host, config, row['inventory'], inventory)
        fetch(host, config, row['summary'], summary)
        bundle_rows.append({'worker': row['worker'], 'inventory': receipt(inventory), 'summary': receipt(summary)})
    bundle = proof / 'BUNDLE.json'; exact_write(bundle, encoded({'workers': bundle_rows}))
    marker = proof / 'GLOBAL_SMOKE_PASS.json'
    validation = proof / 'VALIDATION.json'
    if validation.exists():
        value = json.loads(validation.read_text())
        require(value['bundle'] == receipt(bundle) and value['validator_source'] == config['sources']['smoke'], 'Saved validation binding changed')
        check_pin(marker, value['marker'])
    else:
        output = marker if not marker.exists() else proof / ('REVALIDATED-' + uuid.uuid4().hex + '.json')
        child = subprocess.run([sys.executable, '-B', str(E / PACKAGES['smoke'] / 'validate.py'),
                                '--bundle', str(bundle), '--output', str(output)], capture_output=True, text=True)
        require(child.returncode == 0, 'Global validation failed: ' + child.stdout + child.stderr)
        exact_write(marker, output.read_bytes())
        value = json.loads(marker.read_text())
        require(value.get('status') == 'PASS' and value.get('fit_count') == 6460, 'Validator did not return exact global PASS')
        require(all(value['source_manifest'][key] == config['sources']['smoke'][key] for key in ('bytes', 'sha256')),
                'Validator source changed during validation')
        exact_write(validation, encoded({'bundle': receipt(bundle), 'validator_source': config['sources']['smoke'], 'marker': receipt(marker)}))
    raw = marker.read_bytes(); deployed = {}
    for host in ROOTS:
        result = deliver(host, config, rows, raw)
        exact_write(proof / ('DELIVERED-' + host + '.json'), encoded({'marker': result['marker']}))
        deployed[host] = result
    return {'status': 'PASS', 'proof': str(proof), 'global_marker': receipt(marker),
            'hosts': deployed, 'full_training_launch_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--bindings-sha256', required=True)
    args = parser.parse_args()
    config, raw = load_bindings(args.bindings, args.bindings_sha256)
    proof = HERE / 'global_smoke_barrier_proof_v1' / args.bindings_sha256
    require(proof.resolve().is_relative_to(HERE), 'Proof output escapes the launch planning folder')
    proof.mkdir(parents=True, exist_ok=True)
    with (proof / 'COORDINATOR.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = proof / ('ATTEMPT-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8] + '.json')
        try:
            result = collect(config, raw, proof)
        except Exception as exc:
            result = {'status': 'ERROR', 'error': type(exc).__name__ + ': ' + str(exc), 'proof': str(proof),
                      'prior_outputs_preserved': True, 'full_training_launch_performed': False}
            exact_write(attempt, encoded(result)); print(json.dumps(result)); return 1
        exact_write(attempt, encoded(result)); print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    sys.exit(main())
