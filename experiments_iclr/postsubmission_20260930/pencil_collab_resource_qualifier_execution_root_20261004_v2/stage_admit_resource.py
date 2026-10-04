"""Root-only client for the immutable v2 PENCIL resource attempt.

No work occurs on import. Stage, admit, and launch are separate explicit modes.
"""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'pencil_collab_resource_qualifier_preparation_20261004_v2'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / 'pencil_collab_resource_qualifier_execution_root_20261004_v2'
REMOTE_SOURCE = REMOTE_PHASE / SOURCE.name
SOURCE_MANIFEST = '1285a03bcc4a790162f6cebf1d49548c2c6ac18788b2008776e12038acbbf984'
SOURCE_SEAL = 'dbc31991be21d07e20a2b14363df0bd0701ff55ff376492a5dc4fd7dc4e0f890'
PLAN_SHA = 'cc882b7e1434e1353212a401a39053c85ebbb31846625cb158439fc2425d0f16'
REVIEW_DIRECTORY = 'pencil_collab_resource_qualifier_independent_source_review_20261004_v2'
REVIEW_SEAL = '8eb1d78290351c69ec36f1431e890808a245ce26c26f242d17bfe80f718ef4c7'
REVIEW_MANIFEST = 'c32a4b5e5e93fd1ce74df06b1bf1caa2f3f8f540500d7b472b52f651dc3fef97'
REVIEW_VERDICT = '9ab8e021a1979b5795109914ac0e4195b86daa1dbf0e4364a10b42f7b1294fc4'
TRANSPORT_PINS = [
    dict(path='ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py',
         bytes=10026, sha256='cfb554db15ebdb7e5f82284bc73f0c112c74e1b9bd7dc238b41ba1dce1933276'),
    dict(path='gpu77_connection_recovery_v1/run_gpu77_v3.py', bytes=3261,
         sha256='035b740ceb50eefcfa3cec2aef6dbd1294c769d133e2bc4cf88fb52b088421ff')]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(path, row):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'Pinned file differs: ' + str(path))
    return path


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def transport(identity, code):
    # Reuse the existing native root client and its existing MacLink relay.
    for row in TRANSPORT_PINS:
        pin(PHASE / row['path'], row)
    spec = importlib.util.spec_from_file_location('pencil_v2_native_root_transport', PHASE / TRANSPORT_PINS[0]['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('pencil_resource_v2_' + identity + '_20261004', code)


def context():
    return ('from pathlib import Path\nimport base64,hashlib,json,os,subprocess\n'
            'from datetime import datetime,timezone\n'
            'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
            'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
            'def sha(p):\n h=hashlib.sha256()\n with Path(p).open("rb") as s:\n  for b in iter(lambda:s.read(1048576),b""):h.update(b)\n return h.hexdigest()\n'
            'def save(p,v):\n with p.open("x") as s:\n  json.dump(v,s,indent=2,sort_keys=True,allow_nan=False);s.write("\\n");s.flush();os.fsync(s.fileno())\n')


def packet(folder, expected_seal, expected_manifest=None):
    require(folder.resolve().is_relative_to(PHASE) and not folder.is_symlink(), 'Packet path escaped research root')
    require(re.fullmatch('[a-f0-9]{64}', expected_seal) is not None, 'Explicit seal digest required')
    require(sha(folder / 'SEAL.json') == expected_seal, 'Packet seal digest differs')
    seal = json.loads((folder / 'SEAL.json').read_text())
    manifest_sha = sha(folder / 'MANIFEST.json')
    require(seal['manifest_sha256'] == manifest_sha and (expected_manifest is None or manifest_sha == expected_manifest),
            'Manifest is not authenticated by the exact seal')
    rows = json.loads((folder / 'MANIFEST.json').read_text())['files']
    require(len({row['path'] for row in rows}) == len(rows), 'Duplicate packet path')
    paths = []
    for row in rows:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid packet relative path')
        path = folder / relative
        require(path.resolve().is_relative_to(folder), 'Packet payload escaped')
        paths.append(pin(path, row))
    expected = set(paths + [folder / 'MANIFEST.json', folder / 'SEAL.json'])
    for path in folder.rglob('*'):
        require(not path.is_symlink() and path.resolve().is_relative_to(folder), 'Packet symlink/escape')
        require(path.stat().st_mode & 0o777 == (0o444 if path.is_file() else 0o555), 'Packet is not immutable')
    require(folder.stat().st_mode & 0o777 == 0o555
            and {path for path in folder.rglob('*') if path.is_file()} == expected, 'Packet file set/mode differs')
    return list(expected), seal


def reviewed_plan(review, review_seal, verdict_name):
    packet(SOURCE, SOURCE_SEAL, SOURCE_MANIFEST)
    require(sha(SOURCE / 'PLAN.json') == PLAN_SHA, 'Exact v2 plan required')
    require(review.name == REVIEW_DIRECTORY and review_seal == REVIEW_SEAL
            and verdict_name == 'REVIEW.json', 'Actual sealed v2 technical review binding required')
    paths, seal = packet(review, review_seal, REVIEW_MANIFEST)
    require(Path(verdict_name).name == verdict_name, 'Verdict must be a packet filename')
    verdict_path = review / verdict_name
    require(verdict_path in paths, 'Verdict is not sealed')
    verdict = json.loads(verdict_path.read_text())
    require(sha(verdict_path) == REVIEW_VERDICT, 'Actual review verdict digest differs')
    require(verdict.get('status') == 'PASS' and not verdict.get('blocking_findings')
            and verdict.get('candidate_manifest_sha256') == SOURCE_MANIFEST
            and verdict.get('execution_authorized') is False, 'Actual independent exact v2 PASS required')
    require(seal.get('candidate_manifest_sha256') == SOURCE_MANIFEST, 'Review seal names a different candidate')
    if 'verdict_sha256' in seal:
        require(seal['verdict_sha256'] == sha(verdict_path), 'Seal verdict digest differs')
    if 'review_sha256' in seal:
        require(seal['review_sha256'] == sha(verdict_path), 'Seal review digest differs')
    require(verdict.get('reviewer_context', {}).get('agent_context') == 'REUSED'
            and verdict['reviewer_context'].get('fresh_zero_context_review') is False
            and verdict['reviewer_context'].get('manuscript_or_acceptance_review') is False,
            'Actual reused-context technical review disclosure required')
    plan = json.loads((SOURCE / 'PLAN.json').read_text())
    require(plan['execution_directory'] == HERE.name and REMOTE.name == HERE.name
            and plan['repository'] == str(REPO / 'experiments_iclr')
            and plan['GPU_UUID'] == 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
            and plan['workload']['native_epochs'] == 1 and plan['workload']['TEST_reads'] is False
            and plan['workload']['predictive_metrics'] is False and plan['state_donor'] is False
            and plan['automatic_retry'] is False, 'Resource workload or destination differs')
    binding = dict(source_manifest_sha256=SOURCE_MANIFEST, source_seal_sha256=SOURCE_SEAL,
                   review_directory=review.name, review_seal_sha256=review_seal,
                   review_manifest_sha256=sha(review / 'MANIFEST.json'),
                   verdict=dict(path=str(REMOTE_PHASE / review.name / verdict_name),
                                bytes=verdict_path.stat().st_size, sha256=sha(verdict_path)),
                   review_context_disclosure='Independent technical v2 source review uses reused project context because fresh-child allocation was refused; it is not a fresh manuscript review.')
    return plan, binding


def authenticated_remote(binding):
    code = context() + 'source=Path(' + repr(str(REMOTE_SOURCE)) + ');review=phase/' + repr(binding['review_directory']) + '\n'
    code += 'packets=' + repr([(str(REMOTE_SOURCE), SOURCE_MANIFEST, SOURCE_SEAL),
                              (str(REMOTE_PHASE / binding['review_directory']), binding['review_manifest_sha256'], binding['review_seal_sha256'])]) + '\n'
    code += '''for directory,manifest_sha,seal_sha in packets:
 folder=Path(directory);assert folder.resolve()==folder and not folder.is_symlink()
 assert sha(folder/'MANIFEST.json')==manifest_sha and sha(folder/'SEAL.json')==seal_sha
 seal=json.loads((folder/'SEAL.json').read_text());assert seal['manifest_sha256']==manifest_sha
 rows=json.loads((folder/'MANIFEST.json').read_text())['files'];assert len({r['path'] for r in rows})==len(rows)
 expected={folder/'MANIFEST.json',folder/'SEAL.json'}
 for row in rows:
  p=folder/row['path'];assert p.resolve().is_relative_to(folder) and p.is_file() and not p.is_symlink()
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];expected.add(p)
 assert {p for p in folder.rglob('*') if p.is_file()}==expected
 assert folder.stat().st_mode&0o777==0o555
 for p in folder.rglob('*'):assert not p.is_symlink() and p.stat().st_mode&0o777==(0o444 if p.is_file() else 0o555)
for row in json.loads((source/'INPUT_BINDINGS.json').read_text())['inputs']:
 p=phase/row['path'];assert p.resolve().is_relative_to(phase) and p.is_file() and not p.is_symlink()
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
'''
    code += 'row=' + repr(binding['verdict']) + ';p=Path(row["path"]);assert p.stat().st_size==row["bytes"] and sha(p)==row["sha256"]\n'
    code += 'v=json.loads(p.read_text());assert v.get("status")=="PASS" and not v.get("blocking_findings") and v.get("execution_authorized") is False and v.get("candidate_manifest_sha256")==' + repr(SOURCE_MANIFEST) + '\n'
    return code


def stage(review, review_seal, verdict_name):
    plan, binding = reviewed_plan(review, review_seal, verdict_name)
    files = packet(SOURCE, SOURCE_SEAL, SOURCE_MANIFEST)[0] + packet(review, review_seal)[0]
    for row in json.loads((SOURCE / 'INPUT_BINDINGS.json').read_text())['inputs']:
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE), 'External source pin escaped research root')
        files.append(pin(path, row))
    rows = []
    for path in sorted(set(files)):
        raw = path.read_bytes()
        require(len(raw) < 2_000_000, 'Oversized source payload')
        rows.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=sha(path), data=base64.b64encode(raw).decode()))
    save('SOURCE_STAGE_INVENTORY.json', [{k:v for k,v in row.items() if k != 'data'} for row in rows])
    packed = zlib.compress(json.dumps(rows).encode(), 9)
    encoded = base64.b64encode(packed).decode()
    chunks = [encoded[n:n+40000] for n in range(0, len(encoded), 40000)]
    for number, chunk in enumerate(chunks, 1):
        code = context() + 'staging=root/"source_chunks";staging.mkdir(parents=True,exist_ok=True)\nraw=' + repr(chunk) + '.encode()\n'
        code += 'with (staging/' + repr('part%03d.b64' % number) + ').open("xb") as s:s.write(raw)\n'
        code += 'print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        result = transport('chunk%03d' % number, code)
        require(result['sha256'] == hashlib.sha256(chunk.encode()).hexdigest(), 'Chunk receipt differs')
    save('SOURCE_STAGE_RECEIPT.json', transport('join', stage_join_code(binding, hashlib.sha256(packed).hexdigest(), len(chunks))))
    print('Exact sealed v2 source, review, and 14 bound external source pins staged; no numerical execution.')


def stage_join_code(binding, packed_sha, chunk_count):
    code = context() + 'import zlib\npacked=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(chunk_count+1) + ')),validate=True)\n'
    code += 'assert hashlib.sha256(packed).hexdigest()==' + repr(packed_sha) + '\nrows=json.loads(zlib.decompress(packed))\n'
    code += '''for row in rows:
 p=phase/row['path'];assert p.resolve().is_relative_to(phase)
 raw=base64.b64decode(row['data'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as s:s.write(raw)
'''
    code += 'for name in ' + repr([SOURCE.name, binding['review_directory']]) + ':\n folder=phase/name\n for p in folder.rglob("*"):\n  if p.is_file():p.chmod(0o444)\n for p in sorted((p for p in folder.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)\n folder.chmod(0o555)\n'
    code += 'staged_file_count=len(rows);staged_byte_count=sum(r["bytes"] for r in rows)\n'
    # Reauthenticate closure after joining; no executable source is imported.
    code += authenticated_remote(binding).split('assert Path.cwd()==repo and os.uname().nodename=="peptide"\n', 1)[1]
    code += 'receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),status="EXACT_REVIEWED_PENCIL_V2_SOURCE_STAGED",files=staged_file_count,bytes=staged_byte_count,binding=' + repr(binding) + ',numerical_execution=False,TEST_reads=False)\nsave(root/"SOURCE_STAGE_RECEIPT.json",receipt)\nprint(json.dumps(receipt))\n'
    return code


def admission_code(binding):
    code = authenticated_remote(binding)
    code += '''assert json.loads((root/'SOURCE_STAGE_RECEIPT.json').read_text())['binding']=='''+repr(binding)+'''
plan=json.loads((source/'PLAN.json').read_text());authority=json.loads((source/'metadata/RUNTIME_AUTHORITY.json').read_text())
assert sha(authority['interpreter_path'])==authority['interpreter_sha256']
observed=json.loads((source/'ACTUAL_DEPENDENCY_BINDING.json').read_text());records={}
for name in ('INSTALLED_FILE_INVENTORY','DISTRIBUTION_ADMISSION','INSTALL_RESULT'):
 row=observed[name];p=Path(row['path']);assert p.resolve().is_relative_to(phase) and p.is_file() and not p.is_symlink()
 assert p.stat().st_size<2_000_000 and sha(p)==row['sha256']
 if name!='INSTALLED_FILE_INVENTORY':assert p.stat().st_size==row['bytes']
 records[name]=json.loads(p.read_text())
inventory=records['INSTALLED_FILE_INVENTORY'];distribution=records['DISTRIBUTION_ADMISSION'];result=records['INSTALL_RESULT']
assert result['status']=='COMPLETE_REPO_OVERLAY_INSTALL_ONLY' and result['core_versions_unchanged'] is True
assert result['installed_files']==len(inventory)==3361 and result['installed_bytes']==sum(r['bytes'] for r in inventory)==125404681
assert result['inventory_sha256']==observed['INSTALLED_FILE_INVENTORY']['sha256'] and result['admission_sha256']==observed['DISTRIBUTION_ADMISSION']['sha256']
assert result['overlay']==plan['dependency_overlay'] and result['new_packages']==16
assert distribution['versions']==plan['distribution_versions'] and set(distribution['origins'])==set(plan['added_package_modules'])
overlay=Path(plan['dependency_overlay']);assert overlay.resolve()==overlay and not overlay.is_symlink()
installed=[]
for row in inventory:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 p=overlay/relative;assert p.resolve().is_relative_to(overlay) and p.is_file() and not p.is_symlink()
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 installed.append(dict(row,path=str(p)))
by_path={r['path']:r for r in installed};assert len(by_path)==len(installed)
assert set(by_path)=={str(p) for p in overlay.rglob('*') if p.is_file()}
assert all(not p.is_symlink() for p in overlay.rglob('*'))
added=[]
for name,module in plan['added_package_modules'].items():
 origin=distribution['origins'][name];p=Path(origin['origin'])
 assert p.resolve().is_relative_to(overlay) and str(p) in by_path
 assert all(Path(s).resolve().is_relative_to(overlay) for s in origin['search_locations'])
 added.append(dict(by_path[str(p)],distribution=name,module=module))
admission=json.loads((source/'DEPENDENCY_ADMISSION_TEMPLATE.json').read_text())
assert admission['distribution_versions']==plan['distribution_versions'] and admission['PYTHONPATH']==plan['PYTHONPATH']
assert admission['base_runtime_authority_sha256']==sha(source/'metadata/RUNTIME_AUTHORITY.json')
admission.update(status='ROOT_ADMITTED_FOR_RESOURCE_ONLY',installed_inventory_complete=True,
                 installed_inventory_root=str(overlay),installed_files=installed,added_package_source_pins=added,
                 authenticated_installation_records=observed,UTC=datetime.now(timezone.utc).isoformat(),
                 installation_or_numerical_import_performed=False)
save(root/'ROOT_DEPENDENCY_ADMISSION.json',admission)
gpu=subprocess.run(['nvidia-smi','-i',plan['GPU_UUID'],'--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
lines=gpu.stdout.splitlines();assert len(lines)==1
selected=dict(zip(('uuid','name','total_MiB','free_MiB'),[x.strip() for x in lines[0].split(',')]))
assert selected['uuid']==plan['GPU_UUID'] and selected['name']==authority['gpu_name']
p=root/'ROOT_DEPENDENCY_ADMISSION.json'
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status='EXACT_EXISTING_OVERLAY_ADMITTED_FOR_RESOURCE_ONLY',
 dependency_admission=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)),
 installed_files=len(installed),installed_bytes=sum(r['bytes'] for r in installed),added_package_source_pins=added,
 inventory_sha256=observed['INSTALLED_FILE_INVENTORY']['sha256'],distribution_admission_sha256=observed['DISTRIBUTION_ADMISSION']['sha256'],
 interpreter_sha256=authority['interpreter_sha256'],selected_GPU=selected,numerical_execution=False,TEST_reads=False,unrelated_jobs_modified=False)))
'''
    return code


def gate_command(release_sha):
    runtime = json.loads((SOURCE / 'metadata/RUNTIME_AUTHORITY.json').read_text())
    text = ('import sys;from pathlib import Path;sys.path.insert(0,' + repr(str(REMOTE_SOURCE)) + ');'
            'from common import gate;gate(Path(' + repr(str(REMOTE / 'ROOT_RELEASE.json')) + '),' + repr(release_sha) + ');print("PASS_PENCIL_V2_STDLIB_GATE")')
    return [runtime['interpreter_path'], '-B', '-c', text]


def environment():
    return json.loads((SOURCE / 'PROPOSED_COMMAND.json').read_text())['environment']


def remote_gate(release_sha):
    return ('r=subprocess.run(' + repr(gate_command(release_sha)) + ',cwd=' + repr(str(REPO / 'experiments_iclr'))
            + ',env=dict(os.environ,**' + repr(environment()) + '),capture_output=True,text=True,timeout=60)\n'
            'assert r.returncode==0,(r.returncode,r.stdout,r.stderr)\n')


def admit(review, review_seal, verdict_name):
    plan, binding = reviewed_plan(review, review_seal, verdict_name)
    require(json.loads((HERE / 'SOURCE_STAGE_RECEIPT.json').read_text())['binding'] == binding, 'Staged review binding differs')
    observation = transport('dependencies', admission_code(binding))
    save('DEPENDENCY_ADMISSION_RECEIPT.json', observation)
    release = json.loads((SOURCE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    release.update(status='APPROVED', source_manifest_sha256=SOURCE_MANIFEST, plan_sha256=PLAN_SHA,
                   root_authorization_reference=str(REMOTE / 'ROOT_ADMISSION.json'),
                   independent_source_review=binding['verdict'], dependency_admission=observation['dependency_admission'])
    save('ROOT_RELEASE.json', release)
    save('ROOT_ADMISSION.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         status='APPROVED_ONE_NATIVE_PENCIL_EPOCH_AND_FULL_VALID_RESOURCE_ONLY',binding=binding,
         workload=plan['workload'],caps=plan['caps'],GPU_UUID=plan['GPU_UUID'],
         dependency_admission=observation['dependency_admission'],TEST_reads=False,scores_for_selection=False,
         scientific_fit_released=False,state_donor=False,automatic_retry=False,
         limitations=['Source PASS does not establish native operator/model runtime compatibility.',
                     'GPU free memory and sampled session RSS are observations, not reservations or instantaneous limits.',
                     'Resource receipts cannot establish predictive quality, numerical parity, or a completed 20-epoch fit.',
                     'The source supervisor discloses finite final publication/exit tails after its last cap check.']))
    rows = []
    for name in ('ROOT_RELEASE.json','ROOT_ADMISSION.json'):
        p = HERE / name
        rows.append(dict(path=str(REMOTE / name),sha256=sha(p),data=base64.b64encode(p.read_bytes()).decode()))
    code = authenticated_remote(binding) + 'rows=' + repr(rows) + '\n'
    code += 'for row in rows:\n p=Path(row["path"]);assert p.parent==root and not p.is_symlink();raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with p.open("xb") as s:s.write(raw)\n'
    code += remote_gate(sha(HERE / 'ROOT_RELEASE.json'))
    code += 'print(json.dumps(dict(status="PASS_EXACT_PENCIL_V2_STDLIB_ADMISSION_GATE",stdout=r.stdout,stderr=r.stderr,numerical_execution=False,TEST_reads=False)))\n'
    save('PRENUMERICAL_GATE.json', transport('gate', code))
    print('Exact dependency admission and resource-only root release staged; stdlib gate passed; launch remains a separate mode.')


def launch_code(binding, release_sha, admission_sha):
    command = json.loads((SOURCE / 'PROPOSED_COMMAND.json').read_text())['argv']
    require(command[-1] == 'ROOT_BIND_RELEASE_DIGEST', 'Unexpected command template')
    command[-1] = release_sha
    code = authenticated_remote(binding) + remote_gate(release_sha)
    code += 'assert sha(root/"ROOT_ADMISSION.json")==' + repr(admission_sha) + '\n'
    code += 'assert not any((root/p).exists() for p in ("run01","supervision/run01","RESOURCE_ATTEMPT_SPENT.json","DETACHED_LAUNCH.json","ROOT_LAUNCH_SPENT.json"))\n'
    code += 'save(root/"ROOT_LAUNCH_SPENT.json",dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=' + repr(release_sha) + ',automatic_retry=False))\n'
    code += 'command=' + repr(command) + ';environment=' + repr(environment()) + '\n'
    code += 'with (root/"DETACHED_STDOUT.txt").open("xb") as out,(root/"DETACHED_STDERR.txt").open("xb") as err:\n child=subprocess.Popen(command,cwd=' + repr(str(REPO / 'experiments_iclr')) + ',env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
    code += 'raw=(Path("/proc")/str(child.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[2])==int(f[3])==child.pid\n'
    code += 'v=dict(UTC=datetime.now(timezone.utc).isoformat(),status="OWNED_PENCIL_V2_RESOURCE_SUPERVISOR_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(f[19]),supervisor_session=int(f[3]),supervisor_group=int(f[2]),command=command,environment=environment,source_manifest_sha256=' + repr(SOURCE_MANIFEST) + ',release_sha256=' + repr(release_sha) + ',binding=' + repr(binding) + ',TEST_reads=False,scores_for_selection=False,state_donor=False,automatic_retry=False)\nsave(root/"DETACHED_LAUNCH.json",v)\nprint(json.dumps(v))\n'
    return code


def launch(review, review_seal, verdict_name):
    _, binding = reviewed_plan(review, review_seal, verdict_name)
    require(json.loads((HERE / 'ROOT_ADMISSION.json').read_text())['binding'] == binding, 'Admitted review binding differs')
    require(json.loads((HERE / 'PRENUMERICAL_GATE.json').read_text())['status'] == 'PASS_EXACT_PENCIL_V2_STDLIB_ADMISSION_GATE', 'Admitted stdlib gate required')
    require(not (HERE / 'ROOT_LAUNCH_SPENT.json').exists(), 'Launch attempt already spent; no automatic retry')
    release_sha = sha(HERE / 'ROOT_RELEASE.json')
    save('ROOT_LAUNCH_SPENT.json', dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=release_sha,automatic_retry=False))
    result = transport('launch', launch_code(binding, release_sha, sha(HERE / 'ROOT_ADMISSION.json')))
    save('DETACHED_LAUNCH.json', result)
    print(json.dumps({k:result[k] for k in ('UTC','status','supervisor_PID','supervisor_start_ticks')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('stage','admit','launch'))
    parser.add_argument('--review', required=True, help='Actual sealed review directory basename under this research root')
    parser.add_argument('--review-seal-sha256', required=True, help='Actual seal digest supplied by root after inspecting the review')
    parser.add_argument('--review-verdict', default='REVIEW.json')
    args = parser.parse_args()
    require(Path(args.review).name == args.review, 'Review must be a directory basename')
    globals()[args.mode](PHASE / args.review,args.review_seal_sha256,args.review_verdict)
