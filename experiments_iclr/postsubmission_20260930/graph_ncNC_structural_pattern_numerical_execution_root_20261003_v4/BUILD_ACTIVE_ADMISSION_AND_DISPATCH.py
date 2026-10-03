"""Record the explicit parent decision and build its single numerical dispatch."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import zlib

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
TRANSPORT = PHASE / 'gpu77_connection_recovery_v1'
REMOTE_REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REMOTE_REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE_ROOT = REMOTE_PHASE / ROOT.name
METADATA = PHASE / 'graph_ncNC_structural_pattern_v4_numerical_execution_metadata_preparation_20261003_v1'
GPU = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'


def write_new(name, value):
    content = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    with (ROOT / name).open('xb') as handle:
        handle.write(content)


def pin(path, absolute=False):
    content = path.read_bytes()
    relative = path.relative_to(PHASE)
    return {'path': str(REMOTE_PHASE / relative) if absolute else str(relative),
            'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


candidate = METADATA / 'ROOT_RELEASE_NUMERICAL_CANDIDATE.json'
assert pin(candidate)['sha256'] == '76bb72c5dec9883d8511a31ad911c3937315009a91ba9092f0676b65232d5654'
custody = pin(ROOT / 'SOURCE_CUSTODY.json', True)
physical = pin(ROOT / 'FINAL_PHYSICAL_ADMISSION.json', True)
initial_physical = pin(ROOT / 'CURRENT_PHYSICAL_PREFLIGHT.json', True)
assert custody['sha256'] == '4b4610a147ead4868e92b77565cd5a4bd51dc713114fddb42af25a2673c7bcd8'
assert physical['sha256'] == 'b4e2b8a7c591e94a3adfe0eb7c91d64141b4c409bf8e38fc2165d2fddde3036c'
helper_review = pin(PHASE / 'graph_ncNC_structural_pattern_v4_numerical_helper_root_review_20261003_v1/REVIEW.json')
model_review = pin(PHASE / 'graph_ncNC_structural_pattern_v4_independent_source_review_20261003_v1/REVIEW.json')
assert helper_review['sha256'] == '84b94d7e1850d48de979598eafc958d4d07e1faac3b00d426ee34751cf8c327c'
assert model_review['sha256'] == '7682474372bee7162f969710ad4f4651cdb06cb1bbb507529d4a5826402f5322'
release = json.loads(candidate.read_text())
invocation = release['example_invocations_NOT_AUTHORIZED'][0]
assert invocation == {'stage': 'numerical', 'unit': 'pair', 'base_seed': 0,
                      'output_directory': str(REMOTE_ROOT / 'numerical/run01')}
caps = {'wall_seconds': 1800, 'host_RSS_bytes': 16 * 1024**3,
        'cuda_peak_allocated_bytes': 8 * 1024**3,
        'cuda_peak_reserved_bytes': 8 * 1024**3}
assert release['numerical_caps'] == caps
utc = datetime.now(timezone.utc).isoformat()
authorization = {
    'schema': 'delegate_record_of_explicit_parent_numerical_authorization_v1',
    'recorded_UTC': utc,
    'authorizing_agent': '/root',
    'recorded_by_agent': '/root/initializer_v9_independent_source_review_20261003',
    'record_source': 'Explicit parent task authorization carried in the continuation summary',
    'authorization_summary_not_a_verbatim_quote':
        'Parent authorized one V4 numerical pair using candidate release SHA '
        '76bb72c5dec9883d8511a31ad911c3937315009a91ba9092f0676b65232d5654, '
        'after exact route, interpreter, capacity and source custody checks; '
        'prefer GPU1 and bind it before launch. Publish fresh active admission and release '
        'with the exact reviews and launch the ordinary numerical supervisor once.',
    'candidate_release': pin(candidate),
    'authorized_invocations': [invocation],
    'selected_GPU_UUID': GPU,
    'caps': caps,
    'engineering_optimizer_updates': 14,
    'updates_breakdown': {'existing_replay': 6, 'V3_V4_parity': 8},
    'source_staging': 'Only absent helper, metadata and review files; no differing overwrite',
    'stop_on_physical_source_or_launch_failure': True,
    'automatic_retry': False,
    'alternate_allocation': False,
    'full_graph_authorized': False,
    'scientific_fits_authorized': False,
    'TEST_authorized': False,
    'state_donor_authorized': False,
}
write_new('ROOT_AUTHORIZATION_RECORD.json', authorization)
auth_pin = pin(ROOT / 'ROOT_AUTHORIZATION_RECORD.json', True)
admission = json.loads((METADATA / 'ROOT_NUMERICAL_ADMISSION_CANDIDATE.json').read_text())
admission.update({
    'schema': 'ncnc_pattern_V4_root_numerical_admission_v1',
    'UTC': utc,
    'decision': 'release_one_V4_numerical_pair',
    'authorization_reference': auth_pin,
    'candidate_release': pin(candidate),
    'authorized_stages': ['numerical'],
    'authorized_invocations': [invocation],
    'selected_GPU': GPU,
    'new_helper_independent_review': helper_review,
    'independent_reviews': [model_review, helper_review],
    'fresh_remote_source_custody': custody,
    'fresh_physical_resource_admission': physical,
    'initial_physical_preflight': initial_physical,
    'immediate_dispatch_capacity_recheck_required': True,
    'prior_attempt_history': pin(METADATA / 'PRIOR_ATTEMPTS_AND_COST_CUSTODY.json'),
    'engineering_optimizer_updates': 14,
    'historical_V3_PASS_is_current_admission': False,
    'alternate_allocation': False,
    'TEST_authorized': False,
    'state_donor_authorized': False,
})
write_new('ROOT_NUMERICAL_ADMISSION.json', admission)
release.pop('example_invocations_NOT_AUTHORIZED')
release.update({
    'example_only_NOT_AUTHORIZATION': False,
    'root_authorization_reference': str(REMOTE_ROOT / 'ROOT_NUMERICAL_ADMISSION.json'),
    'authorized_stages': ['numerical'],
    'authorized_invocations': [invocation],
    'cuda_visible_devices': GPU,
    'source_review_status': 'V4_MODEL_AND_NEW_HELPER_REVIEWS_PASSED_ROOT_RELEASED_NUMERICAL_ONLY',
    'independent_reviews': [model_review, helper_review],
    'source_custody': custody,
    'source_upload_or_remote_rehash_completed': True,
    'launch_support_independent_review': helper_review,
    'new_launch_support_review_required': False,
    'new_launch_support_independent_review_completed': True,
    'prepared_source_only_no_root_release': False,
    'candidate_release_descriptor': pin(candidate),
    'active_admission_descriptor': pin(ROOT / 'ROOT_NUMERICAL_ADMISSION.json', True),
    'root_authorization_record': auth_pin,
    'current_physical_resource_admission': physical,
    'prior_attempt_history': pin(METADATA / 'PRIOR_ATTEMPTS_AND_COST_CUSTODY.json'),
    'engineering_optimizer_updates': 14,
    'automatic_retry': False,
    'alternate_allocation': False,
    'TEST_authorized': False,
    'state_donor_authorized': False,
})
assert release['authorized_stages'] == ['numerical'] and len(release['authorized_invocations']) == 1
assert not release['scientific_fits_authorized'] and not release['full_graph_authorized']
write_new('ROOT_RELEASE_NUMERICAL.json', release)
rows = []
for name in ['CURRENT_PHYSICAL_PREFLIGHT.json', 'SOURCE_STAGE_FILE_PINS.json',
             'ROOT_AUTHORIZATION_RECORD.json', 'ROOT_NUMERICAL_ADMISSION.json',
             'ROOT_RELEASE_NUMERICAL.json']:
    path = ROOT / name
    rows.append({**pin(path), 'zlib_base64': base64.b64encode(zlib.compress(path.read_bytes(), 9)).decode()})
sequence = json.loads((METADATA / 'LAUNCH_SEQUENCE.json').read_text())
argv = sequence['supervisor_argv_after_approval']
assert argv[0] == admission['interpreter']['path']
source_value = json.loads((ROOT / 'SOURCE_CUSTODY.json').read_text())
source_extras = json.loads((ROOT / 'SOURCE_STAGE_FILE_PINS.json').read_text())['files']
code = '''from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zlib,base64,subprocess,os,time,socket
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git').resolve()
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'graph_ncNC_structural_pattern_numerical_execution_root_20261003_v4'
assert Path.cwd().resolve()==repo and socket.gethostname()=='peptide'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='db0302db9991bbb5e167a94948ff6fe5f4199551'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(x):
    p=Path(x['path'])
    if not p.is_absolute():p=phase/p
    p=p.resolve();assert p.is_relative_to(phase)
    assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Prelaunch custody differs: '+str(p)
    return p
def physical(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
    return {'PID':pid,'parent_PID':int(f[1]),'process_group':int(f[2]),'session':int(f[3]),'start_time_ticks':int(f[19]),'state':f[0],
            'argv':(p/'cmdline').read_bytes().decode(errors='replace').split('\\0')[:-1],
            'cwd':str((p/'cwd').resolve()),'exe':str((p/'exe').resolve()),
            'namespaces':{n:os.readlink(p/'ns'/n) for n in ('pid','mnt','user','net','uts','cgroup')}}
'''
code += 'source_custody=' + repr(custody) + '\nphysical_admission=' + repr(physical) + '\n'
code += 'packets=' + repr(source_value['verified_packets']) + '\nsource_extras=' + repr(source_extras) + '\n'
code += 'rows=' + repr(rows) + '\nargv=' + repr(argv) + '\nGPU=' + repr(GPU) + '\n'
code += '''verify(source_custody);verify(physical_admission)
for packet in packets:
    manifest=verify(packet['manifest']);d=manifest.parent
    if 'seal' in packet:verify(packet['seal'])
    for row in json.loads(manifest.read_text())['files']:
        p=(d/row['path']).resolve();assert p.is_relative_to(d)
        verify({'path':str(p),'bytes':row.get('bytes',row.get('size')),'sha256':row['sha256']})
for x in source_extras:verify(x)
assert sha(Path(argv[0]))=='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
for x in rows:
    p=phase/x['path'];assert p.resolve().is_relative_to(root)
    assert not p.exists(),'Active execution metadata became present: '+str(p)
for name in ('DETACHED_SUPERVISOR.log','DISPATCH.json','supervision/run01','numerical/run01'):
    assert not (root/name).exists(),'Launch/output path not fresh: '+name
q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
gpu_rows=[[s.strip() for s in line.split(',')] for line in q.stdout.splitlines() if line.strip()]
assert len(gpu_rows)==2 and {x[1] for x in gpu_rows}=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
assert all(x[2]=='NVIDIA A100 80GB PCIe' and int(x[3])==81920 for x in gpu_rows)
selected=next(x for x in gpu_rows if x[1]==GPU)
assert int(selected[5])>=9216,'Selected GPU1 lacks capacity; no alternate allocation authorized'
available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert available>=16*1024**3,'Insufficient available host memory'
for x in rows:
    p=phase/x['path'];b=zlib.decompress(base64.b64decode(x['zlib_base64']))
    assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    verify(x)
release_path=root/'ROOT_RELEASE_NUMERICAL.json'
release=json.loads(release_path.read_text())
assert release['authorized_stages']==['numerical'] and release['cuda_visible_devices']==GPU
assert release['numerical_caps']=={'wall_seconds':1800,'host_RSS_bytes':16*1024**3,'cuda_peak_allocated_bytes':8*1024**3,'cuda_peak_reserved_bytes':8*1024**3}
assert release['authorized_invocations']==[{'stage':'numerical','unit':'pair','base_seed':0,'output_directory':str(root/'numerical/run01')}]
assert release['example_only_NOT_AUTHORIZATION'] is False and not release['full_graph_authorized'] and not release['scientific_fits_authorized']
verify(release['active_admission_descriptor']);verify(release['root_authorization_record'])
env=os.environ.copy();env.update(GNNM_SSH_DESTINATION='shmelev@192.168.18.77',PYTHONDONTWRITEBYTECODE='1')
launcher=physical(os.getpid())
log=root/'DETACHED_SUPERVISOR.log'
with log.open('xb') as output:
    process=subprocess.Popen(['nohup',*argv],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.3)
identity=physical(process.pid) if (Path('/proc')/str(process.pid)).exists() else None
value={'schema':'V4_root_single_detached_numerical_dispatch_v1','UTC':datetime.now(timezone.utc).isoformat(),
       'supervisor_PID':process.pid,'physical_identity':identity,'launcher_physical_identity':launcher,
       'requested_supervisor_argv':argv,'nohup_and_ordinary_session':True,
       'root_release_sha256':sha(release_path),'source_custody':source_custody,'physical_admission':physical_admission,
       'selected_GPU_UUID':GPU,'GPU_rows_immediately_before_dispatch':gpu_rows,'MemAvailable_bytes_immediately_before_dispatch':available,
       'active_metadata_files':[{k:x[k] for k in ('path','bytes','sha256')} for x in rows],
       'single_dispatch':True,'numerical_only':True,'automatic_retry':False,'alternate_allocation':False,
       'other_jobs_signaled':False,'filesystem_isolation':False,'terminal_present':(root/'supervision/run01/SUPERVISOR_TERMINAL.json').exists()}
with (root/'DISPATCH.json').open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\\n');f.flush();os.fsync(f.fileno())
print(json.dumps(value,indent=2))
assert identity is not None,'Supervisor missing after dispatch; preserve failure without retry'
assert identity['argv']==argv and identity['cwd']==str(repo) and identity['exe']==str(Path(argv[0]).resolve()),'Unexpected dispatched physical identity'
assert identity['session']==process.pid and identity['process_group']==process.pid,'Supervisor is not its own ordinary detached session'
assert identity['namespaces']==launcher['namespaces'],'Unexpected namespace isolation'
'''
command = "set -e\ncd /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git\n/usr/bin/python3 - <<'PYREMOTE'\n" + code + 'PYREMOTE\n'
path = TRANSPORT / 'NCNC_V4_NUMERICAL_ROOT_DISPATCH_20261003_v1.txt'
with path.open('x') as handle:
    handle.write(command)
write_new('LOCAL_ACTIVE_METADATA_PINS.json', {'files': [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in rows],
                                           'dispatch_command': pin(path)})
print(json.dumps({'dispatch_command': str(path), 'bytes': len(command.encode()),
                  'release': pin(ROOT / 'ROOT_RELEASE_NUMERICAL.json'),
                  'admission': pin(ROOT / 'ROOT_NUMERICAL_ADMISSION.json'),
                  'selected_GPU_UUID': GPU}, indent=2))
