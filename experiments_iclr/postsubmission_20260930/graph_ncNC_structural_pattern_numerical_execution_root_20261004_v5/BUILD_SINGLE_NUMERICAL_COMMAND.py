"""Build one exact V5 numerical dispatch after both source reviews exist."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import base64
import hashlib
import json
import lzma
import shlex

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REMOTE_REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REMOTE_REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE_ROOT = REMOTE_PHASE / ROOT.name
DRIVER_NAME = 'graph_ncNC_structural_pattern_pilot_preparation_20261004_v5'
SUPERVISOR_NAME = 'graph_ncNC_structural_pattern_normal_supervision_preparation_20261004_v4'
DRIVER_SHA = '9fc539b8f92d7d4e883224c3b0aa85ae583b64c70f2a701a4a648fd818aa32a1'
SUPERVISOR_SHA = 'ab03a87a8f090788d0848b18a012cfdcb87a39a4786ce983441541baf73304bb'
GPU = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'
CAPS = {'wall_seconds': 1800, 'host_RSS_bytes': 16 * 1024**3,
        'cuda_peak_allocated_bytes': 8 * 1024**3, 'cuda_peak_reserved_bytes': 8 * 1024**3}


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(PHASE)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


driver = PHASE / DRIVER_NAME
supervisor = PHASE / SUPERVISOR_NAME
assert pin(driver / 'MANIFEST.json')['sha256'] == DRIVER_SHA
assert pin(supervisor / 'MANIFEST.json')['sha256'] == SUPERVISOR_SHA
review_path = ROOT / 'NUMERICAL_SUPERVISOR_INDEPENDENT_REVIEW.json'
review = json.loads(review_path.read_text())
assert review['status'] == 'PASS' and review['supervisor_manifest_sha256'] == SUPERVISOR_SHA
root_review = json.loads((ROOT / 'ROOT_V5_SOURCE_REVIEW.json').read_text())
assert root_review['status'] == 'SOURCE_REVIEW_PASS' and root_review['driver_manifest_sha256'] == DRIVER_SHA

release = json.loads((driver / 'ROOT_RELEASE_EXAMPLE.json').read_text())
invocation = release['example_invocations_NOT_AUTHORIZED'][0]
assert invocation == {'stage': 'numerical', 'unit': 'pair', 'base_seed': 0,
                      'output_directory': str(REMOTE_ROOT / 'numerical/run01')}
release.pop('example_invocations_NOT_AUTHORIZED')
release.update(example_only_NOT_AUTHORIZATION=False, execution_enabled=True,
               driver_manifest_sha256=DRIVER_SHA, authorized_stages=['numerical'],
               authorized_invocations=[invocation], cuda_visible_devices=GPU,
               root_authorization_reference=str(REMOTE_ROOT / 'ROOT_NUMERICAL_ADMISSION.json'),
               qualification={}, fit_receipts={}, diagnostics_receipt={},
               numerical_caps=CAPS, supervision_manifest_sha256=SUPERVISOR_SHA,
               numerical_driver_path=str(REMOTE_PHASE / DRIVER_NAME / 'pattern_run.py'),
               full_graph_authorized=False, scientific_fits_authorized=False,
               TEST_authorized=False, state_donor_authorized=False, automatic_retry=False)
admission = {'schema': 'root-one-V5-numerical-admission-v1',
             'UTC': datetime.now(timezone.utc).isoformat(), 'authorizing_agent': '/root',
             'decision': 'one_numerical_pair_only', 'driver_manifest_sha256': DRIVER_SHA,
             'supervision_manifest_sha256': SUPERVISOR_SHA, 'caps': CAPS,
             'selected_GPU_UUID': GPU, 'root_source_review': pin(ROOT / 'ROOT_V5_SOURCE_REVIEW.json'),
             'supervisor_independent_review': pin(review_path),
             'runtime_profile_transition': release['runtime_profile_transition'],
             'original_14_engineering_updates_and_tolerance_preserved': True,
             'earlier_diagnostic_or_V4_PASS_substitutes_for_current_qualification': False,
             'fullgraph_or_predictive_fit_authorized': False, 'TEST_authorized': False,
             'ordinary_host_processes_only': True, 'other_jobs_signaled': False,
             'original_failures_and_original_paper_scores_preserved': True}
write_new(ROOT / 'ROOT_NUMERICAL_ADMISSION.json', admission)
write_new(ROOT / 'ROOT_RELEASE_NUMERICAL.json', release)

payloads = []
for folder, expected in [(driver, DRIVER_SHA), (supervisor, SUPERVISOR_SHA)]:
    manifest = json.loads((folder / 'MANIFEST.json').read_text())
    paths = [folder / row['path'] for row in manifest['files']] + [folder / 'MANIFEST.json']
    if (folder / 'SEAL.json').exists():
        paths.append(folder / 'SEAL.json')
    for path in paths:
        row = pin(path)
        for original in manifest['files']:
            if original['path'] == path.name:
                assert row['bytes'] == original['bytes'] and row['sha256'] == original['sha256']
        payloads.append(dict(row, base64=base64.b64encode(path.read_bytes()).decode()))
for name in ['ROOT_V5_SOURCE_REVIEW.json', 'NUMERICAL_SUPERVISOR_INDEPENDENT_REVIEW.json',
             'ROOT_NUMERICAL_ADMISSION.json', 'ROOT_RELEASE_NUMERICAL.json']:
    path = ROOT / name
    payloads.append(dict(pin(path), base64=base64.b64encode(path.read_bytes()).decode()))
packed = base64.b64encode(lzma.compress(json.dumps(payloads).encode(), preset=9)).decode()
code = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,lzma,os,socket,subprocess,time
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='peptide'
assert subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()==str(repo)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='9c7ed8a6192405c64743206f80c13ad0f2a3dcd7'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(path,value):
 with path.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\\n')
def verify_packet(path,wanted):
 assert path.resolve().is_relative_to(phase)
 assert sha(path/'MANIFEST.json')==wanted
 for row in json.loads((path/'MANIFEST.json').read_text())['files']:
  q=(path/row['path']).resolve();assert q.is_relative_to(path.resolve())
  assert q.stat().st_size==row.get('bytes',row.get('size')) and sha(q)==row['sha256']
def identity(pid):
 p=Path('/proc')/str(pid);t=(p/'stat').read_text();v=t[t.rfind(')')+2:].split()
 return {'PID':pid,'start_time_ticks':int(v[19]),'session':int(v[3]),'process_group':int(v[2]),
         'argv':(p/'cmdline').read_bytes().decode().split('\\0')[:-1],
         'cwd':str((p/'cwd').resolve()),'exe':str((p/'exe').resolve())}
'''
code += 'root=phase/' + repr(ROOT.name) + '\n'
code += 'packed=' + repr(packed) + '\n'
code += 'driver_name=' + repr(DRIVER_NAME) + '\nsupervisor_name=' + repr(SUPERVISOR_NAME) + '\n'
code += 'driver_sha=' + repr(DRIVER_SHA) + '\nsupervisor_sha=' + repr(SUPERVISOR_SHA) + '\nGPU=' + repr(GPU) + '\n'
code += '''for name in ('DISPATCH.json','DETACHED_SUPERVISOR.log','supervision/run01','numerical/run01'):
 assert not (root/name).exists(),'Numerical invocation not fresh: '+name
python=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
assert sha(python)=='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
rows=json.loads(lzma.decompress(base64.b64decode(packed,validate=True)))
for row in rows:
 path=(phase/row['path']).resolve();assert path.is_relative_to(phase)
 raw=base64.b64decode(row['base64'],validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==raw,'Existing source differs: '+str(path)
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as handle:handle.write(raw)
verify_packet(phase/driver_name,driver_sha);verify_packet(phase/supervisor_name,supervisor_sha)
release_path=root/'ROOT_RELEASE_NUMERICAL.json';release=json.loads(release_path.read_text())
for row in json.loads((phase/driver_name/'DEPENDENCIES.json').read_text())['sealed_packets']:
 verify_packet(phase/row['packet'],row['manifest_sha256'])
verify_packet(phase/'graph_ncNC_structural_pattern_pilot_preparation_20261003_v4','9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c')
for key in ('data_authority','runtime_authority'):
 p=Path(release[key+'_file']).resolve();assert p.is_relative_to(phase) and sha(p)==release[key+'_sha256']
q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
gpu_rows=[[v.strip() for v in l.split(',')] for l in q.stdout.splitlines() if l.strip()]
assert len(gpu_rows)==2 and {r[1] for r in gpu_rows}=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
assert int(next(r for r in gpu_rows if r[1]==GPU)[5])>=9216
mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert mem>=16*1024**3
argv=[str(python),'-B',str(phase/supervisor_name/'supervise_numerical.py'),'--root-release',str(release_path),'--output',str(root/'supervision/run01')]
env=os.environ.copy();env.update(GNNM_SSH_DESTINATION='shmelev@192.168.18.77',PYTHONDONTWRITEBYTECODE='1')
with (root/'DETACHED_SUPERVISOR.log').open('xb') as log:
 process=subprocess.Popen(['nohup',*argv],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.3)
owned=identity(process.pid) if (Path('/proc')/str(process.pid)).exists() else None
value={'schema':'root-single-V5-numerical-dispatch-v1','UTC':datetime.now(timezone.utc).isoformat(),'physical_identity':owned,'requested_argv':argv,
       'GPU_UUID':GPU,'GPU_rows_before_dispatch':gpu_rows,'MemAvailable_bytes':mem,
       'release_sha256':sha(release_path),'staged_files':len(rows),
       'staged_payloads':[{k:r[k] for k in ('path','bytes','sha256')} for r in rows],
       'ordinary_host_execution':True,'numerical_only':True,'automatic_retry':False,'other_jobs_signaled':False}
write_new(root/'DISPATCH.json',value)
print(json.dumps({k:v for k,v in value.items() if k!='staged_payloads'},indent=2))
assert owned is not None,'Supervisor absent after dispatch. Preserve state and do not retry.'
assert owned['argv']==argv and owned['cwd']==str(repo) and owned['session']==process.pid and owned['process_group']==process.pid
'''
ast.parse(code)
command = 'cd ' + shlex.quote(str(REMOTE_REPO)) + '\n' + shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code]) + '\n'
assert len(command.encode()) < 100000, 'Command too long. Do not dispatch.'
target = PHASE / 'gpu77_connection_recovery_v1/ncnc_V5_single_numerical_launch_20261004_v1_command.txt'
with target.open('x') as handle:
    handle.write(command)
write_new(ROOT / 'ROOT_COMMAND_PREPARATION.json',
          {'UTC': datetime.now(timezone.utc).isoformat(), 'command': pin(target),
           'staged_payload_count': len(payloads), 'LZMA_adapter': True,
           'single_numerical_only': True, 'command_AST_parsed': True, 'executed': False})
print(json.dumps({'command_file':str(target),'command_bytes':len(command.encode()),'payloads':len(payloads)}))
