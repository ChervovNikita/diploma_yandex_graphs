"""Admit one frozen peer-feedback study after successful dropout acquisition."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

HERE = Path(__file__).resolve().parent
P = HERE.parent
S = P / 'meanSAGE_attribute_peer_feedback_execution_source_20261011_v3'
Q = P / 'meanSAGE_attribute_peer_feedback_full_input_qualification_20261011_v2'
R = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'

def receipt(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

manifest = json.loads((S / 'MANIFEST.json').read_text())
assert receipt(S / 'MANIFEST.json')['sha256'] == 'f83cb34bae29bda7bd542075bb3009d973fc7ab3bf494bd2cdfcf0801e704cd4'
for rel, fixed in manifest['files'].items():
    row = receipt(S / rel)
    assert row['bytes'] == fixed['bytes'] and row['sha256'] == fixed['sha256']
binding = json.loads((Q / 'COMPLETE_QUALIFICATION_BINDINGS.json').read_text())
passed = json.loads((Q / 'actual_receipts/qualification_v1/QUALIFICATION.json').read_text())
ended = json.loads((Q / 'actual_receipts/qualification_owner_v1/OWNER_END.json').read_text())
assert passed['complete'] and passed['status'] == 'pass' and ended['status'] == 'COMPLETE_EXIT_0'
assert receipt(Q / 'actual_receipts/qualification_v1/QUALIFICATION.json')['sha256'] == binding['passed_qualification_local']['sha256']
assert receipt(Q / 'actual_receipts/qualification_owner_v1/OWNER_END.json')['sha256'] == binding['passed_owner_local']['sha256']

decision = dict(
    UTC=datetime.now(timezone.utc).isoformat(), scientific_full_fit_admitted=True,
    source_manifest_sha256=receipt(S / 'MANIFEST.json')['sha256'],
    proposal_manifest_sha256='db1dd1aaff443275d37dff8f715b1e830a463f7db48eeee0eddf20eabef6b6c5',
    blocks=[10103, 10427, 10949],
    arms=['P1', 'F1', 'I4', 'S4', 'PA1', 'IA4', 'SA4', 'SM4', 'SX4', 'UX4'],
    banks=30, optimizer_trajectories=48, physical_backbones=57, CAL_endpoints=30,
    qualification_result=binding['passed_qualification_local'],
    qualification_owner=binding['passed_owner_local'],
    first_failed_zero_step_qualification_retained=True,
    qualified_states_reused=False, prediction_metrics_opened_before_decision=False,
    dropout_results_opened_before_decision=False,
    scheduling='After successful whole dropout acquisition owner closure; independent of predictive outcomes.',
    limits=dict(wall_seconds=43200, cuda_bytes=16 * 1024**3, rss_bytes=16 * 1024**3),
    fresh_constructors_and_optimizer_states=True,
    all_48_selected_states_57_bodies_30_exports_before_CAL=True,
    all_30_CAL_15000_updates_before_whole_DEV=True,
    serving='native mean member class probabilities',
    calibration='mean_m softmax(member_log_probs_m/T); CPUfloat64 Adam .05;500updates;latest tied minimum steps0..500',
    scientific_scope='Three paired development protocols on encountered Physics; no unused confirmation or novelty certification.',
    no_automatic_retry=True, no_partial_quality=True, original_paper_scores_unchanged=True,
    reviewed_contracts=['source meanSAGE backbone/TF1 Adam and latest-tied SELECT total',
                       'genuine I4/IA4 separate full own objectives, optimizers and selectors',
                       'UX4 joint live gradients and selector; not genuine independent',
                       'self-only and mean-before-map controls for nonlinear other-member feedback',
                       'passed full-input resource and strict model/optimizer/global/route RNG restore']
)
with (HERE / 'ROOT_DECISION.json').open('x') as handle:
    handle.write(json.dumps(decision, indent=2) + '\n')
payload = dict(decision=decision,
               decision_sha256=receipt(HERE / 'ROOT_DECISION.json')['sha256'],
               decision_data=base64.b64encode((HERE / 'ROOT_DECISION.json').read_bytes()).decode(),
               source_manifest_sha256=receipt(S / 'MANIFEST.json')['sha256'],
               qualification_result_sha256=binding['passed_qualification_local']['sha256'],
               qualification_owner_sha256=binding['passed_owner_local']['sha256'])
(HERE / 'DEPLOY_PAYLOAD.json').write_text(json.dumps(payload, indent=2) + '\n')
remote = r'''
import base64,hashlib,json,os,pathlib,socket,subprocess,sys,time,zlib
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
S=P/'meanSAGE_attribute_peer_feedback_execution_source_20261011_v3';Q=P/'meanSAGE_attribute_peer_feedback_full_input_qualification_20261011_v2';E=P/'meanSAGE_attribute_peer_feedback_acquisition_root_20261011_v1'
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,a):
 with p.open('x') as f:f.write(json.dumps(a,indent=2)+'\n')
def closed(p):
 a=json.loads(p.read_text());assert a['status']=='COMPLETE_EXIT_0' and a['exit_code']==0 and a['direct_child_wait'] and a['child_pid_absent'] and a['owned_cuda_pid_absent'];return a
assert not E.exists(),'Existing admission retained; do not repeat launch'
dropout=P/'Physics_dropout_pilot_acquisition_root_20261011_v1'
closed(dropout/'owner_v1/OWNER_END.json')
assert receipt(S/'MANIFEST.json')['sha256']==payload['source_manifest_sha256']
for rel,fixed in json.loads((S/'MANIFEST.json').read_text())['files'].items():
 got=receipt(S/rel);assert got['bytes']==fixed['bytes'] and got['sha256']==fixed['sha256']
assert receipt(Q/'qualification_v1/QUALIFICATION.json')['sha256']==payload['qualification_result_sha256']
assert receipt(Q/'qualification_owner_v1/OWNER_END.json')['sha256']==payload['qualification_owner_sha256'];closed(Q/'qualification_owner_v1/OWNER_END.json')
qualified=json.loads((Q/'qualification_v1/QUALIFICATION.json').read_text());assert qualified['complete'] and qualified['status']=='pass'
E.mkdir();os.chdir(R)
data=base64.b64decode(payload['decision_data']);assert hashlib.sha256(data).hexdigest()==payload['decision_sha256']
(E/'ROOT_DECISION.json').write_bytes(data)
config=json.loads((Q/'QUALIFY_CONFIG.json').read_text())
config.update(stage='acquire',root_decision=receipt(E/'ROOT_DECISION.json'),qualification=receipt(Q/'qualification_v1/QUALIFICATION.json'),qualification_owner_end=receipt(Q/'qualification_owner_v1/OWNER_END.json'),output=str(E/'acquisition_v1'),owner_output=str(E/'owner_v1'),limits=payload['decision']['limits'])
write(E/'CONFIG.json',config)
command=[str(R/'.venv/bin/python'),'-B','-u',str(S/'owner.py'),str(E/'CONFIG.json')]
with (E/'owner_launcher.log').open('xb') as log:
 owner=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 deadline=time.monotonic()+15
 while not (E/'owner_v1/OWNER_START.json').exists() and owner.poll() is None and time.monotonic()<deadline:time.sleep(.05)
 assert owner.poll() is None and (E/'owner_v1/OWNER_START.json').exists(),'Owner failed before start; retained log'
 start=json.loads((E/'owner_v1/OWNER_START.json').read_text());assert start['direct_child_wait'] and start['owner']['pid']==owner.pid
launch=dict(status='RUNNING',owner_pid=owner.pid,child_pid=start['child']['pid'],banks=30,optimizer_trajectories=48,physical_backbones=57,CAL=30,source_manifest=config['source_manifest'],config=receipt(E/'CONFIG.json'),partial_quality_opened=False,CAL_DEV_truth_read=False,command=command,prior_dropout_owner_end=receipt(dropout/'owner_v1/OWNER_END.json'))
write(E/'LAUNCH.json',launch)
rows=[]
for rel in ['ROOT_DECISION.json','CONFIG.json','LAUNCH.json','owner_v1/OWNER_START.json']:
 p=E/rel;r=receipt(p);r.update(name=rel,data=base64.b64encode(p.read_bytes()).decode());rows.append(r)
print(json.dumps(dict(launch=launch,files=rows)))
'''
(HERE / 'REMOTE_CODE.py.txt').write_text(remote)
spec = importlib.util.spec_from_file_location('transport', P / 'publication/publish_exact_inventory_v19.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', remote, R, 'GPU-44039938-fd82-41d2-fefd-de71514e2fac', 'execute'])
argv = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru', command]
result = module.terminal_transport(argv, input=base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode(), capture_output=True, text=True, timeout=60)
(HERE / 'TRANSPORT.json').write_text(json.dumps(dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2) + '\n')
if result.returncode:
    print(json.dumps(dict(exit_code=result.returncode, stdout=result.stdout[-2500:], stderr=result.stderr[-2500:])))
    raise SystemExit(result.returncode)
value = json.loads(result.stdout)
out = HERE / 'actual_receipts'
out.mkdir()
for row in value['files']:
    data = base64.b64decode(row['data'])
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    path = out / row['name']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
print(json.dumps(value['launch']))
