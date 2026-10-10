"""Stage and run the fixed CPU reader; transfer only integer tables/receipts."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess

H=Path(__file__).resolve().parent;P=H.parent
S=P/'imdb_common_wrong_confidence_bins_source_20261010_v1'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(S/'MANIFEST.json')=='195bcd04f3e7f03cf847b8212614c638944bcb4571975ec101a5db5bd21f1a25'
for row in json.loads((S/'MANIFEST.json').read_text())['files']:
 q=S/row['path'];assert sha(q)==row['sha256'] and q.stat().st_size==row['bytes']
review=json.loads((S/'ROOT_REVIEW_TEMPLATE_DISABLED.json').read_text())
review.update(root_source_review_approved=True,root_stored_confidence_readout_approved=True,
 source_manifest_sha256=sha(S/'MANIFEST.json'),source_seal_sha256=sha(S/'SEAL.json'),
 protocol_sha256=sha(S/'PROTOCOL.json'),input_bindings_sha256=sha(S/'INPUT_BINDINGS.json'),
 UTC=datetime.now(timezone.utc).isoformat(),source_only=True,numerical_or_array_or_server_reads=False,
 scope='Original observed-event probability distances, original masks, complete24 custody and fixed bins; no reconstruction/threshold selection or model.',
 material_blockers=[],owner_source_sha256=sha(H/'run_owned.py'),finite_CPU_owner_reviewed=True)
q=H/'ROOT_REVIEW.json';assert not q.exists();q.write_text(json.dumps(review,indent=2)+'\n')
runtime=json.loads((P/'pubmed_factor1_controls_source_20261010_v1/DATA_AND_RUNTIME.json').read_text())['runtime']
(H/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
payload={}
for q in [H/'run_owned.py',H/'ROOT_REVIEW.json',H/'RUNTIME.json']:
 data=q.read_bytes();payload[str(q.relative_to(P))]=dict(base64=base64.b64encode(data).decode(),sha256=sha(q),bytes=len(data))
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R)
H=P/'imdb_confidence_bins_root_execution_20261010_v1';S=P/'imdb_common_wrong_confidence_bins_source_20261010_v1'
PAYLOAD=__PAYLOAD__
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert not H.exists();H.mkdir()
for n,row in PAYLOAD.items():
 q=P/n;assert q.resolve().is_relative_to(H);data=base64.b64decode(row['base64']);assert hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'];q.write_bytes(data)
review=json.loads((H/'ROOT_REVIEW.json').read_text());assert sha(S/'MANIFEST.json')==review['source_manifest_sha256']
runtime=json.loads((H/'RUNTIME.json').read_text());python=Path(runtime['python']['path']);assert sha(python)==runtime['python']['sha256']
release=json.loads((S/'RELEASE_TEMPLATE_DISABLED.json').read_text())
for k in ['source_manifest_sha256','source_seal_sha256','protocol_sha256','input_bindings_sha256']:release[k]=review[k]
q=H/'ROOT_REVIEW.json'
release.update(enabled=True,root_source_review_approved=True,root_stored_confidence_readout_approved=True,finite_external_owner_bound=True,
 root_review=dict(path=str(q),sha256=sha(q),bytes=q.stat().st_size),runtime_executable=str(python.resolve()),expected_torch_version='2.1.2+cu118',
 output_directory=str(H/'execution'),wall_budget_seconds=300,automatic_retry=False,re_inference_allowed=False)
(H/'RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
result=subprocess.run(['python3','-B',str(H/'run_owned.py')],cwd=R,capture_output=True,text=True,timeout=330)
files={}
for q in [H/'LAUNCH.json',H/'TERMINAL.json',H/'RELEASE.json',H/'execution/TERMINAL.json',H/'execution/COUNTS.json']:
 if q.is_file():
  data=q.read_bytes();assert len(data)<16*1024**2;files[str(q.relative_to(P))]=dict(base64=base64.b64encode(data).decode(),sha256=sha(q),bytes=len(data))
print(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,files=files,
 worker_log_tail=(H/'WORKER.log').read_text()[-5000:] if result.returncode else None)))
'''.replace('__PAYLOAD__',repr(payload))
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15',
 '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -']
result=subprocess.run(argv,input=remote,capture_output=True,text=True,timeout=350)
(H/'TRANSPORT.json').write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
if result.returncode:print(result.stderr);raise SystemExit(result.returncode)
value=json.loads(result.stdout)
for n,row in value['files'].items():
 data=base64.b64decode(row['base64']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 q=H/'fetched'/n;assert q.resolve().is_relative_to((H/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
print(json.dumps(dict(exit_code=value['exit_code'],stdout=value['stdout'],stderr=value['stderr'],worker_log_tail=value['worker_log_tail'],fetched=list(value['files']))))
raise SystemExit(value['exit_code'])
