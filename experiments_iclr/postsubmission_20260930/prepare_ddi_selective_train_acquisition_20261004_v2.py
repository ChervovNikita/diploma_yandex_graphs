"""Bind official DDI metadata and acquire only decoded TRAIN on authorized18.77."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import base64
import json

PHASE = Path(__file__).resolve().parent
HERE = PHASE/'ddi_selective_train_acquisition_execution_root_20261004_v2'
HERE.mkdir(exist_ok=False)
METADATA = PHASE/'dense_graph_auxiliary_transfer_dataset_scout_20261004_v1_followup_metadata/DDI_OFFICIAL_METADATA.json'
meta = json.loads(METADATA.read_text())
assert meta['values']['url']=='http://snap.stanford.edu/ogb/data/linkproppred/ddi.zip'
assert meta['values']['split']=='target' and meta['values']['version']=='1'
plan = dict(UTC=datetime.now(timezone.utc).isoformat(), dataset='ogbl-ddi',
    source_metadata_sha256=hashlib.sha256(METADATA.read_bytes()).hexdigest(),
    official_metadata_url=meta['values']['url'],
    acquisition_url=meta['values']['url'],
    URL_change='Use official metadata URL exactly; prior HTTPS attempt failed before any payload was decoded', prior_attempt='ddi_selective_train_acquisition_execution_root_20261004_v1/MONITOR_01_RESULT.json',
    selected_member_candidates=['ddi/split/target/train.pt','ddi/split/target/train.csv.gz'],
    node_count_member='ddi/raw/num-node-list.csv.gz', expected_nodes=4267,
    archive_bytes_cap=536870912, wall_seconds_cap=600,
    heldout_archive_bytes_downloaded_opaque=True, heldout_split_payloads_decoded=False,
    preserve_official_train_row_order=True, training_or_scoring=False,
    scientific_purpose='Freeze and measure native TRAIN informative support/cost before prospective DDI fitting.',
    automatic_retry=False,
    interpreter='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12',
    interpreter_sha256='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2')
driver = r'''from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,urlopen
import csv,gzip,hashlib,io,json,os,sys,time,zipfile
root=Path(__file__).resolve().parent
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
assert Path.cwd()==repo and root.is_relative_to(repo) and os.uname().nodename=='peptide'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for part in iter(lambda:stream.read(1048576),b''):h.update(part)
 return h.hexdigest()
def save(name,data):
 with (root/name).open('x') as out:json.dump(data,out,indent=2);out.write('\n')
start=time.monotonic();plan=json.loads((root/'PLAN.json').read_text())
result={'UTC_started':datetime.now(timezone.utc).isoformat(),'training_or_scoring':False,'heldout_split_payloads_decoded':False,'decoded_members':[],'automatic_retry':False}
try:
 assert file_sha(Path(sys.executable))==plan['interpreter_sha256']
 save('ACQUISITION_ATTEMPT_SPENT.json',{'UTC':result['UTC_started'],'plan_sha256':file_sha(root/'PLAN.json')})
 archive=root/'official_ddi_v1.zip';h=hashlib.sha256();size=0
 with urlopen(Request(plan['acquisition_url'],headers={'User-Agent':'GNNM-research/1.0'}),timeout=30) as response,archive.open('xb') as stream:
  result['resolved_url']=response.url;result['HTTP_status']=response.status
  declared=response.headers.get('Content-Length')
  if declared:assert int(declared)<=plan['archive_bytes_cap']
  while True:
   part=response.read(1048576)
   if not part:break
   size+=len(part);assert size<=plan['archive_bytes_cap'] and time.monotonic()-start<plan['wall_seconds_cap']
   stream.write(part);h.update(part)
  stream.flush();os.fsync(stream.fileno())
 result.update(archive_bytes=size,archive_sha256=h.hexdigest(),archive_contains_opaque_heldout_bytes=True)
 with zipfile.ZipFile(archive) as z:
  members=[dict(name=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,CRC=x.CRC) for x in z.infolist()]
  save('ARCHIVE_CENTRAL_DIRECTORY.json',members)
  names={x['name'] for x in members};selected=[x for x in plan['selected_member_candidates'] if x in names]
  assert len(selected)==1,'Expected exactly one separately stored official TRAIN payload; do not use a combined split file'
  node=plan['node_count_member'];assert node in names
  assert z.getinfo(selected[0]).file_size<=134217728 and z.getinfo(node).file_size<=1048576
  raw=z.read(selected[0]);node_raw=z.read(node);result['decoded_members']=[selected[0],node]
  with (root/Path(selected[0]).name).open('xb') as stream:stream.write(raw)
  with (root/Path(node).name).open('xb') as stream:stream.write(node_raw)
  num_nodes_rows=list(csv.reader(io.StringIO(gzip.decompress(node_raw).decode())))
  assert len(num_nodes_rows)==1 and len(num_nodes_rows[0])==1
  num_nodes=int(num_nodes_rows[0][0]);assert num_nodes==plan['expected_nodes']
 import torch
 import numpy as np
 assert torch.__version__.split('+')[0]=='2.7.1' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 torch.set_num_threads(2)
 if selected[0].endswith('.pt'):
  # Trusted official OGB split, using the author's legacy-pickle format. Only TRAIN is decoded.
  value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
  assert isinstance(value,dict) and set(value)=={'edge'}
  edge=value['edge']
  if isinstance(edge,np.ndarray):edge=torch.from_numpy(edge)
 else:
  data=np.loadtxt(io.BytesIO(gzip.decompress(raw)),delimiter=',',dtype=np.int64)
  edge=torch.from_numpy(data)
 assert isinstance(edge,torch.Tensor) and edge.ndim==2 and edge.shape[1]==2 and edge.shape[0]>0
 assert edge.dtype in (torch.int32,torch.int64) and edge.device.type=='cpu'
 edge=edge.to(dtype=torch.int64).contiguous();assert int(edge.min())>=0 and int(edge.max())<num_nodes
 canonical=torch.sort(edge,dim=1).values
 unique=torch.unique(canonical,dim=0).shape[0]
 result.update(TRAIN_edges=int(edge.shape[0]),nodes=num_nodes,self_loops=int((edge[:,0]==edge[:,1]).sum()),unique_undirected_edges=int(unique),TRAIN_row_order_preserved=True,TRAIN_tensor_sha256=sha(edge.numpy().tobytes()),selected_member_sha256=sha(raw),node_count_member_sha256=sha(node_raw),torch_version=torch.__version__,data_format=str(edge.dtype))
 torch.save({'edge':edge,'num_nodes':num_nodes},root/'TRAIN_ONLY.pt')
 result.update(status='COMPLETE_OFFICIAL_TRAIN_ACQUISITION_ONLY',TRAIN_ONLY_sha256=file_sha(root/'TRAIN_ONLY.pt'),TRAIN_support_census_completed=False,predictive_experiment=False)
except BaseException as exc:
 result.update(status='FAILED_NO_AUTOMATIC_RETRY',error_type=type(exc).__name__,error=str(exc))
finally:
 result['wall_seconds']=time.monotonic()-start;result['UTC_completed']=datetime.now(timezone.utc).isoformat();save('ACQUISITION_RESULT.json',result)
sys.exit(0 if result['status']=='COMPLETE_OFFICIAL_TRAIN_ACQUISITION_ONLY' else 1)
'''
compile(driver,'acquire_train.py','exec')
for name,raw in [('PLAN.json',(json.dumps(plan,indent=2)+'\n').encode()),('acquire_train.py',driver.encode()),('DDI_OFFICIAL_METADATA.json',METADATA.read_bytes())]:
    (HERE/name).write_bytes(raw)
payload=[dict(path=path.name,data=base64.b64encode(path.read_bytes()).decode(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in HERE.iterdir() if path.is_file()]
spec=importlib.util.spec_from_file_location('ddi_relay',PHASE/'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py')
relay=importlib.util.module_from_spec(spec);spec.loader.exec_module(relay);relay.HERE=HERE
remote='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/'+HERE.name
code="from pathlib import Path\nimport base64,hashlib,json,os,subprocess\nrepo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');assert Path.cwd()==repo and os.uname().nodename=='peptide'\nroot=Path("+repr(remote)+");root.mkdir(exist_ok=False)\nrows="+repr(payload)+"\nfor row in rows:\n raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']\n with (root/row['path']).open('xb') as out:out.write(raw)\nplan=json.loads((root/'PLAN.json').read_text());assert hashlib.sha256(Path(plan['interpreter']).read_bytes()).hexdigest()==plan['interpreter_sha256']\nenv=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(repo/'.gnnm_runtime/buddy_extra_v1/site'))\nwith (root/'DRIVER_STDOUT.txt').open('xb') as out,(root/'DRIVER_STDERR.txt').open('xb') as err:\n child=subprocess.Popen([plan['interpreter'],'-B',str(root/'acquire_train.py')],cwd=repo,env=env,stdout=out,stderr=err,start_new_session=True)\nf=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split();receipt=dict(pid=child.pid,start_ticks=int(f[19]),root=str(root),acquisition_only=True)\nwith (root/'DETACHED_LAUNCH.json').open('x') as out:json.dump(receipt,out,indent=2)\nprint(json.dumps(receipt))\n"
compile(code,'remote_stage_and_launch.py','exec')
launch=relay.run('ddi_train_acquire_v2_launch_20261004',code)
(HERE/'DETACHED_LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
