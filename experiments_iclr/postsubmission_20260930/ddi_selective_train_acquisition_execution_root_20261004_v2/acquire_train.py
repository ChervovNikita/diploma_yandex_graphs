from pathlib import Path
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
