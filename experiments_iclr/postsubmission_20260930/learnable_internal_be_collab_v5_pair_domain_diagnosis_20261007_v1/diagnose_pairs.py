"""Source-array domain metadata only; no export/pair repair/models/scoring."""
import hashlib,io,json,os,resource,socket,subprocess,time,zipfile
from pathlib import Path
started=time.monotonic()
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
SOURCE=PHASE/'learnable_internal_be_contrastive_multitask_suite_20261007_v5'
HERE=Path(__file__).resolve().parent
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[GPU]
assert Path.cwd().resolve()==REPO and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and HERE.is_relative_to(PHASE)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as source:
  for chunk in iter(lambda:source.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
assert sha(SOURCE/'MANIFEST.json')=='ecff016e9a68af1625293bf0166d1428bca20d7c2ac12b64b37914dda5c3af3b'
plan=json.loads((SOURCE/'EXPORT_CUSTODY_PLAN.json').read_text())['collab']
archive=(PHASE/plan['archive']['path']).resolve();assert archive.is_relative_to(PHASE) and sha(archive)==plan['archive']['sha256']
origin_path=PHASE/plan['singleton_transfer_receipt']['path'];assert sha(origin_path)==plan['singleton_transfer_receipt']['sha256']
origin=json.loads(origin_path.read_text());assert origin['GPU_uuid']==GPU and origin['archive_sha256']==plan['archive']['sha256']
import numpy as np
import torch
assert torch.__version__=='2.1.2+cu118' and np.__version__=='1.26.4'
def digest(value):
 value=np.ascontiguousarray(value);h=hashlib.sha256(str((value.shape,value.dtype)).encode());h.update(memoryview(value).cast('B'));return h.hexdigest()
with zipfile.ZipFile(archive) as packed:
 def load(name):
  assert name in plan['members'];data=packed.read(name);assert hashlib.sha256(data).hexdigest()==plan['members'][name]
  return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
 train=load('collab/split/time/train.pt');valid=load('collab/split/time/valid.pt')
def array(v):return v.numpy() if isinstance(v,torch.Tensor) else v
train={k:array(v) for k,v in train.items()};valid={k:array(v) for k,v in valid.items()}
assert set(train)=={'edge','weight','year'} and set(valid)=={'edge','weight','year','edge_neg'}
arrays={'TRAIN_positive':train['edge'],'VALID_positive':valid['edge'],'VALID_negative':valid['edge_neg']}
actual={'TRAIN_edge':digest(train['edge']),'VALID_edge':digest(valid['edge']),'VALID_negative':digest(valid['edge_neg']),'TRAIN_year':digest(train['year']),'VALID_year':digest(valid['year'])}
assert all(actual[k]==plan['expected_public_array_digests'][k] for k in actual)
metadata={key:{'shape':list(value.shape),'dtype':str(value.dtype),'minimum_node_ID':int(value.min()),'maximum_node_ID':int(value.max()),'nonnegative_all':bool((value>=0).all()),'below_public_node_count_all':bool((value<235868).all()),'self_pair_record_count':int((value[:,0]==value[:,1]).sum())} for key,value in arrays.items()}
assert not torch.cuda.is_initialized()
usage=resource.getrusage(resource.RUSAGE_SELF)
receipt={'schema':'internal-be-Collab-v5-pair-domain-failure-diagnosis-v1','source_manifest_sha256':sha(SOURCE/'MANIFEST.json'),'authentic_archive':plan['archive'],'authentic_source_array_digests_matched':actual,'pair_domains':metadata,'CPU_only':True,'cuda_initialized':False,'scientific_fits':0,'models':0,'predictive_scores_computed':False,'TEST_values_parsed':False,'export_retry':False,'records_dropped_or_changed':0,'inclusive_seconds':time.monotonic()-started,'user_CPU_seconds':usage.ru_utime,'system_CPU_seconds':usage.ru_stime,'peak_RSS_bytes':int(usage.ru_maxrss)*1024,'input_blocks':usage.ru_inblock,'output_blocks':usage.ru_oublock,'program_sha256':sha(__file__)}
path=HERE/'DIAGNOSIS.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt))
