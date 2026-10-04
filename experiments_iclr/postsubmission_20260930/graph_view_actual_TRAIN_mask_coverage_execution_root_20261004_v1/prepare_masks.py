"""Construct the declared fixed views on official TRAIN labels, on server CPU.

This produces topology/mask provenance, not a predictive experiment or freeze.
Actual edge arrays stay on the server; only small coverage records are fetched.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,shlex,subprocess

HERE=Path(__file__).resolve().parent
P=HERE.parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
MANIFEST='135e30466c703b1929eadd680b7152330620e4cde75152ec69317458580efa2a'
PROTOCOL='55f4146b450f2e04c518e07df9a118bb21f7dc65e18dc659eb34e6dbe8edfb46'


def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
repo=Path(REPO);phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[UUID]
source=phase/'accuracy_first_graph_view_source_preparation_20261004_v1'
sys.path.insert(0,str(source))
from runtime import verify_package
verify_package(MANIFEST,PROTOCOL)
from views import TrainRole,construct_views,save_coverage,native_edges
import numpy as np
def verify(row):
 p=phase/row['path'];assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 assert p.stat().st_size==row['bytes']
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
 return p
started=time.perf_counter()
data=json.loads(verify(bindings['manifest']).read_text())
assert data['public_graph']==bindings['public_graph'] and data['train_labels']==bindings['train_labels']
with np.load(verify(data['public_graph']),allow_pickle=False) as z:
 raw=z['edge_index'];masks={k:z[k] for k in ('train_mask','val_mask','test_mask')}
 assert set(z.files)=={'features','edge_index','train_mask','val_mask','test_mask'}
assert raw.dtype==np.int64 and all(v.dtype==np.bool_ and v.shape==(24492,10) for v in masks.values())
edges=native_edges(24492,((int(a),int(b)) for a,b in raw.T))
root=phase/'graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1';root.mkdir(exist_ok=True)
records=[]
for split,seed in ((0,17),(1,29),(2,43)):
 ids=tuple(int(x) for x in np.flatnonzero(masks['train_mask'][:,split]))
 with np.load(verify(data['train_labels'][str(split)]),allow_pickle=False) as z:
  assert set(z.files)=={'ids','labels'} and np.array_equal(z['ids'],ids)
  labels=tuple(int(y) for y in z['labels'])
 role=TrainRole(24492,split,ids,labels,tuple(int(x) for x in np.flatnonzero(masks['val_mask'][:,split])),
                tuple(int(x) for x in np.flatnonzero(masks['test_mask'][:,split])))
 bundle=construct_views(role,edges,seed)
 path=root/('split%d_COVERAGE.json'%split)
 coverage=save_coverage(bundle,path,origin='official_TRAIN',execute=True)
 descriptor=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 arrays={name:np.asarray(value,dtype=np.int64).T for name,value in dict(bundle['views'],native=edges).items()}
 archive=root/('split%d_FIXED_VIEWS.npz'%split)
 with archive.open('xb') as f:np.savez_compressed(f,**arrays)
 records.append(dict(split=split,view_seed=seed,coverage=coverage,coverage_descriptor=descriptor,
     topology_archive=dict(path=str(archive.relative_to(phase)),bytes=archive.stat().st_size,
                          sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
     topology_archive_fetched_to_Mac=False))
value=dict(UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=MANIFEST,protocol_sha256=PROTOCOL,
 input_bindings=bindings,route=dict(login=LOGIN,repository=str(repo),GPU_UUID=UUID),splits=records,
 wall_seconds=time.perf_counter()-started,features_decoded=False,VALIDATION_or_TEST_labels_read=False,
 predictive_values_read=False,training_updates=0,views_created=True,scientific_comparison_frozen=False,
 CPU_only=True,numerical_backbone_operations_run=False)
with (root/'ACTUAL_MASK_COVERAGE.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
'''


def main():
    bindings=json.loads((P/'graph_view_train_coverage_census_execution_root_20261004_v1/INPUT_BINDINGS.json').read_text())
    code='REPO='+repr(REPO)+'\nLOGIN='+repr(LOGIN)+'\nUUID='+repr(UUID)+'\nMANIFEST='+repr(MANIFEST)+'\nPROTOCOL='+repr(PROTOCOL)+'\nbindings='+repr(bindings)+'\n'+REMOTE
    compile(code,'<actual-TRAIN-mask-coverage>','exec')
    (HERE/'REMOTE_SOURCE.py.txt').write_text(code)
    command='cd '+shlex.quote(REPO)+' && '+shlex.join([REPO+'/.venv/bin/python','-I','-B','-c',code])
    ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes',
         '-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',LOGIN,command]
    r=subprocess.run(ssh,capture_output=True,text=True,timeout=90)
    save('TRANSPORT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,
         remote_source_sha256=hashlib.sha256(code.encode()).hexdigest(),private_key_contents_read=False))
    assert r.returncode==0,r.stderr
    value=json.loads(r.stdout);save('ACTUAL_MASK_COVERAGE.json',value)
    for record in value['splits']:
        coverage=record['coverage'];save('split%d_COVERAGE.json'%record['split'],coverage)
    print(json.dumps(dict(UTC=value['UTC'],wall_seconds=value['wall_seconds'],
         splits=[dict(split=r['split'],eligible=r['coverage']['scientific_freeze_eligible'],categories=r['coverage']['categories']) for r in value['splits']],
         predictive_values_read=False,scientific_comparison_frozen=False),indent=2))


if __name__=='__main__':main()
