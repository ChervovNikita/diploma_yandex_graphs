"""Bind actual graph-view comparison inputs without fitting or predictive scoring.

Only hashes and compact role/view metadata leave the authorized server. Official
TRAIN labels define views. VALIDATION labels are hashed for the prospective
selection contract. No TEST labels, existing fitted outputs or checkpoints open.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BANK = 'eb34fcb8bc406219007ee39ee9789f1de9bc56acf3cb16164aeab95b4493752f'
BANK_PROTOCOL = '55f4146b450f2e04c518e07df9a118bb21f7dc65e18dc659eb34e6dbe8edfb46'
REFERENCE = '3a054568e54baed6ee6526256b6d37fa5f82b449257d7983c1ef54eace8236e0'
NATIVE = 'fbe1aec5eafb029fce9c9d42ebf9a5b94b98cfbc2ce9881502f0e3fc8e192ca6'

REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
repo=Path(REPO);phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[UUID]
source=phase/'accuracy_first_graph_view_source_preparation_20261004_v2'
sys.path.insert(0,str(source))
from runtime import verify_package
verify_package(BANK,BANK_PROTOCOL)
from views import TrainRole,construct_views,native_edges,digest
import numpy as np
def checked(row):
 p=phase/row['path']
 assert p.resolve().is_relative_to(phase) and not p.is_symlink() and p.stat().st_size==row['bytes']
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
 return p
started=time.perf_counter()
data=json.loads(checked(bindings['manifest']).read_text())
assert data['public_graph']==bindings['public_graph'] and data['train_labels']==bindings['train_labels']
assert data['validation_labels']==bindings['validation_labels']
with np.load(checked(data['public_graph']),allow_pickle=False) as z:
 assert set(z.files)=={'features','edge_index','train_mask','val_mask','test_mask'}
 x=z['features'].copy();raw=z['edge_index'].copy()
 masks={k:z[k].copy() for k in ('train_mask','val_mask','test_mask')}
assert x.shape==(24492,300) and x.dtype==np.float32 and np.isfinite(x).all()
assert raw.dtype==np.int64 and raw.shape[0]==2
assert all(m.dtype==np.bool_ and m.shape==(24492,10) for m in masks.values())
feature={'dtype':'torch.float32','shape':list(x.shape),'logical_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}
edges=native_edges(24492,((int(a),int(b)) for a,b in raw.T))
records={};coverage=[]
for split,seed in ((0,17),(1,29),(2,43)):
 ids={k:tuple(int(i) for i in np.flatnonzero(m[:,split])) for k,m in masks.items()}
 with np.load(checked(data['train_labels'][str(split)]),allow_pickle=False) as z:
  assert set(z.files)=={'ids','labels'} and np.array_equal(z['ids'],ids['train_mask'])
  labels=tuple(int(i) for i in z['labels'])
 with np.load(checked(data['validation_labels'][str(split)]),allow_pickle=False) as z:
  assert set(z.files)=={'ids','labels'} and np.array_equal(z['ids'],ids['val_mask'])
  validation=tuple(int(i) for i in z['labels'])
 assert len(validation)==6123 and all(i in range(5) for i in validation)
 role=TrainRole(24492,split,ids['train_mask'],labels,ids['val_mask'],ids['test_mask'])
 bundle=construct_views(role,edges,seed)
 path=phase/'graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1'/('split%d_COVERAGE.json'%split)
 assert not path.is_symlink() and path.resolve().is_relative_to(phase)
 saved=json.loads(path.read_text())
 assert saved==dict(bundle['coverage'],coverage_origin='official_TRAIN',scientific_freeze_eligible=bundle['coverage']['coverage_eligible'])
 assert saved['scientific_freeze_eligible'] is True
 records[str(split)]={'role':role.identity(),'native_edges_sha256':bundle['coverage']['native_edges_sha256'],
  'feature_identity':feature,'validation_labels_sha256':digest(validation)}
 coverage.append({'split':split,'view_seed':seed,'descriptor':{'path':str(path.relative_to(phase)),
  'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
  'views':bundle['coverage']['views']})
value={'schema':'accuracy-first-paired-input-identities-v1','UTC':datetime.now(timezone.utc).isoformat(),
 'bank_manifest_sha256':BANK,'reference_manifest_sha256':REFERENCE,'native_gnnm_manifest_sha256':NATIVE,
 'input_bindings_by_split':records,'official_TRAIN_coverage_by_split':coverage,'data_bindings':bindings,
 'route':{'login':LOGIN,'repository':str(repo),'GPU_UUID':UUID},'wall_seconds':time.perf_counter()-started,
 'scientific_comparison_frozen':False,'training_updates':0,'features_and_TRAIN_labels_decoded':True,
 'VALIDATION_labels_decoded_for_identity_only':True,'TEST_labels_read':False,
 'predictive_values_read':False,'fitted_outputs_or_checkpoints_read':False,'arrays_fetched_to_Mac':False}
root=phase/OUTPUT;root.mkdir(exist_ok=True)
with (root/'INPUT_IDENTITIES.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
'''


def main():
    manifest = PHASE / 'amazon_ratings_native_warm_execution_root_20261003_v3/data/DATA_MANIFEST.json'
    data = json.loads(manifest.read_text())
    bindings = dict(manifest=dict(path=str(manifest.relative_to(PHASE)), bytes=manifest.stat().st_size,
                    sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),
                    **{key: data[key] for key in ('public_graph','train_labels','validation_labels')})
    values = dict(REPO=REPO, LOGIN=LOGIN, UUID=UUID, BANK=BANK, BANK_PROTOCOL=BANK_PROTOCOL,
                  REFERENCE=REFERENCE, NATIVE=NATIVE, bindings=bindings, OUTPUT=HERE.name)
    code = ''.join(k+'='+repr(v)+'\n' for k,v in values.items()) + REMOTE
    compile(code, '<paired-input-identities>', 'exec')
    with (HERE/'REMOTE_SOURCE.py.txt').open('x') as stream:
        stream.write(code)
    command = 'cd '+shlex.quote(REPO)+' && '+shlex.join([REPO+'/.venv/bin/python','-I','-B','-c',code])
    ssh = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no',
           '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',LOGIN,command]
    result = subprocess.run(ssh,capture_output=True,text=True,timeout=90)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
                   stderr=result.stderr,remote_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
                   stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),private_key_contents_read=False)
    with (HERE/'TRANSPORT.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    assert result.returncode==0, result.stderr
    value=json.loads(result.stdout)
    with (HERE/'INPUT_IDENTITIES.json').open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(UTC=value['UTC'],splits=list(value['input_bindings_by_split']),
                         training_updates=0,predictive_values_read=False,TEST_labels_read=False,
                         scientific_comparison_frozen=False)))


if __name__=='__main__':
    main()
