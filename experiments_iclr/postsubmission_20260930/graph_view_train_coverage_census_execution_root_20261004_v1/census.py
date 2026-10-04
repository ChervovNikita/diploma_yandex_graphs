"""Count TRAIN-defined graph-view support on the authorized one-GPU route.

This reads public topology, official TRAIN masks and compact TRAIN labels only.
It creates no views, imports no model, and reads no predictive or heldout values.
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


def descriptor(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time
started=time.perf_counter()
repo=Path(REPO);phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[UUID]
def verify(row):
 p=phase/row['path'];assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 assert p.stat().st_size==row['bytes']
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
 return p
manifest=json.loads(verify(bindings['manifest']).read_text())
assert manifest['public_graph']==bindings['public_graph'] and manifest['train_labels']==bindings['train_labels']
import numpy as np
with np.load(verify(bindings['public_graph']),allow_pickle=False) as z:
 assert set(z.files)=={'features','edge_index','train_mask','val_mask','test_mask'}
 edge=z['edge_index'];train=z['train_mask']
n=24492
assert edge.dtype==np.int64 and edge.shape[0]==2 and edge.min()>=0 and edge.max()<n
assert train.dtype==np.bool_ and train.shape==(n,10)
raw_loops=int(np.count_nonzero(edge[0]==edge[1]))
pairs=np.unique(np.sort(edge[:,edge[0]!=edge[1]].T,axis=1),axis=0)
assert pairs.shape[1]==2 and np.all(pairs[:,0]<pairs[:,1])
degree=np.bincount(pairs.ravel(),minlength=n)
bins=np.floor(np.log2(np.maximum(degree,1))).astype(np.int64)
pair_bins=np.sort(bins[pairs],axis=1)
rows=[]
for split in (0,1,2):
 ids=np.flatnonzero(train[:,split])
 with np.load(verify(bindings['train_labels'][str(split)]),allow_pickle=False) as z:
  assert set(z.files)=={'ids','labels'} and z['ids'].dtype==z['labels'].dtype==np.int64
  assert np.array_equal(z['ids'],ids);labels=z['labels']
 assert ids.size==12246 and labels.shape==ids.shape and np.all((labels>=0)&(labels<5))
 y=np.full(n,-1,dtype=np.int64);y[ids]=labels
 eligible=(y[pairs[:,0]]>=0)&(y[pairs[:,1]]>=0)
 same=eligible&(y[pairs[:,0]]==y[pairs[:,1]])
 different=eligible&~same
 assert int(same.sum()+different.sum())==int(eligible.sum())
 row=dict(split=split,official_TRAIN_count=int(ids.size),class_counts=np.bincount(labels,minlength=5).tolist(),
          labelled_pair_count=int(eligible.sum()),labelled_fraction_of_native_nonloop_pairs=float(eligible.mean()),categories={})
 for name,mask in [('equal_class',same),('different_class',different)]:
  counts=np.bincount(pairs[mask].ravel(),minlength=n)
  strata,count=np.unique(pair_bins[mask],axis=0,return_counts=True)
  row['categories'][name]=dict(eligible_pairs=int(mask.sum()),
   fraction_of_native_nonloop_pairs=float(mask.mean()),
   official_TRAIN_nodes_with_at_least_one_eligible_pair=int(np.count_nonzero(counts[ids])),
   official_TRAIN_endpoint_pair_count_quantiles=np.quantile(counts[ids],[0,.25,.5,.75,.9,1]).tolist(),
   prospective_floor_10percent_of_category_pairs=int(np.floor(.1*int(mask.sum()))),
   degree_pair_strata=[dict(log2_degree_bins=s.tolist(),eligible_pairs=int(c)) for s,c in zip(strata,count)])
 rows.append(row)
value=dict(schema='graph-view-official-TRAIN-support-census-v1',UTC=datetime.now(timezone.utc).isoformat(),
 route=dict(login=LOGIN,repository=str(repo),GPU_UUID=UUID),input_bindings=bindings,
 node_count=n,raw_edge_shape=list(edge.shape),raw_self_loop_count=raw_loops,
 native_unique_undirected_nonloop_pairs=int(pairs.shape[0]),
 native_directed_entries_with_one_self_loop_per_node=int(2*pairs.shape[0]+n),
 pairing='unique sorted min/max endpoints; equivalent undirected simple graph to native coalescing',
 degree_bins='floor(log2(max(native nonloop degree,1))); unordered endpoint-bin pair',splits=rows,
 numpy_version=np.__version__,wall_seconds=time.perf_counter()-started,
 features_decoded=False,VALIDATION_or_TEST_labels_read=False,predictive_values_read=False,
 models_imported=False,training_updates=0,views_or_mask_assignments_created=False,
 scientific_gain_or_novelty_claim=False)
out=phase/'graph_view_train_coverage_census_execution_root_20261004_v1';out.mkdir(exist_ok=True)
with (out/'CENSUS.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
'''


def main():
    manifest = PHASE/'amazon_ratings_native_warm_execution_root_20261003_v3/data/DATA_MANIFEST.json'
    data = json.loads(manifest.read_text())
    bindings = dict(manifest=descriptor(manifest), public_graph=data['public_graph'], train_labels=data['train_labels'])
    save(HERE/'INPUT_BINDINGS.json', bindings)
    code = 'REPO='+repr(REPO)+'\nLOGIN='+repr(LOGIN)+'\nUUID='+repr(UUID)+'\nbindings='+repr(bindings)+'\n'+REMOTE
    compile(code, '<census-remote>', 'exec')
    (HERE/'REMOTE_SOURCE.py.txt').write_text(code)
    cmd = shlex.join([REPO+'/.venv/bin/python', '-I', '-B', '-c', code])
    ssh = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no',
           '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',LOGIN,
           'cd '+shlex.quote(REPO)+' && '+cmd]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(ssh,capture_output=True,text=True,timeout=90)
    save(HERE/'TRANSPORT.json',dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),
         exit_code=result.returncode,stderr=result.stderr,remote_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
         client=descriptor(Path(__file__)),private_key_contents_read=False))
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    save(HERE/'CENSUS.json',value)
    print(json.dumps({k:value[k] for k in ('UTC','native_unique_undirected_nonloop_pairs','wall_seconds','splits')},indent=2))


if __name__=='__main__':
    main()
