"""One genuine CPU-only public conversion/equality/full-batch geometry check."""
import hashlib,json,os,resource,socket,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO=PHASE.parents[1]
PUBLIC=PHASE/'portable_internal_be_public_interface_20261007_v2'
PUBLIC_SHA='190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
RAW=PHASE/'new_tasks/data/molhiv/provider/ogbg_molhiv'
AUTHORITY=PHASE/'learnable_internal_be_safe_role_export_execution_20261007_v1/outputs/molhiv/ROLE_MANIFEST.json'
AUTHORITY_SHA='c3a10b949409ae37d541f884685bec885254bc2bfebbb7ab41152f0e9df35620'
REVIEW=PHASE/'learnable_internal_be_official_data_root_review_20261007_v1/molhiv_DATA_APPROVAL.json'
REVIEW_SHA='e82d13986d0af6a070e40ddb6cd316627e293016c4bc28c2b87b2d2dde66766b'
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
started=time.monotonic();os.umask(0o077)
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).strip().splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert Path.cwd()==REPO and Path(sys.executable).resolve()==PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert sha(PUBLIC/'MANIFEST.json')==PUBLIC_SHA
for row in json.loads((PUBLIC/'MANIFEST.json').read_text())['files']:
    file=PUBLIC/row['path'];assert file.is_relative_to(PUBLIC) and file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
assert sha(AUTHORITY)==AUTHORITY_SHA and sha(REVIEW)==REVIEW_SHA
authority=json.loads(AUTHORITY.read_text());review=json.loads(REVIEW.read_text())
assert review['approved'] is True and review['data_manifest']['sha256']==AUTHORITY_SHA
sys.path.insert(0,str(PUBLIC))
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
from data_interface import _data
assert not torch.cuda.is_initialized()
stage='public_conversion'
try:
    with (HERE/'export.log').open('x') as log:
        result=subprocess.run([str(PYTHON),'-B',str(PUBLIC/'export.py'),'molhiv','--raw-root',str(RAW),'--output',str(HERE/'roles')],cwd=REPO,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Public conversion exit '+str(result.returncode))
    conversion=json.loads((HERE/'roles/DATA.json').read_text())
    assert conversion['complete'] and conversion['source_bytes_matched_public_pins'] and not conversion['TEST_scoring']
    stage='complete_ordered_author_role_equality'
    fields={};train_payload=None
    for role in ('train','valid'):
        reference=(PHASE/authority['payloads'][role]['path']).resolve()
        assert reference.is_relative_to(PHASE) and sha(reference)==authority['payloads'][role]['sha256']
        keys=_data().ROLE_KEYS['molhiv'][role]
        original=_data().load_npz(reference,keys);converted=_data().load_npz(HERE/'roles'/(role+'.npz'),keys)
        fields[role]={}
        for key in keys:
            a,b=original[key],converted[key]
            assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),role+'/'+key
            fields[role][key]={'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.numel()*a.element_size(),'complete_ordered_equal':True,'raw_tensor_sha256':hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()}
        if role=='train':train_payload=original
    stage='all_fixed_TRAIN_batch_geometry'
    node_counts=train_payload['node_ptr'].diff();edge_counts=train_payload['edge_ptr'].diff();count=len(train_payload['ids']);batch_size=128
    full=count//batch_size;tail=count%batch_size;maxima=[];batches_seen=0
    for seed in (6101,6203,6307):
        best_nodes=best_edges=None
        for epoch in range(1,101):
            generator=torch.Generator().manual_seed(seed+19709+1000*epoch)
            order=torch.randperm(count,generator=generator)
            nodes=torch.cat((node_counts[order[:full*batch_size]].reshape(full,batch_size).sum(1),node_counts[order[full*batch_size:]].sum().reshape(1)))
            edges=torch.cat((edge_counts[order[:full*batch_size]].reshape(full,batch_size).sum(1),edge_counts[order[full*batch_size:]].sum().reshape(1)))
            batches_seen+=len(nodes)
            for criterion,values in [('nodes',nodes),('edges',edges)]:
                index=int(values.argmax());positions=order[index*batch_size:(index+1)*batch_size]
                entry={'criterion':criterion,'seed':seed,'epoch':epoch,'batch_index_zero_based':index,'graphs':len(positions),'nodes':int(nodes[index]),'edges':int(edges[index]),'positions':positions.tolist(),'official_graph_ids':train_payload['ids'][positions].tolist(),'position_int64_bytes_sha256':hashlib.sha256(positions.numpy().tobytes()).hexdigest(),'selection_uses_labels':False}
                if criterion=='nodes' and (best_nodes is None or entry['nodes']>best_nodes['nodes']):best_nodes=entry
                if criterion=='edges' and (best_edges is None or entry['edges']>best_edges['edges']):best_edges=entry
        maxima.extend([best_nodes,best_edges])
    assert batches_seen==3*100*258 and full==257 and tail==5
    scan={'schema':'internal-BE-exact-fixed-Molhiv-TRAIN-batch-geometry-v1','complete':True,'seeds':[6101,6203,6307],'epochs_per_seed':100,'batches_per_epoch':258,'batches_scanned':batches_seen,'permutation_seed_formula':'seed+19709+1000*epoch','batch_size':128,'tail_graphs':tail,'TRAIN_graphs':count,'TRAIN_nodes':int(node_counts.sum()),'TRAIN_edges':int(edge_counts.sum()),'maxima_per_seed':maxima,'global_max_nodes':max(maxima,key=lambda row:row['nodes']),'global_max_edges':max(maxima,key=lambda row:row['edges']),'source_recipe_unchanged':True,'label_based_selection':False,'GPU_execution':False,'worst_GPU_batch_qualified':False}
    write('TRAIN_BATCH_GEOMETRY.json',scan)
    usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    assert not torch.cuda.is_initialized()
    receipt={'schema':'internal-BE-portable-Molhiv-CPU-check-v1','complete':True,'source_manifest_sha256':PUBLIC_SHA,'author_role_manifest_sha256':AUTHORITY_SHA,'author_data_review_sha256':REVIEW_SHA,'public_conversion_source_bytes_matched':True,'complete_ordered_role_equality':fields,'TRAIN_batch_geometry_sha256':sha(HERE/'TRAIN_BATCH_GEOMETRY.json'),'batches_scanned':batches_seen,'public_roles':conversion['roles'],'inclusive_worker_seconds':time.monotonic()-started,'worker_CPU_user_seconds':usage.ru_utime,'worker_CPU_system_seconds':usage.ru_stime,'export_CPU_user_seconds':children.ru_utime,'export_CPU_system_seconds':children.ru_stime,'peak_worker_RSS_bytes':usage.ru_maxrss*1024,'peak_export_RSS_bytes':children.ru_maxrss*1024,'output_storage_bytes':sum(p.stat().st_size for p in (HERE/'roles').iterdir() if p.is_file()),'cuda_initialized':False,'models_constructed':0,'scientific_fits':0,'TEST_scoring':False,'automatic_retry':False,'GPU_or_other_jobs_modified':False,'binaries_transferred':False,'root_review_for_GPU_still_required':True}
    write('CPU_CHECK.json',receipt)
    print(json.dumps({'complete':True,'ordered_equality':True,'batches_scanned':batches_seen,'scores_computed':False}))
except BaseException as error:
    write('FAILURE.json',{'complete':False,'stage':stage,'error_type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-started,'automatic_retry':False,'TEST_scoring':False})
    raise
