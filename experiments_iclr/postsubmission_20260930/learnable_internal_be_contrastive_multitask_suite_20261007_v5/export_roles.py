"""Official TRAIN/VALID numeric-only exporters, disabled until separate release.

Never instantiates provider datasets, processed molecule tensors, TEST split
loaders, TEST target/feature arrays or model predictors. Existing public bytes are hash-bound.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from runtime import ROOT,PHASE,allocation,verify_manifest,bound,sha,runtime_versions


def numeric_rows(path, selected=None):
    """Parse only selected numeric rows; unselected rows remain discarded bytes.

    gzip necessarily decompresses intervening bytes. This is a value boundary,
    not OS isolation or a claim those compressed bytes were never traversed.
    """
    import numpy as np
    rows={}
    with gzip.open(path,'rb') as stream:
        for index,line in enumerate(stream):
            if selected is None or index in selected:
                rows[index]=np.fromstring(line.decode('ascii').strip(),sep=',')
    if selected is not None and set(rows)!=set(selected):raise ValueError('Missing selected official rows')
    return rows


def pack_molecules(raw, train_ids, valid_ids):
    """Original CSV categories + OGB reciprocal edge interleaving, no RDKit."""
    import numpy as np
    node_counts=np.array([int(x[0]) for x in numeric_rows(raw/'num-node-list.csv.gz').values()])
    edge_counts=np.array([int(x[0]) for x in numeric_rows(raw/'num-edge-list.csv.gz').values()])
    if len(node_counts)!=41127 or len(edge_counts)!=41127:raise ValueError('Complete official graph-count metadata')
    node_ptr=np.r_[0,node_counts.cumsum()];edge_ptr=np.r_[0,edge_counts.cumsum()]
    selected=set(train_ids.tolist())|set(valid_ids.tolist())
    nodes=set();edges=set()
    for i in selected:
        nodes.update(range(node_ptr[i],node_ptr[i+1]));edges.update(range(edge_ptr[i],edge_ptr[i+1]))
    xrows=numeric_rows(raw/'node-feat.csv.gz',nodes)
    erows=numeric_rows(raw/'edge.csv.gz',edges)
    brows=numeric_rows(raw/'edge-feat.csv.gz',edges)
    labels=numeric_rows(raw/'graph-label.csv.gz',selected)
    results={}
    for role,ids in (('train',train_ids),('valid',valid_ids)):
        xs=[];es=[];bs=[];ys=[];npointer=[0];epointer=[0]
        for i in ids:
            i=int(i);x=np.stack([xrows[j] for j in range(node_ptr[i],node_ptr[i+1])])
            edge=np.stack([erows[j] for j in range(edge_ptr[i],edge_ptr[i+1])]) if edge_counts[i] else np.empty((0,2))
            bond=np.stack([brows[j] for j in range(edge_ptr[i],edge_ptr[i+1])]) if edge_counts[i] else np.empty((0,3))
            if not np.equal(x,np.floor(x)).all() or not np.equal(edge,np.floor(edge)).all() or not np.equal(bond,np.floor(bond)).all():raise ValueError('Original categorical/local-edge integers')
            reciprocal=np.empty((2,2*len(edge)),dtype=np.int64)
            reciprocal[:,0::2]=edge.T;reciprocal[:,1::2]=edge[:,::-1].T
            xs.append(x.astype(np.int64));es.append(reciprocal);bs.append(np.repeat(bond.astype(np.int64),2,axis=0));ys.append(float(labels[i][0]))
            npointer.append(npointer[-1]+len(x));epointer.append(epointer[-1]+reciprocal.shape[1])
        results[role]=dict(x=np.concatenate(xs),edge_index=np.concatenate(es,axis=1),edge_attr=np.concatenate(bs),
            node_ptr=np.array(npointer,dtype=np.int64),edge_ptr=np.array(epointer,dtype=np.int64),ids=ids.astype(np.int64),y=np.array(ys,dtype=np.float32))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    job=json.loads(Path(args.job).read_text())
    if job.get('data_export_authorized') is not True or job.get('source_review_approved') is not True or job.get('TEST_access') is not False:
        raise ValueError('Separate data-only release required; default disabled')
    allocation(cpu_check=True)
    if job['source_manifest_sha256']!=verify_manifest():raise ValueError('Exact reviewed exporter source')
    review=json.loads(bound(PHASE,job['source_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=job['source_manifest_sha256']:
        raise ValueError('Fresh exporter source review')
    output=(PHASE/job['output_directory']).resolve()
    if not output.is_relative_to(PHASE) or output.exists():raise ValueError('Fresh phase export only')
    output.mkdir();started=time.monotonic()
    try:
        runtime_versions()
        import numpy as np
        import torch
        from data import check_projection,COUNTS
        task=job['task'];custody=json.loads((ROOT/'EXPORT_CUSTODY_PLAN.json').read_text())[task]
        if task=='wikics':
            authority=json.loads(bound(PHASE,custody['authority']).read_text())
            if authority['TEST_labels_available_to_trainer'] is not False:raise ValueError('Existing safe WikiCS authority')
            source=bound(PHASE,authority['available'])
            value=torch.load(source,map_location='cpu',weights_only=True)
            if set(value)!={'x','edge_index','train_ids','train_y','valid_ids','valid_y'}:raise ValueError('Pinned safe WikiCS keys')
            arrays={'train':dict(x=value['x'].numpy(),edge_index=value['edge_index'].numpy(),ids=value['train_ids'].numpy(),y=value['train_y'].numpy()),
                    'valid':dict(ids=value['valid_ids'].numpy(),y=value['valid_y'].numpy())}
            metadata=dict(split_index=0)
        elif task=='collab':
            archive=bound(PHASE,custody['archive'])
            origin=json.loads(bound(PHASE,custody['singleton_transfer_receipt']).read_text())
            if origin['GPU_uuid']!='GPU-44039938-fd82-41d2-fefd-de71514e2fac' or origin['archive_sha256']!=sha(archive):raise ValueError('Authorized singleton acquisition custody')
            with zipfile.ZipFile(archive) as packed:
                members=packed.namelist()
                if len(members)!=len(set(members)):raise ValueError('Duplicate archive member')
                def read(name):
                    if name not in custody['members']:raise ValueError('TEST or nonallowlisted member')
                    value=packed.read(name)
                    if hashlib.sha256(value).hexdigest()!=custody['members'][name]:raise ValueError('Official member changed')
                    return value
                # Exact trusted official NumPy pickles contain only the named
                # official TRAIN/VALID dictionary; never aggregate/test pickles.
                train=torch.load(io.BytesIO(read('collab/split/time/train.pt')),map_location='cpu',weights_only=False)
                valid=torch.load(io.BytesIO(read('collab/split/time/valid.pt')),map_location='cpu',weights_only=False)
                import pandas as pd
                x=pd.read_csv(io.BytesIO(gzip.decompress(read('collab/raw/node-feat.csv.gz'))),header=None).values.astype(np.float32)
                raw=np.loadtxt(io.BytesIO(gzip.decompress(read('collab/raw/edge.csv.gz'))),delimiter=',',dtype=np.int64)
            def arr(v):return v.numpy() if isinstance(v,torch.Tensor) else v
            if set(train)!={'edge','weight','year'} or set(valid)!={'edge','weight','year','edge_neg'}:raise ValueError('Official exact split dictionary')
            train={k:arr(v) for k,v in train.items()};valid={k:arr(v) for k,v in valid.items()}
            if train['year'].max()!=2017 or not (valid['year']==2018).all():raise ValueError('Official temporal split')
            if raw.shape!=train['edge'].shape or not np.array_equal(np.sort(raw.min(1)*235868+raw.max(1)),np.sort(train['edge'].min(1)*235868+train['edge'].max(1))):raise ValueError('Complete duplicate-preserving TRAIN raw correspondence')
            def digest(value):
                h=hashlib.sha256(str((value.shape,value.dtype)).encode());h.update(memoryview(np.ascontiguousarray(value)).cast('B'));return h.hexdigest()
            actual=dict(TRAIN_edge=digest(train['edge']),VALID_edge=digest(valid['edge']),VALID_negative=digest(valid['edge_neg']),raw_features=digest(x),TRAIN_year=digest(train['year']),VALID_year=digest(valid['year']))
            if actual!=custody['expected_public_array_digests']:raise ValueError('Independent official raw/role tensor digests')
            arrays={'train':dict(x=x,positive=train['edge'],positive_year=train['year']), 'valid':dict(positive=valid['edge'],positive_year=valid['year'],negative=valid['edge_neg'])}
            train_pairs=train['edge'].min(1)*235868+train['edge'].max(1)
            valid_pairs=valid['edge'].min(1)*235868+valid['edge'].max(1)
            overlap=np.isin(valid_pairs,train_pairs)
            metadata=dict(valid_negative_count=100000,temporal_roles={'train_max':2017,'valid_only':2018},
                task_target='official_temporal_event_prediction',historical_canonical_pair_overlap=dict(VALID_event_rows=int(overlap.sum()),
                    VALID_distinct_canonical_pairs=int(len(np.unique(valid_pairs[overlap])))),
                VALID_pairs_used_to_filter_TRAIN_support=False)
        elif task=='molhiv':
            authority=json.loads(bound(PHASE,custody['authority']).read_text())
            source_root=PHASE/custody['raw_root']
            # Only explicitly used raw/TRAIN/VALID members are read. The global
            # processed tensor and TEST split file are absent from this allowlist.
            for relative,digest in custody['files'].items():
                if sha(source_root/relative)!=digest:raise ValueError('Official molecule source changed')
            train_ids=np.array([int(x[0]) for x in numeric_rows(source_root/'split/scaffold/train.csv.gz').values()],dtype=np.int64)
            valid_ids=np.array([int(x[0]) for x in numeric_rows(source_root/'split/scaffold/valid.csv.gz').values()],dtype=np.int64)
            if len(train_ids)!=32901 or len(valid_ids)!=4113 or len(np.unique(np.r_[train_ids,valid_ids]))!=37014:raise ValueError('Complete disjoint official scaffold IDs')
            arrays=pack_molecules(source_root/'raw',train_ids,valid_ids)
            metadata=dict(split_kind='official_scaffold',scaffold_group_recomputed=False)
        else:raise ValueError(task)
        manifest=dict(schema='internal-be-official-role-projection-v3' if task=='collab' else 'internal-be-official-role-projection-v2',format='NPZ_numeric_only',task=task,
            train_count=COUNTS[task][0],valid_count=COUNTS[task][1],official_split_preserved=True,TEST_values_in_payload=False,
            source_custody=custody,source_manifest_sha256=job['source_manifest_sha256'],payloads={},**metadata)
        check_projection(task,{k:torch.from_numpy(v) for k,v in arrays['train'].items()},
            {k:torch.from_numpy(v) for k,v in arrays['valid'].items()},manifest)
        for role,value in arrays.items():
            path=output/(role+'.npz');np.savez(path,**{k:np.ascontiguousarray(v) for k,v in value.items()})
            manifest['payloads'][role]=dict(path=str(path.relative_to(PHASE)),sha256=sha(path))
        (output/'ROLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        receipt=dict(complete=True,scientific_fits=0,predictive_scores_computed=False,TEST_target_values_parsed=False,TEST_features_or_labels_supplied_to_training=False,
            all_graph_public_count_metadata_parsed=task=='molhiv',
            molecule_unselected_raw_bytes_traversed=task=='molhiv',inclusive_seconds=time.monotonic()-started,
            role_manifest_sha256=sha(output/'ROLE_MANIFEST.json'),source_manifest_sha256=job['source_manifest_sha256'])
        (output/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    except Exception as error:
        (output/'FAILURE.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),inclusive_seconds=time.monotonic()-started,automatic_retry=False),indent=2)+'\n')
        raise
    print(json.dumps({'export_complete':True,'scientific_fits':0,'TEST_target_values_parsed':False}))


if __name__=='__main__':main()
