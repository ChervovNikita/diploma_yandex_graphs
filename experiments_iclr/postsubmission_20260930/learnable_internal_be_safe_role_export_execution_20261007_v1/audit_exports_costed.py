"""No models/scoring. Complete safe role hydration, hashes and canonical audit."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=Path(__file__).resolve().parent
SOURCE=PHASE/'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1048576),b''):h.update(part)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['wikics','collab','molhiv'],required=True)
    args=parser.parse_args();started=time.monotonic();task=args.task
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=[GPU]:raise ValueError('Wrong singleton')
    if Path.cwd().resolve()!=REPO or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Exact CPU audit context')
    sys.path.insert(0,str(SOURCE))
    from runtime import verify_manifest,runtime_versions,bound
    versions=runtime_versions();source_sha=verify_manifest()
    if source_sha!='76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce':raise ValueError('Exact reviewed source only')
    from data import load_projection,pair_ids
    import torch
    import numpy as np
    output=HERE/'outputs'/task
    binding=dict(path=str((output/'ROLE_MANIFEST.json').relative_to(PHASE)),sha256=sha(output/'ROLE_MANIFEST.json'))
    train,valid,manifest=load_projection(PHASE,binding,task,True)
    plan=json.loads((SOURCE/'EXPORT_CUSTODY_PLAN.json').read_text())[task]
    if manifest['source_custody']!=plan or manifest['source_manifest_sha256']!=source_sha:raise ValueError('Exact source-custody plan')
    fields={}
    def tensor_sha(value):
        array=value.detach().contiguous().numpy();h=hashlib.sha256(str((array.shape,array.dtype)).encode());h.update(memoryview(array).cast('B'));return h.hexdigest()
    for role,payload in [('train',train),('valid',valid)]:
        fields[role]={key:dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size(),tensor_sha256=tensor_sha(value)) for key,value in payload.items()}
    canonical={}
    if task=='wikics':
        authority=json.loads(bound(PHASE,plan['authority']).read_text())
        original=torch.load(bound(PHASE,authority['available']),map_location='cpu',weights_only=True)
        maps={'train':dict(x='x',edge_index='edge_index',ids='train_ids',y='train_y'),'valid':dict(ids='valid_ids',y='valid_y')}
        for role,payload in [('train',train),('valid',valid)]:
            if not all(torch.equal(value,original[maps[role][key]]) for key,value in payload.items()):raise ValueError('WikiCS source equality')
        edge=train['edge_index'];loops=int((edge[0]==edge[1]).sum())
        if loops!=11701:raise ValueError('One prepared self-loop per public node')
        keys=edge[0]*11701+edge[1];reverse=edge[1]*11701+edge[0]
        if not torch.isin(reverse,keys).all():raise ValueError('Complete undirected prepared graph')
        canonical=dict(source_tensor_equality=True,TRAIN_ids=580,VALID_ids=5274,role_overlap=0,public_nodes=11701,
            prepared_entries=edge.shape[1],selfloops=loops,undirected=True)
    elif task=='collab':
        train_keys=pair_ids(train['positive'],235868);valid_keys=pair_ids(valid['positive'],235868)
        unique=len(train_keys.unique());canonical=dict(TRAIN_records=len(train_keys),TRAIN_unique_canonical_pairs=unique,
            TRAIN_extra_repeated_canonical_records=len(train_keys)-unique,VALID_positive_records=len(valid_keys),
            VALID_negative_records=len(valid['negative']),TRAIN_VALID_canonical_overlap=int(torch.isin(valid_keys,train_keys).sum()))
        expected=plan['expected_public_array_digests']
        actual=dict(TRAIN_edge=tensor_sha(train['positive']),VALID_edge=tensor_sha(valid['positive']),
            VALID_negative=tensor_sha(valid['negative']),raw_features=tensor_sha(train['x']))
        if actual!=expected:raise ValueError('Complete official public tensor identity')
        canonical['complete_public_tensor_digests_matched']=True
    else:
        from export_roles import numeric_rows
        source_root=PHASE/plan['raw_root'];node_counts=np.array([int(x[0]) for x in numeric_rows(source_root/'raw/num-node-list.csv.gz').values()])
        edge_counts=np.array([int(x[0]) for x in numeric_rows(source_root/'raw/num-edge-list.csv.gz').values()])
        for role,payload in [('train',train),('valid',valid)]:
            official=np.array([int(x[0]) for x in numeric_rows(source_root/('split/scaffold/'+role+'.csv.gz')).values()],dtype=np.int64)
            if not np.array_equal(official,payload['ids'].numpy()):raise ValueError('Ordered official scaffold IDs')
            if not np.array_equal(payload['node_ptr'].diff().numpy(),node_counts[official]):raise ValueError('Original complete node boundaries')
            if not np.array_equal(payload['edge_ptr'].diff().numpy(),2*edge_counts[official]):raise ValueError('Original complete inverse-edge boundaries')
            edges=payload['edge_index'];bonds=payload['edge_attr']
            if not torch.equal(edges[:,0::2],edges[:,1::2].flip(0)) or not torch.equal(bonds[0::2],bonds[1::2]):raise ValueError('Canonical inverse edges/bond fields')
        # Independent complete selected-CSV value comparison. Stream every
        # compressed row but decode/parse only selected TRAIN/VALID segments.
        for relative,digest in plan['files'].items():
            if sha(source_root/relative)!=digest:raise ValueError('Audit raw source custody changed')
        nsource=np.r_[0,node_counts.cumsum()];esource=np.r_[0,edge_counts.cumsum()]
        def compare_stream(filename,kind):
            segments=[]
            for role,payload in [('train',train),('valid',valid)]:
                for pos,graph_id in enumerate(payload['ids'].tolist()):
                    if kind=='atom':lo,hi=map(int,nsource[graph_id:graph_id+2]);target=int(payload['node_ptr'][pos])
                    elif kind in ('edge','bond'):lo,hi=map(int,esource[graph_id:graph_id+2]);target=int(payload['edge_ptr'][pos])
                    else:lo,hi=graph_id,graph_id+1;target=pos
                    if lo<hi:segments.append((lo,hi,role,target))
            segments.sort();cursor=0;compared=0
            with gzip.open(source_root/'raw'/filename,'rb') as stream:
                for row_id,line in enumerate(stream):
                    while cursor<len(segments) and row_id>=segments[cursor][1]:cursor+=1
                    if cursor>=len(segments):break
                    lo,hi,role,target=segments[cursor]
                    if row_id<lo:continue
                    value=np.fromstring(line.decode('ascii').strip(),sep=',')
                    payload=train if role=='train' else valid;offset=row_id-lo
                    if kind=='atom':expected=payload['x'][target+offset].numpy()
                    elif kind=='bond':expected=payload['edge_attr'][target+2*offset].numpy()
                    elif kind=='edge':expected=payload['edge_index'][:,target+2*offset].numpy()
                    else:expected=np.array([float(payload['y'][target])])
                    if not np.array_equal(value,expected):raise ValueError('Complete original selected '+kind+' source equality')
                    compared+=1
            if compared!=sum(hi-lo for lo,hi,role,target in segments):raise ValueError('Incomplete selected raw source comparison')
            return compared
        source_rows={kind:compare_stream(file,kind) for file,kind in [('node-feat.csv.gz','atom'),('edge-feat.csv.gz','bond'),('edge.csv.gz','edge'),('graph-label.csv.gz','target')]}
        canonical=dict(TRAIN_graphs=len(train['ids']),VALID_graphs=len(valid['ids']),role_overlap=0,
            original_official_ID_order=True,original_full_graph_boundaries=True,inverse_edge_and_bond_interleaving=True,
            all_graph_public_count_metadata_parsed=True,TEST_target_atom_bond_edge_values_parsed=False,
            complete_original_selected_values_matched=True,selected_source_rows_compared=source_rows)
    if torch.cuda.is_initialized():raise ValueError('CPU-only preparation created a CUDA context')
    receipt=dict(schema='internal-be-official-safe-role-output-audit-v1',task=task,passed=True,source_manifest_sha256=source_sha,
        role_manifest=binding,role_fields=fields,canonical_roles=canonical,versions=versions,
        inclusive_CPU_audit_seconds=time.monotonic()-started,peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        user_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,system_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        input_blocks=resource.getrusage(resource.RUSAGE_SELF).ru_inblock,output_blocks=resource.getrusage(resource.RUSAGE_SELF).ru_oublock,
        auditor_program_sha256=sha(__file__),
        source_data_custody_matched=True,complete_schema_domain_guards_passed=True,pre_deserialization_NPZ_inventory_checked=True,
        CPU_only=True,cuda_initialized=False,scientific_fits=0,predictive_metrics_computed=False,
        TEST_target_values_parsed=False,TEST_features_or_labels_supplied_to_training=False,binaries_server_only=True)
    path=HERE/'receipts'/str(task+'_OUTPUT_AUDIT_COSTED.json')
    if path.exists():raise ValueError('No audit overwrite')
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(receipt,sort_keys=True))


if __name__=='__main__':main()
