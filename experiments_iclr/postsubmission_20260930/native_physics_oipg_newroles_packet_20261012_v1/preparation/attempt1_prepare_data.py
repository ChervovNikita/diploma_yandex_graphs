"""Fresh development roles on immutable complete public graphs; no model/score."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def receipt(root,path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return {'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
def write(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    path.chmod(0o444)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True);parser.add_argument('--package',required=True);args=parser.parse_args()
    from source_guard import allocation_guard
    allocation_guard()
    import numpy as np
    import torch
    import scipy.sparse as sp
    root=Path(args.project_root);out=root/args.package;started=time.monotonic();bindings=json.loads((out/'DATA_INPUT_BINDINGS.json').read_text());records={}
    for dataset in ('physics','wikics'):
        binding=bindings[dataset];graph_row=binding['graph'];graph_path=root/graph_row['path']
        if receipt(root,graph_path)!=graph_row:raise ValueError('Immutable graph bytes changed')
        graph=torch.load(graph_path,map_location='cpu',weights_only=True)
        if set(graph)!={'x','edge_index'}:raise ValueError('Graph must contain no target arrays')
        raw_row=binding['raw'];raw_path=root/raw_row['path']
        if receipt(root,raw_path)!=raw_row:raise ValueError('Pinned official raw bytes changed')
        n,f,c,e=binding['geometry'];x,edge=graph['x'],graph['edge_index']
        if tuple(x.shape)!=(n,f) or x.dtype!=torch.float32 or tuple(edge.shape)!=(2,e):raise ValueError('Full raw source geometry differs')
        if dataset=='physics':
            with np.load(raw_path,allow_pickle=False) as archive:
                labels=archive['labels'].astype(np.int64,copy=True)
                features=sp.csr_matrix((archive['attr_data'],archive['attr_indices'],archive['attr_indptr']),shape=tuple(archive['attr_shape']))
                support=sp.csr_matrix((archive['adj_data'],archive['adj_indices'],archive['adj_indptr']),shape=tuple(archive['adj_shape']))
            if labels.shape!=(n,) or features.shape!=(n,f):raise ValueError('Raw Physics feature/label geometry')
            for start in range(0,n,128):
                block=features[start:start+128].toarray().astype(np.float32)
                if not np.array_equal(block,x[start:start+128].numpy()) or not np.logical_or(block==0,block==1).all():raise ValueError('Literal raw binary Physics X differs')
            support=support.maximum(support.T);support.setdiag(0);support.eliminate_zeros();support.setdiag(1);support.sort_indices()
            rows,columns=support.nonzero();raw_edge=np.stack([rows,columns]).astype(np.int64)
            actual=edge.numpy();order=np.lexsort((actual[1],actual[0]))
            if not np.array_equal(raw_edge,actual[:,order]):raise ValueError('Raw reciprocal Physics support/selfloops differ')
            blocks=(109001,109003,109007);role_seeds=[block+5000011+211000003 for block in blocks]
        else:
            raw=json.loads(raw_path.read_text());labels=np.asarray(raw['labels'],dtype=np.int64);features=np.asarray(raw['features'],dtype=np.float32)
            if labels.shape!=(n,) or not np.array_equal(features,x.numpy()):raise ValueError('Raw WikiCS features/labels differ')
            pairs=np.asarray([(u,v) for u,neighbors in enumerate(raw['links']) for v in neighbors if u!=v],dtype=np.int64).T
            support=sp.coo_matrix((np.ones(pairs.shape[1],dtype=np.uint8),pairs),(n,n)).tocsr();support=support.maximum(support.T);support.data[:]=1;support.setdiag(1);support.sort_indices()
            rows,columns=support.nonzero()
            actual=edge.numpy();order=np.lexsort((actual[1],actual[0]))
            if not np.array_equal(np.stack([rows,columns]).astype(np.int64),actual[:,order]):raise ValueError('Raw WikiCS support/selfloops differ')
            blocks=list(range(10));role_seeds=[111001+104729*split for split in blocks]
            # Official raw masks are ignored; this task uses fresh development roles.
            del raw,features,pairs
        folder=out/'data'/dataset;folder.mkdir(parents=True);splits=[]
        for split,(block,seed) in enumerate(zip(blocks,role_seeds)):
            rng=np.random.RandomState(seed);parts={role:[] for role in ('train','select','cal','evaluate')}
            for cls in range(c):
                ids=rng.permutation(np.flatnonzero(labels==cls))
                if len(ids)<=80:raise ValueError('Each class needs80 role labels and development evaluation')
                for role,indices in zip(parts,(ids[:20],ids[20:50],ids[50:80],ids[80:])):parts[role].extend(indices.tolist())
            directory=folder/('split_'+str(split));directory.mkdir();node_inputs={'graph':graph_row};endpoint_inputs={};role_rows={};counts={}
            for role,values in parts.items():
                ids=np.asarray(sorted(values),dtype=np.int64);pack={'ids':torch.from_numpy(ids),'y':torch.from_numpy(labels[ids].copy())}
                path=directory/(role+'.pt');torch.save(pack,path);path.chmod(0o444);row=receipt(root,path);role_rows[role]=row;counts[role]=len(ids)
                if role in ('train','select'):node_inputs[role]=row
                else:
                    q=directory/(role+'_ids.pt');torch.save(pack['ids'],q);q.chmod(0o444);endpoint_inputs[role+'_ids']=receipt(root,q);endpoint_inputs[role+'_targets']=row
            combined=np.concatenate([np.asarray(parts[r]) for r in parts])
            if len(combined)!=n or len(np.unique(combined))!=n:raise ValueError('Whole graph exact disjoint role partition')
            splits.append({'split_id':split,'source_block':int(block),'role_seed':int(seed),'counts':counts,'roles':role_rows,'node_inputs':node_inputs,'endpoint_inputs':endpoint_inputs})
        records[dataset]={'dataset_id':dataset,'graph':graph_row,'raw':raw_row,'geometry':binding['geometry'],'splits':splits,
                          'graph_features_and_support_equal_pinned_raw':True,'new_graph_data_copies':0,
                          'immutable_prepared_edge_order_retained':True,
                          'evaluation_scope':'encountered_development;no_unused_or_protected_confirmation_claim',
                          'official_raw_masks_used':False,'historical_target_archives_or_scores_read':False}
        del graph,x,edge,labels,support
    value={'schema':'fresh-Physics-OIPG-role-custody-v1','datasets':records,'wall_seconds':time.monotonic()-started,
           'actual_input_model_constructions':0,'optimizer_constructions':0,'scientific_fits':0,'scores':0,'GO_created':False}
    write(out/'DATA_CUSTODY.json',value);print(json.dumps({'datasets':{k:{'geometry':v['geometry'],'splits':len(v['splits'])} for k,v in records.items()},'wall_seconds':value['wall_seconds'],'models':0,'fits':0,'GO':False}))
if __name__=='__main__':main()
