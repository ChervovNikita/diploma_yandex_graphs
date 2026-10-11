"""Public Physics preprocessing and fixed scanner roles; no training."""
import hashlib
import os
from pathlib import Path
import time
from common import BLOCKS,receipt,require,write
RAW_SHA='e4d68468eba5fb8f6b280ffd406bd53e6e4295ba1dd2967d6fe99b978e4b3375'
DOWNLOAD_URL='https://raw.githubusercontent.com/shchur/gnn-benchmark/1e72912a0810cdf27ae54fd589a3b43358a2b161/data/npz/ms_academic_phy.npz'

def array_digest(np, array, dtype):
    value = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()

def git_blob_digest(path):
    path = Path(path)
    h = hashlib.sha1(b"blob " + str(path.stat().st_size).encode("ascii") + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def prepare(raw_path,output):
    start=time.monotonic();raw=Path(raw_path).resolve()
    require(raw.stat().st_size==16374005 and receipt(raw)['sha256']==RAW_SHA and git_blob_digest(raw)=='8d8bc12e29aa67f6753a712080eae81079bd7d01','Pinned public raw Physics archive')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import scipy.sparse as sp
    import torch
    from scipy.sparse.csgraph import connected_components
    from torch_geometric.utils import remove_self_loops,add_self_loops,to_undirected
    with np.load(raw,allow_pickle=False) as stored:
        require(tuple(stored['attr_shape'])==(34493,8415) and tuple(stored['adj_shape'])==(34493,34493),'Full provider shapes')
        attributes=sp.csr_matrix((stored['attr_data'],stored['attr_indices'],stored['attr_indptr']),(34493,8415))
        x_tensor=torch.from_numpy(attributes.todense()).to(torch.float);x_tensor[x_tensor>0]=1;x=x_tensor.numpy()
        adjacency=sp.csr_matrix((stored['adj_data'],stored['adj_indices'],stored['adj_indptr']),(34493,34493)).tocoo()
        edge_index=torch.stack((torch.from_numpy(adjacency.row).long(),torch.from_numpy(adjacency.col).long()))
        edge_index,_=remove_self_loops(edge_index);edge_index=to_undirected(edge_index,num_nodes=34493).contiguous()
        require(edge_index.shape==(2,495924),'Provider support count')
        e=edge_index.numpy();require(connected_components(sp.csr_matrix((np.ones(495924,dtype=np.int8),(e[0],e[1])),shape=(34493,34493)),directed=False,return_labels=False)==1,'Already full connected component')
        targets=np.ascontiguousarray(stored['labels'],dtype=np.int64)
    require(targets.shape==(34493,) and np.array_equal(np.unique(targets),np.arange(5)),'Five author classes')
    edge_index,_=add_self_loops(edge_index,num_nodes=34493);edge_index=to_undirected(edge_index,num_nodes=34493).contiguous();edges=edge_index.numpy()
    require(edges.shape==(2,530417) and np.array_equal(np.sort(edges[0,edges[0]==edges[1]]),np.arange(34493)),'Common explicit-self support')
    np.savez_compressed(output/'graph.npz',x=x,edge_index=edges)
    (output/'calibration_only').mkdir();(output/'scorer_only').mkdir();identities={}
    for block in BLOCKS:
        rng=np.random.RandomState(block+5000011+223000019);parts={r:[] for r in ('train','select','cal','dev')}
        for c in range(5):
            ids=rng.permutation(np.sort(np.flatnonzero(targets==c)).astype(np.int64))
            for role,values in zip(parts,(ids[:20],ids[20:50],ids[50:80],ids[80:])):parts[role].append(values)
        roles={r:np.sort(np.concatenate(a)) for r,a in parts.items()}
        require([len(roles[r]) for r in parts]==[100,150,150,34093] and np.array_equal(np.sort(np.concatenate(list(roles.values()))),np.arange(34493)),'Complete fresh partition')
        folder=output/f'block_{block}';folder.mkdir()
        for role in ('train','select'):np.savez(folder/(role+'.npz'),ids=roles[role],y=targets[roles[role]])
        for role in ('select','cal','dev'):np.save(folder/(role+'_ids.npy'),roles[role],allow_pickle=False)
        for role,destination in (('cal','calibration_only'),('dev','scorer_only')):
            path=output/destination/f'block_{block}_{role}.npz';np.savez(path,ids=roles[role],y=targets[roles[role]]);os.chmod(path,0o600)
        identities[str(block)]={r:array_digest(np,a,'<i8') for r,a in roles.items()}
    files={str(p.relative_to(output)):receipt(p) for p in sorted(output.rglob('*')) if p.is_file()}
    result=dict(complete=True,blocks=list(BLOCKS),files=files,roles=identities,raw_archive=receipt(raw),components=1,feature_sha256=array_digest(np,x,'<f4'),graph_sha256=array_digest(np,edges,'<i8'),encountered_graph=True,unused_confirmation_claim=False,role_rng='RandomState(block+5000011+223000019)',predictions_or_scores=0,seconds=time.monotonic()-start)
    require(result['feature_sha256']=='9df20be681bfe193a8565f272f2031f31d73871f09d1f56f24bd11d4d683e945' and result['graph_sha256']=='1b57360470b64c50a463ef8335f03cf4377b4d31a3c7af5a9494d103a47a8801','Full public feature/support payload differs from fixed scientific input');write(output/'CUSTODY.json',result);return result
