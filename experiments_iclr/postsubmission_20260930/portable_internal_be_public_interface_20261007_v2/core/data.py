"""Narrow TRAIN/VALID projections. No OGB/WikiCS raw dataset hydrator.

Root must prepare official role projections separately and bind their custody.
The source loader never opens TEST tensors or a provider containing all labels.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def bound(phase, row):
    phase = Path(phase).resolve()
    path = (phase / row["path"]).resolve(strict=True)
    if not path.is_relative_to(phase) or not path.is_file() or sha(path) != row["sha256"]:
        raise ValueError("Phase custody/hash mismatch")
    return path


def require_keys(tensors, keys):
    if not isinstance(tensors, dict) or set(tensors) != set(keys):
        raise ValueError("Exact role projection keys; extra held-out fields forbidden")
    if not all(isinstance(x, torch.Tensor) for x in tensors.values()):
        raise TypeError("Tensor-only weights_only projection required")


ROLE_KEYS={
 'wikics':{'train':('x','edge_index','ids','y'),'valid':('ids','y')},
 'collab':{'train':('x','positive','positive_year'),'valid':('positive','positive_year','negative')},
 'molhiv':{'train':('x','edge_index','edge_attr','node_ptr','edge_ptr','ids','y'),
           'valid':('x','edge_index','edge_attr','node_ptr','edge_ptr','ids','y')}}
COUNTS={'wikics':(580,5274),'collab':(1179052,60084),'molhiv':(32901,4113)}


def inspect_npz(path, keys):
    """Read ZIP directory and NPY headers ONLY, before any numeric array load."""
    import zipfile
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!={k+'.npy' for k in keys}:
            raise ValueError('Extra/missing/repeated role field rejected before array deserialization')
        for name in names:
            with archive.open(name) as entry:
                version=np.lib.format.read_magic(entry)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(entry)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(entry)
                else:raise ValueError('Unsupported safe array header')
                if dtype.hasobject or order:raise ValueError('No pickle/object/Fortran role arrays')
    return True


def load_npz(path, keys):
    import numpy as np
    inspect_npz(path,keys)
    with np.load(path,allow_pickle=False) as archive:
        return {key:torch.from_numpy(archive[key].copy()) for key in keys}


def load_projection(phase, binding, task, with_valid):
    manifest=json.loads(bound(phase,binding).read_text())
    expected_schema='internal-be-official-role-projection-v3' if task=='collab' else 'internal-be-official-role-projection-v2'
    if manifest.get('schema')!=expected_schema or manifest.get('task')!=task:
        raise ValueError('Exact safe official role authority required')
    if manifest.get('official_split_preserved') is not True or manifest.get('TEST_values_in_payload') is not False or manifest.get('format')!='NPZ_numeric_only':
        raise ValueError('Official split + no TEST numeric values required')
    if set(manifest['payloads'])!={'train','valid'} or not manifest.get('source_custody'):
        raise ValueError('Exact roles and independently admitted official source custody')
    expected=COUNTS[task]
    if (manifest['train_count'],manifest['valid_count'])!=expected:
        raise ValueError('Official complete task population, not self-declared sampled counts')
    train=load_npz(bound(phase,manifest['payloads']['train']),ROLE_KEYS[task]['train'])
    valid=load_npz(bound(phase,manifest['payloads']['valid']),ROLE_KEYS[task]['valid']) if with_valid else None
    check_projection(task,train,valid,manifest)
    return train,valid,manifest


def ids_y(payload,count,classes,integer_label=False,max_id=None):
    ids,y=payload['ids'],payload['y']
    if ids.dtype!=torch.long or ids.ndim!=1 or y.ndim!=1 or len(ids)!=count or len(y)!=count or len(ids.unique())!=count:
        raise ValueError('Complete unique integer IDs/equal one-dimensional labels')
    if integer_label and y.dtype!=torch.long:raise ValueError('Integer classification targets')
    if not integer_label and y.dtype!=torch.float32:raise ValueError('Float32 binary targets')
    if not torch.isfinite(y).all() or (y<0).any() or (y>=classes).any() or not torch.equal(y,y.floor()):
        raise ValueError('Finite label class domain')
    if (ids<0).any() or (max_id is not None and (ids>=max_id).any()):raise ValueError('ID domain')


def validate_event_years(train_year,valid_year,train_count,valid_count):
    if train_year.dtype!=torch.long or train_year.ndim!=1 or len(train_year)!=train_count or (train_year<1963).any() or (train_year>2017).any():
        raise ValueError('Complete official TRAIN event years1963..2017; no future support event')
    if valid_year is not None and (valid_year.dtype!=torch.long or valid_year.ndim!=1 or len(valid_year)!=valid_count or not (valid_year==2018).all()):
        raise ValueError('Complete official VALID2018 event years')


def check_projection(task,train,valid,authority):
    require_keys(train,ROLE_KEYS[task]['train'])
    if valid is not None:require_keys(valid,ROLE_KEYS[task]['valid'])
    if task=='wikics':
        if train['x'].dtype!=torch.float32 or train['x'].shape!=(11701,300) or not torch.isfinite(train['x']).all():
            raise ValueError('Complete finite official WikiCS features')
        edge=train['edge_index']
        if edge.dtype!=torch.long or edge.shape!=(2,442907) or edge.min()<0 or edge.max()>=11701:
            raise ValueError('Complete fixed prepared WikiCS topology')
        if authority['split_index']!=0:raise ValueError('Pilot official split0')
        ids_y(train,580,10,True,11701)
        if valid is not None:
            ids_y(valid,5274,10,True,11701)
            if torch.isin(train['ids'],valid['ids']).any():raise ValueError('TRAIN/VALID overlap')
    elif task=='collab':
        if train['x'].shape!=(235868,128) or train['x'].dtype!=torch.float32 or not torch.isfinite(train['x']).all():
            raise ValueError('Complete finite collab features')
        if len(train['positive'])!=1179052:raise ValueError('Complete official TRAIN records')
        if valid is not None:
            if len(valid['positive'])!=60084 or len(valid['negative'])!=100000 or authority['valid_negative_count']!=100000:
                raise ValueError('Complete official VALID positives/negatives')
            # Canonical pairs may recur as different-year official events.
            # No VALID identities filter historical TRAIN support.
            validate_event_years(train['positive_year'],valid['positive_year'],1179052,60084)
        for payload in (train,valid):
            if payload is None:continue
            for key in ('positive','negative'):
                if key in payload:
                    q=payload[key]
                    if q.dtype!=torch.long or q.ndim!=2 or q.shape[1]!=2 or q.min()<0 or q.max()>=235868 or (key=='positive' and (q[:,0]==q[:,1]).any()):
                        raise ValueError('Integer in-domain pairs; positive records must be nonself')
        # Authentic official shared negatives contain one self-pair; preserve it.
        if type(authority.get('valid_negative_self_pair_records')) is not int or authority['valid_negative_self_pair_records']!=1:
            raise ValueError('Exact official VALID negative self-pair count custody')
        if valid is not None and int((valid['negative'][:,0]==valid['negative'][:,1]).sum())!=authority['valid_negative_self_pair_records']:
            raise ValueError('Complete unchanged official VALID negative self-pair count')
        if valid is None:validate_event_years(train['positive_year'],None,1179052,0)
        if authority.get('temporal_roles')!={'train_max':2017,'valid_only':2018}:raise ValueError('Official temporal identity')
    elif task=='molhiv':
        from ogb.utils.features import get_atom_feature_dims,get_bond_feature_dims
        if authority['split_kind']!='official_scaffold':raise ValueError('Official scaffold split identity')
        for count,payload in ((32901,train),(4113,valid)):
            if payload is None:continue
            ids_y(payload,count,2,False,41127)
            x,edge,attr=payload['x'],payload['edge_index'],payload['edge_attr']
            if x.dtype!=torch.long or x.ndim!=2 or x.shape[1]!=9 or attr.dtype!=torch.long or attr.ndim!=2 or attr.shape[1]!=3:
                raise ValueError('Original categorical atom/bond domains')
            if edge.dtype!=torch.long or edge.ndim!=2 or edge.shape[0]!=2 or attr.shape[0]!=edge.shape[1]:raise ValueError('Bond/edge alignment')
            for values,domains in ((x,get_atom_feature_dims()),(attr,get_bond_feature_dims())):
                if any((values[:,j]<0).any() or (values[:,j]>=limit).any() for j,limit in enumerate(domains)):
                    raise ValueError('OGB category index out of domain')
            nptr,eptr=payload['node_ptr'],payload['edge_ptr']
            if nptr.dtype!=torch.long or eptr.dtype!=torch.long or nptr.shape!=(count+1,) or eptr.shape!=(count+1,):raise ValueError('Integer complete graph boundaries')
            if nptr[0]!=0 or nptr[-1]!=len(x) or eptr[0]!=0 or eptr[-1]!=edge.shape[1] or (nptr.diff()<=0).any() or (eptr.diff()<0).any():raise ValueError('Boundary coverage/order')
            for i in range(count):
                a,b=map(int,eptr[i:i+2]);nodes=int(nptr[i+1]-nptr[i]);local=edge[:,a:b]
                if local.numel() and (local.min()<0 or local.max()>=nodes):raise ValueError('Local edge endpoint domain')
        if valid is not None and torch.isin(train['ids'],valid['ids']).any():raise ValueError('Molecule role overlap')
    else:raise ValueError(task)

def pair_ids(pairs, nodes):
    a, b = torch.minimum(pairs[:, 0], pairs[:, 1]), torch.maximum(pairs[:, 0], pairs[:, 1])
    return a * nodes + b


def support_graph(positives, nodes, masked_queries=None):
    # Remove ALL duplicate records of every selected positive target, BOTH ways.
    # This is deliberately stricter than author's record-only minibatch removal.
    if masked_queries is not None:
        keep = ~torch.isin(pair_ids(positives, nodes), pair_ids(masked_queries, nodes))
        positives = positives[keep]
    edges = torch.cat((positives.t(), positives.flip(1).t()), 1)
    adjacency = SparseTensor(row=edges[0], col=edges[1], sparse_sizes=(nodes, nodes)).coalesce()
    if masked_queries is not None:
        row, col, _ = adjacency.coo()
        if torch.isin(pair_ids(torch.stack((row, col), 1), nodes), pair_ids(masked_queries, nodes)).any():
            raise ValueError("Positive target retained in message/NCN support graph")
    return adjacency


def molecular_batch(payload, positions, device):
    graphs = []
    for i in positions.tolist():
        ns, ne = map(int, payload["node_ptr"][i:i+2])
        es, ee = map(int, payload["edge_ptr"][i:i+2])
        edge = payload["edge_index"][:, es:ee]
        # Stored edges are graph-local; validity checked per graph, no repair.
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= ne-ns):
            raise ValueError("Molecular graph-local edge indexing contract")
        graphs.append(Data(x=payload["x"][ns:ne], edge_index=edge,
                           edge_attr=payload["edge_attr"][es:ee]))
    return {"graph": Batch.from_data_list(graphs).to(device)}, payload["y"][positions].flatten().to(device)


def batches(task, train, recipe, epoch, seed, device):
    gen = torch.Generator().manual_seed(seed + 19709 + 1000*epoch)
    if task == "wikics":
        yield {"x": train["x"].to(device), "edge_index": train["edge_index"].to(device),
               "ids": train["ids"].to(device)}, train["y"].to(device)
    elif task == "molhiv":
        order = torch.randperm(len(train["y"]), generator=gen)
        for ids in order.split(recipe["batch_size"]):
            yield molecular_batch(train, ids, device)
    elif task == "collab":
        positives = train["positive"].to(device)
        raw_edges = torch.cat((positives.t(), positives.flip(1).t()), 1)
        # TRAIN-only negative authority; no future-positive rejection via held labels.
        negatives = negative_sampling(raw_edges, num_nodes=len(train["x"]), num_neg_samples=len(positives)).t()
        if len(negatives) < len(positives):
            raise ValueError("Incomplete negative population, no replacement rescue")
        order = torch.randperm(len(positives), generator=gen)
        for ids in order.split(recipe["batch_size"]):
            # Include complete tail; changed from native drop-last, disclosed.
            p = positives[ids.to(device)]
            q = torch.cat((p, negatives[ids.to(device)]), 0)
            yield {"x": train["x"].to(device), "adj": support_graph(positives, len(train["x"]), p), "query": q}, torch.cat((torch.ones(len(p)), torch.zeros(len(p)))).to(device)
    else:
        raise ValueError(task)


def valid_batches(task, train, valid, recipe, device):
    if task == "wikics":
        yield {"x": train["x"].to(device), "edge_index": train["edge_index"].to(device), "ids": valid["ids"].to(device)}, valid["y"]
    elif task == "molhiv":
        for ids in torch.arange(len(valid["y"])).split(recipe["batch_size"]):
            yield molecular_batch(valid, ids, device)
    elif task == "collab":
        q = torch.cat((valid["positive"], valid["negative"]), 0)
        adj = support_graph(train["positive"].to(device), len(train["x"]))
        for ids in torch.arange(len(q)).split(recipe["eval_batch_size"]):
            yield {"x": train["x"].to(device), "adj": adj, "query": q[ids].to(device)}, ids
