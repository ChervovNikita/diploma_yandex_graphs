"""ACM native graph/features; shared ACM reader opens node/link members only."""
from collections import Counter
import hashlib
import json


LETTERS = {'0':'P','1':'A','2':'C','3':'K'}
EXPECTED_ENDPOINTS = {0:('0','0'),1:('0','0'),2:('0','1'),3:('1','0'),
                      4:('0','2'),5:('2','0'),6:('0','3'),7:('3','0')}
RAW_ORDER = list(range(8))
# DGL canonical etype names are lexical in the author feature traversal.
FEATURE_ETYPE_ORDER = [('A','P','PA'),('C','P','PC'),('P','A','AP'),
                       ('P','C','CP'),('P','P','PP')]


def require(condition,message):
    if not condition:
        raise ValueError(message)


def stream_all_attributes(loader,archive,members):
    schema,attributes,edges = loader.stream_schema(archive,members)
    require({r['raw_id']:(r['source'],r['target']) for r in schema['relations']}==EXPECTED_ENDPOINTS,
            'Complete ACM P/A/C/K eight-relation schema required')
    # The shared reader preserves raw dictionary insertion order from link.dat.
    # Native HGB edge2type overwrites in that same loader encounter order.
    require(list(edges)==RAW_ORDER,'Author ACM raw relation encounter order0..7 differs')
    require(all(r['duplicates_coalesced']==0 for r in schema['relations'])
            and all(w==1. for relation in edges.values() for w in relation.values()),
            'Bound ACM unit support without within-relation duplicates required')
    schema['native_raw_relation_iteration_order'] = list(edges)
    return schema,attributes,edges


def homogeneous_records(schema,edges):
    """Literal author last-raw overwrite; retain raw self labels on replacement."""
    shifts = schema['node_shifts']; total = sum(schema['node_counts'].values())
    require(list(edges)==RAW_ORDER,'Exact author relation iteration order required')
    labels,raw_edges,collisions = {},[],Counter()
    for raw in RAW_ORDER:
        stype,dtype = EXPECTED_ENDPOINTS[raw]
        for source,target in sorted(edges[raw]):
            pair = (source+shifts[stype],target+shifts[dtype])
            if pair in labels and labels[pair]!=raw:
                collisions[f'{labels[pair]}->{raw}'] += 1
            labels[pair] = raw
            raw_edges.append((raw,pair))
    raw_self_labels = Counter(labels[pair] for pair in labels if pair[0]==pair[1])
    for i in range(total):
        labels.setdefault((i,i),8)
    for raw,(source,target) in raw_edges:
        labels.setdefault((target,source),raw+9)
    support = {pair for _,pair in raw_edges}
    support.update((v,u) for u,v in list(support))
    pairs = sorted((u,v) for u,v in support if u!=v)+[(i,i) for i in range(total)]
    etypes = [labels[pair] for pair in pairs]
    fingerprint = hashlib.sha256()
    for pair,etype in zip(pairs,etypes):
        fingerprint.update(f'{pair[0]}\t{pair[1]}\t{etype}\n'.encode())
    return dict(nodes=total,src=[p[0] for p in pairs],dst=[p[1] for p in pairs],
        etype=etypes,num_etypes=17,audit=dict(
            raw_iteration_order=RAW_ORDER,ordered_pair_relation_collisions=sum(collisions.values()),
            overwrite_counts=dict(collisions),collision_policy='literal last raw relation wins',
            released_raw_self_final_labels={str(k):v for k,v in sorted(raw_self_labels.items())},
            self_loops_added=total,homogeneous_edges=len(pairs),
            reverse_labels_synthetic=sum(x>8 for x in etypes),
            edge_type_counts={str(k):v for k,v in sorted(Counter(etypes).items())},
            edge_label_assignment_sha256=fingerprint.hexdigest(),
            edge_order='CSR row/column sorted non-self support followed by ascending self loops'))


def homogeneous_features(schema,attributes,device):
    """Native ACM feats2: paper attributes plus sparse identity for other types."""
    import torch
    require(bool(attributes['0']),'Released ACM paper attributes required')
    features = [torch.tensor(attributes['0'],dtype=torch.float32,device=device)]
    for t in ('1','2','3'):
        count = schema['node_counts'][t]; ids = torch.arange(count,device=device)
        features.append(torch.sparse_coo_tensor(torch.stack([ids,ids]),
            torch.ones(count,device=device),(count,count),device=device).coalesce())
    return features


def normalized_native_adjacencies(schema,attributes,edges):
    """Destination-first raw row; source ACM assertions, PP union+diag, then norm."""
    import numpy as np
    import scipy.sparse as sp
    names = ['PP','PP_r','PA','AP','PC','CP','PK','KP']; raw = {}
    for rid,name in enumerate(names):
        dtype,stype = EXPECTED_ENDPOINTS[rid]; pairs = sorted(edges[rid])
        raw[name] = sp.csr_matrix((np.ones(len(pairs),dtype=np.float32),
            ([p[0] for p in pairs],[p[1] for p in pairs])),
            shape=(schema['node_counts'][dtype],schema['node_counts'][stype]))
    P,A,C = (np.asarray(attributes[t],dtype=np.float32) for t in ('0','1','2'))
    row,col = np.nonzero(P); pk = raw['PK'].tocoo()
    require(np.array_equal(row,pk.row) and np.array_equal(col,pk.col),
            'Source ACM nonzeros(P)==support(PK) assertion differs')
    require(np.array_equal((raw['AP']@raw['PK']).toarray(),A)
            and np.array_equal((raw['CP']@raw['PK']).toarray(),C),
            'Source ACM author/conference attributes do not match field products')
    for forward,reverse in (('PA','AP'),('PC','CP'),('PK','KP')):
        require((raw[forward]!=raw[reverse].T).nnz==0,'Source ACM transpose support differs: '+forward)
    # All entries are unit support. maximum is binary coalescing, not multiplicity.
    PP = raw['PP'].maximum(raw['PP_r']).tocsr()
    PP.setdiag(np.float32(1)); PP.sort_indices()
    adjs = {'PP':PP,'PA':raw['PA'],'AP':raw['AP'],'PC':raw['PC'],'CP':raw['CP']}
    audit = dict(ACM_keep_F=False,attribute_identities_verified=True,
        PP_raw_support=[raw['PP'].nnz,raw['PP_r'].nnz],PP_merged_diagonal_support=PP.nnz,
        PP_rule='binary union/coalesce/set_diag before row normalization',
        operator_convention='destination-first raw file row',retained_operator_order=list(adjs))
    for adj in adjs.values():
        degree = np.diff(adj.indptr)
        adj.data = np.repeat(np.divide(np.float32(1),degree,
            out=np.zeros(len(degree),dtype=np.float32),where=degree!=0),degree)
    return adjs,audit


def feature_channels(schema,attributes,adjs):
    """Native DGL mean propagation/removal traversal for four-hop target P."""
    import numpy as np
    import torch
    state = {letter:{letter:np.asarray(attributes[t],dtype=np.float32)}
             for t,letter in (('0','P'),('1','A'),('2','C'))}
    for hop in range(1,5):
        for stype,dtype,operator in FEATURE_ETYPE_ORDER:
            for key in list(state[stype]):
                if len(key)!=hop or (hop==4 and dtype!='P'):
                    continue
                name = dtype+key
                require(name not in state[dtype],'Duplicate native feature path')
                state[dtype][name] = adjs[operator]@state[stype][key]
        for ntype in ('A','C'):
            state[ntype] = {k:v for k,v in state[ntype].items() if len(k)>hop}
    values = state['P']
    require(len(values)==41 and Counter(len(k)-1 for k in values)=={0:1,1:3,2:5,3:11,4:21},
            'Native ACM feature schema must contain41 channels through4 hops')
    require(all(v.shape==(schema['node_counts']['0'],1902) for v in values.values()),
            'Native ACM feature widths differ')
    return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in values.items()}


def label_products(adjs,hops=4):
    """Qualified left-extension/removal logic, target P and native four hops."""
    require(hops==4,'This ACM recipe binds four-hop labels')
    current = {k:v.copy() for k,v in adjs.items() if k[-1]=='P'}
    for hop in range(2,hops+1):
        new = {}
        for right,adj_r in current.items():
            if len(right)!=hop:
                continue
            for left,adj_l in adjs.items():
                if left[-1]!=right[0] or (hop==hops and left[0]!='P'):
                    continue
                name = left[0]+right
                require(name not in new,'Duplicate metapath product')
                new[name] = (adj_l@adj_r).tocsr()
        current.update(new)
        current = {k:v for k,v in current.items() if k[0]=='P' or len(k)>hop}
    require(len(current)==20 and Counter(len(k)-1 for k in current)=={1:1,2:3,3:5,4:11}
            and all(k[0]==k[-1]=='P' for k in current),
            'Native ACM label schema must contain20 target-to-target channels')
    return current


def train_only_label_channels(products,target_nodes,train_ids,train_labels,classes=3):
    import numpy as np
    import torch
    require(classes==3 and set(train_labels)==set(range(classes)),'All three classes must be established by TRAIN')
    require(len(train_ids)==len(train_labels) and len(set(train_ids))==len(train_ids),'Distinct TRAIN membership required')
    onehot = np.zeros((target_nodes,classes),dtype=np.float32)
    onehot[np.asarray(train_ids),np.asarray(train_labels)] = 1.
    output = {}
    for key,product in products.items():
        corrected = product.copy(); corrected.setdiag(0); corrected.eliminate_zeros()
        output[key] = torch.from_numpy(np.ascontiguousarray(corrected@onehot))
    return output


def tensor_fingerprint(value):
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(value.dtype),shape=list(value.shape)),sort_keys=True).encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def channel_receipt(features,label_features,eval_ids):
    """No target labels or values retained: channel/order/cache and batch hashes."""
    return dict(feature_initialization_order=list(features),feature_stack_order=sorted(features),
        label_initialization_and_stack_order=sorted(label_features),
        feature_shapes={k:list(v.shape) for k,v in features.items()},
        label_shapes={k:list(v.shape) for k,v in label_features.items()},
        feature_sha256={k:tensor_fingerprint(v) for k,v in features.items()},
        TRAIN_only_label_sha256={k:tensor_fingerprint(v) for k,v in label_features.items()},
        evaluation_ID_sha256=hashlib.sha256(json.dumps(eval_ids,separators=(',',':')).encode()).hexdigest(),
        evaluation_nodes=len(eval_ids),TEST_label_reads=0)
