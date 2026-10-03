"""Native DBLP preprocessing; exact development descriptors reuse the sealed loader.

The only ZIP members ever opened are node.dat and link.dat. Attribute propagation
uses all topology, while propagated label inputs are seeded only by TRAIN.
"""
import hashlib
import math
import zipfile


LETTERS = {'0':'A','1':'P','2':'T','3':'V'}
EXPECTED_ENDPOINTS = {0:('0','1'),1:('1','2'),2:('1','3'),
                      3:('1','0'),4:('2','1'),5:('3','1')}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def stream_all_attributes(loader, archive, members):
    schema,target,edges = loader.stream_schema(archive,members)
    require({row['raw_id']:(row['source'],row['target']) for row in schema['relations']}==EXPECTED_ENDPOINTS,
            'Native DBLP A/P/T/V six-direction schema required')
    require(all(weight==1. for relation in edges.values() for weight in relation.values()),
            'Qualified DBLP relation support must have unit weights and no duplicates')
    attributes = {t:[] for t in schema['node_counts']}
    digest = hashlib.sha256()
    # Reopen only the admitted node member to retain the non-target attributes.
    with zipfile.ZipFile(loader.verified(archive)) as zipped:
        with zipped.open(members['nodes']) as stream:
            for line in stream:
                digest.update(line)
                cols = line.decode('utf-8').rstrip('\r\n').split('\t'); t = cols[2]
                if len(cols)==4:
                    values = [float(x) for x in cols[3].split(',')]
                    require(all(math.isfinite(x) for x in values),'Nonfinite provided attribute')
                    attributes[t].append(values)
    require(digest.hexdigest()==schema['member_sha256']['nodes'],'Node bytes changed between feature passes')
    require(attributes['0']==target,'Target attribute pass mismatch')
    schema['challenger_attributes'] = 'all provided A/P/T attributes; absent V -> identity for SeHGNN'
    schema['node_member_passes'] = 2
    return schema,attributes,edges


def homogeneous_records(schema, edges):
    """Native edge2type and undirected union, remove loops then append one per node."""
    shifts = schema['node_shifts']; total = sum(schema['node_counts'].values())
    raw_ids = sorted(edges)
    require(raw_ids==list(range(len(raw_ids))),'Native raw edge labels assume contiguous IDs')
    labels,raw_edges,collisions = {},[],0
    for row in schema['relations']:
        raw = row['raw_id']
        for source,target in sorted(edges[raw]):
            pair = (source+shifts[row['source']],target+shifts[row['target']])
            if pair in labels and labels[pair]!=raw:
                collisions += 1
            labels[pair] = raw; raw_edges.append((raw,pair))
    # Actual byte-bound DBLP has no ambiguous ordered pair. Do not silently repair
    # an endpoint collapse and then claim author correspondence on another graph.
    require(collisions==0,'Parallel raw relation labels on an ordered pair require separate qualification')
    for i in range(total):
        labels.setdefault((i,i),len(raw_ids))
    for raw,(source,target) in raw_edges:
        labels.setdefault((target,source),raw+1+len(raw_ids))
    support = {pair for _,pair in raw_edges}
    support.update((v,u) for u,v in list(support))
    pairs = sorted((u,v) for u,v in support if u!=v)+[(i,i) for i in range(total)]
    return dict(nodes=total,src=[x[0] for x in pairs],dst=[x[1] for x in pairs],
                etype=[labels[x] for x in pairs],num_etypes=2*len(raw_ids)+1,
                audit=dict(ordered_pair_relation_collisions=collisions,self_loops_added=total,
                           homogeneous_edges=len(pairs),reverse_labels_synthetic=sum(labels[x]>len(raw_ids) for x in pairs),
                           edge_order='CSR row/column sorted non-self support followed by ascending self loops'))


def materialize_homogeneous(records, model_module, device):
    import torch
    return model_module.HomogeneousGraph(records['nodes'],
        torch.tensor(records['src'],dtype=torch.long,device=device),
        torch.tensor(records['dst'],dtype=torch.long,device=device),
        torch.tensor(records['etype'],dtype=torch.long,device=device),records['num_etypes'])


def normalized_native_adjacencies(schema, edges):
    """Author SparseTensor uses raw file row as destination, col as source.

    Thus raw0 (A,P) becomes AP, a destination-first A <- P operator. This
    deliberately follows SeHGNN's loader rather than HGT's raw source convention.
    """
    import numpy as np
    import scipy.sparse as sp
    adjs = {}
    for row in schema['relations']:
        raw = row['raw_id']; dtype,stype = row['source'],row['target']
        pairs = sorted(edges[raw]); key = LETTERS[dtype]+LETTERS[stype]
        adj = sp.csr_matrix((np.ones(len(pairs),dtype=np.float32),
            ([p[0] for p in pairs],[p[1] for p in pairs])),
            shape=(schema['node_counts'][dtype],schema['node_counts'][stype]))
        degree = np.diff(adj.indptr)
        adj.data = np.repeat(np.divide(np.float32(1),degree,
            out=np.zeros(len(degree),dtype=np.float32),where=degree!=0),degree)
        require(key not in adjs,'Same endpoint metapath names require a new qualification')
        adjs[key] = adj
    require(set(adjs)=={'AP','PA','PT','PV','TP','VP'},'Native SeHGNN relation set differs')
    return adjs


def feature_channels(schema, attributes, adjs):
    """Only compute channels retained by native two-hop target-A propagation."""
    import numpy as np
    import torch
    raw = {}
    for t in ('0','1','2','3'):
        raw[LETTERS[t]] = np.asarray(attributes[t],dtype=np.float32) if attributes[t] else np.eye(schema['node_counts'][t],dtype=np.float32)
    values = dict(A=raw['A'],AP=adjs['AP']@raw['P'],
        APA=adjs['AP']@(adjs['PA']@raw['A']),
        APT=adjs['AP']@(adjs['PT']@raw['T']),
        APV=adjs['AP']@(adjs['PV']@raw['V']))
    return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in values.items()}


def label_products(adjs, hops=4):
    """Literal native left-extension/removal order for target-start label paths."""
    require(hops==4,'This qualification binds native DBLP four-hop labels')
    current = {k:v.copy() for k,v in adjs.items() if k[-1]=='A'}
    for hop in range(2,hops+1):
        new = {}
        for right,adj_r in current.items():
            if len(right)!=hop:
                continue
            for left,adj_l in adjs.items():
                if left[-1]!=right[0] or (hop==hops and left[0]!='A'):
                    continue
                name = left[0]+right
                require(name not in new,'Duplicate metapath product')
                new[name] = (adj_l@adj_r).tocsr()
        current.update(new)
        current = {k:v for k,v in current.items() if k[0]=='A' or len(k)>hop}
    require(set(current)=={'APA','APAPA','APTPA','APVPA'},'Native label metapath set differs')
    return current


def train_only_label_channels(products, target_nodes, train_ids, train_labels, classes=4):
    import numpy as np
    import torch
    require(classes==4 and set(train_labels)==set(range(classes)),'Classifier schema must be established by TRAIN')
    require(len(train_ids)==len(train_labels) and len(set(train_ids))==len(train_ids),'Distinct TRAIN membership required')
    onehot = np.zeros((target_nodes,classes),dtype=np.float32)
    onehot[np.asarray(train_ids),np.asarray(train_labels)] = 1.
    output = {}
    for key,product in products.items():
        # Remove the diagonal AFTER the complete product, never from each edge
        # adjacency, never renormalize after removal, never seed VAL/TEST labels.
        corrected = product.copy(); corrected.setdiag(0); corrected.eliminate_zeros()
        output[key] = torch.from_numpy(np.ascontiguousarray(corrected@onehot))
    return output


def native_evaluation_ids(target_nodes, split):
    """Freeze BatchNorm composition using topology and development membership only.

    In complete HGB-DBLP all4057 authors have TRAIN/VAL or withheld membership.
    All target features can share the native evaluation batch without accessing
    withheld labels or computing withheld metrics. The literal author ordering is
    sorted TRAIN, sorted VAL, sorted remaining target IDs, one batch (<20000).
    """
    train,validation = split['train_ids'],split['validation_ids']
    used = set(train+validation)
    remaining = [i for i in range(target_nodes) if i not in used]
    ids = train+validation+remaining
    require(len(ids)==target_nodes and len(set(ids))==target_nodes and target_nodes<20000,
            'Qualified native full target evaluation batch differs')
    return ids
