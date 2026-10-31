"""Stream only node/link ZIP members. Development labels are a separate input."""
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified(record):
    require({'path','sha256'}<=set(record)<={'path','sha256','bytes'},'Exact descriptor required')
    path = Path(record['path']); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    require(digest.hexdigest()==record['sha256'],'Input fingerprint mismatch: '+str(path))
    if 'bytes' in record:
        require(path.stat().st_size==record['bytes'],'Input byte count mismatch')
    return path


def read_development_labels(record, archive_sha256):
    value = json.loads(verified(record).read_text())
    require(value['scope']=='TRAIN_VAL_ONLY' and value['archive_sha256']==archive_sha256,
            'Explicit development-label scope/archive required')
    ids,labels,classes = value['node_ids'],value['labels'],value['train_class_schema']
    require(value['target_type']=='0' and len(ids)==len(labels)>0,'Target0 compact labels required')
    require(all(isinstance(i,int) and not isinstance(i,bool) and i>=0 for i in ids),'Invalid label node ID')
    require(ids==sorted(set(ids)),'Sorted distinct development node IDs required')
    require(classes==list(range(len(classes))) and len(classes)>1,'Contiguous TRAIN class schema required')
    require(all(isinstance(y,int) and not isinstance(y,bool) and y in classes for y in labels),'Single class labels required')
    return value


def stream_schema(archive_record, members):
    """No extraction and no opening label.dat or label.dat.test in this loader.

    Native COO->CSR duplicate coalescing is reproduced per raw relation, retaining
    stored zero-sum entries. Native HGT then ignores the weights. All raw records,
    duplicates, raw relation IDs and directed support counts are accounted for.
    """
    require(set(members)=={'nodes','links'},'Only node/link member names admitted')
    require(members['nodes'].endswith('/node.dat') and members['links'].endswith('/link.dat'),
            'Exact node/link ZIP members required')
    archive = verified(archive_record)
    counts,feature_widths = Counter(),{}
    target_attributes,attribute_rows = [],Counter()
    node_types = []
    opened,member_sha256 = [],{}
    with zipfile.ZipFile(archive) as zipped:
        require(len(zipped.namelist())==len(set(zipped.namelist())),'Duplicate ZIP member names refused')
        opened.append(members['nodes'])
        digest = hashlib.sha256()
        with zipped.open(members['nodes']) as member:
            for raw_line in member:
                digest.update(raw_line); line = raw_line.decode('utf-8')
                columns = line.rstrip('\r\n').split('\t')
                require(len(columns) in (3,4),'Node record needs3/4 columns')
                node,type_id = int(columns[0]),int(columns[2])
                require(node==len(node_types) and type_id>=0,'Native contiguous ordered node IDs required')
                node_types.append(type_id); counts[str(type_id)] += 1
                if len(columns)==4:
                    width = columns[3].count(',')+1
                    require(feature_widths.get(str(type_id),width)==width,'Inconsistent attribute dimension')
                    feature_widths[str(type_id)] = width; attribute_rows[str(type_id)] += 1
                    if type_id==0:
                        values = [float(number) for number in columns[3].split(',')]
                        require(all(math.isfinite(x) for x in values),'Nonfinite target features')
                        target_attributes.append(values)
        member_sha256['nodes'] = digest.hexdigest()
        types = sorted(counts,key=int)
        require(types==[str(i) for i in range(len(types))],'Native contiguous node types required')
        shifts = {}; offset = 0
        for t in types:
            shifts[t] = offset
            require(node_types[offset:offset+counts[t]]==[int(t)]*counts[t],'Native contiguous type blocks required')
            require(attribute_rows[t] in (0,counts[t]),'Partial type attributes refused')
            offset += counts[t]
        edges,raw_counts,endpoints = defaultdict(dict),Counter(),{}
        opened.append(members['links'])
        digest = hashlib.sha256()
        with zipped.open(members['links']) as member:
            for raw_line in member:
                digest.update(raw_line); line = raw_line.decode('utf-8')
                columns = line.rstrip('\r\n').split('\t')
                require(len(columns)==4,'Link record needs4 columns')
                source,target,raw = map(int,columns[:3]); weight = float(columns[3])
                require(0<=source<len(node_types) and 0<=target<len(node_types) and raw>=0 and math.isfinite(weight),
                        'Invalid directed link record')
                endpoint = (str(node_types[source]),str(node_types[target]))
                require(endpoints.get(raw,endpoint)==endpoint,'Raw relation has inconsistent endpoints')
                endpoints[raw] = endpoint; raw_counts[raw] += 1
                pair = (source-shifts[endpoint[0]],target-shifts[endpoint[1]])
                edges[raw][pair] = edges[raw].get(pair,0.0)+weight
        member_sha256['links'] = digest.hexdigest()
    raw_rows = []
    for raw in sorted(edges):
        src_type,dst_type = endpoints[raw]
        raw_rows.append(dict(raw_id=raw,source=src_type,target=dst_type,
              name=f'raw{raw}__{src_type}__{dst_type}',raw_records=raw_counts[raw],
              support_edges=len(edges[raw]),duplicates_coalesced=raw_counts[raw]-len(edges[raw]),
              zero_sum_stored_edges=sum(weight==0 for weight in edges[raw].values())))
    schema = dict(node_counts=dict(counts),node_shifts=shifts,provided_attribute_widths=feature_widths,
                  feature_type=2,target='0',target_given_attributes=bool(target_attributes),
                  input_dims={t:feature_widths[t] if t=='0' and target_attributes else counts[t] for t in types},
                  relations=raw_rows,raw_link_records=sum(raw_counts.values()),
                  support_edges=sum(len(value) for value in edges.values()),
                  node_link_members_opened=opened,label_members_opened=[],
                  member_sha256=member_sha256,
                  duplicate_policy='native per-raw-relation COO->CSR support; weights ignored after coalescing',
                  synthetic_reverse_or_self_added=False)
    return schema,target_attributes,edges


def materialize(schema, target_attributes, edges, implementation, row_order, device):
    import torch
    counts = schema['node_counts']
    features = {}
    for t in sorted(counts):
        if t=='0' and target_attributes:
            features[t] = torch.tensor(target_attributes,dtype=torch.float32,device=device)
        else:
            # Native type2: sparse identity for every non-target type, and for a
            # target with no provided attributes (author load_data fallback).
            ids = torch.arange(counts[t],device=device)
            features[t] = torch.sparse_coo_tensor(torch.stack([ids,ids]),torch.ones(counts[t],device=device),
                            (counts[t],counts[t]),device=device).coalesce()
    relations = []
    for row in schema['relations']:
        pairs = sorted(edges[row['raw_id']])
        src = torch.tensor([pair[0] for pair in pairs],dtype=torch.long,device=device)
        dst = torch.tensor([pair[1] for pair in pairs],dtype=torch.long,device=device)
        relations.append(implementation.Relation(row['raw_id'],row['source'],row['target'],src,dst))
    graph = implementation.HeteroGraph(counts,relations,row_order)
    schema = dict(schema,relation_row_order=list(graph.edge_dict),raw_relation_to_row=graph.raw_relation_to_row)
    return graph,features,schema


def verify_split(record, development, seed):
    value = json.loads(verified(record).read_text())
    train,validation = value['train_ids'],value['validation_ids']
    require(value['seed']==seed and value['algorithm']=='NumPy RandomState(seed) shuffle sorted development IDs; first floor(0.2N) validation',
            'Frozen paired native split convention differs')
    require(train==sorted(set(train)) and validation==sorted(set(validation)) and not set(train)&set(validation),
            'Distinct sorted TRAIN/VAL split required')
    require(sorted(train+validation)==development['node_ids'] and len(validation)==int(.2*len(development['node_ids'])),
            'Exact complete development pool/native split ratio required')
    # Verify seed correspondence with a private RNG, never the global stream.
    import numpy as np
    shuffled = np.array(development['node_ids'],dtype=np.int64)
    np.random.RandomState(seed).shuffle(shuffled); cut = int(.2*len(shuffled))
    require(train==sorted(shuffled[cut:].tolist()) and validation==sorted(shuffled[:cut].tolist()),
            'Frozen split does not match its paired seed')
    lookup = dict(zip(development['node_ids'],development['labels']))
    require(set(lookup[i] for i in train)==set(development['train_class_schema']),
            'Classes must be established by TRAIN, not validation/test')
    return value,[lookup[i] for i in train],[lookup[i] for i in validation]
