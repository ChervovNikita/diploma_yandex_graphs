"""Unchanged molecular role-packing functions from frozen V6 export source."""
import gzip


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
