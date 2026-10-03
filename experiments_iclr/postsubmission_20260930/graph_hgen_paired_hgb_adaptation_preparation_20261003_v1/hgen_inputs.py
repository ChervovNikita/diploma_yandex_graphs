"""Complete paired-HGB topology -> native HGEN binary author metapath views."""
import numpy as np
import scipy.sparse as sp


EXPECTED_ENDPOINTS = {0:('0','1'),1:('1','2'),2:('1','3'),
                      3:('1','0'),4:('2','1'),5:('3','1')}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def binary_metapaths(schema,edges):
    require({r['raw_id']:(r['source'],r['target']) for r in schema['relations']}==EXPECTED_ENDPOINTS,
            'Complete HGB-DBLP six-direction A/P/T/V schema required')
    require(all(v==1. for rel in edges.values() for v in rel.values()),
            'Current qualified DBLP unit/no-duplicate link support required')
    raw = {}
    for r in schema['relations']:
        pairs = sorted(edges[r['raw_id']]); raw[r['raw_id']] = sp.csr_matrix(
            (np.ones(len(pairs),dtype=np.bool_),([x[0] for x in pairs],[x[1] for x in pairs])),
            shape=(schema['node_counts'][r['source']],schema['node_counts'][r['target']]))
    ap,pt,pv,pa,tp,vp = [raw[i] for i in range(6)]
    # Boolean reassociation preserves exact reachability and avoids a potentially
    # dense author-by-paper intermediate. Native AddMetaPaths weighted=False
    # retains support only, not path multiplicities, sampling or edge weights.
    values = {'APA':(ap@pa).tocsr(),
              'APTPA':((ap@pt)@(tp@pa)).tocsr(),
              'APCPA':((ap@pv)@(vp@pa)).tocsr()}
    for name,value in values.items():
        value.eliminate_zeros(); value.sort_indices()
        require(value.dtype==np.bool_ and value.shape==(schema['node_counts']['0'],)*2,
                'Binary full-target metapath support required: '+name)
    return values


def normalized_views(supports,implementation,device='cpu',dtype=None):
    import torch
    dtype = dtype or torch.float32
    views,audit = [],{}
    for name in implementation.VIEW_NAMES:
        support = supports[name]
        # Matrix rows are sources in file/metapath products. GCN message matrix
        # rows must be destinations. Remove then insert one unit loop per node,
        # equivalent to PyG add_remaining_self_loops for unweighted supports.
        matrix = support.T.tocsr(copy=True); matrix.setdiag(True); matrix.sort_indices()
        nodes = matrix.shape[0]
        crow = torch.from_numpy(matrix.indptr.astype(np.int64,copy=True))
        column = torch.from_numpy(matrix.indices.astype(np.int64,copy=True))
        degree = crow[1:]-crow[:-1]
        inverse = degree.to(dtype).pow(-.5)
        inverse.masked_fill_(inverse==float('inf'),0.)
        row = torch.repeat_interleave(torch.arange(nodes),degree)
        weights = inverse[column]*inverse[row]
        adjacency = torch.sparse_csr_tensor(crow,column,weights,size=(nodes,nodes),dtype=dtype)
        views.append(implementation.NormalizedView(name,adjacency).to(device))
        audit[name] = dict(binary_metapath_edges=int(support.nnz),existing_metapath_self_edges=int(np.count_nonzero(support.diagonal())),
            normalized_edges=int(matrix.nnz),normalization='PyG GCN defaults: unweighted support,one unit self per node,incoming degree,source_to_target',
            symmetric_support=(support!=support.T).nnz==0,path_count_weights_used=False,sampled=False)
    return views,audit
