"""Frozen randomized DFS forest tours on qualified CPU topology only.

Torch is injected; no graph, tensor, generator, label or numerical import at
module import. Build once before fits. Native support is never modified.
"""
BLOCKS=(10103,10427,10949)
NODES=34493

def build_tours(torch,edge_index,block,*,expected_components):
    """Four fixed tours and full private generator states for root custody.

    Root binds actual component count before calling this constructor. Require CPU because per-node
    root/neighbor shuffling and iterative DFS are preprocessing, not CUDA loops.
    Order: private randperm(N) roots, then randperm(deg(v)) for sorted adjacency
    lists in ascending node ID, including empty lists; no randomness in DFS.
    """
    if type(expected_components) is not int or not 1<=expected_components<=NODES:
        raise ValueError('Root-bound actual component count required before tour draws')
    if (block not in BLOCKS or edge_index.device.type!='cpu' or edge_index.dtype!=torch.int64
            or edge_index.shape!=(2,530417)):
        raise ValueError('Qualified full Physics CPU edge_index and frozen block')
    rows,columns=edge_index.tolist();diagonal=[];off_diagonal=set();entries=0
    for u,v in zip(rows,columns):
        if not (0<=u<NODES and 0<=v<NODES):raise ValueError('Original node IDs required')
        if u==v:diagonal.append(u)
        else:off_diagonal.add((u,v));entries+=1
    if len(off_diagonal)!=entries or sorted(diagonal)!=list(range(NODES)):
        raise ValueError('Coalesced support and one native self record per node')
    if any((v,u) not in off_diagonal for u,v in off_diagonal):
        raise ValueError('Reciprocal native off-diagonal support required')
    neighbors=[[] for _ in range(NODES)]
    for u,v in off_diagonal:neighbors[u].append(v)
    for row in neighbors:row.sort()
    result=[]
    for view in range(4):
        seed=block+337000031+1000003*view
        generator=torch.Generator(device='cpu').manual_seed(seed);before=generator.get_state().clone()
        roots=torch.randperm(NODES,generator=generator,device='cpu').tolist()
        shuffled=[]
        for row in neighbors:
            order=torch.randperm(len(row),generator=generator,device='cpu').tolist()
            shuffled.append([row[i] for i in order])
        after=generator.get_state().clone();visited=[False]*NODES;tokens=[];offsets=[0]
        for root in roots:
            if visited[root]:continue
            visited[root]=True;tokens.append(root);stack=[(root,iter(shuffled[root]))]
            while stack:
                node,iterator=stack[-1];child=next(iterator,None)
                if child is None:
                    stack.pop()
                    if stack:tokens.append(stack[-1][0])
                elif not visited[child]:
                    visited[child]=True;tokens.append(child);stack.append((child,iter(shuffled[child])))
            offsets.append(len(tokens))
        components=len(offsets)-1;length=len(tokens)
        if not all(visited) or components!=expected_components or length!=2*NODES-expected_components or length>68985:
            raise ValueError('Complete forest arrival/return length differs')
        reverse=[];starts=[False]*length
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            starts[lo]=True;reverse.extend(range(hi-1,lo-1,-1))
        ids=torch.tensor(tokens,dtype=torch.int64,device='cpu')
        counts=torch.bincount(ids,minlength=NODES)
        if not bool((counts>0).all()):raise ValueError('Every node must occur')
        result.append(dict(tokens=ids,reverse_index=torch.tensor(reverse,dtype=torch.int64,device='cpu'),starts=torch.tensor(starts,dtype=torch.bool,device='cpu'),occurrences=counts,offsets=torch.tensor(offsets,dtype=torch.int64,device='cpu'),generator_before_roots=before,generator_after_neighbors=after,seed=seed,view=view,components=components,length=length,nodes=NODES,sampling_law='randomized DFS spanning forest; not uniform spanning tree',ordering_uses_labels=False))
    if len({t['components'] for t in result})!=1:raise ValueError('All tours must cover the same graph components')
    return tuple(result)


def to_device(tour,device):
    """Move only fixed model-facing payloads; root retains CPU RNG/offset receipts."""
    return {key:tour[key].to(device) for key in ('tokens','reverse_index','starts','occurrences')}
