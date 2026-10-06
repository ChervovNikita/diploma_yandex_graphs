"""Constructed graph proof checks only; no actual graph/data/model execution."""
import census

def graph(n,edges):
    neighbors=[set() for _ in range(n)]
    for u,v in edges:neighbors[u].add(v);neighbors[v].add(u)
    return neighbors

def oracle(neighbors,edges):
    facts=set(edges);out={}
    for u in range(len(neighbors)):
        for v in range(u+1,len(neighbors)):
            if (u,v) in facts:continue
            common=neighbors[u]&neighbors[v]
            internal=sum(1 for w,z in edges if w in common and z in common)
            if internal:out[u,v]=internal
    return out

def run():
    cases=[('empty',4,[]),('triangle',3,[(0,1),(0,2),(1,2)]),
        ('K4',4,[(u,v) for u in range(4) for v in range(u+1,4)]),
        ('K4_minus_edge',4,[(u,v) for u in range(4) for v in range(u+1,4) if (u,v)!=(0,1)]),
        ('overlapping_witnesses',6,[(u,v) for u in (0,1) for v in (2,3,4,5)]+[(2,3),(3,4),(4,5)]),
        ('cycle',5,[(0,1),(0,4),(1,2),(2,3),(3,4)])]
    result={}
    for name,n,edges in cases:
        edges=sorted(edges);neighbors=graph(n,edges)
        found,known,work,hist=census.enumerate_candidates(neighbors,edges,pair_budget=10000,unique_budget=1000)
        assert found==oracle(neighbors,edges)
        if name=='K4_minus_edge':assert found=={(0,1):1}
        if name=='K4':assert not found and len(known)==6
        if name=='overlapping_witnesses':assert found[0,1]==3
        census.summarize(neighbors,edges,found,known,work,hist,native_draws=1)
        result[name]={'passed':True,'candidates':len(found)}
    try:census.enumerate_candidates(graph(4,cases[3][2]),cases[3][2],pair_budget=0,unique_budget=1000)
    except RuntimeError:pass
    else:raise AssertionError('Work cap must abort rather than silently truncate')
    return {'passed':True,'constructed_cases':result,'work_cap_failure_explicit':True,'actual_graph_execution':False}

if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
