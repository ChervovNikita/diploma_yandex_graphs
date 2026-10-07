"""Deterministic algorithm checks, not scientific data or training evidence."""
import ast
import json
from pathlib import Path
import numpy as np
from context_positive_masks import graph_contexts, positive_masks, row_weights, within_class_permuted, sampled_weights


def main():
    checks = []
    x = np.array([[1.,0.,0.],[.8,.2,0.],[.1,.9,0.],[0.,.2,.8],[0.,0.,1.],[.2,0.,.8]])
    edges = np.array([[0,1,1,2,2,3,3,4,4,5,5,0,1],[1,0,2,1,3,2,4,3,5,4,0,5,2]])
    ids = np.arange(6)
    labels = np.array([0,0,0,1,1,1])
    contexts = graph_contexts(x, edges, ids, chunk_edges=2)
    adjacency = np.zeros((6,6)); adjacency[edges[1],edges[0]]=1;np.fill_diagonal(adjacency,1)
    transition = adjacency/adjacency.sum(1,keepdims=True)
    expected = np.stack([x,x-transition@x,transition@x,transition@transition@x])
    assert np.allclose(contexts,expected)
    checks.append('explicit directed destination convention, duplicate removal, one self-loop and two propagation steps')
    masks = positive_masks(contexts,labels,neighbors=1)
    assert masks.shape == (4,6,6) and np.all(masks.sum(-1)==2)
    assert np.all(~masks | (labels[:,None]==labels[None,:])[None])
    assert np.all(np.diagonal(masks,axis1=1,axis2=2))
    checks.append('exact self and one same-class neighbor per row; no wrong-class positive')
    again = positive_masks(contexts,labels,neighbors=1)
    assert np.array_equal(masks,again)
    checks.append('deterministic relation construction')
    sampled = np.array([0,2,3,5])
    permutation_masks,permutations=within_class_permuted(masks,labels,seed=99221,panel_rows=sampled)
    assert np.all(labels[permutations]==labels[None])
    assert np.array_equal(np.sort(masks.sum(-1),axis=1),np.sort(permutation_masks.sum(-1),axis=1))
    in_panel=np.zeros(len(labels),dtype=bool);in_panel[sampled]=True
    assert np.all(in_panel[permutations]==in_panel[None])
    original_restricted=masks[:,sampled[:,None],sampled[None,:]]
    permuted_restricted=permutation_masks[:,sampled[:,None],sampled[None,:]]
    assert np.array_equal(original_restricted.sum(-1),permuted_restricted.sum(-1))
    checks.append('permuted control preserves positive class, self identity, panel membership and exact per-anchor restricted degree')
    # A denser fixture confirms that the stronger matching does not inherently
    # force identity, while no model outcomes choose a permutation seed.
    rng=np.random.default_rng(44721);labels_large=np.repeat(np.arange(3),12)
    mask_large=positive_masks(rng.normal(size=(4,36,9)),labels_large,neighbors=3)
    panel_large=np.array([i for i in range(36) if i%4!=0])
    perm_large,perm_ids=within_class_permuted(mask_large,labels_large,seed=93321,panel_rows=panel_large)
    before=mask_large[:,panel_large[:,None],panel_large[None,:]]
    after=perm_large[:,panel_large[:,None],panel_large[None,:]]
    assert np.array_equal(before.sum(-1),after.sum(-1)) and np.any(before!=after)
    checks.append('degree-matched panel control can change target identities without changing scored positive counts')
    route = sampled_weights(masks,sampled,'route');common=sampled_weights(masks,sampled,'common')
    assert np.allclose(route.mean(0),common.mean(0))
    assert np.allclose(route.sum(-1),1) and np.allclose(common.sum(-1),1)
    checks.append('sample restriction preserves self; common targets exactly preserve aggregate route mass')
    cycles=np.stack([sampled_weights(masks,sampled,'cycle',update_index=i) for i in range(4)])
    assert np.allclose(cycles.mean(0),common)
    checks.append('four-cycle target identity for one fixed sample; no per-update cycling identity claimed')
    # At exactly equal member scores, scalar loss and aggregate shared score
    # cotangent match. Private route cotangents can differ by construction.
    rng=np.random.default_rng(66321);score=rng.normal(size=(len(sampled),len(sampled)))
    exp=np.exp(score-score.max(-1,keepdims=True));prob=exp/exp.sum(-1,keepdims=True);lp=np.log(prob)
    route_loss=-(route*lp[None]).sum(-1).mean();common_loss=-(common*lp[None]).sum(-1).mean()
    assert np.allclose(route_loss,common_loss)
    assert np.allclose((prob[None]-route).mean(0),(prob[None]-common).mean(0))
    checks.append('equal-output common objective and shared cotangent identity')
    mismatch = float(np.max(np.abs(route-common)))
    if mismatch == 0:
        # Fixture may erase difference through sampling; a direct distinct
        # relation fixture checks the general identity without claiming utility.
        weights=np.zeros((4,3,3))
        for m in range(4):
            for i in range(3):weights[m,i,(i+m)%3]=1
        common_direct=np.broadcast_to(weights.mean(0,keepdims=True),weights.shape)
        mismatch=float(np.max(np.abs(weights-common_direct)))
    assert mismatch > 0
    checks.append('route cotangents can differ despite equal aggregate target/loss')
    singleton=positive_masks(np.zeros((4,3,2)),np.array([0,1,2]),neighbors=16)
    assert np.array_equal(singleton,np.broadcast_to(np.eye(3,dtype=bool),singleton.shape))
    checks.append('singleton and zero-context behavior explicit without dropping own supervision')
    for filename in ['context_positive_masks.py','context_alignment_objective.py']:
        ast.parse((Path(__file__).parent/filename).read_text())
    checks.append('objective/reference Python AST parsing; Torch forward/backward still unqualified')
    report={'status':'NUMPY_ALGORITHM_CHECKS_ONLY','checks':checks,'scientific_data_access':False,
            'scientific_fits':0,'torch_objective_runtime_qualified':False,
            'fullgraph_preprocessing_qualified':False,'cuda_qualified':False}
    (Path(__file__).parent/'MASK_REFERENCE_QUALIFICATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
