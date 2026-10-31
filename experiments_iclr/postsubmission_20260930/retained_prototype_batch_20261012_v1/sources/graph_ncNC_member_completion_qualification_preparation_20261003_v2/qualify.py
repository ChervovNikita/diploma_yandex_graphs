"""Prepared engineering harness: synthetic correspondence, no predictive pilot.

Root must execute in a qualified runtime. Missing native dependencies fail;
no skip, data loader, accuracy gate, hyperparameter selection or installation.
"""
import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
import torch
from graph_ops import Graph, enumerate_neighbors, feature_sum
from prototype import (Recipe, CompletionTwin, FactorLinear, PrivateNorm,
                       clamp_completion, route_weights, permute_members,
                       copy_native_unit_member, training_batch_loss, native_optimizer)
from native_reference import native_modules, verify_seal
from selected_state import selected_bytes, restore_selected_bytes, code_hashes, rng_state, restore_rng


def close(actual, expected):
    # Engineering arithmetic tolerance only, derived from tested dtype. This
    # is neither a predictive success gate nor a selected experiment threshold.
    eps = torch.finfo(actual.dtype).eps
    torch.testing.assert_close(actual, expected, rtol=128*eps, atol=128*eps)


def fixture(dtype=torch.float64, features=4):
    pairs = torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,5],[5,6],[4,6]],dtype=torch.long)
    queries = torch.tensor([[0,3],[1,4],[2,5]],dtype=torch.long)
    x = (torch.arange(7*features,dtype=dtype).reshape(7,features) % 11) / 7 + .05
    return x, pairs, queries


def patterned(model, diverse=False):
    modules=dict(model.named_modules())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parent=modules.get(name.rsplit('.',1)[0])
            if name.endswith(".r") or name.endswith(".s"):
                parameter.fill_(1)
                if diverse:
                    row = torch.arange(parameter.shape[0],device=parameter.device)[:,None]
                    col = torch.arange(parameter.shape[1],device=parameter.device)[None,:]
                    parameter.copy_(1 + .08 * ((row + col) % 3 - 1))
            elif "norm.weight" in name or isinstance(parent,PrivateNorm) and name.endswith("weight"):
                parameter.fill_(1)
            elif isinstance(parent,PrivateNorm):
                # The identical-member limit requires all private affine
                # rows identical too, not merely unit rank-one factors.
                values=torch.arange(parameter.shape[1],device=parameter.device,dtype=parameter.dtype)
                parameter.copy_((.17*torch.sin(values+.3))[None].expand_as(parameter))
            elif name == "decoder.beta":
                parameter.fill_(1)
            else:
                values = torch.arange(parameter.numel(),device=parameter.device,dtype=parameter.dtype).reshape_as(parameter)
                parameter.copy_(.17 * torch.sin(values + .3))


def test_graph_mask_enumeration():
    _, pairs, queries = fixture()
    graph = Graph.mask_train_batch(pairs, torch.tensor([0,3]), 7)
    remaining = {tuple(sorted(pair)) for i,pair in enumerate(pairs.tolist()) if i not in (0,3)}
    assert set(zip(graph.row.tolist(),graph.col.tolist())) == {(a,b) for edge in remaining for a,b in (edge,edge[::-1])}
    for query in (queries, torch.tensor([[0,0],[6,6],[0,6]],dtype=torch.long)):
        neighbors = enumerate_neighbors(graph, query)
        n = {i:set() for i in range(7)}
        for a,b in zip(graph.row.tolist(),graph.col.tolist()): n[a].add(b)
        expected = [[],[],[]]
        for qi,(a,b) in enumerate(query.tolist()):
            for k,ids in enumerate((n[a]&n[b],n[a]-n[b],n[b]-n[a])):
                expected[k].extend((qi,j) for j in sorted(ids))
        for actual,want in zip((neighbors.common,neighbors.left,neighbors.right),expected):
            assert list(zip(actual[0].tolist(),actual[1].tolist())) == want
    # Record masking retains a duplicate if a different record still provides it.
    duplicated = torch.cat((pairs,pairs[:1]),0)
    duplicate_graph = Graph.mask_train_batch(duplicated,torch.tensor([0]),7)
    assert (0,1) in set(zip(duplicate_graph.row.tolist(),duplicate_graph.col.tolist()))


def test_identical_member_limit():
    x,pairs,queries = fixture()
    model = CompletionTwin(Recipe(features=4,hidden=4)).double().eval()
    patterned(model)
    graph = Graph.from_pairs(pairs,7)
    private = model(x,graph,queries,"private")
    pooled = model(x,graph,queries,"pooled_after_clamp")
    close(private,pooled)
    for m in range(1,4): close(private[:,m],private[:,0])
    # Equal evaluation branches are the equality limit; independent dropout
    # masks intentionally make train-mode branches nonidentical.


def test_affine_association():
    weights = torch.tensor([[.8],[.2],[.8],[.2]],dtype=torch.float64)
    response = torch.tensor([[1.],[0.],[1.],[0.]],dtype=torch.float64)
    private = (route_weights(weights,"private") * response).mean()
    pooled = (route_weights(weights,"pooled_after_clamp") * response).mean()
    close(private,torch.tensor(.4,dtype=torch.float64))
    close(pooled,torch.tensor(.25,dtype=torch.float64))
    close(private-pooled,((weights-weights.mean(0))*response).mean())
    identical_response = response.new_ones(response.shape)
    close((weights*identical_response).mean(),(route_weights(weights,"pooled_after_clamp")*identical_response).mean())


def test_after_clamp_and_member_permutation():
    raw = torch.tensor([[5.],[7.],[5.],[7.]],dtype=torch.float64)
    recipe = Recipe()
    weights = clamp_completion(raw,recipe.scale,recipe.offset,recipe.alpha,recipe.pt)
    routed = route_weights(weights,"pooled_after_clamp")
    close(routed,weights.mean(0,keepdim=True).expand_as(weights))
    wrong = clamp_completion(raw.mean(0,keepdim=True),recipe.scale,recipe.offset,recipe.alpha,recipe.pt).expand_as(weights)
    assert not torch.allclose(routed,wrong), "Fixture must distinguish averaging before clamp"
    x,pairs,queries = fixture()
    graph = Graph.from_pairs(pairs,7)
    model = CompletionTwin(Recipe(features=4,hidden=4)).double().eval()
    patterned(model,diverse=True)
    order = [2,0,3,1]
    for mode in ("private","pooled_after_clamp"):
        permuted = copy.deepcopy(model)
        before = model(x,graph,queries,mode)
        permute_members(permuted,order)
        after = permuted(x,graph,queries,mode)
        close(after,before[:,order])
        close(model.serve(before),permuted.serve(after))
    # Weight-only reassignment is not a symmetry: a prescribed swap changes
    # the affine association without selecting the best permutation.
    w = torch.tensor([[.8],[.2],[.8],[.2]],dtype=torch.float64)
    a = torch.tensor([[1.],[0.],[1.],[0.]],dtype=torch.float64)
    assert (w*a).mean() != (w[[1,0,3,2]]*a).mean()


def test_score_detach_feature_gradient():
    x,pairs,queries = fixture()
    model = CompletionTwin(Recipe(features=4,hidden=4)).double().eval()
    patterned(model,diverse=True)
    h = x.clone().requires_grad_(True)
    logits,details = model.decoder(h,Graph.from_pairs(pairs,7),queries,"private",True)
    assert not details["raw_left"].requires_grad and not details["raw_right"].requires_grad
    assert all(not score.requires_grad for both in details["scores"] for score in both)
    for transformed in details["transformed"]: transformed.retain_grad()
    logits.square().sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in details["transformed"])
    assert model.decoder.xlin.ops["0"].weight.grad is not None
    assert model.decoder.xlin.ops["0"].weight.grad.abs().sum() > 0
    # An isolated aggregate verifies the exact surviving feature pullback,
    # even when no-grad scores were functions of those same features.
    features = torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64,requires_grad=True)
    scores = torch.tensor([5.,7.],dtype=torch.float64,requires_grad=True)
    weight = clamp_completion(scores.detach(),2.5,6.,1.05,.1)
    aggregate = feature_sum(features,(torch.tensor([0,0]),torch.tensor([0,1])),1,weight)
    gf,gs = torch.autograd.grad(aggregate.sum(),(features,scores),allow_unused=True)
    close(gf,weight[:,None].expand_as(features)); assert gs is None


def test_native_unit_identity():
    native,_ = native_modules()  # mandatory; dependency failure is not skipped
    from torch_sparse import SparseTensor
    x,pairs,queries = fixture(torch.float32,128)
    recipe = replace(Recipe(),member_count=1)
    portable = CompletionTwin(recipe)
    encoder = native.GCN(128,64,64,1,.1,True,True,-1,"gcn",False,.25,xdropout=.25,taildropout=.05)
    decoder = native.IncompleteCN1Predictor(64,64,1,3,.3,edrop=0.,ln=True,cndeg=-1,
                use_xlin=True,tailact=True,twolayerlin=False,beta=1.,alpha=1.05,
                scale=2.5,offset=6.,trainresdeg=-1,testresdeg=-1,pt=.1,
                learnablept=False,depth=1,splitsize=-1)
    copy_native_unit_member(portable,encoder,decoder)
    graph = Graph.mask_train_batch(pairs,torch.tensor([0,3]),7)
    adjacency = SparseTensor(row=graph.row,col=graph.col,sparse_sizes=(7,7))
    for training in (False,True):
        portable.train(training); encoder.train(training); decoder.train(training)
        native_x = x.clone().requires_grad_(True)
        portable_x = x.clone().requires_grad_(True)
        before = rng_state()
        native_logits = decoder(encoder(native_x,adjacency),adjacency,queries.T).flatten()
        native_logits.square().sum().backward()
        restore_rng(before)
        own_logits = portable(portable_x,graph,queries,"private")[:,0]
        own_logits.square().sum().backward()
        close(own_logits,native_logits); close(portable_x.grad,native_x.grad)
        close(portable.encoder.weight.grad,encoder.convs[0].lin.weight.grad)
        close(portable.decoder.xlin.ops["0"].weight.grad,decoder.xlin[0].weight.grad)
        close(portable.decoder.lin.ops["8"].weight.grad,decoder.lin[8].weight.grad)
        close(portable.encoder.bias.grad,encoder.convs[0].bias.grad)
        close(portable.encoder.norm.weight.grad,encoder.lins[0][0].weight.grad)
        close(portable.encoder.norm.bias.grad,encoder.lins[0][0].bias.grad)
        close(portable.decoder.beta.grad,decoder.beta.grad)
        for name in ("xlin","xcnlin","xijlin","lin","ptlin"):
            own,original=getattr(portable.decoder,name),getattr(decoder,name)
            for index,operation in own.ops.items():
                source=original[int(index)]
                for field in ("weight","bias"):
                    ours=getattr(operation,field).grad
                    theirs=getattr(source,field).grad
                    if theirs is None:
                        assert ours is None
                    else:
                        close(ours,theirs if isinstance(operation,FactorLinear) else theirs[None])
        portable.zero_grad(set_to_none=True); encoder.zero_grad(set_to_none=True); decoder.zero_grad(set_to_none=True)
    # Native utility agreement for masked topology/candidates is additional
    # to the independent explicit-set oracle above.
    from native_reference import native_modules as load_native
    _,utils = load_native()
    cn,left,right = utils.adjoverlap(adjacency,adjacency,queries.T,calresadj=True,cnsampledeg=-1,ressampledeg=-1)
    own = enumerate_neighbors(graph,queries)
    for native_matrix,ours in zip((cn,left,right),(own.common,own.left,own.right)):
        row,col,_ = native_matrix.coo()
        assert torch.equal(row,ours[0]) and torch.equal(col,ours[1])


def assert_state_equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a: assert_state_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b): assert_state_equal(left,right)
    else:
        assert a==b


def test_selected_state_replay():
    x,pairs,queries = fixture()
    model = CompletionTwin(Recipe(features=4,hidden=4)).double()
    patterned(model,diverse=True)
    optimizer = native_optimizer(model)
    # One artificial engineering update populates Adam moments. No benchmark
    # labels, training selection, scientific predictions or new fitting task.
    graph = Graph.from_pairs(pairs,7)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    model(x,graph,queries,"private").square().sum().backward(); optimizer.step()
    for mode in ("private","pooled_after_clamp"):
        blob = selected_bytes(model,optimizer,mode,{"kind":"synthetic_engineering_state","no_validation_selection":True})
        saved_rng = rng_state()
        expected_model = copy.deepcopy(model)
        expected_optimizer = native_optimizer(expected_model)
        expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restore_rng(saved_rng)
        expected_optimizer.zero_grad(set_to_none=True)
        expected = expected_model(x,graph,queries,mode)
        expected.square().sum().backward(); expected_optimizer.step()
        restored,restored_optimizer,payload = restore_selected_bytes(blob)
        assert payload["completion_mode"]==mode and payload["serving_pool"]=="mean_raw_logits"
        assert_state_equal(restored.state_dict(),model.state_dict())
        assert_state_equal(restored_optimizer.state_dict(),optimizer.state_dict())
        restored_optimizer.zero_grad(set_to_none=True)
        actual = restored(x,graph,queries,mode)
        actual.square().sum().backward(); restored_optimizer.step()
        assert torch.equal(actual,expected)
        assert_state_equal(restored.state_dict(),expected_model.state_dict())
        assert_state_equal(restored_optimizer.state_dict(),expected_optimizer.state_dict())
        close(restored.serve(actual),expected_model.serve(expected))


def test_twin_training_schedule():
    x,pairs,queries = fixture()
    base = CompletionTwin(Recipe(features=4,hidden=4)).double()
    patterned(base,diverse=True)
    graph = Graph.from_pairs(pairs,7)
    state = rng_state()
    private,detail_private = base(x,graph,queries,"private",True)
    after_private = rng_state()
    restore_rng(state)
    pooled,detail_pooled = base(x,graph,queries,"pooled_after_clamp",True)
    after_pooled = rng_state()
    assert torch.equal(after_private["torch_cpu"],after_pooled["torch_cpu"])
    for key in ("raw_left","raw_right"):
        assert torch.equal(detail_private[key],detail_pooled[key])
    close(detail_pooled["routed_left"],detail_pooled["raw_left"].mean(0,keepdim=True).expand_as(detail_pooled["raw_left"]))
    # Native loss scaling and the shared encoder invocation are source-bound
    # and independently countable, without a dataset or performance screen.
    invocations=[]
    hook=base.encoder.register_forward_hook(lambda *args:invocations.append(1))
    loss=training_batch_loss(base,x,pairs,torch.tensor([0,3]),torch.tensor([[0,4],[1,6]]),"private")
    hook.remove()
    assert len(invocations)==1 and loss.ndim==0 and torch.isfinite(loss)


TESTS = [test_graph_mask_enumeration,test_identical_member_limit,test_affine_association,
         test_after_clamp_and_member_permutation,test_score_detach_feature_gradient,
         test_native_unit_identity,test_selected_state_replay,test_twin_training_schedule]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    manifest_sha=verify_seal()
    output=Path(args.output)
    if output.exists(): raise RuntimeError("Refuse to overwrite qualification result")
    results=[]
    for test in TESTS:
        try:
            test(); results.append({"test":test.__name__,"status":"PASS"})
        except Exception as error:
            results.append({"test":test.__name__,"status":"FAIL","error":repr(error)})
    passed=all(item["status"]=="PASS" for item in results)
    payload={"schema":"ncnc-completion-engineering-qualification-v1",
             "status":"ENGINEERING_QUALIFICATION_PASSED" if passed else "FAILED",
             "code_hashes":code_hashes(),"torch_version":torch.__version__,"tests":results,
             "prototype_manifest_sha256":manifest_sha,
             "predictive_experiment":False,"benchmark_dataset_or_labels_accessed":False,
             "scientific_success_or_launch_authorized":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"status":payload["status"],"passed":sum(x["status"]=="PASS" for x in results),"required":len(TESTS)}))
    if not passed: raise SystemExit(1)


if __name__=="__main__":main()
