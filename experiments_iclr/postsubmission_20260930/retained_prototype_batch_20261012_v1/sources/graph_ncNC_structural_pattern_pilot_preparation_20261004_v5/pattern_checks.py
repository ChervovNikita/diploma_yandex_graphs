"""Future numerical assertions; never executed during source preparation."""
import json
from itertools import product
from pathlib import Path
import torch
from torch.nn import functional as F
from graph_ops import Graph
from pilot_common import require
from pilot_model import make_factorized, train_flag
from pilot_state import (snapshot, restore_snapshot, state_digest, rng_state, rng_digest,
                         restore_rng, atomic_torch, file_sha, write_journal, read_journal)
from pattern_model import make_pattern
from pattern_teacher import ObservationTeacher
from pattern_objective import losses
from pattern_train import batch_forward, train_batch


def close(actual, expected, label):
    require(actual.shape == expected.shape, "Shape mismatch " + label)
    eps = torch.finfo(actual.dtype).eps
    tolerance = 128 * eps  # unchanged qualified arithmetic rule
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), "Nonfinite comparison " + label)
    require(bool(((actual - expected).abs() <= tolerance + tolerance * expected.abs()).all()), "Qualified arithmetic mismatch " + label)


def nested_close(a, b, label="state"):
    if torch.is_tensor(a):
        if a.is_floating_point(): close(a, b, label)
        else: require(torch.equal(a, b), "Exact integer state differs " + label)
    elif isinstance(a, dict):
        require(set(a) == set(b), "Keys differ " + label)
        for k in a: nested_close(a[k], b[k], label + "." + str(k))
    elif isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), "Container differs " + label)
        for i, (x, y) in enumerate(zip(a, b)): nested_close(x, y, label + "." + str(i))
    else:
        require(a == b, "Scalar/flag differs " + label)


def mathematical_checks(device):
    # Direct products/enumeration and analytical gradients in float64, not the
    # objective implementation used as its own expected answer.
    t = torch.tensor([[.8,-.3,.2],[-.4,.7,-.1],[.5,.4,.9],[-.6,-.2,-.8]], device=device, dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([0,0,1], device=device)
    z = torch.tensor([1.,0.,1.], device=device, dtype=torch.float64)
    v = losses(t, rows, z, 3)
    p = torch.sigmoid(t)
    bern = p * z + (1-p) * (1-z)
    j = torch.stack((-torch.log((bern[:,0]*bern[:,1]).mean())/2,
                     -torch.log(bern[:,2].mean()), t.new_tensor(0.)))
    f = torch.stack((-(torch.log(bern[:,0].mean())+torch.log(bern[:,1].mean()))/2,
                     -torch.log(bern[:,2].mean()), t.new_tensor(0.)))
    close(v["J"], j, "direct joint products"); close(v["F"], f, "direct factorial products")
    actual_j = torch.autograd.grad(v["J"].sum(), t, retain_graph=True)[0]
    expected_j = v["rho"][:, rows].detach() * (p.detach()-z) / v["counts"][rows]
    close(actual_j, expected_j, "joint responsibility gradient")
    actual_f = torch.autograd.grad(v["F"].sum(), t)[0]
    meanp = p.detach().mean(0)
    expected_f = (meanp-z)/(meanp*(1-meanp)) * p.detach()*(1-p.detach()) / (4*v["counts"][rows])
    close(actual_f, expected_f, "factorial marginal gradient")
    permutation = torch.tensor([2,0,3,1], device=device)
    vp = losses(t.detach()[permutation], rows, z, 3)
    close(vp["J"], j, "component permutation J"); close(vp["F"], f, "component permutation F")
    collapsed = t.detach()[0:1].expand(4,-1).clone().requires_grad_()
    cv = losses(collapsed, rows, z, 3)
    close(cv["J"], cv["F"], "collapsed equality")
    cg_j = torch.autograd.grad(cv["J"].sum(), collapsed, retain_graph=True)[0]
    cg_f = torch.autograd.grad(cv["F"].sum(), collapsed)[0]
    close(cg_j, cg_f, "collapsed gradient equality")
    extreme = torch.tensor([[-1000.,1000.]]*4, device=device, dtype=torch.float64, requires_grad=True)
    ev = losses(extreme, torch.tensor([0,0], device=device), torch.tensor([1.,0.], device=device, dtype=torch.float64), 1)
    require(all(bool(torch.isfinite(ev[k]).all()) for k in ("J","F")), "Extreme finite-logit loss")
    require(bool(torch.isfinite(torch.autograd.grad(ev["J"].sum()+ev["F"].sum(), extreme)[0]).all()), "Extreme finite-logit gradient")
    # Conditional joint distributions sum to one over all eight patterns.
    q = p.detach()
    total = sum(torch.prod(torch.where(torch.tensor(bits, device=device, dtype=torch.bool)[None,:], q, 1-q), dim=1).mean() for bits in product((0,1), repeat=3))
    close(total, total.new_tensor(1.), "joint law normalization")
    # Equal first/second moments need not identify higher-order source patterns.
    # This three-bit witness is distinguishable by cardinality and does not
    # establish non-identification for the required cardinality-aware single.
    even = torch.tensor([[0,0,0],[0,1,1],[1,0,1],[1,1,0]], device=device, dtype=torch.float64)
    odd = 1-even
    close(even.mean(0), odd.mean(0), "parity marginal collision")
    close(even.T@even/4, odd.T@odd/4, "parity second-moment collision")
    require(float(even.prod(1).mean()) == 0. and float(odd.prod(1).mean()) == .25, "Parity witness")
    return ["direct_J_F_products", "analytical_J_F_gradients", "permutation_and_empty_support", "collapsed_loss_gradient_equality", "extreme_logits_no_epsilon", "joint_normalization_and_parity_moments"]


def fixture(device):
    generator = torch.Generator(device="cpu").manual_seed(91234)
    pairs = torch.tensor([(0,1),(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,0)], device=device)
    raw = pairs.T.repeat_interleave(2, dim=1); raw[:,1::2] = pairs.T.flip(0)
    return {"x": torch.randn(16,128,generator=generator).to(device), "pairs": pairs, "raw_edge_index": raw,
            "valid_positive": pairs, "valid_negative": torch.tensor([(i//15,(i%15)+1) for i in range(50)], device=device)}


def teacher_checks(data, teacher):
    pairs, device = data["pairs"], data["pairs"].device
    ids = torch.tensor([0,3,4], device=device)
    graph = Graph.mask_train_batch(pairs, ids, teacher.nodes)
    require(bool((graph.row*teacher.nodes+graph.col == 1).any()), "Surviving duplicate lost")
    both = Graph.mask_train_batch(pairs, torch.tensor([0,1],device=device), teacher.nodes)
    require(not bool((both.row*teacher.nodes+both.col == 1).any()), "Removing all duplicates did not remove edge")
    query = torch.tensor([[0,1],[7,8]],device=device)
    from graph_ops import enumerate_neighbors
    neighbors = enumerate_neighbors(graph, query)
    rows, labels = teacher.labels(query, neighbors)
    # Independently form Python membership from supplied observation records.
    original = {tuple(x) for x in pairs.detach().cpu().tolist()}
    original |= {(v,u) for u,v in list(original)}
    expected = []
    for row, node in zip(neighbors.left[0].tolist(), neighbors.left[1].tolist()):
        expected.append(float((int(query[row,1]),node) in original))
    for row, node in zip(neighbors.right[0].tolist(), neighbors.right[1].tolist()):
        expected.append(float((int(query[row,0]),node) in original))
    close(labels, labels.new_tensor(expected), "independent teacher membership")
    require(bool((labels == 1).any()) and bool((labels == 0).any()), "Teacher fixture lacks removals/observation zeros")
    require(not bool((rows == 1).any()), "Isolated query must have empty support")
    return ["source_observation_teacher_membership", "record_duplicate_retention_and_all_record_removal", "synthetic_positive_source_zero_and_empty_support"]


def target_correspondence(mods, data, teacher, device, *, full_graph=False):
    base, base_optimizer = make_factorized(mods, 0, device)
    initial = snapshot(base, base_optimizer)
    model, optimizer = make_pattern(mods, 0, device)
    require(state_digest(snapshot(model,optimizer)) == state_digest(initial), "Pattern constructor/sign/RNG/state differs from qualified F4")
    ids = torch.tensor([0,2,3,5], device=device)
    negatives = data["pairs"].clone()
    negatives[ids] = torch.tensor([[7,9],[8,10],[9,11],[10,12]],device=device)
    restore_snapshot(base,base_optimizer,initial); train_flag(base,True)
    # Reference loss/gradients without Adam, copied byte-identical native batch
    # forward operation rather than an auxiliary implementation oracle.
    graph = Graph.mask_train_batch(data["pairs"],ids,len(data["x"]))
    h = base.encoder(data["x"],graph)
    pos = base.decoder(h,graph,data["pairs"][ids],"private")
    neg = base.decoder(h,graph,negatives[ids],"private")
    loss = -F.logsigmoid(pos).mean()-F.logsigmoid(-neg).mean(); loss.backward()
    reference_rng = rng_digest(rng_state())
    restore_snapshot(model,optimizer,initial); train_flag(model,True)
    main, records = batch_forward(model,data,teacher,negatives,ids)
    close(records[0]["logits"],pos.detach(),"native positive target forward")
    close(records[1]["logits"],neg.detach(),"native negative target forward")
    close(main,loss.detach(),"native main loss")
    tensors = [x for row in records for x in row["detail"]["raw_score_tensors"]]
    isolated = torch.autograd.grad(main,tensors,allow_unused=True,retain_graph=True)
    require(all(g is None for g in isolated), "Target completion scorer is not detached")
    for row in records:
        close(torch.sigmoid(row["detail"]["t"]),row["detail"]["native_q"],"normalized native clamp")
    main.backward(retain_graph=True)
    require(rng_digest(rng_state()) == reference_rng, "Gradient capture changes native dropout schedule")
    for (n,p),(m,q) in zip(base.named_parameters(),model.named_parameters()):
        require(n == m and (p.grad is None) == (q.grad is None), "Main gradient routing differs " + n)
        if p.grad is not None:close(q.grad,p.grad,"native main gradient "+n)
    auxiliary = sum(row["values"]["J"].mean() for row in records)
    aux_grads = torch.autograd.grad(auxiliary,tensors,allow_unused=True)
    require(all(g is not None for g in aux_grads) and all(bool(torch.isfinite(g).all()) for g in aux_grads), "Auxiliary recursive score gradients missing/nonfinite")
    require(any(bool((g != 0).any()) for g in aux_grads), "Auxiliary has no score gradient")
    train_flag(base,False);train_flag(model,False)
    with torch.no_grad():
        bh=base.encoder(data["x"],graph);mh=model.encoder(data["x"],graph)
        routes=model.decoder.diagnostic_routes(mh,graph,data["pairs"][ids])
        close(routes["own"],base.decoder(bh,graph,data["pairs"][ids],"private"),"fixed-bank diagnostic own route")
        close(routes["pooled_clamped_weights"],base.decoder(bh,graph,data["pairs"][ids],"pooled_after_clamp"),"fixed-bank diagnostic pooled route")
        require(set(routes)=={"own","crossed_cyclic_1","crossed_cyclic_2","crossed_cyclic_3","pooled_clamped_weights"},"Fixed routes incomplete")
    # Teacher depends only on TRAIN membership, not feature/VALID/negative y.
    require(teacher.nodes == len(data["x"]),"Teacher node identity")
    return {"checks":["native_constructor_and_initial_RNG", "target_forward_main_gradient_correspondence", "target_scorer_detached_auxiliary_scorer_differentiable", "normalized_clamp_and_drop_schedule", "fixed_bank_own_and_native_pooled_route_correspondence"],
            "full_node_graph":full_graph, "query_records":len(ids), "residual_slots":sum(len(r["rows"]) for r in records)}


def serialized_replay(mods,data,teacher,device,output,*,sign_seed):
    # Two actual updates per arm, split by own trusted serialization, on the
    # complete provided graph when called by full_graph qualification.
    receipts=[]
    for arm in ("J","F"):
        model,opt=make_pattern(mods,20261003,device,engineering_sign_seed=sign_seed)
        train_flag(model,True)
        negatives=data["pairs"].clone()
        ids=torch.tensor([0,2,3,5],device=device)
        negatives[ids]=torch.tensor([[7,9],[8,10],[9,11],[10,12]],device=device)
        train_batch(model,opt,data,teacher,negatives,ids,arm)
        saved=snapshot(model,opt)
        path=Path(output)/("REPLAY_"+arm+".pt")
        pin=atomic_torch(path,{"identity":"engineering_only_no_scientific_donor","state":saved})
        require(file_sha(path)==pin["sha256"],"Own replay serialization changed")
        loaded=torch.load(path,map_location="cpu",weights_only=False)
        require(loaded["identity"]=="engineering_only_no_scientific_donor" and state_digest(loaded["state"])==state_digest(saved),"Serialized state roundtrip differs")
        expected=train_batch(model,opt,data,teacher,negatives,ids,arm)
        after=snapshot(model,opt)
        restored,ropt=make_pattern(mods,20261003,device,engineering_sign_seed=sign_seed)
        restore_snapshot(restored,ropt,loaded["state"])
        actual=train_batch(restored,ropt,data,teacher,negatives,ids,arm)
        replay=snapshot(restored,ropt)
        nested_close(after["models"],replay["models"],"replay model "+arm)
        nested_close(after["optimizer"],replay["optimizer"],"replay Adam "+arm)
        require(state_digest(after["rng"])==state_digest(replay["rng"]) and after["flags"]==replay["flags"],"Replay RNG/flags differ")
        require(expected["support_digest"]==actual["support_digest"],"Replay support/observation labels differ")
        before=rng_digest(rng_state()); train_flag(restored,False)
        graph=Graph.from_pairs(data["pairs"],len(data["x"]))
        with torch.no_grad():
            h=restored.encoder(data["x"],graph)
            score=restored.decoder(h,graph,data["pairs"][ids],"private").mean(1)
        restore_snapshot(model,opt,after); train_flag(model,False)
        with torch.no_grad():
            ref=model.decoder(model.encoder(data["x"],graph),graph,data["pairs"][ids],"private").mean(1)
        close(score,ref,"replay mean-logit serving "+arm)
        require(rng_digest(rng_state())==before,"Replay eval consumed RNG")
        # Engineering saved states cannot be fitted/resumed and are removed
        # after their hashes/value roundtrip are recorded.
        path.unlink()
        receipts.append({"arm":arm,"serialization":pin,"roundtrip_typed_state_equal":True,"next_update_model_Adam_arithmetic_equal":True,"RNG_flags_exact":True,"served_replay_equal":True,"state_file_removed":True})
    return receipts
