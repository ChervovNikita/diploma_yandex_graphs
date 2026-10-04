"""Fabricated native/parity and complete-batch engineering checks.

No data reader or fit loop. Caller supplies admitted runtime and, for batch,
one actual externally authenticated prospective replay batch.
"""
from contextlib import contextmanager
from itertools import product
from pathlib import Path
import gc
import inspect
import math
import resource
import time
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from qualification_common import ARMS, require, atomic_json, file_sha, utc
from graph_ops import Graph, enumerate_neighbors, feature_sum
from count_model import adjacency
from pattern_teacher import ObservationTeacher
import pilot_state as state


class Audit:
    def __init__(self):
        self.comparisons = {}

    def close(self, a, b, label, rule="native"):
        require(a.shape == b.shape and a.dtype == b.dtype, "Shape/dtype differs: " + label)
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "Nonfinite: " + label)
        tolerances = {"native": (1e-6, 1e-5), "vector": (2e-6, 1e-5),
                      "gradient": (2e-6, 2e-5), "density": (2e-10, 2e-10),
                      "adam": (128 * torch.finfo(torch.float32).eps,) * 2}
        atol, rtol = tolerances[rule]
        error = float((a - b).abs().max()) if a.numel() else 0.
        row = self.comparisons.setdefault(rule, {"tensor_comparisons": 0, "max_absolute_error": 0.,
                                                "atol": atol, "rtol": rtol})
        row["tensor_comparisons"] += 1
        row["max_absolute_error"] = max(row["max_absolute_error"], error)
        require(bool(((a - b).abs() <= atol + rtol * b.abs()).all()),
                "Parity failure: " + label + "; max_abs=" + str(error))

    def nested(self, a, b, label, rule="adam"):
        if torch.is_tensor(a):
            require(torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape, label)
            if a.is_floating_point():
                self.close(a, b, label, rule)
            else:
                require(torch.equal(a, b), "Exact integer/tensor differs: " + label)
        elif isinstance(a, dict):
            require(type(a) is type(b) and set(a) == set(b), "Keys differ: " + label)
            for k in a:
                self.nested(a[k], b[k], label + "." + str(k), rule)
        elif isinstance(a, (tuple, list)):
            require(type(a) is type(b) and len(a) == len(b), "Container differs: " + label)
            for i, (x, y) in enumerate(zip(a, b)):
                self.nested(x, y, label + "." + str(i), rule)
        else:
            require(type(a) is type(b) and a == b, "Scalar/flag differs: " + label)


class Resources:
    def __init__(self, context):
        self.context = context
        self.times = {}
        self.arm = None

    def sample(self, *, synchronize=True, enforce=True):
        if synchronize:
            torch.cuda.synchronize(0)
        values = {"host_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "CUDA_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "CUDA_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                  "CUDA_current_allocated_bytes": torch.cuda.memory_allocated(0),
                  "CUDA_current_reserved_bytes": torch.cuda.memory_reserved(0)}
        limits = self.context["plan"]["resource_limits"]
        if enforce:
            require(values["host_peak_RSS_bytes"] <= limits["host_RSS_bytes"]
                    and values["CUDA_peak_allocated_bytes"] <= limits["CUDA_allocated_bytes"]
                    and values["CUDA_peak_reserved_bytes"] <= limits["CUDA_reserved_bytes"], "Resource bound exceeded")
        return values

    def call(self, label, fn):
        atomic_json(self.context["output"] / "PROGRESS.json", {"UTC": utc(), "arm": self.arm,
                    "step": label, "state_donor": False})
        torch.cuda.synchronize(0)
        start = time.monotonic()
        value = fn()
        metrics = self.sample()
        elapsed = time.monotonic() - start
        key = (self.arm or "native") + ":" + label
        self.times[key] = {"wall_seconds": elapsed, **metrics}
        atomic_json(self.context["output"] / "MEASUREMENTS.json", self.times)
        return value


def fixture(device):
    generator = torch.Generator(device="cpu").manual_seed(91234)
    pairs = torch.tensor([(0,1),(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,0)],
                         dtype=torch.long, device=device)
    x = torch.randn((16,128), generator=generator, dtype=torch.float32).to(device)
    ids = torch.tensor([0,3,4], dtype=torch.long, device=device)
    graph = Graph.mask_train_batch(pairs, ids, 16)
    positive = torch.tensor([[0,1],[1,2],[1,3],[7,8]], dtype=torch.long, device=device)
    negative = torch.tensor([[0,4],[1,5],[7,9],[8,10]], dtype=torch.long, device=device)
    return x, pairs, graph, positive, negative


def factory(mods, device, arm, scalar=False):
    if arm == "C_mu":
        return (mods["count_model"].make_count if scalar else mods["vector_count_model"].make_vector_count)(mods, device)
    return mods["pooled_model"].make_pooled(device)


def neighbors_equal(a, b):
    require(a.queries == b.queries, "Support query counts differ")
    for name in ("common", "left", "right"):
        require(all(torch.equal(x, y) for x, y in zip(getattr(a, name), getattr(b, name))),
                "Native ordered support differs: " + name)


def native_support(mods, graph, queries, expected):
    """Every actual native common/left/right coordinate; no sampled subset."""
    adj = adjacency(graph)
    rowptr, col, value = adj.csr()
    require(torch.equal(rowptr, graph.rowptr) and torch.equal(col, graph.col), "Native full CSR differs")
    parts = mods["native_utils"].adjoverlap(adj, adj, queries.T, calresadj=True,
                                          cnsampledeg=-1, ressampledeg=-1)
    for part, name in zip(parts, ("common", "left", "right")):
        rows, nodes, values = part.coo()
        wanted = getattr(expected, name)
        require(torch.equal(rows, wanted[0]) and torch.equal(nodes, wanted[1]), "Actual native coordinates differ: " + name)
        require(values is None or bool((values == 1).all()), "Native unweighted support values differ")
    return {"queries": len(queries), "common": len(expected.common[0]),
            "left": len(expected.left[0]), "right": len(expected.right[0]),
            "every_native_coordinate_exact": True}


def teacher_boundary(pairs, graph, queries):
    teacher = ObservationTeacher.from_train(pairs, graph.nodes)
    all_removed = Graph.mask_train_batch(pairs, torch.tensor([0,1], device=pairs.device), graph.nodes)
    require(bool(((graph.row == 0) & (graph.col == 1)).any())
            and not bool(((all_removed.row == 0) & (all_removed.col == 1)).any()), "Duplicate masking failed")
    neighbors = enumerate_neighbors(graph, queries)
    rows, bits = teacher.labels(queries, neighbors)
    observed = {tuple(v) for v in pairs.cpu().tolist()}
    observed |= {(b, a) for a, b in tuple(observed)}
    expected = []
    for side, opposite in ((neighbors.left, 1), (neighbors.right, 0)):
        expected.extend(float((int(queries[r, opposite]), int(n)) in observed)
                        for r, n in zip(side[0].tolist(), side[1].tolist()))
    require(torch.equal(bits, bits.new_tensor(expected)) and bool((bits == 0).any())
            and bool((bits == 1).any()) and not bool((rows == len(queries)-1).any()),
            "Teacher source-zero/removed-observed/empty fixture incomplete")
    return teacher


def explicit_pool(model, h, graph, query, auxiliary_grad):
    """Independent phase assembly; no CompletionDecoder.forward/route helper."""
    decoder = model.decoder
    neighbors = enumerate_neighbors(graph, query)
    lq, ln = neighbors.left; rq, rn = neighbors.right
    transformed, scores, weights = [], [], []
    for member in range(4):
        outer = h + decoder.xlin.forward_member(h, member)
        transformed.append(outer)
        part = []
        for candidate in (torch.stack((query[lq,1], ln), 1), torch.stack((query[rq,0], rn), 1)):
            if auxiliary_grad:
                with set_checkpoint_early_stop(False):
                    score = checkpoint(decoder.depth_zero, outer, graph, candidate, member,
                                       use_reentrant=False, preserve_rng_state=True, determinism_check="default")
            else:
                with torch.no_grad():
                    score = decoder.depth_zero(outer, graph, candidate, member)
            part.append(score)
        scores.append(part)
        current = []
        for score in part:
            p = torch.sigmoid(decoder.scale * (score.detach() - decoder.offset))
            current.append(decoder.alpha * decoder.pt * p / (decoder.pt * p + 1 - p))
        weights.append(current)
    left, right = (torch.stack([w[s] for w in weights]) for s in (0,1))
    lw, rw = left.mean(0), right.mean(0)
    logits = []
    for member, outer in enumerate(transformed):
        common = feature_sum(outer, neighbors.common, len(query))
        common = common + feature_sum(outer, neighbors.left, len(query), lw)
        common = common + feature_sum(outer, neighbors.right, len(query), rw)
        endpoint = decoder.xijlin.forward_member(h[query[:,0]] * h[query[:,1]], member)
        cn = decoder.xcnlin.forward_member(common, member)
        logits.append(decoder.lin.forward_member(cn * decoder.beta[member] + endpoint, member).flatten())
    raw = torch.cat((torch.stack([s[0] for s in scores]), torch.stack([s[1] for s in scores])), 1)
    return torch.stack(logits, 1), {"neighbors": neighbors, "raw_left": left, "raw_right": right,
                  "routed_left": lw.expand_as(left), "routed_right": rw.expand_as(right),
                  "t": decoder.scale * (raw - decoder.offset) + torch.log(decoder.pt)}


def explicit_pool_loss(t, rows, bits, queries, arm):
    logbit = bits[None] * F.logsigmoid(t) + (1 - bits[None]) * F.logsigmoid(-t)
    result = []
    for i in range(queries):
        current = logbit[:, rows == i]
        if not current.shape[1]:
            result.append(t.sum() * 0.)
        elif arm == "J_P":
            result.append(-(torch.logsumexp(current.sum(1), 0) - math.log(4)) / current.shape[1])
        else:
            result.append(-(torch.logsumexp(current, 0) - math.log(4)).mean())
    return torch.stack(result)


def enumerate_law(model, detail, labels):
    lo = torch.as_tensor(detail["left_offsets"]).cpu().tolist()
    ro = torch.as_tensor(detail["right_offsets"]).cpu().tolist()
    n = detail["left_slots"]
    mu = torch.empty_like(detail["eta"])
    nll, patterns = [], 0
    for i in range(detail["queries"]):
        li = slice(lo[i], lo[i+1]); ri = slice(n + ro[i], n + ro[i+1])
        eta = torch.cat((detail["eta"][li], detail["eta"][ri]))
        a, b = lo[i+1] - lo[i], ro[i+1] - ro[i]
        require(a + b <= 8, "Fabricated enumeration fixture unexpectedly oversized")
        g = model.count_head.potential(detail["context"][i], detail["swapped_context"][i], a, b)
        z = eta.new_tensor(list(product((0.,1.), repeat=a+b))).reshape(2**(a+b), a+b)
        kl, kr = z[:,:a].sum(1).long(), z[:,a:].sum(1).long()
        energy = z @ eta + g[kl, kr]
        logz = torch.logsumexp(energy, 0)
        probability = torch.softmax(energy.detach(), 0)
        m = probability @ z
        mu[li], mu[ri] = m[:a], m[a:]
        bits = torch.cat((labels[li], labels[ri])).to(eta.dtype).detach()
        value = (logz - (eta * bits).sum() - g[int(bits[:a].sum()), int(bits[a:].sum())]) / max(a+b,1)
        nll.append(value if a+b else (eta.sum() + g.sum()) * 0.)
        patterns += len(z)
    return mu.detach(), torch.stack(nll), patterns


def objective(model, records, teacher, arm, mods, *, independent=False, enumeration=False):
    main = -F.logsigmoid(records[0]["logits"]).mean() - F.logsigmoid(-records[1]["logits"]).mean()
    auxiliary = []
    for row in records:
        detail = row["detail"]
        rows, bits = teacher.labels(row["query"], detail["neighbors"])
        row["rows"], row["bits"] = rows, bits
        if arm == "P0":
            values = main.new_zeros(len(row["query"]))
        elif arm == "C_mu":
            values = enumerate_law(model, detail, bits)[1] if enumeration else model.source_nll(detail, bits)
        elif independent:
            values = explicit_pool_loss(detail["t"], rows, bits.to(detail["t"].dtype), len(row["query"]), arm)
        else:
            values = mods["pattern_objective"].losses(detail["t"], rows, bits.to(detail["t"].dtype), len(row["query"]))[arm[0]]
        row["auxiliary_values"] = values
        auxiliary.append(values.mean())
    auxiliary = sum(auxiliary)
    total = main + auxiliary
    require(bool(torch.isfinite(total)), "Nonfinite fixed objective")
    return main, auxiliary, total


def forward(model, x, graph, positive, negative, arm, *, independent=False):
    h = model.encode(x, graph)
    records = []
    for query in (positive, negative):
        logits, detail = (explicit_pool(model, h, graph, query, arm != "P0") if independent
                          else model.query_forward(h, graph, query, auxiliary_grad=arm != "P0"))
        records.append({"query": query, "logits": logits, "detail": detail})
    return records


def detach_probe(main, model, records, arm):
    if arm == "P0":
        require(all(not x.requires_grad for row in records for x in row["detail"]["raw_score_tensors"]),
                "P0 scorer backward enabled")
        return
    inputs = ([x for row in records for x in row["detail"]["raw_score_tensors"]] if arm != "C_mu" else
              [row["detail"][k] for row in records for k in ("eta", "context", "swapped_context")] +
              list(model.count_head.parameters()))
    require(all(x.requires_grad for x in inputs), "Auxiliary differentiability missing")
    require(all(g is None for g in torch.autograd.grad(main, inputs, allow_unused=True, retain_graph=True)),
            "Main crossed detached completion boundary")


def gradient_record(model, *, head_expected):
    names, absent, nonzero, active = [], [], [], 0
    for name, p in model.named_parameters():
        if ".ptlin." in name:
            require(p.grad is None, "Unused native ptlin acquired gradient")
            continue
        if name.startswith("count_head.") and not head_expected:
            require(p.grad is None, "Main trained density head")
            continue
        active += p.numel()
        if p.grad is None:
            absent.append(name)
        else:
            require(bool(torch.isfinite(p.grad).all()), "Nonfinite gradient: " + name)
            names.append(name)
            if bool((p.grad != 0).any()):
                nonzero.append(name)
    require(not absent and names and nonzero, "Missing active gradient paths: " + repr(absent))
    return {"parameters_with_gradient": len(names), "parameters_with_nonzero_gradient": len(nonzero),
            "active_parameter_entries": active, "unused_ptlin_gradient_absent": True}


def compare_gradients(a, b, audit):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    require(set(pa) == set(pb), "Parameter schema differs")
    for name in pa:
        x, y = pa[name].grad, pb[name].grad
        require((x is None) == (y is None), "Gradient route differs: " + name)
        if x is not None:
            audit.close(x, y, "gradient " + name, "gradient")


@torch.no_grad()
def serving(model, x, graph, positive, negative):
    model.eval()
    before = state.rng_digest(state.rng_state())
    h = model.encode(x, graph)
    scores = []
    for queries in (positive, negative):
        logits, _ = model.query_forward(h, graph, queries, auxiliary_grad=False)
        s = model.serve(logits)
        require(s.dtype == torch.float32 and s.shape == (len(queries),) and bool(torch.isfinite(s).all()),
                "Incomplete serving output")
        scores.append(s.detach().cpu())
    require(state.rng_digest(state.rng_state()) == before, "Serving consumed training RNG")
    return scores


def update(model, optimizer, x, graph, positive, negative, teacher, arm, mods):
    model.train(); optimizer.zero_grad(set_to_none=True)
    records = forward(model, x, graph, positive, negative, arm)
    main, auxiliary, total = objective(model, records, teacher, arm, mods)
    detach_probe(main, model, records, arm)
    total.backward()
    gradients = gradient_record(model, head_expected=arm == "C_mu")
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated state")
    return gradients


def own_roundtrip(model, optimizer, mods, device, arm, output, x, graph, positive, negative,
                  teacher, audit, *, next_update):
    saved = state.snapshot(model, optimizer)
    path = Path(output) / ("ENGINEERING_" + arm + ".pt")
    payload = {"schema": "pooled-engineering-own-state-v1", "state_donor": False,
               "arm": arm, "model_seed": 610041, "state": saved}
    pin = state.atomic_torch(path, payload)
    require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Own bytes changed")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    require(loaded["schema"] == payload["schema"] and loaded["state_donor"] is False and loaded["arm"] == arm
            and loaded["model_seed"] == 610041 and state.state_digest(loaded["state"]) == state.state_digest(saved),
            "Own authenticated typed roundtrip differs")
    if next_update:
        update(model, optimizer, x, graph, positive, negative, teacher, arm, mods)
        expected = state.snapshot(model, optimizer)
        restored, ropt = factory(mods, device, arm)
        state.restore_snapshot(restored, ropt, loaded["state"])
        update(restored, ropt, x, graph, positive, negative, teacher, arm, mods)
        actual = state.snapshot(restored, ropt)
        audit.nested(actual["models"], expected["models"], "next-step model")
        audit.nested(actual["optimizer"], expected["optimizer"], "next-step Adam")
        require(state.state_digest(actual["rng"]) == state.state_digest(expected["rng"])
                and actual["flags"] == expected["flags"], "Next-step RNG/flags differ")
        state.restore_snapshot(model, optimizer, expected)
    else:
        expected = saved
        restored, ropt = factory(mods, device, arm)
        state.restore_snapshot(restored, ropt, loaded["state"])
    state.restore_snapshot(model, optimizer, expected)
    reference = serving(model, x, graph, positive, negative)
    state.restore_snapshot(restored, ropt, actual if next_update else loaded["state"])
    replay = serving(restored, x, graph, positive, negative)
    require(all(torch.equal(a, b) for a, b in zip(reference, replay)), "Own exact served-score replay differs")
    path.unlink()
    return {"serialization": pin, "typed_roundtrip_exact": True, "RNG_flags_exact": True,
            "next_update_model_Adam_parity": next_update, "served_scores_exact": True,
            "state_file_removed": True, "state_donor": False,
            "serving_queries_each_replay": [len(positive), len(negative)]}


def direct_native(mods, device, x, graph, queries, audit):
    """Unit-factor native ancestry plus C_mu's exact live native components."""
    (encoder, decoder), _ = mods["count_model"].make_native(mods, 610041, 64, device)
    one = mods["prototype"].CompletionTwin(mods["prototype"].Recipe(member_count=1)).to(device)
    mods["prototype"].copy_native_unit_member(one, encoder, decoder)
    adj = adjacency(graph)
    for train in (False, True):
        one.train(train); encoder.train(train); decoder.train(train)
        rng = state.rng_state()
        native_h = encoder(x, adj)
        h_rng = state.rng_digest(state.rng_state())
        state.restore_rng(rng)
        portable_h = one.encoder(x, graph)
        audit.close(portable_h, native_h, "unit native encoder")
        require(state.rng_digest(state.rng_state()) == h_rng, "Native encoder dropout RNG differs")
        for query in (queries, queries[-1:]):
            rng = state.rng_state()
            native_score = decoder(native_h, adj, query.T, depth=0).flatten()
            end_rng = state.rng_digest(state.rng_state())
            state.restore_rng(rng)
            portable_score = one.decoder.depth_zero(native_h, graph, query, 0)
            audit.close(portable_score, native_score, "unit full native depth0")
            require(state.rng_digest(state.rng_state()) == end_rng, "Native depth0 dropout RNG differs")
            rng = state.rng_state()
            native_target = decoder(native_h, adj, query.T).flatten()
            end_rng = state.rng_digest(state.rng_state())
            state.restore_rng(rng)
            portable_target = one.decoder(native_h, graph, query, "private")[:,0]
            audit.close(portable_target, native_target, "unit nonlinear native depth1 target")
            require(state.rng_digest(state.rng_state()) == end_rng, "Native depth1 dropout/empty schedule differs")
    # C keeps actual native modules. Independently execute its source schedule.
    model, _ = factory(mods, device, "C_mu")
    for train in (False, True):
        model.train(train)
        h = model.encode(x, graph)
        for query in (queries, queries[-1:]):
            calls = []
            handles = [getattr(model.decoder, name).register_forward_pre_hook(
                lambda module, args, name=name: calls.append(name)) for name in ("xlin","xijlin","xcnlin","lin")]
            rng = state.rng_state()
            actual, detail = model.query_forward(h, graph, query, auxiliary_grad=True)
            actual_rng = state.rng_digest(state.rng_state())
            for handle in handles:
                handle.remove()
            require(calls == ["xlin","xlin","xijlin","xcnlin","lin","xlin","xijlin","xcnlin","lin",
                              "xijlin","xcnlin","lin"], "Native C dropout/full-node/empty-call order changed")
            state.restore_rng(rng)
            outer = h + model.decoder.xlin(h)
            lq, ln = detail["neighbors"].left; rq, rn = detail["neighbors"].right
            left = model.decoder(outer, adj, torch.stack((query[lq,1],ln),1).T, depth=0).flatten()
            right = model.decoder(outer, adj, torch.stack((query[rq,0],rn),1).T, depth=0).flatten()
            eta = 2.5 * (torch.cat((left,right)).to(torch.float64) - 6.) + math.log(.1)
            audit.close(detail["eta"], eta, "C direct native scorers", "density")
            labels = torch.zeros_like(eta)
            mu, _, _ = enumerate_law(model, detail, labels)
            audit.close(detail["mu"], mu, "C fixed-context full law unary", "vector")
            common = feature_sum(outer, detail["neighbors"].common, len(query))
            weight = (1.05 * mu).to(outer.dtype).detach()
            common = common + feature_sum(outer, detail["neighbors"].left, len(query), weight[:len(lq)])
            common = common + feature_sum(outer, detail["neighbors"].right, len(query), weight[len(lq):])
            endpoint = model.decoder.xijlin(h[query[:,0]] * h[query[:,1]])
            cn = model.decoder.xcnlin(common)
            expected = model.decoder.lin(cn * model.decoder.beta + endpoint).flatten()[:,None]
            audit.close(actual, expected, "C single nonlinear D(mu) target", "vector")
            require(state.rng_digest(state.rng_state()) == actual_rng, "C native reference RNG differs")
    return {"unit_native_encoder_scorer_target": True, "training_and_eval": True,
            "C_exact_native_scorer_target_and_call_order": True, "empty_calls_paid": True}


def native(context, rt, mods, resources):
    audit = Audit()
    x, pairs, graph, positive, negative = fixture(rt["device"])
    teacher = teacher_boundary(pairs, graph, positive)
    resources.call("every_fabricated_native_coordinate", lambda: native_support(mods, graph,
                     torch.cat((positive, negative)), enumerate_neighbors(graph, torch.cat((positive, negative)))))
    direct = resources.call("direct_native_prediction", lambda: direct_native(mods, rt["device"], x, graph, positive, audit))
    receipts, initial_pool = [], None
    for arm in ARMS:
        resources.arm = arm
        model, optimizer = factory(mods, rt["device"], arm)
        require(set(inspect.signature(model.query_forward).parameters) == {"h","graph","queries","auxiliary_grad"}
                and set(inspect.signature(model.encode).parameters) == {"x","graph"}, "Predictor signature has side information")
        initial = state.snapshot(model, optimizer)
        initial_sha = state.state_digest(initial)
        total_parameters = sum(p.numel() for p in model.parameters())
        require(total_parameters == (88900 if arm == "C_mu" else 43790), "Architecture count differs")
        if arm != "C_mu":
            if initial_pool is None:
                initial_pool = initial_sha
            require(initial_sha == initial_pool, "P0/J/F initialization/Adam/RNG differ")
        else:
            require(sum(p.numel() for p in model.count_head.parameters()) == 50753, "C head count differs")
        reference, ropt = factory(mods, rt["device"], arm, scalar=arm == "C_mu")
        require(state.state_digest(state.snapshot(reference, ropt)) == initial_sha, "Reference fresh state differs")
        state.restore_snapshot(model, optimizer, initial)
        records = resources.call("candidate_forward", lambda: forward(model, x, graph, positive, negative, arm))
        main, auxiliary, total = objective(model, records, teacher, arm, mods)
        detach_probe(main, model, records, arm)
        probe_rng = state.rng_digest(state.rng_state())
        # Main-only must retain every allowed outer native path, never density head.
        main.backward(retain_graph=True)
        gradient_record(model, head_expected=False)
        optimizer.zero_grad(set_to_none=True)
        if arm != "P0":
            nodes = ([t for row in records for t in row["detail"]["raw_score_tensors"]] if arm != "C_mu" else
                     [row["detail"][k] for row in records for k in ("eta","context","swapped_context")])
            node_gradients = torch.autograd.grad(auxiliary, nodes, allow_unused=True, retain_graph=True)
            require(all(g is not None and bool(torch.isfinite(g).all()) for g in node_gradients)
                    and any(bool((g != 0).any()) for g in node_gradients), "Auxiliary output-node gradient absent")
            auxiliary.backward(retain_graph=True)
            gradient_record(model, head_expected=arm == "C_mu")
            optimizer.zero_grad(set_to_none=True)
        require(state.rng_digest(state.rng_state()) == probe_rng, "Gradient probes changed dropout RNG")
        # Teacher is first used after fixed predictors. Alternative labels cannot
        # mutate predictor tensors or consume dropout RNG.
        predictor_sha = state.state_digest([{k: v.detach() for k,v in row["detail"].items()
                        if torch.is_tensor(v)} | {"logits": row["logits"].detach()} for row in records])
        before = state.rng_digest(state.rng_state())
        alternate = ObservationTeacher(graph.nodes, pairs.new_empty(0))
        for row in records:
            _, bits = alternate.labels(row["query"], row["detail"]["neighbors"])
            if arm == "C_mu":
                model.source_nll(row["detail"], bits)
            elif arm != "P0":
                mods["pattern_objective"].losses(row["detail"]["t"], row["rows"], bits, len(row["query"]))
        require(predictor_sha == state.state_digest([{k: v.detach() for k,v in row["detail"].items()
                    if torch.is_tensor(v)} | {"logits": row["logits"].detach()} for row in records])
                and state.rng_digest(state.rng_state()) == before, "Teacher changed fixed predictor/RNG")
        resources.call("candidate_total_backward", total.backward)
        gradients = gradient_record(model, head_expected=arm == "C_mu")
        require(arm != "C_mu" or gradients["active_parameter_entries"] == 84675, "C active parameter count differs")
        candidate_rng = state.rng_digest(state.rng_state())
        state.restore_snapshot(reference, ropt, initial)
        ref_records = resources.call("independent_reference_forward", lambda: forward(reference, x, graph, positive, negative,
                            arm, independent=arm != "C_mu"))
        ref_main, ref_aux, ref_total = objective(reference, ref_records, teacher, arm, mods,
                            independent=True, enumeration=arm == "C_mu")
        for a,b in zip(records, ref_records):
            neighbors_equal(a["detail"]["neighbors"], b["detail"]["neighbors"])
            audit.close(a["logits"], b["logits"], arm + " target", "vector" if arm == "C_mu" else "native")
            for key in (("eta","context","swapped_context","mu","completion_weight") if arm == "C_mu"
                        else ("raw_left","raw_right","routed_left","routed_right","t")):
                audit.close(a["detail"][key], b["detail"][key], arm + " " + key, "vector" if arm == "C_mu" else "native")
            audit.close(a["auxiliary_values"], b["auxiliary_values"], arm + " all-query source NLL", "vector")
        audit.close(main, ref_main, arm + " native BCE")
        audit.close(auxiliary, ref_aux, arm + " auxiliary", "vector")
        resources.call("reference_total_backward", ref_total.backward)
        compare_gradients(model, reference, audit)
        require(state.rng_digest(state.rng_state()) == candidate_rng, "Reference/checkpoint dropout RNG differs")
        optimizer.step(); ropt.step()
        audit.nested(state.snapshot(model, optimizer)["models"], state.snapshot(reference, ropt)["models"], arm + " Adam model")
        audit.nested(state.snapshot(model, optimizer)["optimizer"], state.snapshot(reference, ropt)["optimizer"], arm + " Adam arithmetic")
        # Additional C symmetry and eval partition checks on fixed full supports.
        if arm == "C_mu":
            model.eval()
            with torch.no_grad():
                h = model.encode(x, graph)
                full, d = model.query_forward(h, graph, positive, auxiliary_grad=False)
                reverse, ds = model.query_forward(h, graph, positive.flip(1), auxiliary_grad=False)
                audit.close(full, reverse, "C endpoint exchange", "vector")
                audit.close(d["context"], ds["swapped_context"], "C swapped complete context", "vector")
                parts = [model.query_forward(h, graph, q[None], auxiliary_grad=False)[0] for q in positive]
                audit.close(full, torch.cat(parts), "C eval query partition", "vector")
            model.train()
        serialized = resources.call("own_serialized_next_update", lambda: own_roundtrip(model, optimizer, mods, rt["device"], arm,
                    context["output"], x, graph, positive, negative, teacher, audit, next_update=True))
        receipts.append({"arm": arm, "total_parameters": total_parameters, "initial_typed_state_sha256": initial_sha,
                         "gradients": gradients, "main_detach_and_outer_gradient_paths": True,
                         "auxiliary_only_active_parameter_and_output_node_gradients": arm != "P0",
                         "teacher_label_only_boundary": True, "serialization": serialized})
        del model, optimizer, reference, ropt, records, ref_records, initial, main, auxiliary, total
        gc.collect(); torch.cuda.empty_cache()
    return {"direct_native": direct, "arms": receipts, "comparison_rules_and_errors": audit.comparisons,
            "fabricated_nodes": 16, "fabricated_records": 10, "duplicate_source_zero_empty_checks": True,
            "actual_optimizer_updates_per_arm": 4,
            "update_breakdown_per_arm": {"primary_candidate": 2, "independent_reference": 1, "own_restored_replay": 1},
            "state_donor": False, "complete_batch_feasibility_qualified": False}


def support_stats(detail, mods, count):
    neighbors = detail["neighbors"]
    nl = torch.bincount(neighbors.left[0], minlength=neighbors.queries).cpu().tolist()
    nr = torch.bincount(neighbors.right[0], minlength=neighbors.queries).cpu().tolist()
    row = {"queries": neighbors.queries, "left_slots": sum(nl), "right_slots": sum(nr),
           "common_slots": len(neighbors.common[0]), "max_left": max(nl, default=0),
           "max_right": max(nr, default=0), "empty_residual_queries": sum(a+b == 0 for a,b in zip(nl,nr)),
           "count_pairs": sum((a+1)*(b+1) for a,b in zip(nl,nr)),
           "ESP_reachable_cells": sum((a+1)*(a+2)//2 + (b+1)*(b+2)//2 for a,b in zip(nl,nr)),
           "ESP_square_storage_cells": sum((a+1)**2 + (b+1)**2 for a,b in zip(nl,nr))}
    if count:
        chunks = mods["ragged_density"].partitions(detail["left_offsets"], detail["right_offsets"])
        cells = [len(indices)*((a+1)**2+(b+1)**2+(a+1)*(b+1)) for indices,a,b in chunks]
        row.update(exact_query_chunks=len(chunks), max_nominal_chunk_cells=max(cells, default=0),
                   oversized_whole_queries=sum(c > 262144 for c in cells),
                   padded_cells_per_density_pass=sum(cells), full_query_slot_count_support_retained=True)
    return row


@contextmanager
def potential_meter(mods):
    """Observational call wrapper: unchanged args/results/RNG, no tensor reads."""
    module = mods["ragged_density"]
    original = module.potential
    stats = {"calls": 0, "query_rows": 0, "padded_count_cells": 0}
    def observe(head, context, swapped, nl, nr, **kwargs):
        value = original(head, context, swapped, nl, nr, **kwargs)
        stats["calls"] += 1; stats["query_rows"] += len(nl)
        stats["padded_count_cells"] += value.numel()
        return value
    module.potential = observe
    try:
        yield stats
    finally:
        module.potential = original


def batch(context, rt, mods, resources, data, x, replay):
    audit = Audit()
    teacher = ObservationTeacher.from_train(data["pairs"], data["nodes"])
    records, negatives, expected = replay.query_batch(0)
    require(len(records) == 65536 and x.shape == (235868,128), "True complete native batch/features required")
    before = state.rng_digest(state.rng_state())
    regenerated = resources.call("provider_complete_context_regeneration", lambda: replay.regenerate(0))
    require(state.rng_digest(state.rng_state()) == before, "Provider regeneration consumed model RNG")
    graph = regenerated["graph"]
    positive, negative = regenerated["positive_queries"], regenerated["negative_queries"]
    require(torch.equal(positive, data["pairs"][records]) and torch.equal(negative, negatives[records]), "Replay query order differs")
    actual_native_support = []
    for name, query in (("positive", positive), ("negative", negative)):
        actual_native_support.append(resources.call("every_complete_native_" + name + "_coordinate",
                       lambda name=name, query=query: native_support(mods, graph, query, regenerated[name + "_neighbors"])))
    arms = []
    for arm in ARMS:
        resources.arm = arm
        torch.cuda.reset_peak_memory_stats(0)
        model, optimizer = resources.call("fresh_fixed_factory", lambda: factory(mods, rt["device"], arm))
        initial_sha = state.state_digest(state.snapshot(model, optimizer))
        model.train(); optimizer.zero_grad(set_to_none=True)
        with potential_meter(mods) as meter:
            h = resources.call("complete_node_encoder", lambda: model.encode(x, graph))
            output = []
            for name, query in (("positive", positive),("negative", negative)):
                logits, detail = resources.call("complete_65536_" + name + "_forward",
                           lambda query=query: model.query_forward(h, graph, query, auxiliary_grad=arm != "P0"))
                output.append({"query": query, "logits": logits, "detail": detail})
            resources.call("actual_replay_support_verification", lambda: replay.assert_actual(0, {
                "record_ids": records, "graph": graph, "positive_queries": positive, "negative_queries": negative,
                "positive_neighbors": output[0]["detail"]["neighbors"], "negative_neighbors": output[1]["detail"]["neighbors"]}))
            main, auxiliary, total = resources.call("every_teacher_label_and_fixed_objective",
                            lambda: objective(model, output, teacher, arm, mods))
            detach_probe(main, model, output, arm)
            score_nodes = ([row["detail"]["eta"] for row in output] if arm == "C_mu" else
                           [t for row in output for t in row["detail"]["raw_score_tensors"]])
            for tensor in score_nodes:
                if tensor.requires_grad:
                    tensor.retain_grad()
            support = [support_stats(row["detail"], mods, arm == "C_mu") for row in output]
            resources.call("complete_support_accounting", lambda: atomic_json(context["output"] / ("LIVE_" + arm + ".json"),
                           {"arm": arm, "support": support, "state_donor": False,
                            "main_detach_verified": True, "objective_finite": True, "stage": "BEFORE_BACKWARD"}))
            resources.call("complete_total_backward", total.backward)
            gradients = gradient_record(model, head_expected=arm == "C_mu")
            if arm != "P0":
                require(all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in score_nodes)
                        and any(bool((t.grad != 0).any()) for t in score_nodes), "Full source scorer gradient missing")
            resources.call("native_Adam_update", optimizer.step)
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite full updated state")
            update_meter = dict(meter)
            if arm == "C_mu":
                chunks = sum(row["exact_query_chunks"] for row in support)
                require(update_meter["calls"] == 3 * chunks and update_meter["query_rows"] == 3 * 131072,
                        "Missing marginal/NLL/checkpoint recomputation calls")
            # Release large retained backward graphs before own-state replay.
            del h, output, main, auxiliary, total, score_nodes, logits, detail, tensor
            gc.collect()
            roundtrip = resources.call("own_state_and_complete_serving_replay", lambda: own_roundtrip(model, optimizer, mods,
                        rt["device"], arm, context["output"], x, graph, positive, negative, teacher, audit, next_update=False))
        arms.append({"arm": arm, "model_seed": 610041, "initial_typed_state_sha256": initial_sha,
                     "optimizer_updates": 1, "positive_queries": 65536, "negative_queries": 65536,
                     "support": support, "gradients": gradients, "actual_density_potential_calls_update": update_meter,
                     "actual_density_potential_calls_including_serving": dict(meter),
                     "serialization": roundtrip, "resources": resources.sample(), "state_donor": False})
        atomic_json(context["output"] / ("ARM_" + arm + ".json"), arms[-1])
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()
    return {"arms": arms, "epoch": 1, "batch": 0, "provider_batch_receipt": expected,
            "actual_native_support": actual_native_support,
            "teacher_identity": teacher.receipt(), "comparison_rules_and_errors": audit.comparisons,
            "complete_record_and_support_regeneration_RNG_exact": True,
            "full_TRAIN_epochs": 0, "VALID_evaluations": 0, "fits": 0, "state_donor": False}
