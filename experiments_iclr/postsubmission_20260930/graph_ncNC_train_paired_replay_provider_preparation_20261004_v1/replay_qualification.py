"""Prepared common-core and complete real TRAIN stream/support audits only."""
from pathlib import Path
from itertools import zip_longest
import random
from replay_common import file_sha, load_module, require
from replay_provider import (EpochReplay, batch_context, coordinate_chunks, draw_epoch,
                             epoch_seed, isolated_rng, validate_arrays, write_epoch)


def same_context(rt, a, b):
    torch = rt["torch"]
    for key in ("record_ids", "positive_queries", "negative_queries"):
        require(torch.equal(a[key], b[key]), "Full integer query/mask mismatch: " + key)
    require(a["graph"].nodes == b["graph"].nodes, "Graph node count differs")
    for name in ("row", "col", "rowptr"):
        require(torch.equal(getattr(a["graph"], name), getattr(b["graph"], name)), "Full integer graph mismatch")
    for side in ("positive", "negative"):
        an, bn = a[side+"_neighbors"], b[side+"_neighbors"]
        require(an.queries == bn.queries, "Query count differs")
        for kind in ("common", "left", "right"):
            for x, y in zip(getattr(an, kind), getattr(bn, kind)):
                require(torch.equal(x, y), "Full ordered integer support mismatch")
        for common in (False, True):
            aa = coordinate_chunks(rt, a[side+"_queries"], an, common)
            bb = coordinate_chunks(rt, b[side+"_queries"], bn, common)
            for x, y in zip_longest(aa, bb):
                require(x is not None and y is not None, "Coordinate stream block count differs")
                require(torch.equal(x, y), "Full streamed coordinate integer mismatch")


def fixture(rt):
    torch, device = rt["torch"], rt["device"]
    pairs = torch.tensor([[0,1],[0,1],[0,2],[1,2],[2,3],[3,4],[4,5]], dtype=torch.long, device=device)
    raw = torch.stack([p if direction == 0 else p.flip(0) for p in pairs for direction in (0,1)], dim=1)
    identity = {"nodes": 12, "records": len(pairs), "ordered_raw_entries": raw.shape[1],
                "pairs_sha256": rt["tensor_sha"](pairs), "raw_sha256": rt["tensor_sha"](raw), "batch_size": 3,
                "fixture_only": True}
    return {"pairs": pairs, "raw": raw, "nodes": 12, "batch_size": 3,
            "identity": dict(rt["identity"], TRAIN=identity)}


def oracle(rt, data, records, queries):
    """Independent Python multiset mask and ordered support reference; fixture only."""
    torch, device = rt["torch"], data["pairs"].device
    removed = set(records.cpu().tolist())
    arcs = set()
    for i, (u, v) in enumerate(data["pairs"].cpu().tolist()):
        if i not in removed:
            arcs.add((u, v)); arcs.add((v, u))
    adjacency = {n: set() for n in range(data["nodes"])}
    for u, v in arcs:
        adjacency[u].add(v)
    result = {kind: ([], []) for kind in ("common", "left", "right")}
    for row, (u, v) in enumerate(queries.cpu().tolist()):
        sets = {"common": adjacency[u] & adjacency[v], "left": adjacency[u]-adjacency[v], "right": adjacency[v]-adjacency[u]}
        for kind in sets:
            for node in sorted(sets[kind]):
                result[kind][0].append(row); result[kind][1].append(node)
    return {kind: tuple(torch.tensor(x, dtype=torch.long, device=device) for x in coordinates) for kind, coordinates in result.items()}


def expect_failure(action):
    try:
        action()
    except (RuntimeError, ValueError, FileNotFoundError):
        return
    raise RuntimeError("Expected fail-closed rejection did not happen")


def perturb_rng(rt):
    import numpy as np
    random.random(); np.random.random(5)
    rt["torch"].rand(11)
    rt["torch"].rand(19, device=rt["device"])


def fabricated(rt, master_seed, epoch, output):
    data, torch = fixture(rt), rt["torch"]
    p1, n1 = draw_epoch(rt, data, master_seed, epoch)
    perturb_rng(rt)
    p2, n2 = draw_epoch(rt, data, master_seed, epoch)
    require(torch.equal(p1, p2) and torch.equal(n1, n2), "Ambient RNG changes provider integers")
    before = rt["state"].rng_digest(rt["state"].rng_state())
    def exceptional():
        with isolated_rng(rt, epoch_seed(master_seed, epoch)):
            perturb_rng(rt)
            raise ValueError("fixture exception")
    expect_failure(exceptional)
    require(rt["state"].rng_digest(rt["state"].rng_state()) == before, "Exception leaked RNG")
    trace = Path(output)/"fixture_trace"
    before = rt["state"].rng_digest(rt["state"].rng_state())
    receipt = write_epoch(rt, data, p1, n1, master_seed, epoch, trace)
    epoch_sha = file_sha(trace/"EPOCH.json")
    reader = EpochReplay(rt, data, trace, epoch_sha)
    require(torch.equal(reader.permutation, p1) and torch.equal(reader.negatives, n1), "Saved integer arrays changed")
    require(rt["state"].rng_digest(rt["state"].rng_state()) == before, "Writer/reader consumed model RNG")
    contexts = []
    for batch in range(receipt["full_batches"]):
        a = reader.regenerate(batch)
        b = batch_context(rt, data, p1, n1, batch)
        same_context(rt, a, b)
        for side in ("positive", "negative"):
            ref = oracle(rt, data, a["record_ids"], a[side+"_queries"])
            for kind, pair in ref.items():
                for x, y in zip(pair, getattr(a[side+"_neighbors"], kind)):
                    require(torch.equal(x, y), "Independent fixture oracle differs")
        contexts.append(a)
    # Duplicate record survival and isolated empty supports are explicit cases.
    g1 = rt["graphs"].Graph.mask_train_batch(data["pairs"], torch.tensor([0], device=rt["device"]), data["nodes"])
    g2 = rt["graphs"].Graph.mask_train_batch(data["pairs"], torch.tensor([0,1], device=rt["device"]), data["nodes"])
    require(bool(((g1.row == 0) & (g1.col == 1)).any()) and not bool(((g2.row == 0) & (g2.col == 1)).any()), "Duplicate record mask semantics changed")
    empty = rt["graphs"].enumerate_neighbors(g1, torch.tensor([[10,11]], dtype=torch.long, device=rt["device"]))
    require(all(len(getattr(empty, k)[0]) == 0 for k in ("common", "left", "right")), "Empty support not retained")
    # Teacher is imported only here, after provider contexts already exist.
    teacher_module = load_module("pattern_teacher", rt["source_paths"]["pattern_teacher"])
    teacher = teacher_module.ObservationTeacher.from_train(data["pairs"], data["nodes"])
    for batch, value in enumerate(contexts):
        for side in ("positive", "negative"):
            rows, labels = teacher.labels(value[side+"_queries"], value[side+"_neighbors"])
            require(len(rows) == len(labels), "Label-only fixture alignment differs")
        reader.assert_actual(batch, value)
    mask = torch.tensor([0,1,3], dtype=torch.long, device=rt["device"])
    graph = rt["graphs"].Graph.mask_train_batch(data["pairs"], mask, data["nodes"])
    probes = torch.tensor([[0,1],[0,5]], dtype=torch.long, device=rt["device"])
    neighbors = rt["graphs"].enumerate_neighbors(graph, probes)
    _, labels = teacher.labels(probes, neighbors)
    require(bool((labels == 0).any()) and bool((labels == 1).any()), "Fixture must distinguish source-unobserved and removed-observed bits")
    bad_p = p1.clone(); bad_p[1] = bad_p[0]
    expect_failure(lambda: validate_arrays(rt, data, bad_p, n1))
    bad_n = n1.clone(); bad_n[0] = 0
    expect_failure(lambda: validate_arrays(rt, data, p1, bad_n))
    bad_n = n1.clone(); bad_n[0] = data["pairs"][0]
    expect_failure(lambda: validate_arrays(rt, data, p1, bad_n))
    # Failure checks operate only on own fixture copies, with no source mutation.
    changed_data = dict(data, identity=dict(data["identity"], TRAIN=dict(data["identity"]["TRAIN"], nodes=13)))
    expect_failure(lambda: EpochReplay(rt, changed_data, trace, epoch_sha))
    expect_failure(lambda: EpochReplay(rt, data, trace, "0"*64))
    pin = receipt["files"]["permutation"]
    f = trace/pin["path"]
    original = f.read_bytes()
    f.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    expect_failure(lambda: EpochReplay(rt, data, trace, epoch_sha))
    f.write_bytes(original)
    EpochReplay(rt, data, trace, epoch_sha)
    return {"complete_batches": receipt["full_batches"], "full_integer_stream_equality": True,
            "full_integer_support_equality": True, "fixture_geometry": data["identity"]["TRAIN"],
            "checks": ["native sampler/permutation common core", "Python/NumPy/CPU/CUDA RNG preservation including exception",
                       "ambient-model-RNG independence", "full .npy integer roundtrip", "every full mask and dropped tail",
                       "independent Python support oracle", "duplicate survival", "isolated empty support",
                       "teacher-after-context label-only boundary and source-zero semantics",
                       "wrong identity/receipt pin/corrupted payload/duplicate permutation/self-link/forbidden-negative rejection"],
            "teacher_label_only": True, "trace": str(trace), "epoch_sha256": epoch_sha}


def full_batch(rt, data, master_seed, epoch, output):
    torch = rt["torch"]
    p1, n1 = draw_epoch(rt, data, master_seed, epoch)
    perturb_rng(rt)
    with isolated_rng(rt, epoch_seed(master_seed, epoch)):
        frozen_data = {"pairs": data["pairs"], "raw_edge_index": data["raw"], "x": range(data["nodes"])}
        n2, iterator, _ = rt["frozen_epoch_stream"](frozen_data, {"native_utils": rt["utils"]}, rt["sampler"])
        p2 = iterator.idx.clone()
        frozen_masks = [ids.clone() for ids in iterator]
    require(torch.equal(p1, p2) and torch.equal(n1, n2), "Full native integer stream differs")
    require(len(frozen_masks) == 17, "Complete real epoch required")
    require(torch.equal(p1[17*65536:], p2[17*65536:]), "Full dropped tail differs")
    trace = Path(output)/"complete_TRAIN_trace"
    receipt = write_epoch(rt, data, p1, n1, master_seed, epoch, trace)
    epoch_sha = file_sha(trace/"EPOCH.json")
    reader = EpochReplay(rt, data, trace, epoch_sha)
    require(torch.equal(reader.permutation, p2) and torch.equal(reader.negatives, n2), "Full persisted arrays differ")
    for batch, records in enumerate(frozen_masks):
        actual = reader.regenerate(batch)
        require(torch.equal(actual["record_ids"], records), "Actual full record mask differs")
        graph = rt["graphs"].Graph.mask_train_batch(data["pairs"], records, data["nodes"])
        positive, negative = data["pairs"][records], n2[records]
        ref = {"record_ids": records, "graph": graph, "positive_queries": positive, "negative_queries": negative,
               "positive_neighbors": rt["graphs"].enumerate_neighbors(graph, positive),
               "negative_neighbors": rt["graphs"].enumerate_neighbors(graph, negative)}
        same_context(rt, actual, ref)
    return {"TRAIN_identity": data["identity"]["TRAIN"], "complete_batches": 17,
            "full_integer_stream_equality": True, "full_integer_support_equality": True,
            "checks": ["entire native negative array", "entire permutation and dropped tail", "all17 native masks",
                       "every integer graph/query/common/left/right support", "every streamed coordinate block", "RNG restoration"],
            "work": {"native_sampler_draws": 2, "native_permutations": 2, "record_graph_rebuilds": 51,
                     "full_query_support_enumerations": 102, "supervised_queries_per_support_pass": 65536,
                     "optimizer_updates": 0, "model_forwards": 0, "VALID_traversals": 0, "TEST_reads": 0},
            "trace": str(trace), "epoch_sha256": epoch_sha, "trace_integer_file_bytes": sum(v["bytes"] for v in receipt["files"].values()),
            "coordinate_bytes_if_materialized": sum(b[s][k]["materialized_coordinate_bytes"]
               for b in receipt["batches"] for s in ("positive", "negative") for k in ("residual_coordinates", "common_coordinates"))}
