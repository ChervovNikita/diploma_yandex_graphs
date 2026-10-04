"""Prospective compact integer stream; exact model-free support regeneration.

No Torch import at module load. No teacher, model, metric or split reader.
Caller must authenticate source/runtime and the immutable TRAIN inputs first.
RNG isolation requires exclusive use of global RNG during this synchronous call.
"""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import json
import os
import sys
import tempfile
from replay_common import atomic_json, file_sha, require, utc

COORDINATE_CHUNK = 131072
COORDINATE_SCHEMA = {"residual": ["query_row", "left0_right1", "query_u", "query_v", "candidate", "counterpart"],
                     "common": ["query_row", "query_u", "query_v", "candidate"]}


def epoch_seed(master_seed, epoch):
    require(type(master_seed) is int and 0 <= master_seed < 2**63 and type(epoch) is int and 1 <= epoch <= 100, "Invalid prospective seed/epoch")
    key = f"ncnc-TRAIN-replay-v1:{master_seed}".encode()
    # Distinct uint32 epoch seeds support the bound legacy NumPy seed_all path.
    return (int.from_bytes(sha256(key).digest()[:4], "little") + epoch - 1) % 2**32


@contextmanager
def isolated_rng(rt, seed):
    torch, state = rt["torch"], rt["state"]
    torch.cuda.synchronize(0)
    before = state.rng_state()
    before_sha = state.rng_digest(before)
    try:
        rt["seed_all"](seed)
        yield
    finally:
        try:
            torch.cuda.synchronize(0)
        finally:
            state.restore_rng(before)
            require(state.rng_digest(state.rng_state()) == before_sha, "Provider failed to restore model/global RNG")


def validate_arrays(rt, data, permutation, negatives):
    torch = rt["torch"]
    pairs, raw, nodes, bs = data["pairs"], data["raw"], data["nodes"], data["batch_size"]
    require(pairs.dtype == raw.dtype == permutation.dtype == negatives.dtype == torch.long, "int64 arrays required")
    require(permutation.ndim == 1 and len(permutation) == len(pairs), "Full permutation missing")
    require(torch.equal(permutation.sort().values, torch.arange(len(pairs), device=permutation.device)), "Not a complete record permutation")
    require(negatives.ndim == 2 and negatives.shape[1] == 2 and len(negatives) >= len(pairs), "Full native negative draw missing")
    require(bool((negatives[:, 0] != negatives[:, 1]).all()) and int(negatives.min()) >= 0 and int(negatives.max()) < nodes, "Negative endpoint contract differs")
    keys = (raw[0] * nodes + raw[1]).sort().values
    sampled = negatives[:, 0] * nodes + negatives[:, 1]
    idx = torch.searchsorted(keys, sampled)
    require(not bool(((idx < len(keys)) & (keys[idx.clamp(max=len(keys)-1)] == sampled)).any()), "Native negatives contain forbidden raw edges")
    require(bs > 0 and len(pairs) // bs > 0, "No complete training batches")


def verify_TRAIN_identity(rt, data):
    identity = data["identity"]["TRAIN"]
    require(identity["nodes"] == data["nodes"] and identity["records"] == len(data["pairs"])
            and identity["ordered_raw_entries"] == data["raw"].shape[1] and identity["batch_size"] == data["batch_size"],
            "TRAIN geometry identity differs")
    require(rt["tensor_sha"](data["pairs"]) == identity["pairs_sha256"]
            and rt["tensor_sha"](data["raw"]) == identity["raw_sha256"], "Immutable TRAIN integer identity differs")


def draw_epoch(rt, data, master_seed, epoch):
    """Native sampler then native permutation, with model RNG restored exactly."""
    with isolated_rng(rt, epoch_seed(master_seed, epoch)):
        native = rt["sampler"](data["raw"], data["nodes"])
        iterator = rt["PermIterator"](data["pairs"].device, len(data["pairs"]), data["batch_size"])
        permutation = iterator.idx.clone()
        batches = [row.clone() for row in iterator]
    negatives = native.T.contiguous()
    validate_arrays(rt, data, permutation, negatives)
    bs = data["batch_size"]
    require(len(batches) == len(data["pairs"]) // bs, "Native full-batch/tail behavior differs")
    for batch, records in enumerate(batches):
        require(rt["torch"].equal(records, permutation[batch*bs:(batch+1)*bs]), "Native batch order is not the stored permutation prefix")
    return permutation, negatives


def graph_pin(rt, graph):
    return {"nodes": graph.nodes, "directed_entries": len(graph.row),
            **{name + "_sha256": rt["tensor_sha"](getattr(graph, name)) for name in ("row", "col", "rowptr")}}


def coordinate_chunks(rt, queries, neighbors, common=False):
    """All coordinates in source order; bounded materialization, no sampling."""
    torch = rt["torch"]
    sides = [(None, neighbors.common)] if common else [(0, neighbors.left), (1, neighbors.right)]
    for side, (rows, candidates) in sides:
        for start in range(0, len(rows), COORDINATE_CHUNK):
            r, c = rows[start:start+COORDINATE_CHUNK], candidates[start:start+COORDINATE_CHUNK]
            if common:
                yield torch.stack((r, queries[r, 0], queries[r, 1], c), dim=1)
            else:
                yield torch.stack((r, torch.full_like(r, side), queries[r, 0], queries[r, 1], c, queries[r, 1-side]), dim=1)


def coordinate_pin(rt, queries, neighbors, common=False):
    import numpy as np
    rows = len(neighbors.common[0]) if common else len(neighbors.left[0]) + len(neighbors.right[0])
    width = 4 if common else 6
    digest = sha256(str(((rows, width), np.dtype("int64"))).encode())
    for block in coordinate_chunks(rt, queries, neighbors, common):
        array = block.detach().cpu().contiguous().numpy()
        digest.update(memoryview(array).cast("B"))
    return {"shape": [rows, width], "dtype": "int64", "sha256": digest.hexdigest(), "materialized_coordinate_bytes": rows*width*8}


def support_pin(rt, queries, neighbors):
    require(neighbors.queries == len(queries), "Native support query count differs")
    return {"query_sha256": rt["tensor_sha"](queries), "queries": len(queries),
            "left_slots": len(neighbors.left[0]), "right_slots": len(neighbors.right[0]), "common_slots": len(neighbors.common[0]),
            "residual_coordinates": coordinate_pin(rt, queries, neighbors),
            "common_coordinates": coordinate_pin(rt, queries, neighbors, common=True)}


def batch_context(rt, data, permutation, negatives, batch):
    bs = data["batch_size"]
    require(type(batch) is int and 0 <= batch < len(permutation)//bs, "Batch outside complete prefix")
    records = permutation[batch*bs:(batch+1)*bs].clone()
    graph = rt["graphs"].Graph.mask_train_batch(data["pairs"], records, data["nodes"])
    positive, negative = data["pairs"][records], negatives[records]
    pn = rt["graphs"].enumerate_neighbors(graph, positive)
    nn = rt["graphs"].enumerate_neighbors(graph, negative)
    return {"record_ids": records, "graph": graph, "positive_queries": positive, "negative_queries": negative,
            "positive_neighbors": pn, "negative_neighbors": nn}


def batch_pin(rt, value):
    return {"record_ids_sha256": rt["tensor_sha"](value["record_ids"]), "graph": graph_pin(rt, value["graph"]),
            **{side: support_pin(rt, value[side+"_queries"], value[side+"_neighbors"]) for side in ("positive", "negative")}}


def save_integer_array(path, value):
    import numpy as np
    require(sys.byteorder == "little", "Qualified little-endian runtime required")
    array = value.detach().cpu().contiguous().numpy()
    require(str(array.dtype) == "int64", "Only integer trace arrays may be saved")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array.astype("<i8", copy=False), allow_pickle=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path),
            "shape": list(array.shape), "dtype": "<i8", "payload_bytes": array.size*8}


def write_epoch(rt, data, permutation, negatives, master_seed, epoch, output):
    output = Path(output).resolve()
    require(not output.exists(), "Fresh epoch output required")
    output.mkdir(parents=True, mode=0o700)
    verify_TRAIN_identity(rt, data)
    validate_arrays(rt, data, permutation, negatives)
    files = {"permutation": save_integer_array(output/"PERMUTATION.npy", permutation),
             "negative_pairs": save_integer_array(output/"NEGATIVE_PAIRS.npy", negatives)}
    bs, total = data["batch_size"], len(permutation)
    batches = []
    for batch in range(total//bs):
        batches.append(batch_pin(rt, batch_context(rt, data, permutation, negatives, batch)))
    receipt = {"schema": "ncnc-TRAIN-paired-replay-epoch-v1", "UTC": utc(), "identity": data["identity"],
               "master_seed": master_seed, "epoch": epoch, "provider_seed": epoch_seed(master_seed, epoch),
               "batch_size": bs, "full_batches": total//bs, "supervised_records": total//bs*bs, "dropped_tail_records": total%bs,
               "full_negative_rows": len(negatives), "files": files,
               "native_negative_draw_sha256": rt["tensor_sha"](negatives.T),
               "permutation_sha256": rt["tensor_sha"](permutation),
               "dropped_tail_sha256": rt["tensor_sha"](permutation[total//bs*bs:]), "batches": batches,
               "coordinate_schema": COORDINATE_SCHEMA,
               "stored_teacher_labels": False, "coordinates_stored": False,
               "coordinate_regeneration": "exact source-bound integer graph/query operations; equality qualification mandatory",
               "old_run_matching_claim": False}
    atomic_json(output/"EPOCH.json", receipt)
    return receipt


class EpochReplay:
    """Reads actual arrays; regenerates or checks complete native contexts."""
    def __init__(self, rt, data, root, expected_epoch_sha256):
        import numpy as np
        self.rt, self.data, self.root = rt, data, Path(root).resolve()
        require(sys.byteorder == "little", "Qualified little-endian runtime required")
        require(type(expected_epoch_sha256) is str and len(expected_epoch_sha256) == 64
                and file_sha(self.root/"EPOCH.json") == expected_epoch_sha256, "Externally bound epoch receipt required")
        verify_TRAIN_identity(rt, data)
        r = json.loads((self.root/"EPOCH.json").read_text())
        require(r.get("schema") == "ncnc-TRAIN-paired-replay-epoch-v1" and r["identity"] == data["identity"], "Epoch source/runtime/TRAIN identity differs")
        require(r["stored_teacher_labels"] is False and r["coordinates_stored"] is False and r["old_run_matching_claim"] is False, "Epoch scope differs")
        require(r["coordinate_schema"] == COORDINATE_SCHEMA, "Coordinate schema differs")
        arrays = {}
        for name, pin in r["files"].items():
            require(name in ("permutation", "negative_pairs") and Path(pin["path"]).name == pin["path"], "Unexpected trace payload")
            f = self.root/pin["path"]
            require(f.stat().st_size == pin["bytes"] and file_sha(f) == pin["sha256"], "Epoch array corrupted")
            mapped = np.load(f, mmap_mode="r", allow_pickle=False)
            require(pin["dtype"] == "<i8" and str(mapped.dtype) == "int64" and list(mapped.shape) == pin["shape"]
                    and mapped.size*8 == pin["payload_bytes"], "Trace dtype/shape differs")
            arrays[name] = rt["torch"].from_numpy(np.array(mapped, copy=True)).to(data["pairs"].device)
        require(set(arrays) == {"permutation", "negative_pairs"}, "Trace arrays incomplete")
        self.permutation, self.negatives = arrays["permutation"], arrays["negative_pairs"]
        validate_arrays(rt, data, self.permutation, self.negatives)
        bs, total = data["batch_size"], len(self.permutation)
        require(r["provider_seed"] == epoch_seed(r["master_seed"], r["epoch"]) and r["batch_size"] == bs
                and r["full_batches"] == total//bs and r["supervised_records"] == total//bs*bs
                and r["dropped_tail_records"] == total%bs and len(r["batches"]) == total//bs
                and r["full_negative_rows"] == len(self.negatives), "Epoch reduction/tail geometry differs")
        require(r["native_negative_draw_sha256"] == rt["tensor_sha"](self.negatives.T)
                and r["permutation_sha256"] == rt["tensor_sha"](self.permutation)
                and r["dropped_tail_sha256"] == rt["tensor_sha"](self.permutation[total//bs*bs:]), "Native full integer identities differ")
        self.receipt = r

    def query_batch(self, batch):
        """Native adapter; full negative tensor is borrowed and must not be mutated."""
        bs = self.data["batch_size"]
        require(type(batch) is int and 0 <= batch < self.receipt["full_batches"], "Bad batch")
        records = self.permutation[batch*bs:(batch+1)*bs].clone()
        return records, self.negatives, self.receipt["batches"][batch]

    def regenerate(self, batch):
        before = self.rt["state"].rng_digest(self.rt["state"].rng_state())
        value = batch_context(self.rt, self.data, self.permutation, self.negatives, batch)
        self.assert_actual(batch, value)
        require(self.rt["state"].rng_digest(self.rt["state"].rng_state()) == before, "Support regeneration consumed model RNG")
        return value

    def assert_actual(self, batch, value):
        # Actual native decoder neighbors can be checked without enumerating twice.
        require(type(batch) is int and 0 <= batch < self.receipt["full_batches"], "Bad actual-context batch")
        require(batch_pin(self.rt, value) == self.receipt["batches"][batch], "Actual native graph/query/support differs from prospective trace")
