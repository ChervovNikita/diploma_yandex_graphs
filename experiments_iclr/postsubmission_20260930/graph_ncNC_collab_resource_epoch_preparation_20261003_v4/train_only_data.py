"""Direct exact-file TRAIN/raw loader. Called only after stdlib admission."""
import gzip
import codecs
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from guards import require


def tensor_sha(tensor):
    # Exact acquired metadata convention from cache_builder.py:19-23.
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def csv_digest(path):
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def load_train_only(context, meter):
    root, pins = Path(context["admission"]["dataset_root"]), context["bindings"]["allowed_data_files"]
    for relative in ("raw/node-feat.csv.gz", "raw/edge.csv.gz"):
        size, digest = meter.call("cold_csv_decompression_hash", csv_digest, root / relative)
        require(size == pins[relative]["csv_bytes"] and digest == pins[relative]["csv_sha256"], "Uncompressed custody digest differs")
    # This authenticated official TRAIN dictionary is the sole torch.load.
    def read_official_train():
        # Authenticated OGB legacy torch.save stores NumPy int64 arrays.
        # Keep weights_only=True with exactly the observed reconstruction
        # globals and the int64 dtype class, scoped to this single read.
        allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                   type(np.dtype(np.int64)), codecs.encode]
        with torch.serialization.safe_globals(allowed):
            stored = torch.load(root / "split/time/train.pt", map_location="cpu", weights_only=True)
        require(type(stored) is dict and set(stored) == {"edge", "weight", "year"}, "Unexpected official TRAIN dictionary")
        for name, value in stored.items():
            require(type(value) is np.ndarray and value.dtype == np.dtype(np.int64), "Official TRAIN storage type differs: " + name)
        # Native OGB replace_numpy_with_torchtensor uses torch.from_numpy.
        # Preserve record order and values. Complete tensor hashes are checked below.
        return {name: torch.from_numpy(value) for name, value in stored.items()}
    train = meter.call("cold_train_read", read_official_train)
    pairs = train["edge"]
    require(torch.is_tensor(pairs) and pairs.dtype == torch.long and tuple(pairs.shape) == (1179052, 2), "TRAIN record shape/dtype differs")
    require(torch.is_tensor(train["year"]) and int(train["year"].max()) == 2017, "TRAIN year authority differs")
    del train
    # Match inspected OGB pandas conversion, with no propagated BUDDY features.
    feature_array = meter.call("cold_feature_csv_read", lambda: pd.read_csv(root / "raw/node-feat.csv.gz", compression="gzip", header=None).values.astype(np.float32))
    raw_pairs = meter.call("cold_edge_csv_read", lambda: pd.read_csv(root / "raw/edge.csv.gz", compression="gzip", header=None).values.T.astype(np.int64))
    # Exact native reciprocal interleaving, before any graph coalescing.
    def reciprocal_interleaving():
        expanded = np.repeat(raw_pairs, 2, axis=1)
        expanded[0, 1::2] = expanded[1, 0::2]
        expanded[1, 1::2] = expanded[0, 0::2]
        return expanded
    raw_edge_index = meter.call("native_raw_reciprocal_graph_reconstruction", reciprocal_interleaving)
    x, raw = torch.from_numpy(feature_array), torch.from_numpy(raw_edge_index)
    require(tuple(x.shape) == (235868, 128) and x.dtype == torch.float32, "Raw all-node features differ")
    require(tuple(raw.shape) == (2, 2358104) and raw.dtype == torch.long, "Raw native graph order/shape differs")
    require(bool(torch.isfinite(x).all()), "Nonfinite raw features")
    require(int(pairs.min()) >= 0 and int(pairs.max()) < len(x), "TRAIN endpoint out of range")
    require(int(raw.min()) >= 0 and int(raw.max()) < len(x), "Raw endpoint out of range")
    digests = meter.call("data_tensor_hash_and_transfer", lambda: {"train_records": tensor_sha(pairs), "raw_features": tensor_sha(x), "ordered_raw_graph": tensor_sha(raw)})
    require(digests == context["bindings"]["data_tensor_digests"], "Acquired tensor custody identity differs")
    # A multiset check includes duplicate records, unlike an edge-set check.
    raw_records = torch.from_numpy(raw_pairs.T.copy())
    def canonical_keys(records):
        a, b = records.min(1).values, records.max(1).values
        return (a * len(x) + b).sort().values
    same_records = meter.call("train_raw_record_multiset_check", lambda: torch.equal(canonical_keys(pairs), canonical_keys(raw_records)))
    require(same_records, "Raw graph and official TRAIN record multisets differ")
    del raw_records, raw_pairs, raw_edge_index, feature_array
    return {"x": x, "pairs": pairs, "raw_edge_index": raw, "digests": digests}


def transfer_data(data, device, meter):
    meter.counts["explicit_host_to_device_tensor_bytes"] += sum(data[name].numel() * data[name].element_size() for name in ("x", "pairs", "raw_edge_index"))
    return meter.call("initial_full_data_host_to_device", lambda: {**data,
        "x": data["x"].to(device), "pairs": data["pairs"].to(device),
        "raw_edge_index": data["raw_edge_index"].to(device)})


def epoch_stream(data, native_utils, sampler, meter):
    # Preserve native call order and default arguments. No count override,
    # undirected switch, future-positive filter, or fixed BUDDY negative stream.
    negatives = meter.call("native_epoch_negative_sampling", sampler, data["raw_edge_index"], len(data["x"]))
    require(negatives.dtype == torch.long and negatives.ndim == 2 and negatives.shape[0] == 2, "Native negative sampler shape differs")
    require(negatives.shape[1] >= len(data["pairs"]), "Native default negative draw cannot index every TRAIN record")
    def validate_negatives():
        require(int(negatives.min()) >= 0 and int(negatives.max()) < len(data["x"]), "Native negative endpoint out of range")
        require(bool((negatives[0] != negatives[1]).all()), "Native negative sampler returned self-links")
        keys = data["raw_edge_index"][0] * len(data["x"]) + data["raw_edge_index"][1]
        keys = keys.sort().values
        sampled = negatives[0] * len(data["x"]) + negatives[1]
        index = torch.searchsorted(keys, sampled)
        forbidden = (index < len(keys)) & (keys[index.clamp(max=len(keys)-1)] == sampled)
        require(not bool(forbidden.any()), "Native sampler returned forbidden TRAIN edges")
    meter.call("native_negative_guard", validate_negatives)
    iterator = meter.call("native_train_permutation", native_utils.PermIterator, data["pairs"].device, len(data["pairs"]), 65536)
    require(len(iterator) == 17 and len(iterator.idx) == 1179052, "Native epoch coverage differs")
    meter.counts["explicit_device_to_host_tensor_bytes"] += negatives.numel() * negatives.element_size() + (len(iterator.idx) + 64940) * iterator.idx.element_size()
    identities = meter.call("native_stream_hashes_and_host_transfer", lambda: {"negative_draw_sha256": tensor_sha(negatives), "permutation_sha256": tensor_sha(iterator.idx), "dropped_tail_sha256": tensor_sha(iterator.idx[17 * 65536:])})
    return negatives.T, iterator, {**identities, "negative_rows_drawn": len(negatives.T), "full_batches": 17,
        "supervised_records": 1114112, "dropped_tail_records": 64940, "all_positive_coverage": False}
