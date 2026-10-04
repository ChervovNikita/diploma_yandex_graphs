"""Authenticated direct TRAIN/raw/VALID reads. No TEST path or split accessor."""
from pathlib import Path
from hashlib import sha256
import codecs
from common import DATA_FILES, sha as file_sha, require


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = sha256(str((array.shape, array.dtype)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def load_data(authority):
    import numpy as np
    import pandas as pd
    import torch
    root = Path(authority["dataset_root"]).resolve()
    for relative in DATA_FILES:
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Data path escaped authority")
        pin = authority["files"][relative]
        require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Data custody differs: " + relative)
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.int64)), codecs.encode]
    def read_split(name):
        with torch.serialization.safe_globals(allowed):
            stored = torch.load(root / ("split/time/" + name + ".pt"), map_location="cpu", weights_only=True)
        specs = authority["expected_arrays"][name]
        require(type(stored) is dict and set(stored) == set(specs), "Split keys differ: " + name)
        result = {}
        for key, value in stored.items():
            spec = specs[key]
            require(type(value) is np.ndarray and str(value.dtype) == spec["dtype"] and list(value.shape) == spec["shape"], "Split array contract differs")
            tensor = torch.from_numpy(value)
            require(tensor_sha(tensor) == spec["sha256"], "Split tensor identity differs")
            result[key] = tensor
        return result
    train, valid = read_split("train"), read_split("valid")
    require(set(train) == {"edge", "weight", "year"} and set(valid) == {"edge", "weight", "year", "edge_neg"}, "Official collab split fields differ")
    pairs = train["edge"]
    require(tuple(pairs.shape) == (1179052, 2) and pairs.dtype == torch.long, "Complete TRAIN records differ")
    require(int(train["year"].max()) == 2017 and bool((valid["year"] == 2018).all()), "Official temporal authority differs")
    x = torch.from_numpy(pd.read_csv(root / "raw/node-feat.csv.gz", compression="gzip", header=None).values.astype(np.float32))
    raw = pd.read_csv(root / "raw/edge.csv.gz", compression="gzip", header=None).values.T.astype(np.int64)
    expanded = np.repeat(raw, 2, axis=1)
    expanded[0, 1::2] = expanded[1, 0::2]
    expanded[1, 1::2] = expanded[0, 0::2]
    raw_edges = torch.from_numpy(expanded)
    require(tuple(x.shape) == (235868, 128) and tuple(raw_edges.shape) == (2, 2358104), "Complete node/raw graph shape differs")
    require(bool(torch.isfinite(x).all()), "Nonfinite official features")
    digests = {"train_records": tensor_sha(pairs), "raw_features": tensor_sha(x), "ordered_raw_graph": tensor_sha(raw_edges)}
    require(digests == authority["train_raw_tensor_digests"], "Existing TRAIN/raw tensor custody differs")
    require(int(pairs.min()) >= 0 and int(pairs.max()) < len(x) and int(raw_edges.min()) >= 0 and int(raw_edges.max()) < len(x), "TRAIN/raw endpoint out of range")
    original = torch.from_numpy(raw.T.copy())
    def keys(edges):
        lo, hi = edges.min(1).values, edges.max(1).values
        return (lo * len(x) + hi).sort().values
    require(torch.equal(keys(original), keys(pairs)), "Raw/official TRAIN record multiset differs")
    for query in (valid["edge"], valid["edge_neg"]):
        require(query.dtype == torch.long and query.ndim == 2 and query.shape[1] == 2 and len(query) > 0, "VALID query shape differs")
        require(int(query.min()) >= 0 and int(query.max()) < len(x), "VALID endpoint out of range")
    require(len(valid["edge_neg"]) >= 50, "Official VALID negative pool incomplete")
    return {"x": x, "raw_edge_index": raw_edges, "split_edge": {"train": train, "valid": valid},
            "digests": {**digests, "valid_positive": tensor_sha(valid["edge"]), "valid_negative": tensor_sha(valid["edge_neg"])}}
