"""Export the existing Roman loader's processed edge order for an audit."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from datasets import load_dataset  # noqa: E402

data, train_masks, val_masks, test_masks, *_ = load_dataset(
    "roman-empire", add_self_loops=True, device="cpu", data_dir=str(repo / "data")
)
edge = np.ascontiguousarray(data.edge_index.numpy(), dtype=np.int64)
out = repo / "experiments_iclr" / "roman_dgl_processed_edge_index.npy"
np.save(out, edge, allow_pickle=False)

def array_sha(value: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()

record = {
    "source": "datasets.load_dataset('roman-empire', add_self_loops=True, device='cpu')",
    "edge_shape": list(edge.shape),
    "edge_dtype": str(edge.dtype),
    "edge_array_sha256": array_sha(edge),
    "edge_file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "features_shape": list(data.x.shape),
    "features_sha256": array_sha(data.x.numpy()),
    "train_mask_shape": list(train_masks.shape),
    "train_mask_sha256": array_sha(train_masks.numpy()),
    "valid_mask_sha256": array_sha(val_masks.numpy()),
    "test_mask_sha256": array_sha(test_masks.numpy()),
}
meta = out.with_suffix(".json")
meta.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
