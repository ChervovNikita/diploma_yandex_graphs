"""Canonical, content-addressed contracts for the spectral v2 prototype."""

import base64
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
SCHEMA = "gnnm-output-spectral-neighbor-v2"
CODE_FILES = ("contracts.py", "adapters.py", "spectral_core.py", "collector.py", "replay.py")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def json_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_bytes(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    metadata = {"dtype": str(value.dtype), "shape": list(value.shape),
                "byteorder": sys.byteorder}
    return metadata, raw


def tensor_fingerprint(tensor):
    metadata, raw = tensor_bytes(tensor)
    return metadata | {"sha256": hashlib.sha256(canonical_bytes(metadata) + raw).hexdigest()}


def pack_float64(tensor):
    value = tensor.detach().cpu().to(torch.float64).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Only finite float64 evidence can be serialized")
    metadata, raw = tensor_bytes(value)
    return metadata | {
        "sha256": hashlib.sha256(canonical_bytes(metadata) + raw).hexdigest(),
        "encoding": "base64-raw-contiguous",
        "data": base64.b64encode(raw).decode("ascii"),
    }


def unpack_float64(payload):
    if set(payload) != {"dtype", "shape", "byteorder", "sha256", "encoding", "data"}:
        raise ValueError("Unexpected tensor payload fields")
    if (payload["dtype"] != "torch.float64" or payload["byteorder"] != sys.byteorder
            or payload["encoding"] != "base64-raw-contiguous"):
        raise ValueError("Unsupported dtype, byte order, or tensor encoding")
    shape = payload["shape"]
    if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError("Invalid tensor shape")
    raw = base64.b64decode(payload["data"], validate=True)
    if len(raw) != 8 * math.prod(shape):
        raise ValueError("Tensor byte count disagrees with shape")
    metadata = {key: payload[key] for key in ("dtype", "shape", "byteorder")}
    if hashlib.sha256(canonical_bytes(metadata) + raw).hexdigest() != payload["sha256"]:
        raise ValueError("Tensor content hash mismatch")
    if not raw:
        result = torch.empty(shape, dtype=torch.float64)
    else:
        result = torch.frombuffer(bytearray(raw), dtype=torch.float64).clone().reshape(shape)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite decoded tensor")
    return result


def state_hash(model):
    return json_hash({name: tensor_fingerprint(value)
                      for name, value in model.state_dict().items()})


def code_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in CODE_FILES}


def seal_record(record):
    result = {key: value for key, value in record.items() if key != "record_sha256"}
    result["record_sha256"] = json_hash(result)
    return result


def verify_seal(record):
    if record.get("schema") != SCHEMA:
        raise ValueError("Unsupported spectral record schema")
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != json_hash(payload):
        raise ValueError("Record content hash mismatch")


def save_new_json(value, path, artifact_root=HERE):
    """Never overwrite evidence; a caller may explicitly set its study root."""
    path, root = Path(path).resolve(), Path(artifact_root).resolve()
    if root != path.parent and root not in path.parents:
        raise ValueError("Output must remain within the explicit artifact root")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
