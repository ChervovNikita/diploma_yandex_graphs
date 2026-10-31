"""Prepare sealed official collab TEST queries and separate index/target custody.

Preparation alone may decode the authenticated official split. No model,
optimizer, score, calibration, train accessor or all-split accessor is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import secrets
import socket
import subprocess
import zipfile

ARCHIVE_SHA = "c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):digest.update(block)
    return digest.hexdigest()


def receipt(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError("Confine endpoint source/output custody")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def shuffle_records(torch, numpy, positive, negative):
    """Opaque row IDs; no seed or label-derived position is exposed to export."""
    p, n = len(positive), len(negative)
    nonce = secrets.token_bytes(32)
    order = torch.from_numpy(numpy.random.default_rng(int.from_bytes(nonce, "big")).permutation(p+n)).long()
    ids = torch.arange(p+n, dtype=torch.long)
    pairs = torch.cat((positive, negative), 0)[order].contiguous()
    mask = torch.cat((torch.ones(p, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)))[order]
    original = torch.cat((torch.arange(p), torch.arange(n)))[order].long()
    return {"ids": ids, "pairs": pairs}, {"ids": ids.clone(), "is_positive": mask, "original_index": original}, hashlib.sha256(nonce).hexdigest()


def prepare(*, project_root, archive_path, fit_manifest_path, output):
    root, archive, output = Path(project_root).resolve(), Path(archive_path).resolve(), Path(output).resolve()
    if not archive.is_relative_to(root) or not output.is_relative_to(root) or output == root or output.exists():
        raise ValueError("Use project-confined authenticated archive and fresh sealed output")
    if archive.stat().st_size != 121625147 or file_sha(archive) != ARCHIVE_SHA:raise ValueError("Official archive bytes differ")
    fit = json.loads(Path(fit_manifest_path).read_text())
    if fit.get("schema") != "gnnm-link-collab-fit-data-v1" or fit["input_audit"]["nodes"] != 235868:
        raise ValueError("Bind complete existing collab TRAIN/SELECT custody")
    import numpy as np
    import torch
    with zipfile.ZipFile(archive) as sealed:
        matches = [name for name in sealed.namelist() if name.endswith("/split/time/test.pt")]
        if len(matches) != 1:raise ValueError("Expected one official sealed TEST member")
        raw = sealed.read(matches[0])
    split = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if type(split) is not dict or set(split) != {"edge", "weight", "year", "edge_neg"}:raise ValueError("Official TEST fields differ")
    arrays = {}
    for key, value in split.items():
        if type(value) is not np.ndarray or value.dtype != np.int64:raise ValueError("Official TEST dtype differs")
        arrays[key] = torch.from_numpy(value)
    positive, negative = arrays["edge"], arrays["edge_neg"]
    for pairs in (positive, negative):
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or int(pairs.min()) < 0 or int(pairs.max()) >= 235868:
            raise ValueError("Official TEST endpoint records differ")
    if len(negative) != 100000 or arrays["weight"].shape != (len(positive),) or arrays["year"].shape != (len(positive),):
        raise ValueError("Official TEST shared pool/metadata coverage differs")
    query, target, nonce_hash = shuffle_records(torch, np, positive, negative)
    output.mkdir(parents=True)
    query_path, target_path = output/"EVALUATE_QUERIES.pt", output/"SEALED_EVALUATE_TARGETS.pt"
    torch.save(query, query_path);torch.save(target, target_path)
    manifest = {"schema": "gnnm-link-sealed-evaluate-v1", "schema_version": 1, "UTC": datetime.now(timezone.utc).isoformat(),
        "dataset": "ogbl-collab", "complete": True, "endpoint_inputs": {"evaluate_queries": receipt(root, query_path), "evaluate_targets": receipt(root, target_path)},
        "query_records": len(query["ids"]), "positive_records": len(positive), "negative_records": len(negative), "nodes": 235868,
        "metric": fit["metric"], "fit_manifest": receipt(root, Path(fit_manifest_path)), "preparation_program": receipt(root, Path(__file__)),
        "archive": receipt(root, archive), "archive_url": "https://snap.stanford.edu/ogb/data/linkproppred/collab.zip",
        "official_member": {"name": matches[0], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "official_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()} for key,value in arrays.items()},
        "year_min": int(arrays["year"].min()), "year_max": int(arrays["year"].max()),
        "shuffle": {"algorithm": "secret 256-bit nonce; NumPy permutation; only nonce SHA retained", "nonce_sha256": nonce_hash,
            "opaque_query_ids": True, "positive_negative_source_order_hidden": True},
        "role_protocol": "official TEST records preserved; query-only shuffled export; separate target and original within-pool indices",
        "calibration": "raw_only_absent; no confidence claims", "support_policy": "full_TRAIN_only; temporal pair recurrence preserved",
        "test_decoded_by_preparation_only": True, "training_or_selection_or_export_launched": False,
        "model_constructed": False, "optimizer_created": False, "scoring_performed": False, "scientific_launch_enabled": False,
        "author_protocol_replication": False}
    (output/"MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "archive", "fit-manifest", "output"):parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines() != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise RuntimeError("Sealed preparation requires the authorized allocation")
    value = prepare(project_root=args.project_root, archive_path=args.archive, fit_manifest_path=args.fit_manifest, output=args.output)
    print(json.dumps({key:value[key] for key in ("schema","complete","query_records","positive_records","negative_records","scoring_performed","scientific_launch_enabled")},indent=2))


if __name__ == "__main__":main()
