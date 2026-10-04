#!/usr/bin/env python3
"""Prospective CPU-only public feature equality check; requires root admission."""
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path
import pickle
import platform
import socket
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
PHASE = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930")
FEATURE = PHASE / "pubmed_heart_official_acquisition_server_20261004_v2/available/pubmed/gnn_feature"
FEATURE_SHA256 = "c895f9e8e2d96eae8d610e7be8740f77fe1cb6b40e02a73046b2f02aa63a8dd5"
PUBLIC_IDENTITIES = ROOT / "RAW_FEATURE_PUBLIC_INPUT_IDENTITIES.json"
RAW_ALLOWLIST = {"ind.pubmed.allx", "ind.pubmed.tx", "ind.pubmed.test.index"}
NODES, COLUMNS = 19717, 500


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def server_guard():
    phase = PHASE.resolve(strict=True)
    if platform.system() != "Linux" or socket.gethostname() != "peptide" or not Path(__file__).resolve().is_relative_to(phase):
        raise RuntimeError("Run only inside the pinned peptide phase")
    if FEATURE.is_symlink() or not FEATURE.resolve(strict=True).is_relative_to(phase):
        raise RuntimeError("Supplied feature path is unqualified")
    return phase


def check_admission(job_path, inspection_path):
    phase = server_guard()
    for path in (job_path, inspection_path):
        if not path.resolve(strict=True).is_relative_to(phase):
            raise ValueError("Review or inspection receipt leaves phase")
    job = json.loads(job_path.read_text())
    inspection_receipt = json.loads(inspection_path.read_text())
    if job.get("source_review_approved") is not True or job.get("script_sha256") != sha256(Path(__file__)):
        raise ValueError("Exact root source review is absent")
    if job.get("public_input_identities_sha256") != sha256(PUBLIC_IDENTITIES):
        raise ValueError("Public input identities changed/unreviewed")
    if job.get("inspection_receipt_sha256") != sha256(inspection_path):
        raise ValueError("Inspection receipt changed/unreviewed")
    if inspection_receipt.get("available_geometry_checks_pass") is not True:
        raise ValueError("Available-input inspection did not pass")
    feature = inspection_receipt.get("feature", {})
    if feature.get("shape") != [NODES, COLUMNS] or feature.get("dtype") != "torch.float32" or feature.get("finite") is not True:
        raise ValueError("Prospective 19717x500 finite float32 condition is absent")
    if inspection_receipt.get("files", {}).get("gnn_feature", {}).get("sha256") != FEATURE_SHA256 or sha256(FEATURE) != FEATURE_SHA256:
        raise ValueError("Supplied feature identity differs")
    return job, inspection_receipt


def acquire_raw_features(directory):
    identities = json.loads(PUBLIC_IDENTITIES.read_text())
    entries = identities["feature_inputs"]
    if len(entries) != 3 or {Path(item["path"]).name for item in entries} != RAW_ALLOWLIST:
        raise ValueError("Public feature input allowlist differs")
    paths, receipts = {}, []
    for item in entries:
        name = Path(item["path"]).name
        expected_url = "https://raw.githubusercontent.com/kimiyoung/planetoid/" + identities["planetoid_commit"] + "/data/" + name
        if item["public_url"] != expected_url or not (0 < item["size"] <= 8_000_000):
            raise ValueError("Public feature input URL/size differs")
        with urlopen(Request(expected_url, headers={"User-Agent": "Pubmed-feature-equivalence-check"}), timeout=45) as response:
            payload = response.read(item["size"] + 1)
        git_blob_sha1 = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if len(payload) != item["size"] or git_blob_sha1 != item["sha"]:
            raise ValueError("Public raw feature Git blob identity mismatch: " + name)
        path = directory / name
        with path.open("xb") as stream:
            stream.write(payload)
        paths[name] = path
        receipts.append({"name": name, "public_url": expected_url, "bytes": len(payload),
                         "git_blob_sha1": git_blob_sha1, "sha256": sha256(path)})
    return paths, receipts, identities


class RestrictedFeatureUnpickler(pickle.Unpickler):
    """Only authenticated CSR matrices and their NumPy storage, no fallback."""
    def find_class(self, module, name):
        import numpy as np
        from scipy.sparse import csr_matrix
        allowed = {
            ("scipy.sparse.csr", "csr_matrix"): csr_matrix,
            ("scipy.sparse._csr", "csr_matrix"): csr_matrix,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unapproved raw feature pickle global: " + module + "." + name)
        return allowed[(module, name)]


def reconstruct_features(paths):
    import numpy as np
    import torch
    from scipy.sparse import csr_matrix
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Native torch.Tensor float32 default differs")
    matrices = []
    for name in ("ind.pubmed.allx", "ind.pubmed.tx"):
        matrix = RestrictedFeatureUnpickler(io.BytesIO(paths[name].read_bytes()), encoding="latin1").load()
        if not isinstance(matrix, csr_matrix) or matrix.ndim != 2 or matrix.shape[1] != COLUMNS or not (0 < matrix.shape[0] <= NODES):
            raise ValueError("Raw public CSR feature schema differs")
        matrix.check_format(full_check=True)
        if not np.isfinite(matrix.data).all():
            raise ValueError("Raw feature values are not finite")
        # Pinned PyG read_file: todense() followed by torch.Tensor(out).
        matrices.append(torch.Tensor(matrix.todense()))
    allx, tx = matrices
    index_text = paths["ind.pubmed.test.index"].read_text()
    if not index_text.endswith("\n"):
        raise ValueError("Pinned PyG node-order text parser requires final newline")
    # Pinned read_txt_array/parse_txt_array with dtype=torch.long.
    index = torch.tensor([[int(value) for value in line.split()] for line in index_text.split("\n")[:-1]]).to(torch.long).squeeze()
    sorted_index = index.sort()[0]
    if index.ndim != 1 or allx.size(0) + tx.size(0) != NODES or index.numel() != tx.size(0) or not torch.equal(sorted_index, torch.arange(allx.size(0), NODES)):
        raise ValueError("Public node-order index/population differs")
    # Pinned Pubmed non-NELL, non-CiteSeer feature branch; no labels or graph.
    raw = torch.cat([allx, tx], dim=0)
    raw[index] = raw[sorted_index]
    if tuple(raw.shape) != (NODES, COLUMNS) or raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
        raise ValueError("Reconstructed feature schema differs")
    return raw, {"allx_shape": list(allx.shape), "tx_shape": list(tx.shape), "node_order_index_rows": index.numel(),
                 "ordering": "cat(allx,tx); x[node_order_index]=x[sort(node_order_index)]"}


def compare(supplied, reference):
    import numpy as np
    import torch
    first, second = supplied.contiguous().numpy(), reference.contiguous().numpy()
    return {"equal_float32_values": bool(torch.equal(supplied, reference)),
            "equal_float32_bits": bool(np.array_equal(first.view(np.uint32), second.view(np.uint32))),
            "mismatched_float32_values": int(torch.count_nonzero(supplied != reference)),
            "max_absolute_difference": float((supplied - reference).abs().max()),
            "reference_dense_C_order_sha256": hashlib.sha256(second.tobytes(order="C")).hexdigest()}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--inspection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase = server_guard()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(phase) or not output.parent.is_dir():
        raise ValueError("Use one fresh output inside phase")
    job, inspection_receipt = check_admission(args.job, args.inspection_receipt)
    import numpy as np
    import scipy
    import torch
    torch.set_num_threads(1)
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError("Require weights_only=True; no supplied-feature pickle fallback")
    payload = torch.load(FEATURE, map_location="cpu", weights_only=True)
    supplied = payload.get("entity_embedding") if isinstance(payload, dict) else None
    if not isinstance(supplied, torch.Tensor) or supplied.device.type != "cpu" or supplied.layout != torch.strided or tuple(supplied.shape) != (NODES, COLUMNS) or supplied.dtype != torch.float32 or not bool(torch.isfinite(supplied).all()):
        raise ValueError("Supplied CPU feature schema differs from inspection")
    output.mkdir()
    paths, acquisitions, identities = acquire_raw_features(output)
    raw, reconstruction = reconstruct_features(paths)
    raw_comparison = compare(supplied, raw)
    # Explicit PyG 2.2.0 NormalizeFeatures alternative, fixed before comparison.
    normalized = raw - raw.min()
    normalized.div_(normalized.sum(dim=-1, keepdim=True).clamp_(min=1.))
    normalized_comparison = compare(supplied, normalized)
    receipt = {"schema": "pubmed_public_raw_feature_equivalence_v1", "supplied_feature_sha256": FEATURE_SHA256,
               "shape": [NODES, COLUMNS], "dtype": "torch.float32", "device": "cpu", "torch_threads": 1,
               "pyg_source_commit": identities["pyg_commit"], "raw_source_commit": identities["planetoid_commit"],
               "public_raw_feature_acquisitions": acquisitions, "reconstruction": reconstruction,
               "comparisons": {"raw_Planetoid_float32": raw_comparison, "PyG_2.2.0_NormalizeFeatures": normalized_comparison},
               "NormalizeFeatures_policy": "subtract global minimum; divide each row by row sum clamped to at least1",
               "native_loader_transform": "Pinned native Pubmed Planetoid call defaults to transform=None; normalization is a separate comparison",
               "literal_exporter_origin": "Unknown; value equivalence does not establish the historical exporter",
               "root_feature_authority_adopted": False, "training_admission": False,
               "scientific_models_or_link_metrics": False, "inclusive_elapsed_seconds": time.monotonic() - started,
               "job_sha256": sha256(args.job), "inspection_receipt_sha256": sha256(args.inspection_receipt),
               "script_sha256": sha256(Path(__file__)), "public_input_identities_sha256": sha256(PUBLIC_IDENTITIES),
               "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__}}
    (output / "FEATURE_EQUIVALENCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
