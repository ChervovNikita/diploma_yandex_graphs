"""Runtime-only complete acquired train-target and full-P CSR interfaces."""
from .contract import checked


def load_inputs(phase_ref):
    import json
    from pathlib import Path
    from propagation_cost_engineering_v1.data_contract import load_acquired
    from propagation_cost_engineering_v1.prepare import audit_prepared
    phase = json.loads(checked(phase_ref).read_text())
    export, lineage = load_acquired(phase)
    metadata, directory = audit_prepared(Path(phase["repo_root"]) / phase["prepared_manifest_relative_to_repo"], phase, lineage)
    return export, lineage, metadata, directory


def load_family(metadata, directory, arm, *, device="cuda:0"):
    import numpy as np
    import torch
    from propagation_cost_engineering_v1.complete_models import CompleteCSR, NoAdjointCSR, CompleteFamily, OperationLedger
    from .counts import ARMS
    if arm not in ARMS:
        raise ValueError("Unknown shared-message packet arm")
    ledger, matrices = OperationLedger(), []
    directions = ("forward",) if arm == "detached_shared_mean" else ("forward", "adjoint")
    for direction in directions:
        arrays = []
        for suffix in ("rowptr", "columns", "weights"):
            row = metadata["arrays"]["full_%s_%s" % (direction, suffix)]
            file = directory / row["file"]
            stat = file.stat()
            if file.parent != directory or file.is_symlink() or stat.st_size != row["bytes"] or stat.st_mtime_ns != row["verified_mtime_ns"]:
                raise ValueError("Prepared full operator changed after complete hash audit")
            array = np.load(file, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
                raise ValueError("Prepared operator shape/dtype changed")
            arrays.append(torch.tensor(array, device=device))
            del array
        matrices.append(torch.sparse_csr_tensor(*arrays, size=(metadata["nodes"], metadata["nodes"]), dtype=torch.float32, device=device))
    operator = (NoAdjointCSR("full", matrices[0], ledger, metadata["nodes"]) if len(matrices) == 1
                else CompleteCSR("full", *matrices, ledger, metadata["nodes"]))
    return CompleteFamily({"full": operator}, metadata, ledger, device)
