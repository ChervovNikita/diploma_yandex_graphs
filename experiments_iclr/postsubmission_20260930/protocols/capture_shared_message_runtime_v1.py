"""Read current authorized interpreter/GPU metadata without Torch or data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from propagation_cost_engineering_v1.contract import live_context, within
    live = live_context(args.supervisor, Path(__file__))
    output = within(args.output, PHASE / "protocols")
    if output.exists():
        raise FileExistsError(output)
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = [row.split(",") for row in query.stdout.strip().splitlines()]
    if len(rows) != 1:
        raise ValueError("Only one explicitly authorized GPU is in scope")
    values = [value.strip() for value in rows[0]]
    if values[1] != "GPU-44039938-fd82-41d2-fefd-de71514e2fac":
        raise ValueError("Authorized device identity differs")
    receipt = {"schema": "gnnm-current-shared-message-runtime-metadata-v1",
               "UTC": datetime.now(timezone.utc).isoformat(), "live": live,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "python": platform.python_version(), "interpreter": sys.executable,
               "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "torch-geometric")},
               "gpu": dict(zip(("index", "uuid", "name", "driver", "memory_used_MiB", "memory_total_MiB", "utilization_percent"), values)),
               "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
               "Torch_or_CUDA_imported": False, "dataset_or_target_read": False,
               "scientific_fit_or_quality_score": False}
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"schema": receipt["schema"], "package_versions": receipt["package_versions"], "gpu": receipt["gpu"]}))


if __name__ == "__main__":
    main()
