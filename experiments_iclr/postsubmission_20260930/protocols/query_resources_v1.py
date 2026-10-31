"""Record aggregate compute availability without inspecting other users' jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if phase not in output.parents or output.exists():
        raise ValueError("Require a new output within the authorized research phase")
    start = datetime.now(timezone.utc).isoformat()
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"], text=True, capture_output=True, check=False)
    disk = shutil.disk_usage(phase)
    record = {
        "schema": "gnnm-aggregate-resources-v1", "query_start_utc": start,
        "query_end_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_query_exit": query.returncode, "gpu_query_stdout": query.stdout,
        "gpu_query_stderr": query.stderr, "cpu_logical_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)), "load_average": os.getloadavg(),
        "physical_memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        "disk_free_bytes": disk.free, "GPU_context_or_training_created": False,
        "individual_processes_or_other_project_files_inspected": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
