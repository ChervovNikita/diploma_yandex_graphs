"""Root resource receipt: phase stat inventory and metadata-only artifact audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

PHASE = Path(__file__).resolve().parents[1]


def confined(value):
    p = PHASE / value
    if p.is_absolute() and p.resolve().is_relative_to(PHASE) and not p.is_symlink():
        return p.resolve()
    raise ValueError("Path outside phase or symlink")


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def ref(p):
    return dict(path=str(p.relative_to(PHASE)), sha256=sha(p))


def inventory(root):
    logical = allocated = count = links = 0
    for folder, dirs, names in os.walk(root, followlinks=False):
        for n in dirs[:]:
            if (Path(folder) / n).is_symlink():
                dirs.remove(n)
                links += 1
        for n in names:
            s = (Path(folder) / n).lstat()
            if stat.S_ISLNK(s.st_mode):
                links += 1
            elif stat.S_ISREG(s.st_mode):
                logical += s.st_size
                allocated += s.st_blocks * 512
                count += 1
    return dict(logical_regular_file_bytes=logical, allocated_regular_file_bytes=allocated,
                regular_files=count, skipped_symlinks=links)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--supervisor', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    request_path = confined(args.request)
    r = json.loads(request_path.read_text())
    if r['source_sha256'] != sha(Path(__file__)) or r['root_authorized_metadata_only'] is not True:
        raise ValueError('Root source-bound resource request required')
    outer = Path(os.environ['GNNM_BOUND_START_JSON'])
    s = json.loads(outer.read_text())
    if s['root_request'] != ref(request_path):
        raise ValueError('Outer request differs')
    out = confined(r['output_relative_to_phase'])
    out.mkdir(exist_ok=False)
    audits = []
    for value in r['artifact_manifests']:
        p = confined(value['path'])
        if sha(p) != value['sha256']:
            raise ValueError('Manifest identity differs')
        m = json.loads(p.read_text())
        count = total = 0
        for item in m['files']:
            f = confined(item['path'])
            # These are exclusively artificial metadata outputs, never real tensors.
            if not f.is_relative_to(p.parent) or sha(f) != item['sha256']:
                raise ValueError('Declared metadata artifact differs')
            count += 1
            total += f.stat().st_size
        audits.append(dict(manifest=ref(p), verified_files=count, logical_verified_bytes=total))
    scoped = {value: inventory(confined(value)) for value in r['scoped_directories']}
    phase = inventory(PHASE)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    result = dict(schema='gnnm-shared-message-completed-resource-receipt-v1', UTC=datetime.now(timezone.utc).isoformat(),
                  request=ref(request_path), source=ref(Path(__file__)), outer_start=ref(outer),
                  supervisor_directory=str(args.supervisor.relative_to(PHASE)), metadata_artifact_audits=audits,
                  scoped_disk=scoped, whole_phase_disk=phase, filesystem_free_bytes=shutil.disk_usage(PHASE).free,
                  physical_host_memory_bytes=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'),
                  available_physical_pages_bytes=os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'),
                  host_CPU_affinity_count=len(os.sched_getaffinity(0)), GPU_query=gpu.stdout,
                  elapsed_seconds=time.monotonic()-start,
                  data_targets_predictions_models_or_checkpoints_opened=False,
                  allocation_is_stat_blocks_times512_not_a_filesystem_quota=True,
                  point_in_time_receipt_plus_supervisors_and_future_metadata_need_reserve=True)
    with (out/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='completed_metadata_resource_receipt', verified_files=sum(a['verified_files'] for a in audits),
                          phase_allocated_bytes=phase['allocated_regular_file_bytes'], seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
