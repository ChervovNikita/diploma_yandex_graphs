"""Measure only current research-phase file storage for resource admission."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat

PHASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PHASE / "protocols") or output.exists():
        raise ValueError("Require new confined phase resource record")
    apparent = allocated = files = skipped_links = 0
    for folder, dirs, names in os.walk(PHASE, followlinks=False):
        for name in dirs[:]:
            if (Path(folder) / name).is_symlink():
                dirs.remove(name)
                skipped_links += 1
        for name in names:
            info = (Path(folder) / name).lstat()
            if stat.S_ISLNK(info.st_mode):
                skipped_links += 1
            elif stat.S_ISREG(info.st_mode):
                apparent += info.st_size
                allocated += info.st_blocks * 512
                files += 1
    usage = shutil.disk_usage(PHASE)
    value = {"schema": "gnnm-confined-phase-disk-measurement-v1",
             "UTC": datetime.now(timezone.utc).isoformat(),
             "phase": str(PHASE), "apparent_regular_file_bytes": apparent,
             "allocated_regular_file_bytes": allocated, "regular_file_count": files,
             "symlink_entries_not_followed": skipped_links,
             "filesystem_free_bytes": usage.free,
             "scope": "Whole phase including earlier studies and failures; conservative charge for new-study disk cap",
             "dataset_contents_read": False, "heldout_targets_read": False,
             "snapshot_is_point_in_time": True}
    with output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"apparent_bytes": apparent, "allocated_bytes": allocated,
                      "files": files, "symlink_entries_not_followed": skipped_links}))


if __name__ == "__main__":
    main()
