"""Create a reviewed final transfer tar, hashing each input while streaming it once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

from sync_common import allocation_guard, checked, fresh, read_json, safe


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def write_bundle(root, rows, directories, output):
    """No scientific I/O or input copies; a failed partial archive is preserved."""
    rows = [checked(row) for row in rows]
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate archive member")
    output = Path(output)
    started = time.monotonic()
    with tarfile.open(output, "x", format=tarfile.PAX_FORMAT) as archive:
        for row in rows:
            source = safe(root, row["path"])
            info = archive.gettarinfo(str(source), arcname=row["path"])
            if not info.isfile() or info.size != row["bytes"]:
                raise ValueError("Artifact size/type changed: " + row["path"])
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o555 if os.access(source, os.X_OK) else 0o444
            with source.open("rb") as stream:
                reader = HashingReader(stream)
                archive.addfile(info, reader)
                if reader.bytes != row["bytes"] or reader.digest.hexdigest() != row["sha256"] or stream.read(1):
                    raise ValueError("Artifact bytes changed while streaming: " + row["path"])
        for relative in directories:
            directory = safe(root, relative, file=False)
            if not directory.is_dir() or any(directory.iterdir()):
                raise ValueError("Declared initial directory changed: " + relative)
            info = tarfile.TarInfo(relative + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.mtime = 0
            archive.addfile(info)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    output.chmod(0o444)
    return {"bytes": output.stat().st_size, "sha256": digest.hexdigest(), "files": len(rows),
            "input_stream_hash_passes_per_path": 1, "archive_hash_passes": 1,
            "wall_seconds": time.monotonic() - started, "tensor_deserialization": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True, help="Fresh project-relative tar output")
    parser.add_argument("--receipt-output", required=True)
    args = parser.parse_args()
    allocation_guard()
    root = Path(args.project_root).absolute()
    manifest, manifest_receipt = read_json(root, args.manifest, sha256=args.manifest_sha256)
    audit, audit_receipt = read_json(root, args.audit, sha256=args.audit_sha256)
    if (manifest.get("scientific_launch_enabled") is not False or manifest.get("GO_created") is not False
            or audit.get("plan_hash") != manifest["plan_hash"] or audit.get("manifest") != manifest_receipt
            or audit.get("all_phase_reads_within_manifest_union") is not True or audit.get("scientific_fits") != 0):
        raise ValueError("Exact disabled manifest source-loading audit required before packaging")
    rows = {row["path"]: checked(row) for row in manifest["files"]}
    for row in (manifest_receipt, audit_receipt):
        if row["path"] in rows and rows[row["path"]] != row:
            raise ValueError("Conflicting metadata archive binding")
        rows[row["path"]] = row
    output = safe(root, args.output, file=False)
    result = write_bundle(root, [rows[k] for k in sorted(rows)], manifest["empty_directories"], output)
    result.update(schema="exact-disabled-final-master-transfer-bundle-v1", path=args.output,
                  plan_hash=manifest["plan_hash"], manifest=manifest_receipt, source_loading_audit=audit_receipt,
                  server77_access=False, package_installs=0, scientific_fits=0, GO_created=False,
                  target_host_runtime_provider_ABI_and_resource_admission_required=True)
    fresh(safe(root, args.receipt_output, file=False), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
