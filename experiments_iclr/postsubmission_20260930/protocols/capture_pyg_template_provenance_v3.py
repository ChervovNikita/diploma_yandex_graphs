"""Read installed PyG generation source and project-only SAGEConv temp code.

Standard-library metadata only: no Torch/PyG import, target or model access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
NAMES = ("torch_geometric.nn.conv.sage_conv_SAGEConv_propagate",
         "torch_geometric.nn.conv.sage_conv_SAGEConv_edge_updater")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def confined(path):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(PHASE) or path == PHASE:
        raise ValueError("Confine project evidence to the phase")
    current = PHASE
    for item in path.relative_to(PHASE).parts:
        current /= item
        if current.is_symlink():
            raise ValueError("No symlink evidence paths")
    return path.resolve()


def read(path):
    a = path.stat()
    if a.st_size > 512 * 1024:
        raise ValueError("Source metadata is bounded to 512 KiB per file")
    data = path.read_bytes()
    b = path.stat()
    if (a.st_ino, a.st_size, a.st_mtime_ns, a.st_ctime_ns) != (b.st_ino, b.st_size, b.st_mtime_ns, b.st_ctime_ns):
        raise ValueError("Source changed during read")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    args = p.parse_args()
    request_path = confined(args.request)
    request_data = read(request_path)
    request = json.loads(request_data)
    start = json.loads(confined(os.environ["GNNM_BOUND_START_JSON"]).read_text())
    if (request.get("schema") != "gnnm-pyg-generated-source-metadata-request-v1"
            or request.get("source_metadata_only") is not True
            or request.get("Torch_or_PyG_import_authorized") is not False
            or start["root_request"] != {"path": str(request_path.relative_to(PHASE)), "sha256": sha(request_data)}
            or start["inner_supervisor_directory"] != str(confined(args.supervisor))
            or start["whole_cap_seconds"] != request["wall_cap_seconds"]
            or request["entrypoint_sha256"] != sha(Path(__file__).read_bytes())):
        raise ValueError("Require actual bounded source-only root request")
    failed_ref = request["failed_synthetic_terminal"]
    failed_path = confined(PHASE / failed_ref["path"])
    failed_bytes = read(failed_path)
    if sha(failed_bytes) != failed_ref["sha256"]:
        raise ValueError("Failed synthetic envelope changed")
    failed = json.loads(failed_bytes)
    failed_start_bytes = read(failed_path.parent / "START.json")
    if (failed.get("complete") is not False or failed.get("child_exit_code") != 1
            or sha(failed_start_bytes) != failed["START_sha256"]):
        raise ValueError("Require the actual preserved failed synthetic chain")
    failed_start = json.loads(failed_start_bytes)
    lower = datetime.fromisoformat(failed_start["start_UTC"]).timestamp() - 2.0
    upper = datetime.fromisoformat(failed["terminal_UTC"]).timestamp() + 2.0
    output = confined(PHASE / request["output_relative_to_phase"])
    output.mkdir(parents=True, exist_ok=False)
    package = importlib.metadata.distribution("torch-geometric")
    if package.version != "2.7.0":
        raise ValueError("Installed PyG differs from the frozen runtime")
    selected = []
    for item in package.files or []:
        s = str(item)
        if s in ("torch_geometric/template.py", "torch_geometric/inspector.py",
                 "torch_geometric/nn/conv/message_passing.py", "torch_geometric/nn/conv/sage_conv.py") or (
                 s.startswith("torch_geometric/nn/conv/") and item.name in ("propagate.jinja", "edge_updater.jinja", "collect.jinja")):
            selected.append(item)
    if len(selected) != 7:
        raise ValueError("Expected exactly seven PyG generator/template source files")
    installed = []
    for item in selected:
        data = read(Path(package.locate_file(item)))
        target = output / "installed_sources" / str(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(data)
        installed.append({"package_relative_path": str(item), "sha256": sha(data),
                          "bytes": len(data), "copy_relative_to_phase": str(target.relative_to(PHASE))})
    temp_root = confined(os.environ["TMPDIR"])
    generated = []
    for name in NAMES:
        files = sorted(p for p in temp_root.glob(name + "_*.py")
                       if lower <= p.stat().st_mtime <= upper)
        if len(files) > 64:
            raise ValueError("Too many generated candidates for bounded metadata capture")
        for file in files:
            file = confined(file)
            data = read(file)
            target = output / "generated_sources" / file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(data)
            generated.append({"module_name": name, "original_relative_to_phase": str(file.relative_to(PHASE)),
                              "sha256": sha(data), "bytes": len(data),
                              "copy_relative_to_phase": str(target.relative_to(PHASE))})
    if not any(r["module_name"] == NAMES[0] for r in generated):
        raise ValueError("No project-only generated SAGEConv propagate source found")
    report = {"schema": "gnnm-pyg-generated-source-metadata-v1", "UTC": datetime.now(timezone.utc).isoformat(),
              "request": start["root_request"], "entrypoint_sha256": sha(Path(__file__).read_bytes()),
              "supervisor_relative_to_phase": str(confined(args.supervisor).relative_to(PHASE)),
              "outer_START_sha256": sha(confined(os.environ["GNNM_BOUND_START_JSON"]).read_bytes()),
              "package": "torch-geometric", "package_version": package.version,
              "installed_sources": installed, "generated_sources": generated,
              "TMPDIR_relative_to_phase": str(temp_root.relative_to(PHASE)),
              "failed_synthetic_terminal": failed_ref,
              "generated_selection": {"rule": "mtime within actual failed synthetic UTC envelope plus/minus two seconds", "lower_epoch": lower, "upper_epoch": upper},
              "Torch_or_PyG_imported": any(k == "torch" or k == "torch_geometric" for k in sys.modules),
              "dataset_or_target_read": False, "scientific_or_numerical_run": False}
    with (output / "RESULT.json").open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(output), "installed_sources": len(installed),
                      "generated_sources": len(generated), "Torch_or_PyG_imported": report["Torch_or_PyG_imported"]}))


if __name__ == "__main__":
    main()
