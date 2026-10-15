"""Pure argv transform for the reviewed reporting entrypoint.

Call inside the reservation-owning backend.start, before its original Popen:
argv = transform(task, output, checkpoint, states, original_argv).
This module never starts a process or overrides backend.start.
"""
import hashlib
import json
from pathlib import Path


def receipt(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ObservedArgvTransform:
    def __init__(self, registry_index, registry_sha256, project_root):
        self.root = Path(project_root).resolve()
        self.index_path = Path(registry_index).resolve()
        if not self.index_path.is_relative_to(self.root) or receipt(self.index_path)["sha256"] != registry_sha256:
            raise ValueError("Pinned own reporting registry required")
        self.index = json.loads(self.index_path.read_text())
        self.presets = {Path(row["path"]).stem: row for row in self.index["preset_files"]}
        self.entry = receipt(Path(__file__).with_name("runtime_entry.py"))

    def __call__(self, task, output, checkpoint, states, original_argv):
        if task["worker"] != "allocation/gpu0":
            raise ValueError("This bridge is allocation-only; no77authorization or portability")
        preset_pin = self.presets[task["cell_id"]]
        if receipt(preset_pin["path"]) != preset_pin or receipt(self.entry["path"]) != self.entry:
            raise ValueError("Frozen reporting source/registry changed")
        preset = json.loads(Path(preset_pin["path"]).read_text())
        spec = preset["fit_specs"][task["id"]] if task["kind"] == "fit" else preset["I4_specs"][task["id"]]
        execution = task["binding"]["execution"]
        expected_entry = spec["runtime_entrypoint"] if task["kind"] == "fit" else spec["aggregator_source"]
        if execution["runtime_entrypoint"] != expected_entry:
            raise ValueError("Admitted underlying runtime differs from reporting profile")
        if task["kind"] == "fit" and execution["job_descriptor"] != spec["descriptor"]:
            raise ValueError("Admitted descriptor differs from the actual compiler identity")
        if not original_argv or original_argv[0] != execution["python_executable"] or execution["runtime_entrypoint"]["path"] not in original_argv:
            raise ValueError("Original reservation-owned argv does not match its admitted execution")
        argv = [original_argv[0], "-B", "-u", self.entry["path"], "--project-root", str(self.root),
            "--preset-file", preset_pin["path"], "--preset-sha256", preset_pin["sha256"], "--task-id", task["id"], "--output", str(output)]
        if task["kind"] == "fit":
            if checkpoint:
                if receipt(checkpoint["path"]) != checkpoint:
                    raise ValueError("Own selected resume authority changed")
                argv += ["--resume", checkpoint["path"]]
        else:
            if task["dependencies"] != spec["component_fit_ids"]:
                raise ValueError("I4 component identities/order changed")
            members = [states[key]["selected_VALID_export"] for key in task["dependencies"]]
            for member in members:
                if receipt(member["path"]) != member:
                    raise ValueError("I4 member selected export changed")
            argv += ["--members", *[member["path"] for member in members]]
        return argv
