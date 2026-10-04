"""Stdlib-only custody and prospective, model-free release gate."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import importlib.util
import json
import os
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STAGES = ("fabricated", "full_batch", "generate_epoch")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_pin(path, pin):
    path = Path(path).resolve()
    require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Source pin mismatch: " + str(path))
    return path


def load_module(name, path):
    path = Path(path).resolve()
    prior = sys.modules.get(name)
    if prior is not None:
        require(Path(prior.__file__).resolve() == path, "Module shadowed: " + name)
        return prior
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def preflight(release_path, stage, epoch, master_seed, output):
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text())
    require(release.get("schema") == "ncnc-TRAIN-paired-replay-root-release-v1", "Separate replay release required")
    require(release.get("execution_enabled") is True and release.get("root_authorization_reference"), "Source preparation is not execution authority")
    require(stage in STAGES and stage in release.get("authorized_stages", []), "Stage not released")
    require(type(master_seed) is int and 0 <= master_seed < 2**63 and type(epoch) is int and 1 <= epoch <= 100, "Prospective seed/epoch invalid")
    require("torch" not in sys.modules, "Standalone preflight must precede Torch import")
    manifest_sha = file_sha(HERE / "MANIFEST.json")
    require(release.get("provider_manifest_sha256") == manifest_sha, "Release targets another provider")
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for pin in manifest["files"]:
        path = (HERE / pin["path"]).resolve()
        require(path.is_relative_to(HERE), "Provider payload path escaped")
        read_pin(path, pin)
    research = Path(release["research_root"]).resolve()
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    sources = {}
    for pin in bindings["sources"]:
        path = (research / pin["relative_path"]).resolve()
        require(path.is_relative_to(research), "Dependency path escaped")
        sources[pin["key"]] = read_pin(path, pin)
    invocation = {"stage": stage, "epoch": epoch, "master_seed": master_seed, "output_directory": str(Path(output).resolve())}
    require(invocation in release.get("authorized_invocations", []), "Exact invocation not released")
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(HERE), "Fresh output outside source required")
    require(not any((a / "MANIFEST.json").exists() for a in (output, *output.parents)), "Output belongs to a sealed packet")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == release.get("cuda_visible_devices"), "Normally visible device must be root bound before launch")
    require(release.get("runtime_profile") == "ordinary-authenticated-then-V5-deterministic", "Wrong process runtime profile")
    require(release.get("prospective_trace_only") is True and release.get("fits_authorized") is False, "This release can admit only prospective model-free trace work")
    return {"release": release, "release_sha256": file_sha(release_path), "stage": stage, "invocation": invocation,
            "sources": sources, "bindings": bindings, "provider_manifest_sha256": manifest_sha, "output": output}


def qualification_pins(context, identity):
    require(context["stage"] in ("full_batch", "generate_epoch"), "Wrong qualification admission stage")
    required = ("fabricated",) if context["stage"] == "full_batch" else ("fabricated", "full_batch")
    for stage in required:
        pin = context["release"]["qualification"][stage]
        receipt = json.loads(read_pin(pin["path"], pin).read_text())
        require(receipt.get("schema") == "ncnc-TRAIN-replay-qualification-v1" and receipt.get("stage") == stage
                and receipt.get("status") == "PASS" and receipt.get("provider_manifest_sha256") == context["provider_manifest_sha256"], "Missing provider qualification")
        require(receipt.get("runtime_identity") == identity["runtime"] and receipt.get("source_identity") == identity["source"], "Qualification source/runtime differs")
        require(receipt.get("no_model_or_fit") is True and receipt.get("no_VALID_or_TEST") is True, "Qualification scope differs")
        if stage == "full_batch":
            require(receipt.get("TRAIN_identity") == identity["TRAIN"] and receipt.get("complete_batches") == 17
                    and receipt.get("full_integer_stream_equality") is True and receipt.get("full_integer_support_equality") is True,
                    "Complete real TRAIN stream/support qualification required")
