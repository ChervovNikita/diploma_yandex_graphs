"""Standard-library admission and source checks, before numerical imports."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_file(path, pin):
    path = Path(path)
    require(path.is_file(), "Required pinned file absent: " + str(path))
    if "bytes" in pin:
        require(path.stat().st_size == pin["bytes"], "Pinned byte count differs: " + str(path))
    require(file_sha(path) == pin["sha256"], "Pinned file digest differs: " + str(path))
    return path


def verify_preparation():
    manifest = read_json(HERE / "MANIFEST.json")
    listed = set()
    for pin in manifest["files"]:
        relative = Path(pin["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid sealed relative path")
        checked_file(HERE / relative, pin)
        listed.add(str(relative))
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob("*.py")}
    require(actual <= listed, "Unsealed Python source present")
    return file_sha(HERE / "MANIFEST.json")


def verify_external_sources(research_root, bindings):
    research_root = Path(research_root).resolve()
    for pin in bindings["external_metadata_and_source_files"]:
        checked_file(research_root / pin["path"], pin)
    prototype = research_root / bindings["prototype_directory"]
    manifest_path = checked_file(prototype / "MANIFEST.json", {"sha256": bindings["prototype_manifest_sha256"]})
    for pin in read_json(manifest_path)["files"]:
        relative = Path(pin["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid prototype member path")
        checked_file(prototype / relative, pin)
    qa = read_json(research_root / bindings["cpu_qualification_path"])
    require(qa["status"] == "ENGINEERING_QUALIFICATION_PASSED", "CPU engineering qualification not passed")
    require(len(qa["tests"]) == 8 and all(row["status"] == "PASS" for row in qa["tests"]), "Incomplete CPU engineering certificate")
    supplement = read_json(research_root / bindings["supplemental_oracles_path"])
    require(supplement["status"] == "SUPPLEMENTAL_ENGINEERING_ORACLES_PASSED", "Supplemental oracles not passed")
    require(supplement["private_factor_analytic_gradient_oracle"] and supplement["shared_weight_and_bias_analytic_gradient_oracle"] and supplement["twice_balanced_BCE_loss_and_parameter_gradient_oracle"], "Missing factor/loss numerical oracle")
    require(supplement["loss_parameter_gradients_checked"] == 41, "Supplemental gradient coverage differs")
    for certificate in (qa, supplement):
        require(certificate["prototype_manifest_sha256"] == bindings["prototype_manifest_sha256"], "Engineering evidence belongs to another prototype")
    require(supplement["native_float64_encoder_parity_claim"] is False, "Unsupported native float64 encoder claim")
    return prototype


def preflight(admission_path, stage, output_path):
    started = time.perf_counter()
    manifest_sha = verify_preparation()
    bindings = read_json(HERE / "BINDINGS.json")
    admission = read_json(admission_path)
    require(admission["schema"] == "ncnc-collab-root-resource-admission-v2", "Wrong root admission schema")
    require(admission["status"] == "ROOT_ADMITTED" and admission["stage"] == stage, "Stage is not admitted by root")
    require(admission["preparation_manifest_sha256"] == manifest_sha, "Root admitted different preparation bytes")
    require(admission["scope"] == "collab_complete_train_resource_only", "Wrong root data scope")
    require(admission["root_owned_device_admission"] is True, "No root device scheduling admission")
    require(admission["scientific_predictive_execution"] is False, "This driver is resource engineering only")
    require(sys.platform == "linux", "The admitted CUDA/host-RSS execution profile is Linux")
    require(isinstance(admission["engineering_rng_identity"], str) and admission["engineering_rng_identity"], "Missing root engineering RNG identity")
    seed = admission["engineering_rng_seed"]
    require(type(seed) is int and 0 <= seed < 2**32, "Root must supply an explicit engineering seed")
    interpreter = Path(sys.executable).resolve()
    require(interpreter == Path(admission["interpreter_path"]).resolve(), "Root interpreter path differs")
    checked_file(interpreter, {"sha256": admission["interpreter_sha256"]})
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == admission["cuda_visible_devices"], "GPU visibility differs from root admission")
    require(bool(admission["cuda_visible_devices"]) and "," not in admission["cuda_visible_devices"], "Require one root-isolated visible CUDA device")
    require(admission["cuda_logical_device"] == 0, "Require the single visible logical device0")
    required_versions = {"torch", "torch-geometric", "torch-sparse", "torch-scatter", "numpy", "pandas"}
    require(required_versions <= set(admission["distribution_versions"]), "Incomplete root runtime version binding")
    versions = {name: importlib.metadata.version(name) for name in admission["distribution_versions"]}
    require(versions == admission["distribution_versions"], "Runtime distribution versions differ")
    source_modules = {pin["module"] for pin in admission["runtime_source_pins"]}
    require({"torch_sparse", "torch_scatter", "torch_geometric.nn.conv.gcn_conv"} <= source_modules, "Missing root sparse/GCN runtime source pins")
    for pin in admission["runtime_source_pins"]:
        checked_file(pin["path"], pin)
    require(admission["runtime_binary_files"] == bindings["observed_runtime_binary_files"], "Root binary admission differs from observed sparse/scatter custody")
    for pin in admission["runtime_binary_files"]:
        checked_file(pin["path"], pin)
    require(admission["negative_sampler"]["qualified_native_default_semantics"] is True, "Native sampler source must be reviewed by root")
    require(admission["negative_sampler"]["sha256"] == bindings["static_native_sampler_metadata"]["source_sha256"], "Native sampler differs from static installed source custody")
    require(admission["negative_sampler"]["function_sha256"] == bindings["observed_sampler_function_sha256"], "Native sampler function differs from observed runtime metadata")
    checked_file(admission["negative_sampler"]["path"], admission["negative_sampler"])
    prototype = verify_external_sources(admission["research_root"], bindings)
    require(Path(admission["dataset_root"]).resolve() == Path(bindings["dataset_root"]).resolve(), "Use exact existing TRAIN/features custody")
    data_pins = bindings["allowed_data_files"]
    require(set(data_pins) == {"split/time/train.pt", "raw/node-feat.csv.gz", "raw/edge.csv.gz"}, "Data allowlist changed")
    custody_hash_seconds = time.perf_counter()
    for relative, pin in data_pins.items():
        checked_file(Path(admission["dataset_root"]) / relative, pin)
    custody_hash_seconds = time.perf_counter() - custody_hash_seconds
    output = Path(output_path).resolve()
    require(output == Path(admission["output_directory"]).resolve(), "Output differs from root admission")
    require(not output.exists(), "Refuse to overwrite output directory")
    parity = None
    if stage == "resource_epoch":
        require("gpu_parity_receipt" in admission, "Full epoch requires admitted GPU parity receipt")
        pin = admission["gpu_parity_receipt"]
        parity = read_json(checked_file(pin["path"], pin))
        require(parity["schema"] == "ncnc-collab-TRAIN-resource-qualification-v2", "GPU parity receipt schema differs")
        require(parity["status"] == "GPU_TRAIN_PARITY_PASSED", "GPU TRAIN parity has not passed")
        require(parity["preparation_manifest_sha256"] == manifest_sha, "GPU parity uses other preparation")
        require(parity["prototype_manifest_sha256"] == bindings["prototype_manifest_sha256"], "GPU parity uses other prototype")
        require(parity["engineering_rng_identity"] == admission["engineering_rng_identity"] and parity["engineering_rng_seed"] == seed, "GPU parity RNG admission differs")
        require(parity["native_batch_size"] == 65536 and parity["node_count"] == 235868 and parity["train_records"] == 1179052, "GPU parity did not use full graph/native batch")
        require(parity["complete_graph_directed_entries"] == 1935264, "GPU parity original TRAIN topology count differs")
        require(parity["data_tensor_digests"] == bindings["data_tensor_digests"], "GPU parity data differs")
        require(parity["all_required_checks_passed"] is True, "GPU parity checks incomplete")
        from qualification_status import qualification_succeeded, validate_accounting
        require(qualification_succeeded(parity), "GPU parity final accounting/status is not qualified")
        validate_accounting(parity["setup_or_GPU_parity_accounting"])
        validate_accounting(parity["final_process_accounting"])
    output.mkdir(parents=True, mode=0o700)
    return {"admission": admission, "bindings": bindings, "prototype": prototype, "output": output,
            "gpu_parity": parity, "manifest_sha256": manifest_sha,
            "admission_sha256": file_sha(admission_path), "versions": versions,
            "preflight_seconds": time.perf_counter() - started,
            "custody_hash_seconds": custody_hash_seconds}


def import_prototype(path):
    require(not any(name in sys.modules for name in ("prototype", "graph_ops", "native_reference", "selected_state")), "Refuse preloaded prototype namespace")
    sys.path.insert(0, str(path))
    import prototype
    import graph_ops
    import native_reference
    import selected_state
    for module in (prototype, graph_ops, native_reference, selected_state):
        require(Path(module.__file__).resolve().parent == path.resolve(), "Imported prototype path differs")
    return prototype, graph_ops, native_reference, selected_state


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
