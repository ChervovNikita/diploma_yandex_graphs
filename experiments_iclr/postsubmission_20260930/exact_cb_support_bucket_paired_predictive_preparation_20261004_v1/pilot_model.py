"""Deferred qualified model/runtime loading and fixed scientific initialization."""
import importlib
import importlib.util
import sys
from pathlib import Path
from pilot_common import require


def modules(context):
    root = context["paths"]["prototype_root"]
    sys.path.insert(0, str(root))
    names = ("graph_ops", "prototype", "native_reference", "selected_state")
    loaded = {name: importlib.import_module(name) for name in names}
    for name, module in loaded.items():
        require(Path(module.__file__).resolve() == root / (name + ".py"), "Model module shadowed: " + name)
    native, native_utils = loaded["native_reference"].native_modules()
    loaded.update(native=native, native_utils=native_utils)
    spec = importlib.util.spec_from_file_location("ncnc_pilot_design_spec", context["paths"]["design_root"] / "design_spec.py")
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    loaded["design"] = design
    return loaded


def ordinary_runtime(context):
    import torch
    import os
    from pilot_common import file_sha
    release = context["release"]
    authority = context["runtime"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(authority.get("ordinary_host_execution") is True, "Normal host runtime authority required")
    admitted_visible = set(authority["allowed_cuda_visible_devices"])
    require(admitted_visible <= {"0", "1", *context["authority"]["physical_GPU_UUIDs"]} and admitted_visible, "Unknown GPU authority")
    require(visible in admitted_visible and visible == release["cuda_visible_devices"], "Use the root-bound normal single-device CUDA_VISIBLE_DEVICES")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One normally visible CUDA device required")
    require(authority["cuda_version"] == "12.6" and torch.__version__ == authority["torch_version"] and torch.version.cuda == authority["cuda_version"], "Qualified Torch/CUDA build differs")
    torch.cuda.set_device(0)
    require(torch.get_default_dtype() == torch.float32, "Plain float32 required")
    require(torch.are_deterministic_algorithms_enabled() is False, "Qualified kernel profile differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    require(not torch.is_autocast_enabled(), "Autocast is not admitted")
    for pin in authority["runtime_source_pins"]:
        module = importlib.import_module(pin["module"])
        require(Path(module.__file__).resolve() == Path(pin["path"]).resolve() and file_sha(module.__file__) == pin["sha256"], "Runtime source mismatch")
    for pin in authority["runtime_binary_files"]:
        path = Path(pin["path"])
        require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Runtime binary mismatch")
    binary_paths = {Path(pin["path"]).resolve() for pin in authority["runtime_binary_files"]}
    extension_dirs = {Path(pin["path"]).resolve().parent for pin in authority["runtime_source_pins"] if pin["module"] in ("torch_sparse", "torch_scatter")}
    loaded_paths = {Path(path).resolve() for path in torch.ops.loaded_libraries if Path(path).resolve().parent in extension_dirs}
    require(loaded_paths and loaded_paths <= binary_paths and all(any(p.parent == d for p in loaded_paths) for d in extension_dirs), "Sparse/scatter binary admission differs")
    from torch_geometric.utils import negative_sampling
    import inspect
    from hashlib import sha256
    pin = authority["negative_sampler"]
    source = inspect.getsourcefile(negative_sampling)
    require(Path(source).resolve() == Path(pin["path"]).resolve() and file_sha(source) == pin["sha256"], "Sampler source mismatch")
    require(sha256(inspect.getsource(negative_sampling).encode()).hexdigest() == pin["function_sha256"], "Sampler function differs")
    signature = inspect.signature(negative_sampling)
    require(signature.parameters["num_neg_samples"].default is None and signature.parameters["method"].default == "sparse" and signature.parameters["force_undirected"].default is False, "Native sampler defaults differ")
    properties = torch.cuda.get_device_properties(0)
    require(properties.name == authority["gpu_name"] and properties.total_memory == authority["gpu_total_memory_bytes"], "Device properties differ")
    torch.cuda.synchronize(0)
    return torch.device("cuda:0"), negative_sampling


def runtime_settings():
    import os
    import torch
    return {"torch_version": str(torch.__version__), "CUDA_version": torch.version.cuda,
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "TF32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "autocast_CUDA": torch.is_autocast_enabled("cuda"),
            "autocast_CPU": torch.is_autocast_enabled("cpu"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")}


def observe_runtime(context):
    """Actual scalar profile/full RNG digests; never restores or compares fit RNG."""
    import pilot_state as state
    from pilot_common import HERE
    require(Path(state.__file__).resolve() == HERE / "pilot_state.py", "Runtime RNG helper shadowed")
    value = state.rng_state()
    actual = runtime_settings()
    expected = context.get("runtime_profile_transition", {}).get("actual_profile_after")
    return {"actual_runtime_profile": actual, "actual_RNG_sha256": state.rng_digest(value),
            "actual_RNG_component_sha256": {key: state.state_digest(child) for key, child in value.items()},
            "runtime_profile_exact": actual == expected if expected is not None else None}


def runtime_custody(context):
    """Recheck authenticated runtime read-only; retain current legitimate fit RNG."""
    import inspect
    import torch
    from hashlib import sha256
    from pilot_common import (driver_module_custody, runtime_stdlib, read_bound_json, file_sha,
                              DATA_AUTHORITY_SHA, RUNTIME_AUTHORITY_SHA)
    driver_module_custody(context)
    require(runtime_stdlib(context) == context["versions"], "Admitted runtime versions changed")
    require(read_bound_json(context["release"]["data_authority_file"], DATA_AUTHORITY_SHA) == context["authority"]
            and read_bound_json(context["release"]["runtime_authority_file"], RUNTIME_AUTHORITY_SHA) == context["runtime"],
            "Admitted runtime/data authority changed")
    authority = context["runtime"]
    for pin in authority["runtime_source_pins"]:
        module = sys.modules.get(pin["module"])
        require(module is not None and Path(module.__file__).resolve() == Path(pin["path"]).resolve()
                and file_sha(module.__file__) == pin["sha256"], "Runtime source custody differs")
    for pin in authority["runtime_binary_files"]:
        path = Path(pin["path"])
        require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Runtime binary custody differs")
    binary_paths = {Path(pin["path"]).resolve() for pin in authority["runtime_binary_files"]}
    extension_dirs = {Path(pin["path"]).resolve().parent for pin in authority["runtime_source_pins"]
                      if pin["module"] in ("torch_sparse", "torch_scatter")}
    loaded_paths = {Path(path).resolve() for path in torch.ops.loaded_libraries if Path(path).resolve().parent in extension_dirs}
    require(loaded_paths and loaded_paths <= binary_paths
            and all(any(p.parent == d for p in loaded_paths) for d in extension_dirs), "Sparse/scatter binary custody differs")
    device, sampler = context["_authenticated_runtime"]
    from torch_geometric.utils import negative_sampling
    require(sampler is negative_sampling, "Admitted sampler binding changed")
    pin = authority["negative_sampler"]
    source = inspect.getsourcefile(sampler)
    require(Path(source).resolve() == Path(pin["path"]).resolve() and file_sha(source) == pin["sha256"]
            and sha256(inspect.getsource(sampler).encode()).hexdigest() == pin["function_sha256"], "Sampler source custody differs")
    signature = inspect.signature(sampler)
    require(signature.parameters["num_neg_samples"].default is None and signature.parameters["method"].default == "sparse"
            and signature.parameters["force_undirected"].default is False, "Native sampler defaults changed")
    properties = torch.cuda.get_device_properties(0)
    require(str(device) == "cuda:0" and torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0
            and properties.name == authority["gpu_name"] and properties.total_memory == authority["gpu_total_memory_bytes"],
            "Admitted device changed")
    observed = observe_runtime(context)
    require(observed["runtime_profile_exact"] is True, "Admitted deterministic runtime profile changed")
    return observed


def runtime(context):
    """Authenticate ordinary False once, then use one explicit fail-closed profile."""
    from pilot_common import driver_module_custody, RUNTIME_PROFILE_TRANSITION, RUNTIME_AUTHORITY_SHA
    if "_authenticated_runtime" in context:
        require(context["runtime_profile_transition"]["status"] == "TRANSITION_VERIFIED", "Runtime transition incomplete")
        runtime_custody(context)
        return context["_authenticated_runtime"]
    driver_module_custody(context)
    environment = context["preimport_environment"]
    require(environment["Torch_absent_before_configuration"] is True
            and environment["Torch_absent_after_configuration"] is True
            and environment["CUBLAS_WORKSPACE_CONFIG_after"] == ":4096:8", "Pre-import workspace configuration missing")
    device, sampler = ordinary_runtime(context)
    context["_authenticated_runtime"] = (device, sampler)
    before = observe_runtime(context)
    profile = before["actual_runtime_profile"]
    require(profile["deterministic_algorithms"] is False and profile["deterministic_warn_only"] is False
            and profile["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            and profile["autocast_CPU"] is False, "Authenticated ordinary pre-transition profile differs")
    transition = {"schema": "ncnc-pattern-deterministic-profile-transition-v1", "identity": context["identity"],
                  "status": "IN_PROGRESS", "declared_transition": RUNTIME_PROFILE_TRANSITION,
                  "ordinary_runtime_authority_sha256": RUNTIME_AUTHORITY_SHA,
                  "ordinary_runtime_source_binary_admission_completed": True,
                  "preimport_environment": environment, "actual_profile_before": profile,
                  "RNG_before_sha256": before["actual_RNG_sha256"],
                  "RNG_components_before_sha256": before["actual_RNG_component_sha256"]}
    context["runtime_profile_transition"] = transition
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    after = observe_runtime(context)
    transition.update(actual_profile_after=after["actual_runtime_profile"],
                      RNG_after_sha256=after["actual_RNG_sha256"],
                      RNG_components_after_sha256=after["actual_RNG_component_sha256"],
                      RNG_exactly_unchanged=before["actual_RNG_sha256"] == after["actual_RNG_sha256"]
                          and before["actual_RNG_component_sha256"] == after["actual_RNG_component_sha256"],
                      actual_profile_matches_exact_transition=after["actual_runtime_profile"]
                          == dict(profile, deterministic_algorithms=True, deterministic_warn_only=False))
    require(transition["RNG_exactly_unchanged"] and transition["actual_profile_matches_exact_transition"],
            "Explicit deterministic transition changed RNG or another runtime setting")
    transition["status"] = "TRANSITION_VERIFIED"
    return device, sampler


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_native(mods, seed, width, device):
    import torch
    seed_all(seed)
    native = mods["native"]
    encoder = native.GCN(128, width, width, 1, .1, True, True, -1, "gcn", False, .25, xdropout=.25, taildropout=.05).to(device)
    decoder = native.IncompleteCN1Predictor(width, width, 1, 3, .3, edrop=0., ln=True, cndeg=-1,
        use_xlin=True, tailact=True, twolayerlin=False, beta=1., alpha=1.05, scale=2.5, offset=6.,
        trainresdeg=-1, testresdeg=-1, pt=.1, learnablept=False, depth=1, splitsize=-1).to(device)
    optimizer = torch.optim.Adam([{"params": encoder.parameters(), "lr": .0082},
                                  {"params": decoder.parameters(), "lr": .0037}],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
    result = (encoder, decoder)
    expected = mods["design"].native_parameter_count(width)
    require(sum(p.numel() for _, p in named_parameters(result)) == expected, "Native parameter count differs")
    return result, optimizer


def make_factorized(mods, seed, device, *, engineering_sign_seed=None):
    import torch
    seed_all(seed)
    prototype = mods["prototype"]
    model = prototype.CompletionTwin(prototype.Recipe()).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mods["design"].factor_sign_seed(seed) if engineering_sign_seed is None else engineering_sign_seed)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if name.endswith(".r") or name.endswith(".s"):
                signs = 2 * torch.randint(0, 2, parameter.shape, dtype=torch.int64, generator=generator, device="cpu") - 1
                parameter.copy_(signs.to(device=parameter.device, dtype=parameter.dtype))
    optimizer = prototype.native_optimizer(model)
    require(sum(p.numel() for p in model.parameters()) == mods["design"].factorized_parameter_count(), "Factorized parameter count differs")
    return model, optimizer


def named_parameters(model):
    if isinstance(model, tuple):
        return [(prefix + "." + name, value) for prefix, part in zip(("encoder", "decoder"), model) for name, value in part.named_parameters()]
    return list(model.named_parameters())


def train_flag(model, flag):
    for part in model if isinstance(model, tuple) else (model,):
        part.train(flag)
