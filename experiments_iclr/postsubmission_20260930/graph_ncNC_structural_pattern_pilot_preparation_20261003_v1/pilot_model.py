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


def runtime(context):
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
