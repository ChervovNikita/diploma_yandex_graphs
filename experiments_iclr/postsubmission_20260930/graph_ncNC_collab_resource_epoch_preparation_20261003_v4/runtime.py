"""Admitted single-device numerical runtime and inclusive work accounting."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib
import inspect
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from guards import file_sha, checked_file, require


def qualify_runtime(context):
    admission = context["admission"]
    require(torch.__version__ == admission["torch_version"], "Torch build differs from root admission")
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled(), "Require plain float32 without autocast")
    require(torch.are_deterministic_algorithms_enabled() == admission["deterministic_algorithms"], "Deterministic-kernel profile differs from root admission")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Require one root-isolated available CUDA device")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for pin in admission["runtime_source_pins"]:
        module = importlib.import_module(pin["module"])
        require(Path(module.__file__).resolve() == Path(pin["path"]).resolve(), "Imported runtime source path differs")
        require(file_sha(module.__file__) == pin["sha256"], "Imported runtime source digest differs")
    for pin in admission["runtime_binary_files"]:
        checked_file(pin["path"], pin)
    binary_paths = {Path(pin["path"]).resolve() for pin in admission["runtime_binary_files"]}
    extension_directories = {Path(pin["path"]).resolve().parent for pin in admission["runtime_source_pins"]
                             if pin["module"] in ("torch_sparse", "torch_scatter")}
    loaded_paths = {Path(path).resolve() for path in torch.ops.loaded_libraries
                    if Path(path).resolve().parent in extension_directories}
    require(loaded_paths and loaded_paths <= binary_paths, "Loaded sparse/scatter extension is not admitted")
    for directory in extension_directories:
        require(any(path.parent == directory for path in loaded_paths), "Admitted sparse/scatter extension family was not loaded")
    from torch_geometric.utils import negative_sampling
    sampler_pin = admission["negative_sampler"]
    source_file = inspect.getsourcefile(negative_sampling)
    require(source_file is not None and file_sha(source_file) == sampler_pin["sha256"], "Native sampler runtime file differs")
    require(Path(source_file).resolve() == Path(sampler_pin["path"]).resolve(), "Native sampler import path differs")
    require(hashlib.sha256(inspect.getsource(negative_sampling).encode()).hexdigest() == sampler_pin["function_sha256"], "Native sampler function differs")
    signature = inspect.signature(negative_sampling)
    require(signature.parameters["num_neg_samples"].default is None and signature.parameters["method"].default == "sparse" and signature.parameters["force_undirected"].default is False, "Native sampler default signature differs")
    properties = torch.cuda.get_device_properties(0)
    require(properties.name == admission["gpu_name"] and properties.total_memory == admission["gpu_total_memory_bytes"], "GPU properties differ from root admission")
    identity = {"name": properties.name, "total_memory_bytes": properties.total_memory,
        "cuda_visible_devices": admission["cuda_visible_devices"], "logical_device": 0,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "distribution_versions": context["versions"], "runtime_source_pins": admission["runtime_source_pins"],
        "runtime_binary_files": admission["runtime_binary_files"],
        "loaded_sparse_scatter_binary_paths": sorted(str(path) for path in loaded_paths),
        "negative_sampler": sampler_pin, "dtype": "torch.float32", "TF32": False, "mixed_precision": False}
    identity["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    if context["gpu_parity"] is not None:
        require(context["gpu_parity"]["runtime_identity"] == identity, "GPU parity/runtime/device identity differs")
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device("cuda:0"), negative_sampling, identity


def set_engineering_rng(admission):
    seed = admission["engineering_rng_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": torch.cuda.get_rng_state(0).clone()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], 0)


def rng_digest(state):
    digest = hashlib.sha256(repr(state["python"]).encode())
    digest.update(repr((state["numpy"][0], state["numpy"][2:])).encode())
    digest.update(state["numpy"][1].tobytes())
    digest.update(state["torch_cpu"].cpu().numpy().tobytes())
    digest.update(state["torch_cuda"].cpu().numpy().tobytes())
    return digest.hexdigest()


class Meter:
    """Host elapsed times with CUDA sync; nested totals are inclusive."""
    def __init__(self):
        self.seconds, self.calls, self.counts = defaultdict(float), Counter(), Counter()
        self.enumerations = []
        self.scope, self.query_pass = "outer", None
        self.started = time.perf_counter()

    def synchronize(self):
        torch.cuda.synchronize(0)
        self.counts["explicit_timing_synchronizations"] += 1

    def call(self, label, function, *args, **kwargs):
        self.synchronize()
        started = time.perf_counter()
        self.calls[label] += 1
        try:
            return function(*args, **kwargs)
        finally:
            self.synchronize()
            self.seconds[label] += time.perf_counter() - started

    def finite(self, tensor, name):
        self.counts["finite_guard_scalar_synchronizations"] += 1
        require(bool(torch.isfinite(tensor).all().item()), "Nonfinite engineering tensor at " + name)

    def receipt(self):
        self.synchronize()
        # Linux ru_maxrss is KiB; the resource execution profile is Linux/CUDA.
        return {"inclusive_seconds": dict(self.seconds), "calls": dict(self.calls),
            "work_counts": dict(self.counts), "enumerations": self.enumerations,
            "wall_seconds": time.perf_counter() - self.started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "timings_are_nested_inclusive": True, "profiling_and_guards_charged": True,
            "final_cuda_synchronization_completed": True,
            "opaque_library_transfers_and_synchronizations": "charged in enclosing elapsed time; wrapper counters count explicit/source-observable work only"}


@contextmanager
def instrument_portable(prototype, model, meter):
    """Count actual immutable prototype calls; no candidate arrays retained."""
    originals, instance_originals = {}, []
    def replace_global(name, replacement):
        originals[name] = getattr(prototype, name)
        setattr(prototype, name, replacement)
    enumeration = prototype.enumerate_neighbors
    def enumerate_counted(graph, query):
        result = meter.call(meter.scope + "_enumeration", enumeration, graph, query)
        meter.counts[meter.scope + "_enumeration_calls"] += 1
        meter.counts["source_graph_rows_scalar_synchronizations"] += 2
        meter.enumerations.append({"scope": meter.scope, "query_pass": meter.query_pass, "queries": len(query),
            "common": len(result.common[0]), "left": len(result.left[0]), "right": len(result.right[0])})
        meter.counts[meter.scope + "_enumerated_queries"] += len(query)
        return result
    aggregation = prototype.feature_sum
    def aggregate_counted(*args, **kwargs):
        result = meter.call(meter.scope + "_feature_aggregation", aggregation, *args, **kwargs)
        meter.counts[meter.scope + "_aggregation_calls"] += 1
        meter.finite(result, meter.scope + "_aggregate")
        return result
    clamp = prototype.clamp_completion
    def clamp_counted(*args, **kwargs):
        result = meter.call("native_completion_clamp", clamp, *args, **kwargs)
        meter.counts["completion_clamp_calls"] += 1
        meter.finite(result, "clamped_completion")
        return result
    replace_global("enumerate_neighbors", enumerate_counted)
    replace_global("feature_sum", aggregate_counted)
    replace_global("clamp_completion", clamp_counted)
    def replace_method(instance, name, replacement):
        instance_originals.append((instance, name, name in instance.__dict__, instance.__dict__.get(name)))
        setattr(instance, name, replacement)
    recursive = model.decoder.completion_scores
    def recursive_counted(*args, **kwargs):
        previous = meter.scope
        meter.scope = "recursive"
        meter.counts["recursive_score_calls"] += 1
        try:
            result = meter.call("recursive_complete_base_scorer", recursive, *args, **kwargs)
            require(not result.requires_grad, "Native recursive score escaped no_grad")
            meter.finite(result, "recursive_score")
            return result
        finally:
            meter.scope = previous
    replace_method(model.decoder, "completion_scores", recursive_counted)
    xlin = model.decoder.xlin.forward_member
    def xlin_counted(x, member):
        require(len(x) == 235868, "Full-node xlin population reduced")
        label = meter.scope + "_full_node_xlin"
        meter.counts[label + "_calls"] += 1
        meter.counts[label + "_rows"] += len(x)
        meter.counts[label + "_linear_map_calls"] += 2
        result = meter.call(label, xlin, x, member)
        meter.finite(result, label)
        if meter.scope == "recursive":
            require(not result.requires_grad, "Recursive full-node xlin retained autograd")
        return result
    replace_method(model.decoder.xlin, "forward_member", xlin_counted)
    decode = model.decoder.decode
    def decode_counted(*args, **kwargs):
        meter.counts[meter.scope + "_nonlinear_decode_calls"] += 1
        result = meter.call(meter.scope + "_nonlinear_decode", decode, *args, **kwargs)
        meter.finite(result, meter.scope + "_decoded_logits")
        return result
    replace_method(model.decoder, "decode", decode_counted)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(prototype, name, original)
        for instance, name, existed, original in reversed(instance_originals):
            if existed:
                setattr(instance, name, original)
            else:
                delattr(instance, name)


def finite_gradients_and_state(model, optimizer, meter, gradients=True):
    for name, parameter in model.named_parameters():
        meter.finite(parameter, "parameter:" + name)
        if gradients:
            if name.startswith("decoder.ptlin."):
                require(parameter.grad is None, "Unused fixed-pt branch unexpectedly trained")
            else:
                require(parameter.grad is not None, "Missing expected active gradient:" + name)
                meter.finite(parameter.grad, "gradient:" + name)
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                meter.finite(value, "Adam:" + name)


def state_bytes(model, optimizer):
    parameters = sum(value.numel() * value.element_size() for value in model.parameters())
    buffers = sum(value.numel() * value.element_size() for value in model.buffers())
    gradients = sum(value.grad.numel() * value.grad.element_size() for value in model.parameters() if value.grad is not None)
    adam = sum(value.numel() * value.element_size() for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
    return {"parameter_bytes_including_unused_ptlin": parameters, "buffer_bytes": buffers, "gradient_bytes": gradients, "Adam_tensor_bytes": adam}
