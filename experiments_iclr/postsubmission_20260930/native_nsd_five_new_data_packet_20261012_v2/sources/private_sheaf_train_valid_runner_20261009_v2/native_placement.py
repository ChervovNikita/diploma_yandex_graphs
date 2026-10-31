# SPDX-License-Identifier: Apache-2.0
"""Explicit static tensor placement around the unchanged original constructor."""

BUILDER_TENSORS = ("edge_index", "full_left_right_idx", "left_right_idx",
                   "vertex_tril_idx", "diag_indices", "tril_indices", "deg",
                   "fixed_diag_indices", "fixed_tril_indices")


def make_native_placed_factory(torch, adapter, cpu_edge_index, final_edge_index, native_args):
    """Build exact author topology on CPU; transfer plain static attrs before use.

    Author constructors initialize learned Linear/epsilon parameters on CPU in
    either placement. Source topology arithmetic is unchanged; no map, degree,
    normalization, SVD, sparse propagation or forward method is replaced.
    This supports only the sealed general-map/no-LP-HP scope. Numerical/static
    index parity against direct final-device construction must be qualified.
    """
    if cpu_edge_index.device.type != "cpu":
        raise ValueError("CPU topology construction requires CPU canonical support")
    device = torch.device(native_args["device"])
    if final_edge_index.device != device:
        raise ValueError("Final graph/device mismatch")
    if not torch.equal(cpu_edge_index, final_edge_index.cpu()):
        raise ValueError("CPU and final canonical graph/order differ")
    args_cpu = dict(native_args, device="cpu")
    author_cpu_factory = adapter.make_native_factory(cpu_edge_index, args_cpu)
    final_graph_version = final_edge_index._version

    def factory():
        if final_edge_index._version != final_graph_version:
            raise RuntimeError("Final static support/order was changed")
        model = author_cpu_factory()  # original class and original initialization
        if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
               for parameter in model.parameters()):
            raise RuntimeError("Qualify the native CPU/float32 initialization environment")
        model.to(device)  # learned parameters / registered buffers only
        model.edge_index = final_edge_index
        model.time_range = model.time_range.to(device)
        model.device = device
        builder = model.laplacian_builder
        for name in BUILDER_TENSORS:
            tensor = getattr(builder, name, None)
            if tensor is not None:
                setattr(builder, name, final_edge_index if name == "edge_index" else tensor.to(device))
        builder.device = device
        # Fail rather than silently leave a new plain tensor on its old device.
        for module in model.modules():
            for name, value in vars(module).items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    raise RuntimeError("Unmoved native plain tensor: " + name)
        if model.edge_index is not builder.edge_index:
            raise RuntimeError("Native model/builder graph identity diverged")
        return model

    return factory
