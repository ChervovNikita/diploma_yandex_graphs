"""Complete-TRAIN graph/native-batch GPU arithmetic qualification, no updates."""
from dataclasses import replace
import hashlib
import torch
from torch.nn import functional as F
from guards import require
from runtime import capture_rng, restore_rng, rng_digest, instrument_portable
from train_only_data import epoch_stream, tensor_sha


def state_digest(state):
    return hashlib.sha256("\n".join(name + ":" + tensor_sha(value) for name, value in sorted(state.items())).encode()).hexdigest()


def initial_engineering_state(prototype, device, meter):
    # This is the sealed unit-factor engineering identity, not a new study
    # initialization. The sole seed is supplied externally by root.
    model = meter.call("four_member_engineering_initialization", lambda: prototype.CompletionTwin(prototype.Recipe()).to(device))
    for name, value in model.named_parameters():
        meter.finite(value, "initial_parameter:" + name)
    state = meter.call("initial_state_device_to_host", lambda: {name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    initial_bytes = sum(value.numel() * value.element_size() for value in state.values())
    meter.counts["explicit_device_to_host_tensor_bytes"] += initial_bytes
    meter.counts["explicit_host_to_device_tensor_bytes"] += initial_bytes
    identity = meter.call("initial_state_hash", state_digest, state)
    rng = capture_rng()
    del model
    return state, rng, identity


def compare_float(actual, expected, name, checks):
    require(actual.dtype == expected.dtype == torch.float32 and actual.shape == expected.shape, "Parity shape/dtype differs at " + name)
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), "Nonfinite parity tensor at " + name)
    # Exact inherited qualify.py arithmetic rule; no tolerance search/new gate.
    tolerance = 128 * torch.finfo(torch.float32).eps
    difference = (actual - expected).abs()
    passed = bool((difference <= tolerance + tolerance * expected.abs()).all())
    checks.append({"check": name, "shape": list(actual.shape), "passed": passed,
        "maximum_absolute_arithmetic_difference": float(difference.max()) if difference.numel() else 0.0,
        "rtol": tolerance, "atol": tolerance})
    require(passed, "Inherited arithmetic tolerance failed at " + name)


def mapped_gradients(prototype, portable, encoder, decoder, native, meter):
    pairs = [("encoder.weight", portable.encoder.weight, encoder.convs[0].lin.weight),
        ("encoder.bias", portable.encoder.bias, encoder.convs[0].bias),
        ("encoder.norm.weight", portable.encoder.norm.weight, encoder.lins[0][0].weight),
        ("encoder.norm.bias", portable.encoder.norm.bias, encoder.lins[0][0].bias),
        ("decoder.beta", portable.decoder.beta, decoder.beta)]
    private_norm_names = set()
    for name in ("xlin", "xcnlin", "xijlin", "lin", "ptlin"):
        own, original = getattr(portable.decoder, name), getattr(decoder, name)
        for index, operation in own.ops.items():
            for field in ("weight", "bias"):
                label = "decoder." + name + ".ops." + index + "." + field
                pairs.append((label, getattr(operation, field), getattr(original[int(index)], field)))
                if isinstance(operation, prototype.PrivateNorm):
                    private_norm_names.add(label)
    require(len(pairs) == 31, "Corresponding native parameter gradient count differs")
    result = {}
    for name, ours, theirs in pairs:
        gradient = theirs.grad if native else ours.grad
        if gradient is None:
            result[name] = None
        else:
            if not native and name in private_norm_names:
                gradient = gradient[0]
            meter.counts["explicit_device_to_host_tensor_bytes"] += gradient.numel() * gradient.element_size()
            result[name] = gradient.detach().cpu().clone()
    return result


def native_modules(prototype, reference, device, meter):
    native, utils = reference.native_modules()
    encoder = native.GCN(128, 64, 64, 1, .1, True, True, -1, "gcn", False, .25, xdropout=.25, taildropout=.05).to(device)
    decoder = native.IncompleteCN1Predictor(64, 64, 1, 3, .3, edrop=0., ln=True, cndeg=-1,
        use_xlin=True, tailact=True, twolayerlin=False, beta=1., alpha=1.05, scale=2.5, offset=6.,
        trainresdeg=-1, testresdeg=-1, pt=.1, learnablept=False, depth=1, splitsize=-1).to(device)
    portable = prototype.CompletionTwin(replace(prototype.Recipe(), member_count=1)).to(device)
    prototype.copy_native_unit_member(portable, encoder, decoder)
    meter.counts["explicit_host_to_device_tensor_bytes"] += sum(value.numel() * value.element_size() for model in (encoder, decoder, portable) for value in model.state_dict().values())
    return encoder, decoder, portable, utils


def train_objective(positive, negative):
    return -F.logsigmoid(positive).mean() - F.logsigmoid(-negative).mean()


def run(context, prototype, graph_ops, reference, data, device, sampler, meter, progress):
    from torch_sparse import SparseTensor
    initial, epoch_rng, initial_sha = initial_engineering_state(prototype, device, meter)
    del initial
    progress["initial_engineering_state_sha256"] = initial_sha
    progress["epoch_start_rng_sha256"] = rng_digest(epoch_rng)
    _, utils = reference.native_modules()
    negatives, iterator, stream = epoch_stream(data, utils, sampler, meter)
    progress["native_stream"] = stream
    record_ids = next(iter(iterator))
    require(len(record_ids) == 65536, "Parity did not select a complete native TRAIN minibatch")
    positive, negative = data["pairs"][record_ids], negatives[record_ids]
    graph = meter.call("parity_full_masked_graph_rebuild", graph_ops.Graph.mask_train_batch, data["pairs"], record_ids, len(data["x"]))
    def build_native():
        keep = torch.ones(len(data["pairs"]), device=device, dtype=torch.bool)
        keep[record_ids] = False
        remaining = data["pairs"][keep].T
        return SparseTensor.from_edge_index(remaining, sparse_sizes=(235868, 235868)).to_device(device, non_blocking=True).to_symmetric()
    adjacency = meter.call("parity_native_full_masked_graph_rebuild", build_native)
    row, col, _ = adjacency.coo()
    require(torch.equal(row, graph.row) and torch.equal(col, graph.col), "GPU native/portable complete masked adjacency differs")
    progress["masked_graph_directed_entries"] = len(graph.row)
    for label, query in (("positive", positive), ("negative", negative)):
        cn, left, right = meter.call("native_full_candidate_enumeration", utils.adjoverlap, adjacency, adjacency, query.T, calresadj=True, cnsampledeg=-1, ressampledeg=-1)
        own = meter.call("portable_full_candidate_enumeration", graph_ops.enumerate_neighbors, graph, query)
        for original, ours in zip((cn, left, right), (own.common, own.left, own.right)):
            native_row, native_col, _ = original.coo()
            require(torch.equal(native_row, ours[0]) and torch.equal(native_col, ours[1]), "GPU complete native candidate order differs")
        progress.setdefault("full_batch_candidate_counts", {})[label] = {"queries": len(query), "common": len(own.common[0]), "left": len(own.left[0]), "right": len(own.right[0])}
        del cn, left, right, own, native_row, native_col
    dropout_rng = capture_rng()
    encoder, decoder, portable, _ = meter.call("source_matched_native_portable_initialization", native_modules, prototype, reference, device, meter)
    checks = progress.setdefault("arithmetic_checks", [])
    for training in (False, True):
        profile = "training" if training else "evaluation"
        progress["current_profile"] = profile
        encoder.train(training); decoder.train(training); portable.train(training)
        encoder.zero_grad(set_to_none=True); decoder.zero_grad(set_to_none=True); portable.zero_grad(set_to_none=True)
        native_x = data["x"].detach().clone().requires_grad_(True)
        hooks = []
        def native_guard(name):
            def guard(module, inputs, output):
                if torch.is_tensor(output):
                    meter.finite(output, "native:" + name)
            return guard
        for prefix, model in (("encoder", encoder), ("decoder", decoder)):
            for name, module in model.named_modules():
                hooks.append(module.register_forward_hook(native_guard(prefix + "." + name)))
        restore_rng(dropout_rng)
        try:
            native_h = meter.call("native_" + profile + "_full_node_encoder", encoder, native_x, adjacency)
            native_pos = meter.call("native_" + profile + "_positive_decoder", decoder, native_h, adjacency, positive.T).flatten()
            native_neg = meter.call("native_" + profile + "_negative_decoder", decoder, native_h, adjacency, negative.T).flatten()
            native_loss = meter.call("native_" + profile + "_TRAIN_loss", train_objective, native_pos, native_neg)
            meter.finite(native_loss, "native_train_loss")
            meter.call("native_" + profile + "_backward", native_loss.backward)
            native_rng_after = rng_digest(capture_rng())
            meter.counts["explicit_device_to_host_tensor_bytes"] += sum(value.numel() * value.element_size() for value in (native_pos, native_neg, native_x.grad, native_loss))
            reference_values = meter.call("native_parity_tensor_device_to_host", lambda: {"positive": native_pos.detach().cpu().clone(), "negative": native_neg.detach().cpu().clone(),
                "input_gradient": native_x.grad.detach().cpu().clone(), "loss": native_loss.detach().cpu().clone(),
                "gradients": mapped_gradients(prototype, portable, encoder, decoder, True, meter)})
        finally:
            for hook in hooks:
                hook.remove()
        del native_h, native_pos, native_neg, native_loss, native_x
        own_x = data["x"].detach().clone().requires_grad_(True)
        restore_rng(dropout_rng)
        with instrument_portable(prototype, portable, meter):
            own_h = meter.call("portable_" + profile + "_full_node_encoder", portable.encoder, own_x, graph)
            meter.finite(own_h, "portable_encoder")
            meter.query_pass = "positive"
            own_pos = meter.call("portable_" + profile + "_positive_decoder", portable.decoder, own_h, graph, positive, "private")[:, 0]
            meter.query_pass = "negative"
            own_neg = meter.call("portable_" + profile + "_negative_decoder", portable.decoder, own_h, graph, negative, "private")[:, 0]
            own_loss = meter.call("portable_" + profile + "_TRAIN_loss", train_objective, own_pos, own_neg)
            meter.finite(own_loss, "portable_train_loss")
            meter.call("portable_" + profile + "_backward", own_loss.backward)
        require(rng_digest(capture_rng()) == native_rng_after, "GPU native/portable RNG consumption differs")
        meter.counts["explicit_device_to_host_tensor_bytes"] += sum(value.numel() * value.element_size() for value in (own_pos, own_neg, own_x.grad, own_loss))
        own_values = meter.call("portable_parity_tensor_device_to_host", lambda: {"positive": own_pos.detach().cpu(), "negative": own_neg.detach().cpu(), "input_gradient": own_x.grad.detach().cpu(), "loss": own_loss.detach().cpu(), "gradients": mapped_gradients(prototype, portable, encoder, decoder, False, meter)})
        for name in ("positive", "negative", "input_gradient", "loss"):
            meter.call("parity_arithmetic_comparison", compare_float, own_values[name], reference_values[name], profile + ":" + name, checks)
        for name, original in reference_values["gradients"].items():
            ours = own_values["gradients"][name]
            if original is None:
                require(ours is None and name.startswith("decoder.ptlin."), "Unused native parameter gradient differs")
                checks.append({"check": profile + ":" + name, "both_gradients_none": True, "passed": True})
            else:
                require(ours is not None, "Missing GPU portable gradient at " + name)
                meter.call("parity_arithmetic_comparison", compare_float, ours, original, profile + ":" + name, checks)
        for name, parameter in portable.named_parameters():
            if parameter.grad is not None:
                meter.finite(parameter.grad, "portable_gradient:" + name)
            else:
                require(name.startswith("decoder.ptlin."), "Missing GPU active private-factor gradient")
        progress.setdefault("profiles_completed", []).append(profile)
        del own_values, reference_values, own_h, own_pos, own_neg, own_loss, own_x
    progress["matching_native_parameter_gradients_per_profile"] = 31
    progress["optimizer_updates"] = 0
    progress["native_float64_encoder_parity_claim"] = False
    progress["all_required_checks_passed"] = len(progress["profiles_completed"]) == 2 and len(checks) == 70 and all(check["passed"] for check in checks)
    require(progress["all_required_checks_passed"], "Incomplete GPU numerical checks")
    return "GPU_TRAIN_PARITY_PASSED"
