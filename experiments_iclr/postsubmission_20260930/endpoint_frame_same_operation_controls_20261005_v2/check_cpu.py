"""One bounded fabricated CPU component check, prepared for root execution.

Fixed seed/7-node operands; zero fits and exactly one native Adam update.
No data/checkpoint/graph payload reads, performance metrics, retries or server IO.
On failure the sealed parent, adapter and control source are never repaired here.
"""
import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_PINS = {
    "graph_ncNC_member_completion_qualification_preparation_20261003_v2/graph_ops.py": "37edca94c7980432c8b8c850a4e05fcdba855323136d996877d20e5d65713be5",
    "graph_ncNC_member_completion_qualification_preparation_20261003_v2/prototype.py": "1f954a340a65d02089661b4dab567cd0c2fa7efadcfa5b03cbe4f228c5e1bda0",
    "native_endpoint_private_factor_mechanism_assessment_20261005_v1/adapter.py": "e275f83f8cd65c89d53e0d5580369c7dd4e0e59673e88bf567337d276fc8fc22",
    "endpoint_frame_same_operation_controls_20261005_v2/controls.py": "4667d87c013f2397bbc6fd5963b42288707b0b595c888eecf87b3bfd11c8b7b0",
}
CONSTRUCTOR_SEED = 2174
FACTOR_SIGN_SEED = 1956882699
OPTIMIZER_UPDATES = 0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_sources():
    for relative, expected in SOURCE_PINS.items():
        require(sha256((ROOT / relative).read_bytes()).hexdigest() == expected,
                f"Pinned source bytes differ: {relative}")
    names = ("graph_ops", "endpoint_controls_pinned_prototype",
             "endpoint_controls_pinned_adapter", "endpoint_controls_pinned_controls")
    modules = []
    for name, relative in zip(names, SOURCE_PINS):
        require(name not in sys.modules, f"Fresh process required; module already loaded: {name}")
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    graph_ops, parent, adapter, controls = modules
    return parent, graph_ops, controls.make_control_classes(parent, graph_ops, adapter)


def run_fixture():
    global OPTIMIZER_UPDATES
    import torch
    torch.set_num_threads(1)
    require(torch.get_default_dtype() == torch.float32, "Fixture requires default float32")
    parent, graph_ops, classes = load_sources()
    Shared = classes["shared_frame_f4"]
    Single = classes["same_four_frames_native_single"]
    Independent = classes["independent_native_single_one_frame"]
    recipe = parent.Recipe()
    single_recipe = parent.Recipe(member_count=1)
    tolerance = 128 * torch.finfo(torch.float32).eps

    # Exact predecessor operands, including its separate positive decoder h.
    pairs = torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,5],[5,6],[4,6]], dtype=torch.long)
    queries = torch.tensor([[0,3],[1,4],[2,5]], dtype=torch.long)
    x = (torch.arange(7 * recipe.features, dtype=torch.float32).reshape(7, recipe.features) % 11) / 7 + .05
    h = (torch.arange(7 * recipe.hidden, dtype=torch.float32).reshape(7, recipe.hidden) % 11) / 7 + .05
    graph = graph_ops.Graph.from_pairs(pairs, 7)
    record_ids = torch.tensor([0,3], dtype=torch.long)
    negative_pairs = queries[:2]

    def close(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def initialize_signs(model):
        before = torch.get_rng_state().clone()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(FACTOR_SIGN_SEED)
        with torch.no_grad():
            for name, parameter in sorted(model.named_parameters()):
                if name.endswith(".r") or name.endswith(".s"):
                    values = torch.randint(0, 2, parameter.shape, dtype=torch.int64, generator=generator)
                    parameter.copy_((2 * values - 1).to(dtype=parameter.dtype, device=parameter.device))
        require(torch.equal(before, torch.get_rng_state()), "Dedicated factor signs consumed training RNG")

    torch.manual_seed(CONSTRUCTOR_SEED)
    before = torch.get_rng_state().clone()
    baseline = parent.CompletionTwin(recipe)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    shared = Shared(recipe)
    require(torch.equal(after, torch.get_rng_state()), "Shared-frame constructor consumed extra RNG")
    initialize_signs(baseline)
    initialize_signs(shared)
    base_state, shared_state = baseline.state_dict(), shared.state_dict()
    require(set(shared_state) == set(base_state) | {"decoder.v"}, "Shared F4 added unexpected state")
    for name, value in base_state.items():
        require(torch.equal(value, shared_state[name]), f"Signed shared F4 initial state differs: {name}")
    require(torch.equal(shared.decoder.v, torch.eye(1, recipe.hidden)), "Common frame is not e0")
    for training in (False, True):
        baseline.train(training)
        shared.train(training)
        for mode in ("private", "pooled_after_clamp"):
            before = torch.get_rng_state().clone()
            expected = baseline(x, graph, queries, mode)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual = shared(x, graph, queries, mode)
            require(torch.equal(actual, expected), "Signed shared F4 initial logits differ")
            require(torch.equal(after, torch.get_rng_state()), "Shared F4 native forward RNG differs")
            require(torch.equal(shared.serve(actual), baseline.serve(expected)), "Shared F4 serving differs")

    # Same constructor seed, explicit axes, and separate ordinary ownership.
    independent = []
    for axis in range(4):
        torch.manual_seed(CONSTRUCTOR_SEED)
        independent.append(Independent(single_recipe, axis_index=axis))
    for axis, model in enumerate(independent):
        expected_axis = torch.eye(recipe.hidden)[axis:axis+1]
        require(torch.equal(model.decoder.v, expected_axis), "Explicit independent axis differs")
        require(torch.equal(model.decoder.endpoint_product(h, queries, 0),
                            h[queries[:,0]] * h[queries[:,1]]), "Independent axis product differs")
    parameter_owners = [set(id(p) for p in model.parameters()) for model in independent]
    storage_owners = [set(p.data_ptr() for p in model.parameters()) for model in independent]
    for left in range(4):
        for right in range(left+1,4):
            require(not parameter_owners[left] & parameter_owners[right], "Independent parameters are shared")
            require(not storage_owners[left] & storage_owners[right], "Independent storage is shared")

    torch.manual_seed(CONSTRUCTOR_SEED)
    before = torch.get_rng_state().clone()
    native_single = Independent(single_recipe, axis_index=0)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    four = Single(single_recipe)
    require(torch.equal(after, torch.get_rng_state()), "Four-frame single constructor consumed extra RNG")
    require(torch.equal(four.decoder.v, torch.eye(4, recipe.hidden)), "Four-frame initial axes differ")
    products = four.decoder.endpoint_product(h,queries,0)
    original_product = h[queries[:,0]] * h[queries[:,1]]
    for frame in range(4):
        require(torch.equal(products[:,frame*recipe.hidden:(frame+1)*recipe.hidden], original_product),
                "Initial four-frame product block differs")
    native_state, four_state = native_single.state_dict(), four.state_dict()
    require(set(native_state) == set(four_state), "Four-frame single added unexpected state keys")
    for name, value in native_state.items():
        if name == "decoder.v":
            continue
        if name == "decoder.xijlin.ops.0.weight":
            expected_blocks = (value, value/4, value/4, -value/2)
            for frame, block in enumerate(expected_blocks):
                require(torch.equal(four_state[name][:,frame*recipe.hidden:(frame+1)*recipe.hidden], block),
                        "Adopted Wnative,C,C,-2C initializer differs")
        else:
            require(torch.equal(value, four_state[name]), f"Four-frame native initial state differs: {name}")
    for training in (False, True):
        native_single.train(training)
        four.train(training)
        before = torch.get_rng_state().clone()
        expected = native_single(x, graph, queries, "private")
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        actual = four(x, graph, queries, "private")
        close(actual, expected)
        close(four.serve(actual), native_single.serve(expected))
        require(torch.equal(after, torch.get_rng_state()), "Single native RNG/dropout order differs")

    models = [shared, four, *independent]
    for model in models:
        require(all(p.device.type == "cpu" for p in model.parameters()), "CPU-only fixture required")
        require(type(model.decoder).depth_zero is parent.CompletionDecoder.depth_zero, "depth_zero overridden")
        require(type(model.decoder).completion_scores is parent.CompletionDecoder.completion_scores,
                "completion_scores overridden")
        require(type(model.decoder).decode is parent.CompletionDecoder.decode, "decode overridden")
        require(type(model).forward is parent.CompletionTwin.forward and type(model).serve is parent.CompletionTwin.serve,
                "Native twin forward/serving changed")
    for model in [four, *independent]:
        require(not any(name.endswith(".r") or name.endswith(".s") for name,_ in model.named_parameters()),
                "Native single retains BE r/s parameters")
        require(not any(isinstance(module, parent.FactorLinear) for module in model.modules()),
                "Native single retains FactorLinear")

    # Observe actual inherited d,d,4d call order, restoring the method afterward.
    four.eval()
    native_single.eval()
    linear = four.decoder.xijlin.ops["0"]
    original_method = linear.forward_member
    widths = []
    def observe_width(value, member):
        widths.append(value.shape[-1])
        return original_method(value, member)
    linear.forward_member = observe_width
    try:
        four.decoder(h, graph, queries, "private")
    finally:
        del linear.forward_member
    require(widths == [recipe.hidden, recipe.hidden, 4*recipe.hidden], "Inherited recursion/outer width dispatch differs")
    close(four.decoder.depth_zero(h, graph, queries, 0), native_single.decoder.depth_zero(h, graph, queries, 0))

    def off_axis_flags(model):
        gradient = model.decoder.v.grad
        require(gradient is not None and bool(torch.isfinite(gradient).all()),
                "Frame derivative is absent/nonfinite")
        return [bool((gradient[frame,torch.arange(recipe.hidden) != frame] != 0).any())
                for frame in range(4)]

    def positive_h_probe(model):
        # Native backward-only decoder objective, independent of encoded ReLU
        # zero channels. No parameter assignment or optimizer update occurs.
        model.eval()
        model.zero_grad(set_to_none=True)
        positive = model.decoder(h,graph,queries,"private")
        negative = model.decoder(h,graph,pairs[:3],"private")
        probe = -parent.F.logsigmoid(positive).mean() - parent.F.logsigmoid(-negative).mean()
        require(bool(torch.isfinite(probe)), "Fixed-positive-h objective is nonfinite")
        probe.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), f"Probe derivative is nonfinite: {name}")
        return off_axis_flags(model)

    initial_positive_h_flags = positive_h_probe(four)
    require(all(initial_positive_h_flags), "Initial fixed-positive-h vector path is inaccessible")
    encoded_train = []

    def native_backward_counts(model, total, active, capture_encoder=False):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(CONSTRUCTOR_SEED)
        # The unchanged helper retains native record masking and M=1 loss scale.
        handle = None
        if capture_encoder:
            handle = model.encoder.register_forward_hook(
                lambda _module,_arguments,result: encoded_train.append(result.detach().clone()))
        try:
            loss = parent.training_batch_loss(model, x, pairs, record_ids, negative_pairs, "private")
        finally:
            if handle is not None:
                handle.remove()
        require(bool(torch.isfinite(loss)), "Native fixture objective is nonfinite")
        loss.backward()
        actual_total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bearing = sum(p.numel() for p in model.parameters() if p.grad is not None)
        require(actual_total == total and trainable == total, "Actual total/trainable parameter count differs")
        require(bearing == active, "Actual grad-bearing parameter count differs")
        for name, parameter in model.named_parameters():
            if ".ptlin." in name:
                require(parameter.grad is None, "Unused fixed-pt parameter unexpectedly bears gradient")
            else:
                require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                        f"Active parameter gradient absent/nonfinite: {name}")
        return {"total":actual_total,"trainable":trainable,"grad_bearing":bearing}

    counts = {"shared_frame_f4":native_backward_counts(shared,43854,38857)}
    counts["independent_native_single_one_frame"] = [native_backward_counts(model,38211,33986) for model in independent]
    counts["same_four_frames_native_single"] = native_backward_counts(four,50691,46466,capture_encoder=True)
    require(len(encoded_train) == 1, "Expected one native encoder pass")
    encoded_train_flags = off_axis_flags(four)
    encoded_axis_all_zero = [bool((encoded_train[0][:,frame] == 0).all()) for frame in range(4)]
    for frame in range(4):
        if encoded_axis_all_zero[frame]:
            require(not encoded_train_flags[frame], "All-zero encoded axis unexpectedly has a vector derivative")

    # Exactly one actual native Adam step, on the full masked seven-node fixture.
    optimizer = parent.native_optimizer(four)
    optimizer.step()
    OPTIMIZER_UPDATES += 1
    require(all(bool(torch.isfinite(p).all()) for p in four.parameters()), "Updated parameter is nonfinite")

    # All four post-update paths, using only a backward probe: no second step.
    post_positive_h_flags = positive_h_probe(four)
    require(all(post_positive_h_flags), "Post-update fixed-positive-h vector path is inaccessible")
    require(OPTIMIZER_UPDATES == 1, "Fixture update count differs")
    return {
        "fixture":"endpoint-frame-three-controls-one-cpu-component-v2",
        "status":"passed",
        "fit_count":0,
        "optimizer_updates":OPTIMIZER_UPDATES,
        "constructor_seed":CONSTRUCTOR_SEED,
        "factor_sign_seed":FACTOR_SIGN_SEED,
        "actual_parameter_counts":counts,
        "single_r_s_absent":True,
        "independent_axis_products_and_ownership_checked":True,
        "shared_signed_f4_initial_state_logits_rng_exact":True,
        "inherited_single_pair_widths":widths,
        "single_initial_native_function_tolerance":{"rtol":tolerance,"atol":tolerance},
        "initial_fixed_positive_h_off_axis_paths":initial_positive_h_flags,
        "encoded_train_initial_off_axis_nonzero":encoded_train_flags,
        "encoded_train_axis_all_zero":encoded_axis_all_zero,
        "post_one_update_fixed_positive_h_off_axis_paths":post_positive_h_flags,
        "performance_metrics":False,
        "data_checkpoint_graph_outcome_payloads_read":False,
    }


if __name__ == "__main__":
    try:
        result = run_fixture()
    except Exception as error:
        print(json.dumps({"fixture":"endpoint-frame-three-controls-one-cpu-component-v2",
                          "status":"failed","fit_count":0,"optimizer_updates":OPTIMIZER_UPDATES,
                          "error_type":type(error).__name__,"error":str(error)},indent=2,sort_keys=True))
        raise
    print(json.dumps(result,indent=2,sort_keys=True))
