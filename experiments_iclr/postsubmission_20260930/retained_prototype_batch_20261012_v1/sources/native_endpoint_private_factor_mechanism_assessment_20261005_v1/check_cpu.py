"""One bounded fabricated CPU engineering fixture; root executes if adopted.

No dataset, fit, optimizer step, checkpoint, source outcome or remote access.
Exact initial function/RNG/state correspondence and derivative accessibility
are engineering checks, not a task-quality or diversification result.
"""
import importlib.util
import json
import sys

from adapter import make_adapter_classes, verify_parent_sources


def load_pinned_parent():
    root = verify_parent_sources()
    modules = []
    for name, filename in (
        ("graph_ops", "graph_ops.py"),
        ("endpoint_reflection_pinned_prototype", "prototype.py"),
    ):
        if name in sys.modules:
            raise RuntimeError(f"Fixture requires a fresh process; module exists: {name}")
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[1], modules[0]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_fixture():
    import torch
    torch.set_num_threads(1)
    parent, graph_ops = load_pinned_parent()
    Decoder, Twin = make_adapter_classes(parent, graph_ops)
    recipe = parent.Recipe()
    torch.manual_seed(2174)
    constructor_start = torch.get_rng_state().clone()
    base = parent.CompletionTwin(recipe)
    constructor_end = torch.get_rng_state().clone()
    torch.set_rng_state(constructor_start)
    candidate = Twin(recipe)
    require(torch.equal(constructor_end, torch.get_rng_state()), "Adapter constructor consumed extra RNG")
    require(all(p.device.type == "cpu" for p in candidate.parameters()), "Fixture is CPU only")
    require(torch.equal(candidate.decoder.v, torch.eye(recipe.member_count, recipe.hidden)), "Axis initialization differs")
    require(Decoder.depth_zero is parent.CompletionDecoder.depth_zero, "Recursive scorer was overridden")
    require(Decoder.completion_scores is parent.CompletionDecoder.completion_scores, "Completion scoring was overridden")
    require(Decoder.decode is parent.CompletionDecoder.decode, "Native decode was overridden")
    require(Twin.forward is parent.CompletionTwin.forward and Twin.serve is parent.CompletionTwin.serve,
            "Native twin forward/mean raw serving changed")

    def identical_inherited_state():
        original, extended = base.state_dict(), candidate.state_dict()
        require(set(extended) == set(original) | {"decoder.v"}, "Unexpected added state")
        for name, tensor in original.items():
            require(torch.equal(tensor, extended[name]), f"Inherited state differs: {name}")
    identical_inherited_state()

    pairs = torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,5],[5,6],[4,6]], dtype=torch.long)
    queries = torch.tensor([[0,3],[1,4],[2,5]], dtype=torch.long)
    x = (torch.arange(7 * recipe.features, dtype=torch.float32).reshape(7, recipe.features) % 11) / 7 + .05
    h = (torch.arange(7 * recipe.hidden, dtype=torch.float32).reshape(7, recipe.hidden) % 11) / 7 + .05
    graph = graph_ops.Graph.from_pairs(pairs, 7)
    for member in range(recipe.member_count):
        require(torch.equal(candidate.decoder.endpoint_product(h, queries, member),
                            h[queries[:,0]] * h[queries[:,1]]), "Axis target product is not exact")

    # Both modes, eval and active native training dropout; reset the same RNG.
    for training in (False, True):
        base.train(training)
        candidate.train(training)
        for mode in ("private", "pooled_after_clamp"):
            before = torch.get_rng_state().clone()
            expected, expected_details = base(x, graph, queries, mode, True)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual, actual_details = candidate(x, graph, queries, mode, True)
            require(torch.equal(actual, expected), "Axis initial target logits are not exact")
            require(torch.equal(after, torch.get_rng_state()), "Native dropout/RNG call order changed")
            require(torch.equal(candidate.serve(actual), base.serve(expected)), "Mean raw serving changed")
            require(set(actual_details) == set(expected_details), "return_details contract changed")
            for key in ("raw_left", "raw_right", "routed_left", "routed_right"):
                require(torch.equal(actual_details[key], expected_details[key]), f"Completion details differ: {key}")
            for lhs, rhs in zip(actual_details["transformed"], expected_details["transformed"]):
                require(torch.equal(lhs, rhs), "Context transform changed")
            for lhs, rhs in zip(actual_details["scores"], expected_details["scores"]):
                require(all(torch.equal(a,b) for a,b in zip(lhs,rhs)), "Recursive scores changed")
            require(not actual_details["raw_left"].requires_grad and not actual_details["raw_right"].requires_grad,
                    "Completion scores ceased to be detached")
            identical_inherited_state()

    # Direct target interface: deterministic anisotropic cotangent, no task law.
    candidate.eval()
    candidate.zero_grad(set_to_none=True)
    operands = torch.stack((torch.arange(1, recipe.hidden + 1, dtype=torch.float32),
                            torch.arange(2, recipe.hidden + 2, dtype=torch.float32)))
    one_query = torch.tensor([[0,1]], dtype=torch.long)
    cotangent = torch.arange(1, recipe.hidden + 1, dtype=torch.float32)
    pullback = sum((candidate.decoder.endpoint_product(operands, one_query, member) * cotangent).sum()
                   for member in range(recipe.member_count))
    pullback.backward()
    for member in range(recipe.member_count):
        grad = candidate.decoder.v.grad[member]
        off_axis = torch.arange(recipe.hidden) != member
        require(bool(torch.isfinite(grad).all()) and bool((grad[off_axis] != 0).any()),
                "Off-axis target derivative is inaccessible")

    # Same accessibility through the real native decode and served raw mean.
    candidate.zero_grad(set_to_none=True)
    served = candidate.serve(candidate.decoder(h, graph, queries, "private"))
    served.square().sum().backward()
    for member in range(recipe.member_count):
        grad = candidate.decoder.v.grad[member]
        off_axis = torch.arange(recipe.hidden) != member
        require(bool(torch.isfinite(grad).all()) and bool((grad[off_axis] != 0).any()),
                "Off-axis served native derivative is inaccessible in the fixed fixture")
    require(torch.equal(candidate.decoder.v.detach(), torch.eye(recipe.member_count, recipe.hidden)),
            "Fixture updated reflection parameters")
    identical_inherited_state()
    return {
        "fixture": "one-bounded-fabricated-cpu-check-v1",
        "axis_initial_target_logits_exact": True,
        "inherited_state_exact": True,
        "constructor_and_forward_rng_exact": True,
        "recursive_completion_context_contract_retained": True,
        "off_axis_product_and_served_derivatives_accessible": True,
        "optimizer_steps": 0,
        "fits": 0,
        "data_checkpoint_outcome_payloads_read": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2, sort_keys=True))
