"""Stdlib-only static custody/syntax check. Does not import implementation.

This is the packet's only executable preparation command. PASS does not
qualify arithmetic, tensors, gradients, random draws, data or resource use.
"""
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PLAN_MANIFEST = "aa70f5a4d89a394ef059a431c78d333af95e6ab1c0c4e8c8dbbd028ee4b8d2f9"
PLAN_SEAL = "348fd165d465c2e84777cf0093262b1899f796c815381b88232812cd95c0664c"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(path, entry):
    require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Source pin differs: " + str(path))


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def assignment(function_node, name):
    return next(node for node in function_node.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def signature(node):
    return [arg.arg for arg in node.args.args] + [arg.arg for arg in node.args.kwonlyargs]


class BindWidth(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Constant(value=64), node) if node.id == "width" else node


def main():
    entries, trees, texts = [], {}, {}
    for path in sorted(HERE.glob("*.py")):
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        compile(tree, str(path), "exec")  # parse/compile only, NEVER eval/exec
        entries.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "AST_compile": "PASS"})
        trees[path.stem], texts[path.stem] = tree, text
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for entry in bindings["external_sources"]:
        verify(HERE.parent / entry["path"], entry)
    for entry in bindings["copied_sources"]:
        verify(HERE / entry["copied_path"], entry)
        verify(HERE.parent / entry["path"], entry)
    require(digest(HERE / "FROZEN_PLAN" / "MANIFEST.json") == PLAN_MANIFEST, "Frozen plan manifest differs")
    require(digest(HERE / "FROZEN_PLAN" / "SEAL.json") == PLAN_SEAL, "Frozen plan seal differs")
    plan = json.loads((HERE / "FROZEN_PLAN" / "MANIFEST.json").read_text())
    for entry in plan["payload"]:
        verify(HERE / "FROZEN_PLAN" / entry["path"], entry)
    native_source = HERE.parent / "graph_ncNC_structural_pattern_pilot_preparation_20261003_v3" / "pilot_model.py"
    native_tree = ast.parse(native_source.read_text())
    current = function(trees["cardinality_model"], "make_single")
    prior = function(native_tree, "make_native")
    for name in ("encoder", "decoder"):
        expected = BindWidth().visit(assignment(prior, name))
        require(ast.dump(assignment(current, name)) == ast.dump(expected), "Exact native constructor differs: " + name)
    # Signatures make the implemented teacher boundary visible to review.
    require(signature(function(trees["cardinality_model"], "predict_law")) == ["self", "h", "graph", "queries"], "Prediction interface changed")
    require(signature(function(trees["cardinality_density"], "completion_draws")) == ["law", "queries", "counterpart_pairs", "keys", "route"], "Sampler interface changed")
    require(signature(function(trees["cardinality_teacher"], "labels")) == ["self", "counterpart_pairs"], "Teacher cannot access predicted law")
    for stem, name in (("cardinality_model", "predict_law"), ("cardinality_model", "usable_context"),
                       ("cardinality_density", "completion_draws")):
        require(not any(arg in signature(function(trees[stem], name)) for arg in ("teacher", "labels", "target", "K", "split")), "Forbidden predictor input")
    tokens = {
        "cardinality_density": ["torch.logaddexp", "ctx.save_for_backward(u, offsets, observed_count)", "table = log_esp_prefix(u[start:end])  # paid recomputation",
                                "reverse_esp(u, table, pi, probability_readout=True)", "r.clamp_min(1)", "torch.where(r > 0", "ROUNDING_LOG_INCLUDE = 1e-10", "del table"],
        "cardinality_counter": ["DENOMINATOR = 2**53", "math.log1p((value - DENOMINATOR + .5) / DENOMINATOR)", "purpose={purpose}", "canonical_pair(counterpart)"],
        "cardinality_model": ["torch.random.fork_rng(devices=[])", 'generator.manual_seed(separated_seed("count-init"))', "nn.Linear(270, 16", "nn.Linear(16, 16", "nn.Linear(16, 1",
                              "left_queries.T, depth=0", "right_queries.T, depth=0", "1.05 * draw", "weight.detach()", "logits.mean(1)"],
        "cardinality_training": ["ordinal_start", "route=\"D4\"", "F.logsigmoid(records[0][\"logits\"]).mean()", "records[0][\"nll_per_query\"].mean()", "total = main + 1. * auxiliary", "len(rows) == 17"],
        "cardinality_serving": ["self.completed_selection_states == 100", "complete_d4_hits50 > self.best_value", "positive > torch.topk(negative, 50).values[-1]", "loaded_checkpoint_sha256 == selected_bank.checkpoint_sha256", "131072, False"],
    }
    for stem, tokens_for_module in tokens.items():
        for token in tokens_for_module:
            require(token in texts[stem], "Required static semantic signature absent: " + stem + ":" + token)
    require(not function(trees["cardinality_model"], "predict_law").decorator_list, "Auxiliary scorer cannot be blanket detached")
    for stem, tree in trees.items():
        if stem != "source_check":
            require(not any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                            and any(isinstance(value, ast.Name) and value.id == "__name__" for value in ast.walk(node.test))
                            for node in ast.walk(tree)), "Runtime CLI/entry point is not admitted")
    require(sum(a * b + b for a, b in ((270, 16), (16, 16), (16, 1))) == 4625, "Head integer formula differs")
    pending = json.loads((HERE / "QUALIFICATION_PENDING.json").read_text())
    require(pending["status"] == "SOURCE_ONLY_UNQUALIFIED_UNRELEASED" and pending["runtime_fixture_gates_executed"] == 0
            and pending["numerical_or_full_graph_or_fit_launches"] == 0 and pending["scientific_execution_release"] is None, "Execution claim/authorization unexpectedly present")
    for name in sys.modules:
        require(name.split(".")[0] not in ("torch", "numpy", "scipy", "pandas", "torch_sparse", "torch_scatter", "torch_geometric"), "Static check imported a runtime module")
    result = {"schema": "ncnc-cardinality-single-static-source-check-v1", "UTC": datetime.now(timezone.utc).isoformat(),
              "status": "PASS_SOURCE_ONLY", "python_sources": entries, "external_pins_verified": len(bindings["external_sources"]),
              "unchanged_copies_verified": len(bindings["copied_sources"]), "plan_manifest_sha256": PLAN_MANIFEST,
              "plan_seal_sha256": PLAN_SEAL, "exact_native_constructor_AST": "PASS",
              "source_formula_head_parameters": 4625, "runtime_implementation_imported": False,
              "model_instantiated": False, "numerical_fixture_executed": False, "project_data_read": False,
              "runtime_qualification_claimed": False, "scientific_or_engineering_release_created": False,
              "limits": "AST/signatures and hashes only; runtime arithmetic, gradients, tensors, RNG and resource gates remain pending"}
    (HERE / "STATIC_SOURCE_CHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "python_sources": len(entries), "external_pins": result["external_pins_verified"],
                      "unchanged_copies": result["unchanged_copies_verified"], "runtime_executed": False}, indent=2))


if __name__ == "__main__":
    main()
