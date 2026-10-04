"""Bounded stdlib-only syntax, source-preservation and protocol checks.

This script parses source text. It never imports the runner, models, datasets,
numerical packages, or checkpoints, and makes no performance claim.
"""

import ast
import difflib
import hashlib
import json
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parent


def tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def function(module, name, class_name=None):
    scope = module.body
    if class_name:
        scope = next(n.body for n in scope if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def check():
    results = []
    py_files = sorted(ROOT.rglob("*.py"))
    parsed = {str(p.relative_to(ROOT)): tree(p) for p in py_files}
    results.append(f"Parsed all {len(py_files)} Python source files without imports.")

    manifest = json.loads((ROOT / "PINNED_SOURCE_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / "pinned" / entry["repository_path"]
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"], path
        git_blob = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        assert git_blob == entry["expected_git_blob_sha1"], path
    results.append("Seven author files match their pinned SHA256 and Git blob identities.")

    original = parsed["pinned/OGB/model.py"]
    adapted = parsed["native/model.py"]
    for name in ("__init__", "param_init", "create_input_feat", "calculate_loss", "train"):
        assert ast.dump(function(original, name, "BaseModel")) == ast.dump(function(adapted, name, "BaseModel")), name
    for name in ("create_input_layer", "create_gnn_layer", "create_predictor_layer", "adjust_lr"):
        assert ast.dump(function(original, name)) == ast.dump(function(adapted, name)), name
    for name in ("layer.py", "loss.py", "negative_sample.py"):
        a = (ROOT / "pinned/OGB" / name).read_text().replace("from utils import *", "from .utils import *")
        assert a == (ROOT / "native" / name).read_text(), name
    for name in ("get_pos_neg_edges", "gcn_normalization", "adj_normalization", "generate_neg_dist_table"):
        assert ast.dump(function(parsed["pinned/OGB/utils.py"], name)) == ast.dump(function(parsed["native/utils.py"], name)), name
    results.append("Native construction, initialization, training, loss, sampler, encoder and predictor bodies are preserved.")

    validate = function(adapted, "validate", "BaseModel")
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "test" for n in ast.walk(adapted))
    assert not any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.lower() == "test"
                   for n in ast.walk(validate))
    assert len(calls(validate, "get_pos_neg_edges")) == 1
    assert calls(validate, "get_pos_neg_edges")[0].args[0].value == "valid"
    assert len(calls(validate, "self.encoder")) == 1
    assert not calls(parsed["train_ddi.py"], "model.test")
    assert not any(isinstance(n, ast.FunctionDef) and n.name in ("evaluate_hits", "evaluate_mrr")
                   for n in ast.walk(parsed["native/utils.py"]))
    results.append("Executable evaluation is VALID-only; no native TEST evaluator remains.")

    runner = parsed["train_ddi.py"]
    recipe = ast.literal_eval(next(n.value for n in runner.body if isinstance(n, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == "AUTHOR_RECIPE" for t in n.targets)))
    config = json.loads((ROOT / "config.json").read_text())
    assert config["release_enabled"] is False and config["recipe"] == recipe
    assert config["artifact_contract"]["path"] is None
    assert config["artifact_contract"]["train_weight_present"] is None
    assert all(config["qualification"][k] is False for k in ("train_valid_contract", "runtime", "full_budget"))
    defaults = {}
    for call in calls(function(parsed["pinned/OGB/main.py"], "argument"), "parser.add_argument"):
        name = call.args[0].value.removeprefix("--")
        default = next((kw.value for kw in call.keywords if kw.arg == "default"), None)
        if default is not None:
            if isinstance(default, ast.BinOp) and isinstance(default.op, ast.Mult):
                defaults[name] = ast.literal_eval(default.left) * ast.literal_eval(default.right)
            else:
                defaults[name] = ast.literal_eval(default)
    readme = (ROOT / "pinned/README.md").read_text()
    command = next(line for line in readme.splitlines() if line.startswith("python main.py --data_name ogbl-ddi "))
    tokens = shlex.split(command)[2:]
    author = dict(defaults)
    for i in range(0, len(tokens), 2):
        name, value = tokens[i].removeprefix("--"), tokens[i + 1]
        author[name] = type(defaults[name])(value)
    assert all(author[k] == value for k, value in recipe.items())
    assert recipe["epochs"] == 500 and recipe["batch_size"] == 65536 and recipe["eval_steps"] == 5
    results.append("Configuration equals the pinned README command plus main.py defaults; release is disabled.")

    seed_fn = function(runner, "run_seed")
    assert len(calls(seed_fn, "BaseModel")) == 1 and len(calls(runner, "BaseModel")) == 1
    assert len(calls(seed_fn, "model.param_init")) == 1
    assert calls(seed_fn, "random.seed")[0].lineno < calls(seed_fn, "BaseModel")[0].lineno
    assert calls(seed_fn, "np.random.seed")[0].lineno < calls(seed_fn, "BaseModel")[0].lineno
    assert calls(seed_fn, "torch.manual_seed")[0].lineno < calls(seed_fn, "BaseModel")[0].lineno
    main = function(runner, "main")
    seed_loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.target) == "seed")
    assert len(calls(seed_loop, "run_seed")) == 1
    checkpoint = calls(seed_fn, "torch.save")[0]
    assert checkpoint.lineno > calls(seed_fn, "model.validate")[0].lineno
    selection = next(n for n in ast.walk(seed_fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "metric > best_metric")
    assert calls(selection, "torch.save") and not any(isinstance(n, ast.Break) for n in ast.walk(seed_fn))
    assert not calls(seed_fn, "torch.load")
    load = calls(runner, "torch.load")
    assert len(load) == 1 and ast.unparse(load[0].args[0]) == "artifact_path"
    assert any(kw.arg == "weights_only" and kw.value.value is True for kw in load[0].keywords)
    release_call = calls(main, "read_release_config")[0]
    imports = [n for n in main.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(n.lineno > release_call.lineno for n in imports)
    results.append("Fresh model/optimizer per seeded call; strict VALID Hits@20 selection, no early stopping or checkpoint loading.")

    for name in ("model.py", "layer.py", "loss.py", "utils.py", "negative_sample.py"):
        a, b = ROOT / "pinned/OGB" / name, ROOT / "native" / name
        expected = "".join(difflib.unified_diff(a.read_text().splitlines(keepends=True),
                                               b.read_text().splitlines(keepends=True),
                                               fromfile=f"pinned/OGB/{name}", tofile=f"native/{name}"))
        assert (ROOT / "diffs" / (name + ".diff")).read_text() == expected
    results.append("All five pinned-to-native unified diffs are exact.")
    a, b = ROOT / "pinned/OGB/main.py", ROOT / "train_ddi.py"
    expected = "".join(difflib.unified_diff(a.read_text().splitlines(keepends=True),
                                           b.read_text().splitlines(keepends=True),
                                           fromfile="pinned/OGB/main.py", tofile="train_ddi.py"))
    assert (ROOT / "diffs/main.py_to_train_ddi.py.diff").read_text() == expected
    results.append("The pinned-main-to-adapter-entrypoint unified diff is exact.")
    return {"status": "passed", "scope": "stdlib AST/source only", "checks": results,
            "runtime_imports": False, "artifact_reads": False, "training_or_scoring": False}


if __name__ == "__main__":
    outcome = check()
    (ROOT / "SOURCE_CHECKS.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))
