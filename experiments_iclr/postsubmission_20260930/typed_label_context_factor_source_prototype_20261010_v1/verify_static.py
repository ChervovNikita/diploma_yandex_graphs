"""Only AST/JSON source inspection; never imports prototype/provider/model code."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    sources = {p.name: p.read_text() for p in ROOT.glob("*.py") if p.name != "verify_static.py"}
    trees = {name: ast.parse(text, filename=name) for name, text in sources.items()}
    for name, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ""])
                if any(x.split(".")[0] in {"torch", "numpy", "dgl", "torch_sparse"} for x in modules):
                    raise AssertionError("No provider imports at module scope: " + name)
    caps = json.loads((ROOT / "RUNTIME_TEMPLATE_DISABLED.json").read_text())
    assert caps["enabled"] is False
    assert all(value is False for value in caps["capabilities"].values())
    assert caps["root_review_sha256"] is None
    model = sources["model.py"]
    context = sources["context.py"]
    objective = sources["objective.py"]
    assert 'CONTEXT_SITES = ("feature_projection.0", "feature_projection.4")' in model
    assert 'self.context_U = torch.nn.Parameter(torch.zeros(' in model
    assert 'h.uniform_(-bound, bound, generator=generator)' in model
    assert 'self.private_names if shared else self.slow_names+self.private_names' in model
    assert 'object.__setattr__(site, "_context_field", None)' in model
    assert 'source[ctx_ids] = y[ctx_positions]' in context
    assert 'query_ids = plan.halves[1-index]' in context
    assert 'signed_mass = 2 * labels[key] - mass' in context
    assert 'for key in LABEL_PATHS' in context
    assert '(len(view.query_ids)/total_queries)' in objective
    assert 'if bank.shared else sum(member_losses)' in objective
    assert 'torch.stack(context_logits).mean(dim=0)' in objective
    assert 'member_bank.mean(dim=0)' in objective
    assert not any(x in objective for x in ("optimizer.step", ".backward(", "torch.save", "subprocess"))
    assert '"scientific"' in objective.split('def paired_epoch_loss', 1)[1].split('def serving_logits', 1)[0]
    result = {"status": "static_source_checks_passed", "modules_AST_parsed": len(trees),
              "prototype_modules_imported": False, "providers_imported": False,
              "model_or_tensor_or_data_execution": False, "scientific_cap_enabled": False,
              "claim_limit": "Source structure only; numerical identity/gradient/quality unqualified"}
    (ROOT / "STATIC_CHECKS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
