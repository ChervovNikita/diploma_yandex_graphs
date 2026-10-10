"""AST/JSON/string inspection only; never imports prototype/provider/model code."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check():
    sources = {p.name: p.read_text() for p in ROOT.glob("*.py") if p.name != "verify_static.py"}
    trees = {name: ast.parse(text, filename=name) for name, text in sources.items()}
    for name, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ""])
                assert not any(x.split(".")[0] in {"torch", "numpy", "dgl", "torch_sparse", "sklearn"} for x in modules), name
    template = json.loads((ROOT / "RUNTIME_TEMPLATE_DISABLED.json").read_text())
    assert template["enabled"] is False and all(value is False for value in template["capabilities"].values())
    assert template["root_review"] is None and template["scientific_execution_approved"] is False
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert protocol["execution_enabled"] is False and protocol["scientific_admission"] is False
    assert protocol["additional_information_dim"] == 3 and protocol["local_field_dim"] == 18
    assert protocol["source_derived_generator_parameters"] == 2*4*37*(18*32+32*512) == 2*37*(18*128+128*512)
    assert len(protocol["conditions"]) == 6 and len(protocol["seed_specs"]) == 3
    model, context, objective, provider, driver, gate = (sources[n] for n in
            ("model.py", "context.py", "objective.py", "provider.py", "driver.py", "runtime_gate.py"))
    for name, text in (("model", model), ("provider", provider), ("objective", objective), ("driver", driver), ("gate", gate)):
        assert '.caps import CLOSED' in text, name
    assert 'CONTEXT_SITES = ("feature_projection.0", "feature_projection.4")' in model
    assert 'self.context_U = torch.nn.Parameter(torch.zeros(' in model
    assert 'h.uniform_(-bound, bound, generator=generator)' in model
    assert 'object.__setattr__(site, "_context_field", None)' in model
    assert 'self.private_names if shared else self.slow_names+self.private_names' in model
    assert 'source[ctx_ids] = y[ctx_positions]' in context
    assert 'query_ids = plan.halves[1-index]' in context
    assert 'signed_mass = 2 * labels[key] - mass' in context
    assert 'for key in LABEL_PATHS' in context
    assert 'tuple(node for node in ids if node in queries)' in objective
    assert 'ids = view.query_ids + inputs.validation.ids' in objective
    assert 'not set(ids) & set(view.context_ids)' in objective
    assert 'torch.stack(valid_values).mean(dim=0)' in objective
    assert 'loader.load(' in provider and 'engine.prepare_static(rt, data, costs)' in provider
    assert 'torch.utils.data.DataLoader(' in provider and 'generator=' not in provider.split('torch.utils.data.DataLoader(', 1)[1].split('plan = ', 1)[0]
    assert provider.index('train_loader = ') < provider.index('plan = ')
    assert 'same native shuffle' not in provider  # No source-only equivalence certificate.
    assert 'self.scalar.scale(own).backward()' in driver
    assert 'own = weighted/len(self.bank.members) if self.bank.shared else weighted' in driver
    assert 'loss*(len(query)/len(self.inputs.train.ids))' in driver
    assert 'for train_batch in self.inputs.train_loader' in driver
    assert 'self.caps.require("scientific")' in driver
    assert 'for epoch in range(1 if qualify else 200)' in driver
    assert 'scores["VALID"]["BCE"] < best_loss' in driver and 'epoch-best_epoch > 50' in driver
    assert 'torch.allclose(output, base, **tolerance)' in driver
    assert 'torch.equal(output, base)' not in driver
    assert '"atol": .001, "rtol": .001' in driver
    assert '"atol": 1e-6, "rtol": 1e-5' in driver
    assert 'weights_only=True' in driver and 'verify_shared_state_dict(saved["bank_state"])' in driver
    assert 'engine.exact(torch, self.scalar.state_dict(), saved["scaler_state"])' in driver
    assert 'engine.exact(torch, self.member_rng, saved["member_rng"])' in driver
    assert 'member_logits.mean(dim=0)' in driver
    assert 'TRAIN_contexts_per_row=1, VALID_contexts_per_row=2' in driver
    assert 'calls == 2*len(self.bank.members)' in driver
    assert 'scientific_novelty_or_superiority_or_acceptance_admitted' in driver
    assert 'ordinary_independent_four_body_required' in gate
    assert 'same_information_independently_selected_four_body_required' in gate
    assert not any(token in driver+provider for token in ('subprocess', 'os.system', 'ssh', 'sudo', 'teacher_logits', 'masked_reconstruction', 'label.dat.test'))
    execute = next(n for n in trees['driver.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    assert isinstance(execute.body[1], ast.Expr) and isinstance(execute.body[1].value, ast.Call)
    assert execute.body[1].value.func.attr == 'require'  # Before release/provider/data calls.
    assert not any(isinstance(n, ast.If) and '__main__' in ast.unparse(n.test) for n in trees['driver.py'].body)
    result = {"status": "static_source_checks_passed", "modules_AST_parsed": len(trees),
              "prototype_modules_imported": False, "providers_imported": False,
              "model_tensor_or_data_execution": False, "all_execution_capabilities_closed": True,
              "TRAIN_complement_and_VALID_two_context_source_structure": True,
              "source_only_claim_limit": "Syntax/declared structure only; runtime, gradients, replay, quality and costs unqualified"}
    (ROOT / "STATIC_CHECKS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    check()
