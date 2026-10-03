"""Independent stdlib-only source check; never imports candidate/scientific code."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CANDIDATE = BASE / "graph_ncNC_structural_pattern_parity_diagnostic_source_preparation_20261004_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return {"path": str(path.relative_to(BASE)), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def source(path):
    return ast.parse(path.read_text()), path.read_text()


def fn(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "expected":
            node.id = "reference"
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in ("state", "parity"):
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and node.value.value.id == "bound":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def norm(node):
    # Parse/unparse makes a private AST copy before name normalization.
    return ast.dump(Normalize().visit(ast.parse(ast.unparse(node))), include_attributes=False)


def run():
    checks = {}
    inspected = []
    plan = json.loads((CANDIDATE / "PLAN.json").read_text())
    manifest = json.loads((CANDIDATE / "MANIFEST.json").read_text())
    seal = json.loads((CANDIDATE / "SEAL.json").read_text())
    checks["candidate_manifest_matches_seal"] = digest(CANDIDATE / "MANIFEST.json") == seal["manifest"]["sha256"]
    for row in manifest["files"]:
        path = CANDIDATE / row["path"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], row["path"]
        inspected.append(pin(path))
    checks["all_candidate_manifest_payload_bytes_match"] = True
    inspected += [pin(CANDIDATE / name) for name in ("MANIFEST.json", "SEAL.json")]
    for packet in plan["source_packets"]:
        root = BASE / packet["root"]
        predecessor_manifest = root / "MANIFEST.json"
        assert digest(predecessor_manifest) == packet["manifest_sha256"]
        inspected.append(pin(predecessor_manifest))
        # Hash source only; do not open preserved numerical results/logs.
        for row in json.loads(predecessor_manifest.read_text())["files"]:
            if Path(row["path"]).suffix == ".py":
                path = root / row["path"]
                assert path.stat().st_size == row.get("bytes", row.get("size")) and digest(path) == row["sha256"]
                inspected.append(pin(path))
    checks["all_pinned_predecessor_manifests_and_python_sources_match"] = True
    for row in plan["authority_files"] + plan["relevant_source_files"] + [plan["preserved_failure"]["verification"]]:
        path = BASE / row["path"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
        inspected.append(pin(path))
    checks["plan_authority_relevant_source_and_failure_description_pins_match"] = True
    v4 = BASE / plan["V4_root"]
    v3 = BASE / plan["V3_root"]
    tree, text = source(CANDIDATE / "parity_diagnostic.py")
    parity, parity_text = source(v4 / "checkpoint_parity.py")
    comparator, _ = source(v4 / "pattern_checks.py")
    actual = fn(parity, "actual_step")
    measured = fn(tree, "measured_step")
    pred = next(n for n in ast.walk(fn(comparator, "close")) if isinstance(n, ast.Compare)
                and isinstance(n.ops[0], ast.LtE))
    recorded_pred = next(n.value for n in fn(tree, "tensor_difference").body
                        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == "passed")
    checks["qualified_predicate_AST_identical_after_expected_to_reference_rename"] = norm(pred) == norm(recorded_pred)
    assignments = ("main", "scores", "isolated", "auxiliary", "total", "forward", "support", "before_backward", "gradients")
    def assigned(function, name):
        return next(n for n in function.body if isinstance(n, ast.Assign)
                    and any((isinstance(t, ast.Name) and t.id == name)
                            or (isinstance(t, ast.Tuple) and any(isinstance(x, ast.Name) and x.id == name for x in t.elts))
                            for t in n.targets))
    checks["native_math_assignments_AST_identical_after_helper_qualification"] = all(
        norm(assigned(actual, name)) == norm(assigned(measured, name)) for name in assignments)
    def calls(function):
        wanted = ("optimizer.zero_grad", "total.backward", "finite", "optimizer.step")
        result = []
        for n in function.body:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                normalized = Normalize().visit(ast.parse(ast.unparse(n))).body[0]
                if ast.unparse(normalized.value.func) in wanted:
                    result.append(norm(normalized))
        return result
    checks["zero_grad_backward_finite_and_native_step_order_AST_identical"] = calls(actual) == calls(measured)
    checks["exact_original_tolerance_expression"] = "tolerance = 128 * eps" in text and "eps = torch.finfo(actual.dtype).eps" in text
    checks["one_optimizer_step_call_site"] = sum(isinstance(n, ast.Call) and ast.unparse(n.func) == "optimizer.step" for n in ast.walk(tree)) == 1
    checks["fixed_two_arms_three_variants_two_updates"] = all(s in text for s in (
        'for arm in ("J", "F"):', '(("V3_A", reference_factory), ("V3_B", reference_factory),',
        '("V4_C", bound["pattern_model"].make_pattern)', 'for number in (1, 2):'))
    checks["restore_then_exact_snapshot_digest_before_every_trajectory"] = all(s in text for s in (
        'state.restore_snapshot(model, optimizer, initial)', 'restored = state.snapshot(model, optimizer)',
        'state.state_digest(restored) == report["initial_state_sha256"]'))
    checks["no_CLI_or_main_entrypoint"] = not any(isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(tree))
    imports = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)] + [alias.name for n in tree.body if isinstance(n, ast.Import) for alias in n.names]
    checks["top_level_imports_stdlib_only"] = set(imports) <= {"hashlib", "importlib", "json", "os", "pathlib", "time"}
    checks["no_tensor_or_pickle_serialization_call"] = not any(isinstance(n, ast.Call) and ast.unparse(n.func) in {"torch.save", "torch.load", "pickle.dump", "pickle.load", "atomic_torch"} for n in ast.walk(tree))
    failure = json.loads((BASE / plan["preserved_failure"]["verification"]["path"]).read_text())
    checks["caps_equal_original_failed_qualification"] = plan["caps"] == failure["caps"]
    checks["work_count_matches_plan"] = plan["work_and_cost_custody"]["planned_optimizer_updates"] == 2 * 3 * 2 == 12 and plan["work_and_cost_custody"]["planned_member_trajectory_updates"] == 48
    checks["separate_diagnostic_admission_and_no_scientific_scope"] = all(s in text for s in (
        'ncnc-pattern-parity-diagnostic-root-admission-v1', 'admission.get("status") == "APPROVED"',
        'admission.get("scientific_fit_admitted") is False', 'admission.get("state_donor") is False',
        'admission.get("TEST_supported") is False'))
    v3_model, _ = source(v3 / "pattern_model.py")
    v4_model, _ = source(v4 / "pattern_model.py")
    checks["V3_V4_factory_constructor_AST_identical"] = all(
        ast.dump(fn(v3_model, name), include_attributes=False) == ast.dump(fn(v4_model, name), include_attributes=False)
        for name in ("make_pattern", "__init__"))
    assert all(checks.values()), {key: value for key, value in checks.items() if not value}
    evidence = {"schema": "independent-ncnc-diagnostic-source-checks-v1", "status": "PASS_SOURCE_ONLY",
                "scientific_or_candidate_module_imported": False, "runtime_or_GPU_executed": False,
                "checks": checks,
                "limitations": ["AST and source byte checks establish static properties only.",
                    "No native optimizer, Torch tensor, CUDA arithmetic, dataset, saved score, checkpoint, or TEST was opened or executed.",
                    "Only predecessor manifest metadata and Python source were hash-verified; preserved numerical receipts/logs were not opened."]}
    (HERE / "CHECKS.json").write_text(json.dumps(evidence, indent=2) + "\n")
    unique = {row["path"]: row for row in inspected}
    (HERE / "INSPECTED_HASHES.json").write_text(json.dumps({"files": [unique[key] for key in sorted(unique)]}, indent=2) + "\n")
    print(json.dumps({"checks_passed": len(checks), "source_and_metadata_files_hash_verified": len(unique),
                      "candidate_manifest_sha256": digest(CANDIDATE / "MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    run()
