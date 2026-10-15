"""Source-only preparation checks: no numerical import or native execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent


def main():
    pins = json.loads((HERE / 'SOURCES.json').read_text())['source_files']
    for row in pins:
        path = PHASE / row['path_relative_to_phase']
        data = path.read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    for path in HERE.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    module = ast.parse((HERE / 'session.py').read_text())
    owner = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'BUDDYContextSession')
    step = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_step')
    calls = [n for n in ast.walk(step) if isinstance(n, ast.Call)]
    auxiliary = [n for n in calls if ast.unparse(n.func) == 'self.objective.context_auxiliary']
    assert len(auxiliary) == 1
    assert any(k.arg == 'reduction' and isinstance(k.value, ast.Constant) and k.value.value == 'joint' for k in auxiliary[0].keywords)
    assert sum(ast.unparse(n.func) == 'loss.backward' for n in calls) == 1
    assert sum(ast.unparse(n.func) == 'n.optimizer.step' for n in calls) == 1
    assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'loss' for t in n.targets)
        and ast.unparse(n.value) == "F + extra['loss']" for n in step.body)
    assert not any(isinstance(n, ast.Call) and (ast.unparse(n.func).endswith('.add_module')
        or ast.unparse(n.func).startswith(('self.t.nn.', 't.nn.'))) for n in ast.walk(owner))
    result = {'schema': 'BUDDY-C3-source-preparation-checks-v1', 'status': 'passed-source-checks-only',
        'dependency_files_exact': len(pins), 'all_new_Python_sources_compile': True,
        'auxiliary_reduction_is_joint': True, 'one_training_backward_and_original_optimizer_step': True,
        'native_F_plus_exact_qualified_auxiliary': True, 'no_new_model_modules_or_Parameters': True,
        'factual_context_and_tensor_core_reused_unchanged': True,
        'native_constructions': 0, 'native_forward_calls': 0, 'optimizer_steps': 0, 'TEST_accessed': False,
        'native_qualification': 'pending; this check does not execute the callback, source sampler, loss, checkpoint or serving'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
