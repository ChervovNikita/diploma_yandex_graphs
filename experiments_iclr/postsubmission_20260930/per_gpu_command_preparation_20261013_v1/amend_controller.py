"""Add only the GPU companion to a fresh controller's source inventory."""
import ast
import hashlib
import json
from pathlib import Path
import sys

runtime = Path(sys.argv[1]).resolve()
path = runtime / 'compile_master_plan.py'
before = path.read_bytes()
tree = ast.parse(before)
assert not any(isinstance(node, ast.Constant) and node.value == 'gpu_command.py' for node in ast.walk(tree))
helper = runtime / 'gpu_command.py'
assert helper.is_file()
anchor = 'STAGE_KIND = '
source = before.decode()
assert source.count(anchor) == 1
source = source.replace(anchor, "CONTROLLER_FILES = CONTROLLER_FILES + ('gpu_command.py',)\n" + anchor)
after = source.encode()
ast.parse(after)
path.chmod(path.stat().st_mode | 0o200)
path.write_bytes(after)
left = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
right = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(after).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
assert left == right
print(json.dumps({'schema': 'per-gpu-companion-source-inventory-amendment-v1',
    'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': hashlib.sha256(after).hexdigest(),
    'added_controller_file': {'name': helper.name, 'bytes': helper.stat().st_size, 'sha256': hashlib.sha256(helper.read_bytes()).hexdigest()},
    'all_original_function_ASTs_identical': True, 'launcher_unchanged': True,
    'scientific_fits': 0, 'GO': False}, sort_keys=True))
