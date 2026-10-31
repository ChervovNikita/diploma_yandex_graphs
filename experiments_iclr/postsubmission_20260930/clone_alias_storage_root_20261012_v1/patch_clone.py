"""Preserve identical tensor views within each independent CPU state snapshot."""
from pathlib import Path
import ast

REPLACEMENT = '''def _clone_tree(torch, value):
    # A shared module has distinct state_dict keys pointing at the same tensor.
    # Memoize only identical dense views; different views and independent bodies
    # retain separate clones. Every top-level call owns a fresh CPU snapshot.
    dense = {}

    def clone(item):
        if torch.is_tensor(item):
            if item.layout == torch.strided and not item.is_quantized:
                storage = item.untyped_storage()
                key = (item.device, item.dtype, storage.data_ptr(), storage.nbytes(),
                       item.storage_offset(), tuple(item.shape), tuple(item.stride()),
                       item.is_conj(), item.is_neg())
                if key not in dense:
                    dense[key] = item.detach().cpu().clone()
                return dense[key]
            return item.detach().cpu().clone()
        if isinstance(item, Mapping):
            result = type(item)((copy.deepcopy(k), clone(v)) for k, v in item.items())
            if hasattr(item, "_metadata"):
                result._metadata = copy.deepcopy(item._metadata)
            return result
        if isinstance(item, tuple):
            return tuple(clone(v) for v in item)
        if isinstance(item, list):
            return [clone(v) for v in item]
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise TypeError("Checkpoint state must be a plain tensor/container tree")

    return clone(value)
'''

def apply(path):
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_clone_tree')
    lines = source.splitlines(keepends=True)
    successor = ''.join(lines[:node.lineno-1]) + REPLACEMENT + ''.join(lines[node.end_lineno:])
    before = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(successor).body if isinstance(n, ast.FunctionDef)}
    assert [k for k in before if before[k] != after[k]] == ['_clone_tree']
    path.write_text(successor)
    return {'changed_AST_functions': ['_clone_tree'], 'source_optimizer_or_training_law_changed': False}
