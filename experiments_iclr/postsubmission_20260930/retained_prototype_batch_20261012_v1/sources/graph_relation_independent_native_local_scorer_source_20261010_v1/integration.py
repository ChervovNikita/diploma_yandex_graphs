"""Two constructor/start-check substitutions in a private original train module."""
import ast
import copy
import hashlib
from types import SimpleNamespace

from initializer import initialize, require, stable_receipt, tensor_sha


def independent_public(constructor, public, hook_root, seed):
    """Keep original Session AST; change only the verified install callback."""
    facade = constructor.adapted_public(public, hook_root)
    original_verified = constructor.verified_hook

    def verified(root):
        hook = original_verified(root)
        original_install = hook.install_private_local_attention

        def install(body, members=4):
            controller = original_install(body, members)
            controller.independent_native_start = initialize(hook.torch, controller, seed)
            return controller

        hook.install_private_local_attention = install
        return hook

    def factory(*args, **kwargs):
        require(len(args) >= 3 and args[2] == seed, 'Exact initializer/task optimizer seed')
        constructor.verified_hook = verified
        try:
            session = facade.Session(*args, **kwargs)
        finally:
            constructor.verified_hook = original_verified
        session.independent_native_start = session.private_local_attention.independent_native_start
        require(session.steps == 0 and len(session.optimizers) == 1 and not session.optimizers[0].state,
            'Scorers initialized before original fresh Adam, zero history')
        return session

    return SimpleNamespace(Session=factory, recipe=facade.recipe)


def require_start(session, name, parameter):
    require(session.independent_native_start['before_original_Adam'] is True,
        'Independent native start receipt exists before optimizer')
    parts = name.split('.')
    layer = int(parts[4]); side = parts[6]
    expected = [row for row in session.independent_native_start['rows'] if row['layer'] == layer and row['side'] == side]
    require(len(expected) == 4 and all(tensor_sha(parameter[row['member']]) == row['row_sha256'] for row in expected),
        'Exact prospective independently generated scorer rows')


def install(base, constructor_source):
    """Reuse full driver/replay/objective/partition; only alter initial scorer rows."""
    tree = ast.parse(constructor_source.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_session')
    function = copy.deepcopy(original)
    counts = dict(constructor=0, copied_start=0)
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'constructor.adapted_public':
            node.value = ast.parse('_independent_public(constructor, public, phase_path(pins["attention_manifest"]["path"]).parent, seed)').body[0].value
            counts['constructor'] += 1
        if isinstance(node, ast.If) and ast.unparse(node.test) == "session.relation_partition['roles'][name] == 'private_local_scorer'":
            require(len(node.body) == 1 and 'All private scorer rows must copy' in ast.unparse(node.body[0]), 'Unique original copied-start assertion')
            node.body = ast.parse('_require_independent_start(session, name, parameter)').body
            counts['copied_start'] += 1
    require(counts == dict(constructor=1, copied_start=1), 'Exactly two reviewed initialization substitutions')
    adapted = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(base), _independent_public=independent_public, _require_independent_start=require_start)
    exec(compile(adapted, str(constructor_source) + ':native-scorer-initialization', 'exec'), namespace)
    base.make_session = namespace['make_session']
    return hashlib.sha256(ast.dump(adapted, include_attributes=False).encode()).hexdigest()


def receipt(session):
    return stable_receipt(session.independent_native_start)
