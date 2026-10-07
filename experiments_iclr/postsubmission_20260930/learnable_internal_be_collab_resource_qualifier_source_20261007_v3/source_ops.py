"""Named, unmodified scopes from the sealed v4 runner; no framework import.

The source has nested helpers rather than a public training-step API. This
wrapper compiles only reviewed named scopes. It never calls run.main/admit or
the epoch loop. All globals/captures are explicit in NAMESPACE_KEYS below.
"""
import ast
import hashlib
from pathlib import Path

NAMESPACE_KEYS = ('job', 'config', 'output', 'PHASE', 'ROOT', 'json', 'random',
    'np', 'torch', 'Ensemble', 'native_sources', 'load_projection', 'batches',
    'valid_batches', 'bound', 'objective', 'own_supervision', 'factor_counts',
    'local_transition', 'finite_predictions', 'finite_state',
    'LinkEvaluator', 'GraphEvaluator')
HELPERS = ('forward', 'serving', 'evaluate', 'cpu_tree', 'save')


def scopes(path):
    tree = ast.parse(Path(path).read_text())
    main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
    body = next(x for x in main.body if isinstance(x, ast.Try)).body
    start = next(i for i, x in enumerate(body) if isinstance(x, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'seed' for t in x.targets))
    stream_loop = next(i for i, x in enumerate(body) if isinstance(x, ast.For)
        and ast.unparse(x.target) == 'm' and ast.unparse(x.iter) == 'range(model.members)')
    bootstrap = body[start:stream_loop+1]
    helpers = [next(x for x in body if isinstance(x, ast.FunctionDef) and x.name == name)
               for name in HELPERS]
    epoch = next(x for x in body if isinstance(x, ast.For) and ast.unparse(x.target) == 'epoch')
    batch = next(x for x in epoch.body if isinstance(x, ast.For)
                 and ast.unparse(x.target) == '(batch, y)')
    first = next(i for i, x in enumerate(epoch.body) if isinstance(x, ast.If)
                 and ast.unparse(x.test) == 'not ordinary_independent and metric > best')
    last = next(i for i, x in enumerate(epoch.body) if isinstance(x, ast.If)
                and ast.unparse(x.test) == 'model.independent')
    result = {'bootstrap': bootstrap, 'helpers': helpers, 'TRAIN_batch': batch.body,
              'checkpoint_branches': epoch.body[first:last+1]}
    if not any(isinstance(x, ast.FunctionDef) and x.name == 'optimizer' for x in bootstrap):
        raise ValueError('Exact source optimizer capture absent')
    return result


def contract(path):
    return {name: {'first_line': min(x.lineno for x in nodes),
            'last_line': max(x.end_lineno for x in nodes),
            'ast_sha256': hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),
                include_attributes=False).encode()).hexdigest()}
            for name, nodes in scopes(path).items()}


def execute(nodes, namespace, source_path):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(source_path), 'exec'), namespace)


def bootstrap(path, namespace):
    missing = set(NAMESPACE_KEYS)-set(namespace)
    if missing:
        raise ValueError('Missing explicit v4 captures: '+str(sorted(missing)))
    selected = scopes(path)
    execute(selected['bootstrap']+selected['helpers'], namespace, path)
    namespace['steps'] = 0
    return selected
