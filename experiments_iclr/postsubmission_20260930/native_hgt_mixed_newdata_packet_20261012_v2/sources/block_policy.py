"""Module-semantic block gradients; no model, optimizer, RNG or data changes."""
import ast
import hashlib
import inspect

POLICIES = ('own/own', 'pool/pool', 'pool/own', 'own/pool')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def partition(implementation, model):
    """Exhaustive identity partition of the immutable global_BE module tree.

    Names are recorded after ownership is established, never used to choose a
    block. None gradients stay None, including unused final non-target maps.
    """
    require(type(model) is implementation.PrivateHGT, 'Exact PrivateHGT required')
    require(type(model.core) is implementation.NativeHGT, 'Exact shared native core required')
    require(set(model._modules) == {'core', 'factors'} and not model._parameters,
            'Unexpected top-level parameter ownership')
    require(type(model.factors) is implementation.nn.ModuleList and len(model.factors) == len(model.core.gcs),
            'Exactly one private factor module per native layer required')
    width = model.core.out.in_features
    private = []
    for factor in model.factors:
        require(type(factor) is implementation.PrivateFactors and factor.mode == 'be',
                'Only global_BE fast factors; CP excluded')
        require(set(factor._parameters) == {'a', 'b'} and not factor._modules and not factor._buffers,
                'Private modules must contain only member a/b tables')
        require(tuple(factor.a.shape) == tuple(factor.b.shape) == (4, width),
                'Four route-private a/b rows required')
        private.extend(factor.parameters())
    shared = list(model.core.parameters())
    named = list(model.named_parameters(remove_duplicate=False))
    require(named and all(p.requires_grad for _, p in named), 'All frozen parameters must remain trainable')
    all_ids = [id(p) for _, p in named]
    shared_ids, private_ids = {id(p) for p in shared}, {id(p) for p in private}
    require(len(all_ids) == len(set(all_ids)) and not shared_ids.intersection(private_ids)
            and shared_ids.union(private_ids) == set(all_ids)
            and len(shared) == len(shared_ids) and len(private) == len(private_ids),
            'Aliased, missing or overlapping block parameters')
    names = {id(p): name for name, p in named}
    receipt = dict(schema='HGT_global_BE_semantic_partition_v1', members=4,
        shared_owner='model.core: adapters, typed maps, relation maps/prior, skip, norms, classifier',
        private_owner='each model.factors layer: be-mode member a/b tables',
        shared=[dict(name=names[id(p)], shape=list(p.shape), numel=p.numel()) for p in shared],
        private=[dict(name=names[id(p)], shape=list(p.shape), numel=p.numel(), member_axis=0) for p in private],
        buffers=[dict(name=name, shape=list(value.shape)) for name, value in model.named_buffers()],
        exhaustive=True, disjoint=True, alias_free=True, CP_excluded=True)
    return dict(shared=tuple(shared), private=tuple(private), names=names, receipt=receipt)


def backward(torch, policy, blocks, logits, ids, labels, own_loss):
    """One existing forward; original own normalization; one later optimizer step."""
    require(policy in POLICIES, 'Unknown objective assignment')
    require(logits.ndim == 3 and logits.shape[0] == 4, 'Four native member logits required')
    if policy == 'own/own':
        # Training uses immutable driver.fit directly. This branch is only the
        # focused step witness and has the identical original backward call.
        own_loss.backward()
        return
    pooled = torch.nn.functional.cross_entropy(logits.mean(0)[ids], labels)
    require(bool(torch.isfinite(pooled)), 'Nonfinite TRAIN pooled CE')
    if policy == 'pool/pool':
        pooled.backward()
        return
    shared_loss, private_loss = (pooled, own_loss) if policy == 'pool/own' else (own_loss, pooled)
    shared_grad = torch.autograd.grad(shared_loss, blocks['shared'], retain_graph=True, allow_unused=True)
    private_grad = torch.autograd.grad(private_loss, blocks['private'], allow_unused=True)
    for parameters, gradients in ((blocks['shared'], shared_grad), (blocks['private'], private_grad)):
        for parameter, gradient in zip(parameters, gradients):
            # No zeros for unused leaves: AdamW skips them exactly as native
            # backward does. Neither gradient source advances buffers or RNG.
            parameter.grad = gradient


def fit_tree(source):
    """Replace exactly one original backward expression; every other AST node stays."""
    tree = ast.parse(source)
    original = ast.dump(tree, include_attributes=False)
    found = []
    class Replace(ast.NodeTransformer):
        def visit_Expr(self, node):
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == 'train_loss' and value.func.attr == 'backward'):
                require(not value.args and not value.keywords, 'Original backward signature drift')
                replacement = ast.Expr(value=ast.Call(func=ast.Name(id='_mixed_backward', ctx=ast.Load()),
                    args=[ast.Name(id=name, ctx=ast.Load()) for name in ('model', 'logits', 'ids', 'labels', 'train_loss')],
                    keywords=[]))
                found.append(replacement)
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)
    changed = Replace().visit(tree)
    require(len(found) == 1, 'Exactly one immutable fit backward call required')
    ast.fix_missing_locations(changed)
    derived = ast.dump(changed, include_attributes=False)
    # Independent structural reverse projection of the inspectable generated AST.
    class Reverse(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == '_mixed_backward'):
                require(ast.unparse(node.value) == '_mixed_backward(model, logits, ids, labels, train_loss)',
                        'Derived hook signature drift')
                return ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='train_loss', ctx=ast.Load()),
                    attr='backward', ctx=ast.Load()), args=[], keywords=[]))
            return self.generic_visit(node)
    restored = Reverse().visit(ast.parse(ast.unparse(changed)))
    require(ast.dump(restored, include_attributes=False) == original, 'Non-backward fit body mutation')
    custody = dict(schema='immutable_fit_one_backward_substitution_v1', substitutions=1,
        original_fit_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_fit_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_fit_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        reverse_projection_exact=True, own_own='imported original driver.fit, no substitution',
        replacement='_mixed_backward(model, logits, ids, labels, train_loss)',
        unchanged='forward, original mean_member_ce, optimizer/OneCycle, validation selector, checkpoint, RNG and selected replay')
    return changed, custody


def fit_function(torch, implementation, driver, model, policy, blocks):
    if policy == 'own/own':
        return driver.fit
    tree, _ = fit_tree(inspect.getsource(driver.fit))
    def hook(actual_model, logits, ids, labels, train_loss):
        require(actual_model is model, 'Hook model identity changed')
        backward(torch, policy, blocks, logits, ids, labels, train_loss)
    namespace = dict(driver.__dict__, _mixed_backward=hook)
    exec(compile(tree, str(inspect.getsourcefile(driver.fit)) + ':mixed_backward_only', 'exec'), namespace)
    return namespace['fit']
