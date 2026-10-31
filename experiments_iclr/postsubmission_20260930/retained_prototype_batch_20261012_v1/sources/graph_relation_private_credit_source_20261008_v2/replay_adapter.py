"""Adapt exactly two sites in the public replay; retain its full RNG/Adam engine."""
import ast
import copy
import hashlib
from types import MethodType
from permissions import unchanged

PUBLIC_REPLAY_SHA = '120d6863fe57da53e115c78d8adc9ff39864ae932b0479df84686479f19f86a6'
MODE = 'graph_relation_credit_two_disjoint_member_view_VJPs_v1'


def output_credit(session, outputs, labels):
    unchanged(session)
    torch = session.torch; la, ha, lb, hb = outputs
    losses = session.core['objectives']
    Fa = losses.own_supervision(la, labels, 'wikics').mean()
    Fb = losses.own_supervision(lb, labels, 'wikics').mean()
    La = session.relation_pool(la, labels, 'wikics', torch)
    Lb = session.relation_pool(lb, labels, 'wikics', torch)
    F = .5 * (Fa + Fb); L = .5 * (La + Lb)
    J = .5 * (.5 * Fa + .5 * La) + .5 * (.5 * Fb + .5 * Lb)
    if not all(torch.isfinite(value) for value in (F, L, J)):
        raise FloatingPointError('Nonfinite full-TRAIN F/L/J')
    cotangents = {}
    # Every condition charges both output collections. The two derived fields
    # support the matched diagnostic and each group's fixed F/J choice.
    for name, loss, retain in (('F', F, True), ('J', J, False)):
        raw = torch.autograd.grad(loss, outputs, retain_graph=retain, create_graph=False, allow_unused=True)
        session.execution_totals['output_cotangent_collections'] += 1
        values = tuple(torch.zeros_like(value) if grad is None else grad.detach()
                       for value, grad in zip(outputs, raw))
        if any(not torch.isfinite(value).all() for value in values):
            raise FloatingPointError('Nonfinite F/J output cotangent')
        cotangents[name] = values
    session.last_relation_losses = dict(F=float(F.detach()), pool=float(L.detach()), J=float(J.detach()),
        auxiliary=0., risk_beta=.5, reported_scalars_are_diagnostics_for_block_fields=True)
    diagnostic = J if session.relation_partition['policy'] == 'allJ' else F
    return dict(loss=diagnostic.detach(), own_mean=F.detach(), auxiliary=F.detach() * 0), outputs, cotangents


def collect(session, view, member, logits, representation, cotangents):
    torch = session.torch
    groups = session.relation_partition['groups']; selectors = session.relation_partition['selectors']
    for index, (group, loss_name) in enumerate(zip(groups, selectors)):
        weights = cotangents[loss_name]
        gradients = torch.autograd.grad((logits, representation), tuple(p for _, p in group),
            grad_outputs=(weights[2 * view][member], weights[2 * view + 1][member]),
            retain_graph=index == 0, create_graph=False, allow_unused=True)
        session.execution_totals['member_reverse_collections'] += 1
        for (_, parameter), gradient in zip(group, gradients):
            if gradient is None:
                continue  # Native inactive global/head paths stay None, not fabricated zero gradients.
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite disjoint block VJP')
            if parameter.grad is None:
                parameter.grad = gradient.detach()
            else:
                parameter.grad.add_(gradient.detach())


def compiled_replay(original, filename):
    source = filename.read_text()
    if hashlib.sha256(filename.read_bytes()).hexdigest() != PUBLIC_REPLAY_SHA:
        raise ValueError('Exact public RNG/old-state replay source required')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_step']
    if len(functions) != 1:
        raise ValueError('Unique public replay function required')
    function = copy.deepcopy(functions[0])
    starts = [i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'size' for t in n.targets)]
    ends = [i for i, n in enumerate(function.body) if isinstance(n, ast.Delete)
            and [t.id for t in n.targets if isinstance(t, ast.Name)] == ['total', 'own', 'auxiliary', 'raw']]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError('Exact public output-loss fragment changed')
    replacement = ast.parse('result, outputs, cotangents = _output_credit(session, (la, ha, lb, hb), labels)').body
    function.body[starts[0]:ends[0] + 1] = replacement

    class Replace(ast.NodeTransformer):
        backwards = increments = 0
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and ast.unparse(call.func) == 'torch.autograd.backward':
                self.backwards += 1
                return ast.copy_location(ast.parse(
                    '_relation_collect(session, view, member, logits, representation, cotangents)').body[0], node)
            return self.generic_visit(node)
        def visit_AugAssign(self, node):
            if ast.unparse(node.target) == "session.execution_totals['member_reverse_collections']":
                self.increments += 1
                return None
            return self.generic_visit(node)

    transform = Replace(); function = transform.visit(function)
    if transform.backwards != 1 or transform.increments != 1:
        raise ValueError('Exactly one public member reverse site/counter must be replaced')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    namespace = dict(vars(original), _output_credit=output_credit, _relation_collect=collect)
    exec(compile(module, str(filename) + ':graph-relation-credit', 'exec'), namespace)
    return namespace['train_step'], digest


def install(session, original, filename):
    if session.model.contrastive:
        raise ValueError('No embedding auxiliary can enter this study')
    original.install(session, diagnostics=False)
    implementation, digest = compiled_replay(original, filename)
    session.train_step = MethodType(implementation, session)
    session.relation_replay_AST_sha256 = digest
    session.relation_replay_mode = MODE
    return session
