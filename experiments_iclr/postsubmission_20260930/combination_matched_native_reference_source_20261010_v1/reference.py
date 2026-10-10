"""One pre-Adam insertion; numerical Session/driver/selector sources unchanged."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from initializer import initialize, require, stable


def constructor_tree(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Session')
    original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    function = copy.deepcopy(original)
    positions = [index for index, node in enumerate(function.body)
                 if isinstance(node, ast.FunctionDef) and node.name == 'optimizer']
    require(len(positions) == 1, 'Unique original pre-Adam insertion')
    function.body.insert(positions[0], ast.parse('_matched_native_scorers(self)').body[0])
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def adapted_public(public, paired_seed, member_index, original_row_seed, expected_public_sha):
    source = Path(public.__file__)
    require(hashlib.sha256(source.read_bytes()).hexdigest() == expected_public_sha,
            'Pinned original native constructor')
    tree = constructor_tree(source.read_text())
    ast_sha = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()

    def install(session):
        session.matched_native_start = initialize(session, paired_seed, member_index, original_row_seed)
        session.matched_native_constructor_AST_sha256 = ast_sha

    namespace = dict(vars(public), _matched_native_scorers=install)
    exec(compile(tree, str(source)+':one-native-scorer-insertion', 'exec'), namespace)
    native_session = type('MatchedNativeM1Session', (public.Session,),
        {'__init__': namespace['__init__'], '__module__': __name__})
    return SimpleNamespace(Session=native_session, recipe=public.recipe, constructor_AST_sha256=ast_sha)


def install_accounting(session):
    require(session.model.members == len(session.optimizers) == 1 and session.steps == 0
            and not session.optimizers[0].state and not session.model.contrastive,
            'Fresh M1 F-only body and one original private Adam')
    owners = [id(parameter) for group in session.optimizers[0].param_groups for parameter in group['params']]
    require(len(owners) == len(set(owners)) and set(owners) == {id(p) for p in session.model.parameters()},
            'Every native parameter has exactly one optimizer owner')
    optimizer = session.optimizers[0]
    require(all(group['lr'] == .001 and group['eps'] == 1e-8 and group['weight_decay'] == 0
                and tuple(group['betas']) == (.9, .999) for group in optimizer.param_groups),
            'Original native Adam settings, no factor or ensemble division')
    session.matched_reference_work = dict(TRAIN_attempts=0, completed_TRAIN_updates=0,
        actual_TRAIN_member_view_forwards=0, differentiated_own_CE_view_terms=0,
        original_total_backward_invocations=0, optimizer_steps=0, VALID_evaluations=0,
        VALID_member_forwards=0)
    original_forward, original_step = session.forward, session.train_step
    session._matched_inside_TRAIN = False
    session._matched_failed = False

    def forward(batch):
        result = original_forward(batch)
        if session._matched_inside_TRAIN:
            require(result[0].shape[:2] == (1, 580), 'Complete M1 TRAIN view')
            session.matched_reference_work['actual_TRAIN_member_view_forwards'] += 1
        return result

    def step(batch, labels):
        require(not session._matched_failed and not session._matched_inside_TRAIN
                and len(labels) == 580 and batch['ids'].numel() == 580,
                'One complete own-label update; failed updates cannot be retried')
        work = session.matched_reference_work
        work['TRAIN_attempts'] += 1
        before_steps = session.steps
        before_forwards = work['actual_TRAIN_member_view_forwards']
        session._matched_inside_TRAIN = True
        try:
            result = original_step(batch, labels)  # Unchanged .5CE(view0)+.5CE(view1), one backward/Adam.
            require(session.steps == before_steps+1
                    and work['actual_TRAIN_member_view_forwards'] == before_forwards+2,
                    'Original two-own-view old-state update completed')
            work['completed_TRAIN_updates'] += 1
            work['differentiated_own_CE_view_terms'] += 2
            work['original_total_backward_invocations'] += 1
            work['optimizer_steps'] += 1
            return result
        except BaseException:
            session._matched_failed = True
            raise
        finally:
            session._matched_inside_TRAIN = False

    session.forward, session.train_step = forward, step
    session.matched_optimizer_parameter_ids = set(owners)


def descriptor(session, spec, manifest_sha, train_sha):
    return dict(schema='matched-native-own-reference-body-v1', spec=spec,
        source_manifest_sha256=manifest_sha, train_program_sha256=train_sha,
        native_scorer_start=stable(session.matched_native_start),
        constructor_AST_sha256=session.matched_native_constructor_AST_sha256,
        core=session.core_provenance, native=session.native_provenance,
        body_count=1, optimizer_count=1, dense_parameterization='Ordinary free native matrices; no BE factors',
        objective='.5 CE(view0)+.5 CE(view1); original M1 own.sum; all native coordinates on own risk',
        backward_count_definition='One original total.backward per update differentiates two CE view terms',
        work=dict(session.matched_reference_work), selected_definition='Original M1 strict-first complete VALID maximum',
        own_local_restore_and_live_end_local_streams=True, independent_body_no_J_or_pooled_gradient=True,
        bitwise_parity_claimed=False, qualification_is_separate_release_custody=True, exact_resume_supported=False)
