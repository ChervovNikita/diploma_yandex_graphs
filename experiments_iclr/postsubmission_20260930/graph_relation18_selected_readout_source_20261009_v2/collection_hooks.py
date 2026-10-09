"""Small reconstruction hook over the sealed Wiki12 collect_cell loop."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import time
from types import SimpleNamespace


def adapt_tree(source):
    """Change only construction/restore and the alphaF cohort label, in memory."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_cell']
    if len(functions) != 1:
        raise ValueError('Unique original collect_cell required')
    function = copy.deepcopy(functions[0])
    block = next(n for n in function.body if isinstance(n, ast.Try)).body
    starts = [i for i, n in enumerate(block) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets)]
    ends = [i for i, n in enumerate(block) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'phase' for t in n.targets)
            and isinstance(n.value, ast.Constant) and n.value.value == 'inference_and_CPU_transfer']
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise ValueError('Original construction/restore boundary changed')
    replacement = ast.parse("""
phase='checkpoint_load_and_reconstruction'; phase_started=time.monotonic()
model,streams,metadata=_relation_restore(torch,public,record,source_pins,cost)
record['selected_metadata']=metadata
torch.cuda.synchronize()
record['checkpoint_load_and_restore_seconds']=time.monotonic()-phase_started
""").body
    block[starts[0]:ends[0]] = replacement
    count = {'baseline': 0, 'path': 0}
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and node.value == 'plain':
            node.value = 'alphaF'; count['baseline'] += 1
        if isinstance(node, ast.Constant) and node.value == 'plain_cohorts_':
            node.value = 'alphaF_cohorts_'; count['path'] += 1
        if isinstance(node, ast.keyword) and node.arg in ('plain_cell', 'plain_archive'):
            node.arg = node.arg.replace('plain', 'alphaF')
    if count != {'baseline': 1, 'path': 1}:
        raise ValueError('Original one baseline cohort freeze boundary changed')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def bind_collector(gate, pins, relation, legacy):
    """Reuse every original inference, loss, partial-save and failure statement."""
    path = gate.bound(pins['reuse']['legacy_collect'])
    tree = adapt_tree(path.read_text())
    namespace = dict(vars(legacy))
    namespace['MAX_FORWARDS'] = 48
    namespace['_relation_restore'] = restore_factory(gate, pins, relation)
    exec(compile(tree, str(path) + ':relation18-selected-hook', 'exec'), namespace)
    legacy.collect_cell = namespace['collect_cell']
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def restore_factory(gate, pins, relation):
    def restore(torch, public, record, source_pins, cost):
        checkpoint = gate.bound(record['selected_checkpoint'])
        began = time.monotonic()
        cost['checkpoint_deserialization_attempts'] += 1
        cost['checkpoint_bytes_submitted_to_deserializer'] += record['selected_checkpoint']['bytes']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        gate.bound(record['selected_checkpoint'])
        record['checkpoint_deserialization_seconds'] = time.monotonic() - began
        run = state['run']; policy = record['condition']; method = relation.identity(policy)
        expected = copy.deepcopy(public.recipe('wikics'))
        expected['arms'] = [method]
        expected['contrastive'].update(alignment_weight=0., residual_weight=0.)
        expected['graph_relation_credit'] = dict(policy=policy, risk_beta=.5, auxiliary=False)
        gate.require(state.get('config') == expected and run['task'] == 'wikics'
            and run['seed'] == record['seed'] and run['arm'] == run['method_identity'] == method
            and run['TEST_scoring'] is False and run['risk_beta'] == .5
            and run['source_manifest_sha256'] == pins['reuse']['relation_manifest']['sha256']
            and run['driver_sha256'] == pins['relation_source_pins']['public_driver']['sha256'],
            'Original selected relation run/seed/config/source identity')
        gate.require(run['data']['train_npz_sha256'] == source_pins['train']['sha256']
            and run['data']['valid_npz_sha256'] == source_pins['development']['sha256']
            and run['native']['polynormer_model_sha256'] == source_pins['polynormer']['sha256']
            and run['core'] == pins['public_core_sha256'], 'Selected source and full TRAIN/development role hashes')
        gate.require(type(state.get('epoch')) is int and 1 <= state['epoch'] <= 1100
            and type(state.get('global')) is bool
            and state.get('checkpoint_kind') == 'strict-first-maximum complete VALID joint snapshot'
            and math.isfinite(state['selected_VALID']) and len(state['member_VALID']) == 4
            and all(math.isfinite(x) for x in state['member_VALID']), 'Original selected epoch/mode/selector')
        streams = state['streams']
        gate.require(len(streams) == 4 and all(set(s) == {'cpu', 'cuda'}
            and all(t.device.type == 'cpu' and t.dtype == torch.uint8 for t in s.values()) for s in streams),
            'Original four selected member RNG streams for inference')
        # Observe actual fresh Adam constructions, including failed attempts.
        original_init = torch.optim.Adam.__init__
        def counted_init(instance, *args, **kwargs):
            started = time.monotonic()
            cost['optimizer_construction_attempts'] += 1
            record['optimizer_construction_attempts'] = record.get('optimizer_construction_attempts', 0) + 1
            try:
                original_init(instance, *args, **kwargs)
                cost['optimizer_constructions'] += 1
                record['optimizer_constructions'] = record.get('optimizer_constructions', 0) + 1
            finally:
                seconds = time.monotonic() - started
                cost['optimizer_construction_seconds'] += seconds
                record['optimizer_construction_seconds'] = record.get('optimizer_construction_seconds', 0.) + seconds
        original_factory = relation.make_session
        def counted_factory(public_arg, constructor, replay, pool, source, chosen_policy, seed, device, native):
            def adapted_public(*args, **kwargs):
                facade = constructor.adapted_public(*args, **kwargs)
                class ObservedSession(facade.Session):
                    def __init__(self, *session_args, **session_kwargs):
                        started = time.monotonic()
                        cost['Session_construction_attempts'] += 1
                        record['Session_construction_attempts'] = record.get('Session_construction_attempts', 0) + 1
                        try:
                            super().__init__(*session_args, **session_kwargs)
                            cost['Session_constructions'] += 1
                            record['Session_constructions'] = record.get('Session_constructions', 0) + 1
                        finally:
                            seconds = time.monotonic() - started
                            record['construction_seconds'] = record.get('construction_seconds', 0.) + seconds
                            cost['Session_construction_seconds'] += seconds
                return SimpleNamespace(Session=ObservedSession, recipe=facade.recipe)
            proxy = SimpleNamespace(adapted_public=adapted_public)
            return original_factory(public_arg, proxy, replay, pool, source, chosen_policy, seed, device, native)
        began = time.monotonic()
        torch.optim.Adam.__init__ = counted_init
        relation.make_session = counted_factory
        try:
            session = relation.reconstruct_selected(state, device='cuda:0')
        finally:
            torch.optim.Adam.__init__ = original_init
            relation.make_session = original_factory
            record['reconstruction_seconds'] = time.monotonic() - began
        gate.require(session.steps == 0 and len(session.optimizers) == 1
            and not session.optimizers[0].state and session.model.members == 4 and not session.model.independent
            and session.model.models[0].body._global is state['global']
            and session.model.contrastive is False, 'Fresh serving-only Session, zero optimizer history/steps')
        session.core['selection'].finite_state(session.model, [])
        session.model.eval()
        metadata = dict(epoch=state['epoch'], selected_epochs=[state['epoch']], global_mode=state['global'],
            selected_member_modes=[state['global']] * 4, stored_accuracy=state['selected_VALID'],
            stored_member_accuracy=state['member_VALID'], method_identity=method, saved_member_streams_used=True,
            optimizer_or_training_RNG_history_restored=False, reselection=False, bitwise_parity_claimed=False,
            parameter_counts=session.core['factors'].factor_counts(session.model),
            stored_model_tensor_bytes=sum(t.numel() * t.element_size() for t in session.model.state_dict().values()),
            graph_relation_credit=state['graph_relation_credit'])
        model = session.model
        del state, session
        return model, streams, metadata
    return restore
