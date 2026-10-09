"""Operator constructor/mode hooks over the unchanged variable-member collector."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import time


def collector_tree(source):
    tree = ast.parse(source)
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_cell'))
    count = {'constructor': 0, 'restore': 0, 'baseline': 0, 'cohort_path': 0}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if any(isinstance(t, ast.Name) and t.id == 'model' for t in node.targets) and 'Ensemble' in ast.unparse(call):
                node.value = ast.parse('_qk_construct(torch,state,cost,record)').body[0].value; count['constructor'] += 1
            if isinstance(call.func, ast.Name) and call.func.id == 'restore':
                node.value = ast.parse('_qk_restore(torch,model,saved,state,meta,config,cost,record)').body[0].value; count['restore'] += 1
        if isinstance(node, ast.If) and ast.unparse(node.test) == "state['arm'] == 'be_init'":
            node.test = ast.parse("state['operator'] == 'native_tied'").body[0].value; count['baseline'] += 1
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cohort_path' for t in node.targets):
            node.value = ast.parse("output/'raw'/('native_tied_cohorts_'+state['arm']+'_'+str(seed)+'.npz')").body[0].value
            count['cohort_path'] += 1
    if count != {'constructor': 1, 'restore': 1, 'baseline': 1, 'cohort_path': 1}:
        raise ValueError('Unique sealed collection hooks required')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def configure(g, pins, routing, route_id, collector):
    tree = collector_tree(g.bound(pins['reuse']['variable_member_collect']).read_text())
    namespace = dict(vars(collector)); namespace['MAX_FORWARDS'] = 108
    namespace['_qk_construct'] = construct_factory(g, pins, routing, route_id)
    namespace['_qk_restore'] = restore_factory(g, pins)
    exec(compile(tree, collector.__file__ + ':QK36-hooks', 'exec'), namespace)
    collector.collect_cell = namespace['collect_cell']
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def construct_factory(g, pins, routing, route_id):
    def construct(torch, state, cost, record):
        begin = time.monotonic(); original_init = torch.optim.Adam.__init__
        def observed_init(instance, *args, **kwargs):
            started = time.monotonic(); cost['optimizer_construction_attempts'] += 1
            record['optimizer_construction_attempts'] = record.get('optimizer_construction_attempts', 0) + 1
            try:
                original_init(instance, *args, **kwargs)
                cost['optimizer_constructions'] += 1
                record['optimizer_constructions'] = record.get('optimizer_constructions', 0) + 1
            finally:
                seconds = time.monotonic() - started; cost['optimizer_construction_seconds'] += seconds
                record['optimizer_construction_seconds'] = record.get('optimizer_construction_seconds', 0.) + seconds
        cost['Session_factory_attempts'] += 1; record['Session_factory_attempts'] = 1
        torch.optim.Adam.__init__ = observed_init
        try:
            session = routing.make_session(route_id=route_id, operator=state['operator'], kind=state['kind'],
                seed=state['seed'], device='cuda:0', later_execution_authorized=True)
            cost['Session_factory_completions'] += 1; record['Session_factory_completions'] = 1
        finally:
            torch.optim.Adam.__init__ = original_init
            record['native_Session_and_operator_install_seconds'] = time.monotonic() - begin
        bodies = 4 if state['kind'] == 'independent4' else 1
        g.require(session.steps == 0 and len(session.optimizers) == bodies and all(not opt.state for opt in session.optimizers)
            and record['optimizer_constructions'] == 2 * bodies
            and session.model.members == state['members'] and not session.model.contrastive,
            'Same operator installed before loading; both fresh Adam generations charged, zero steps/history')
        record['serving_operator_binding'] = dict(session.operator_binding)
        record['installed_before_state_load'] = True
        def refusal(*args, **kwargs):
            raise RuntimeError('Postclosure selected-state serving only; no training or resume')
        session.train_step = refusal
        model = session.model
        del session
        return model
    return construct


def restore_factory(g, pins):
    def restore(torch, model, saved, state, unused_meta, config, cost, record):
        g.require(type(saved) is dict and saved['config'] == config and saved['owner_context'] == state['original_owner_context'], 'Exact selected original recipe/owner/cell')
        run = saved['run']; context = state['original_owner_context']; binding = saved['operator_binding']
        g.require(run['task'] == 'wikics' and run['arm'] == state['kind'] and run['seed'] == state['seed']
            and run['TEST_scoring'] is False and run['owner_context'] == context
            and run['driver_sha256'] == pins['reuse']['public_driver']['sha256']
            and run['core'] == pins['public_core_sha256']
            and run['data']['train_npz_sha256'] == pins['plan']['role_hashes']['train']
            and run['data']['valid_npz_sha256'] == pins['plan']['role_hashes']['valid'], 'Selected exact task/seed/source/roles')
        installed = record['serving_operator_binding']
        mathematical = ('operator', 'kind', 'seed', 'adapter_manifest_sha256', 'adapter_program_sha256', 'native_sha256',
            'factor_sha256', 'bias_outside_scale', 'ordinary_global_factor_coordinates_added', 'operator_delta_or_gamma_scalars',
            'parameter_count', 'optimizer_count', 'large_qk_projections_per_global_layer_member', 'objective', 'F_engine', 'copied_full_qk_start')
        g.require(all(binding[k] == installed[k] == state['original_operator_binding'][k] for k in mathematical)
            and binding['route_id'] == state['original_training_route'] and binding['mathematical_source_changed'] is False,
            'Same installed operator and parameter inventory; original route provenance remains explicit')
        model.load_state_dict(saved['model'], strict=True)
        stored = saved['selected_VALID']; g.require(math.isfinite(stored), 'Original selected accuracy finite')
        if state['kind'] == 'independent4':
            g.require(saved.get('evaluation_only') is True and saved.get('candidate') == 'individual_best_bank_only'
                and model.independent and len(model.models) == 4, 'Exact truly independent own-selected evaluation-only bank')
            modes = saved['body_global']
            g.require(type(modes) is list and len(modes) == 4 and all(type(x) is bool for x in modes), 'Four actual saved own-selected modes')
            for body, mode in zip(model.models, modes):
                body.set_global(mode)
            bank = g.read(g.bound(state['own_bank_metrics']))
            g.require(bank['VALID'] == stored and len(bank['members']) == 4 and all(math.isfinite(x) for x in bank['members']), 'Original own-best bank metric custody')
            epochs = []; diagnostics = []; begin = time.monotonic()
            for member, row in enumerate(state['own_selected_checkpoints']):
                cost['checkpoint_deserialization_attempts'] += 1; cost['checkpoint_file_bytes_submitted_to_deserializer'] += row['bytes']
                own = torch.load(g.bound(row), map_location='cpu', weights_only=False); g.bound(row)
                g.require(own['owner_context'] == context and own['operator_binding']['operator'] == state['operator']
                    and type(own['global']) is bool and own['global'] is modes[member]
                    and type(own['epoch']) is int and 1 <= own['epoch'] <= 1100 and math.isfinite(own['metric']), 'Original own-selected epoch/mode/state authority')
                epochs.append(own['epoch']); diagnostics.append(dict(member_index0=member, own_epoch=own['epoch'],
                    own_saved_VALID=own['metric'], final_bank_member_VALID=bank['members'][member],
                    final_bank_minus_own_saved_VALID=bank['members'][member] - own['metric'], selected_global=own['global']))
                del own
            record['own_epoch_mode_metadata_deserialization_seconds'] = time.monotonic() - begin
            per = bank['members']; streams = None; selector = 'independently_selected_evaluation_only_bank'
        else:
            g.require(saved.get('evaluation_only') is not True and type(saved.get('global')) is bool
                and type(saved.get('epoch')) is int and 1 <= saved['epoch'] <= 1100
                and saved['checkpoint_kind'] == 'strict-first-maximum complete VALID joint snapshot'
                and len(saved['member_VALID']) == state['members'] and all(math.isfinite(x) for x in saved['member_VALID']), 'Coherent original joint selected state')
            model.set_global(saved['global']); modes = [saved['global']] * state['members']; epochs = [saved['epoch']]
            per = saved['member_VALID']; streams = saved['streams']; diagnostics = None; selector = 'joint_strict_first_VALID_maximum'
            g.require(len(streams) == state['members'] and all(set(s) == {'cpu', 'cuda'}
                and all(v.device.type == 'cpu' and v.dtype == torch.uint8 for v in s.values()) for s in streams), 'Saved selected member RNG streams for coherent inference')
        body_modes = [body.body._global for body in model.models]
        member_modes = [body_modes[m if model.independent else 0] for m in range(model.members)]
        g.require(member_modes == modes, 'Actual restored local/global modes; no forced global selection')
        model.eval()
        meta = dict(selected_epochs=epochs, selected_member_modes=modes, selector=selector, stored_accuracy=stored,
            stored_member_accuracy=per, own_selected_member_diagnostics=diagnostics,
            original_training_route=state['original_training_route'], serving_route=installed['route_id'],
            original_operator_binding=binding, serving_operator_binding=installed,
            optimizer_or_training_RNG_history_restored=False, reselection=False, bitwise_parity_claimed=False,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            stored_model_tensor_bytes=sum(t.numel() * t.element_size() for t in model.state_dict().values()))
        record['selected_metadata'] = meta
        record['historical_selected_metadata'] = dict(selected_VALID=stored, member_VALID=per, selected_epochs=epochs,
            member_global=modes, selection=selector, own_selected_member_diagnostics=diagnostics)
        return body_modes, member_modes, streams
    return restore
