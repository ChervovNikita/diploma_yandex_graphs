"""Source-only teacher fit/qualification and equal bounded selection CLI.

Authoring never imports this file. Runtime fits require a separate complete
admission. No command here accepts correction or final-pool labels.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
import correction_screen_driver as core

PARENT_PREPARE_SHA = '376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253'


def protocol():
    return core.read_json(Path(__file__).resolve().parent.parent/'TEACHER_PROTOCOL.json')


def verify_cell(record):
    path = core.verified(record)
    freeze = core.read_json(path)
    core.require(freeze['schema'] == 'modern-teacher-cell-freeze-v1' and freeze['report_eligible'],
        'Only completed full schedule teacher cells are selectable')
    core.check_implementation(freeze['implementation_sha256'])
    core.verify_tree(path.parent, freeze['payload'])
    return freeze, path.parent


def selection_body(args):
    request_record = core.descriptor(args.request)
    request = core.read_json(args.request)
    core.require(request['teacher_protocol'] == protocol(), 'Wrong frozen teacher protocol')
    core.check_implementation(request['implementation_sha256'])
    family, backbone = request['family'], request['backbone']
    core.require(family in protocol()['families'] and backbone in protocol()['native'], 'Unknown family/backbone')
    core.require(len(request['cells']) == 12, 'Equal four configs by three paired seeds required')
    cells, seen = [], set()
    for record in request['cells']:
        freeze, root = verify_cell(record)
        spec = freeze['specification']
        key = (spec['config'], spec['seed'])
        core.require(spec['family'] == family and spec['backbone'] == backbone and key not in seen,
            'Mixed or repeated teacher cells')
        core.require(freeze['teacher_protocol'] == request['teacher_protocol'] and
            freeze['implementation_sha256'] == request['implementation_sha256'] and
            freeze['environment'] == request['environment'] and freeze['graph'] == request['graph'],
            'Cell protocol/runtime/graph differs')
        core.require(freeze['label_scope'] == ['train', 'validation'] and not freeze['final_labels_read'],
            'Wrong teacher label scope')
        seen.add(key)
        cells.append(dict(freeze=record, specification=spec, role_freeze=freeze['role_freeze'],
            primary_validation_nll=freeze['selection']['primary_validation_nll'],
            checkpoint=core.descriptor(root/'teacher_checkpoint.pt'),
            saved_logits=core.descriptor(root/'teacher_member_logits.npy'),
            preprocessing=freeze['selection']['preprocessing'], costs=freeze['costs']))
    expected = {(c,s) for c in range(4) for s in protocol()['seeds']}
    core.require(seen == expected, 'Complete paired grid required; no partial/favorable selection')
    for seed in protocol()['seeds']:
        roles = [row['role_freeze'] for row in cells if row['specification']['seed'] == seed]
        core.require(all(record == roles[0] for record in roles), 'Config roles are not paired')
    values = [sum(row['primary_validation_nll'] for row in cells
        if row['specification']['config'] == config)/3 for config in range(4)]
    core.require(all(math.isfinite(v) for v in values), 'Nonfinite selection value')
    chosen = min(range(4), key=lambda config: (values[config], config))
    result = dict(schema='modern-teacher-family-selection-v1', request=request_record,
        family=family, backbone=backbone, graph=request['graph'], teacher_protocol=request['teacher_protocol'],
        implementation_sha256=request['implementation_sha256'], environment=request['environment'],
        selected_config=chosen, mean_validation_nll_by_config=values, all_cells=cells,
        selected_teachers=[row for row in cells if row['specification']['config'] == chosen],
        label_scope=['train', 'validation'], final_labels_read=False, secondary_selected=False,
        selection='one config for all three seeds; earliest config exact tie; joint independent checkpoint selection')
    core.verified(request_record)
    core.write_json(Path(args.output).resolve(), result, exclusive=True)


def study_selection(args):
    """Close all 72 equal cells before any control/correction final reporting."""
    request_record = core.descriptor(args.request)
    request = core.read_json(args.request)
    core.require(len(request['selections']) == 6, 'Two graphs by three paired families required')
    expected = {(b,f) for b in protocol()['native'] for f in protocol()['families']}
    seen, role_pairs, records = set(), {}, []
    for record in request['selections']:
        selection = core.read_json(core.verified(record))
        key = (selection['backbone'], selection['family'])
        core.require(selection['schema'] == 'modern-teacher-family-selection-v1' and key in expected and
            key not in seen and selection['teacher_protocol'] == protocol() and
            selection['implementation_sha256'] == request['implementation_sha256'] and
            selection['environment'] == request['environment'], 'Wrong/repeated family selection or mixed runtime')
        core.require(len(selection['all_cells']) == 12 and len(selection['selected_teachers']) == 3,
            'Full grid required')
        cell_keys = set()
        for row in selection['all_cells']:
            freeze, root = verify_cell(row['freeze'])
            spec = row['specification']
            cell_keys.add((spec['config'], spec['seed']))
            core.require(freeze['specification'] == spec and spec['backbone'] == key[0] and spec['family'] == key[1],
                'Mixed cell family')
            core.require(row['checkpoint'] == core.descriptor(root/'teacher_checkpoint.pt') and
                row['saved_logits'] == core.descriptor(root/'teacher_member_logits.npy') and
                row['role_freeze'] == freeze['role_freeze'] and
                row['primary_validation_nll'] == freeze['selection']['primary_validation_nll'],
                'Cell payload/selection record differs')
            pair = (key[0], spec['seed'])
            if pair in role_pairs:
                core.require(role_pairs[pair] == row['role_freeze'], 'Cross-family controls must share exact roles')
            else:
                role_pairs[pair] = row['role_freeze']
        core.require(cell_keys == {(c,s) for c in range(4) for s in protocol()['seeds']}, 'Incomplete family grid')
        values = [sum(row['primary_validation_nll'] for row in selection['all_cells']
            if row['specification']['config'] == c)/3 for c in range(4)]
        config = min(range(4), key=lambda c: (values[c],c))
        core.require(selection['selected_config'] == config and selection['mean_validation_nll_by_config'] == values and
            selection['selected_teachers'] == [row for row in selection['all_cells'] if row['specification']['config'] == config],
            'Family selection differs from registered source criterion')
        seen.add(key)
        records.append(dict(selection=record, backbone=key[0], family=key[1], graph=selection['graph']))
    core.require(seen == expected, 'Missing family/graph control')
    core.check_implementation(request['implementation_sha256'])
    core.verified(request_record)
    core.write_json(Path(args.output).resolve(), dict(schema='modern-teacher-study-selection-v1',
        request=request_record, teacher_protocol=protocol(), implementation_sha256=request['implementation_sha256'],
        environment=request['environment'],
        family_selections=records, complete_family_cells=72, source_selection_closed=True,
        label_scope=['train','validation'], final_labels_read=False, secondary_selected=False), exclusive=True)


def qualification_parity(torch, adapter, spec, graph, train):
    """Prospective complete-graph native/body equivalence and common-gradient gate."""
    from backbone_boundary_adapter import set_boundary_identity_
    device = graph.teacher_input.device
    single_spec = adapter.specification(spec['backbone'], 'single_author', spec['config'], spec['seed'])
    boundary_spec = adapter.specification(spec['backbone'], 'gnnm_boundary_4', spec['config'], spec['seed'])
    single = adapter.TeacherFamily(single_spec).to(device).eval()
    shared = adapter.TeacherFamily(boundary_spec).to(device).eval()
    set_boundary_identity_(shared.boundary)
    rows = []
    stages = (False, True) if spec['backbone'] == 'polynormer_r' else (False,)
    for stage in stages:
        single.set_global_stage(stage)
        shared.set_global_stage(stage)
        with torch.no_grad():
            original, cloned = single(graph), shared(graph)
        # Frozen tolerance; failed checks remain failed, no tolerance rescue.
        for m in range(4):
            torch.testing.assert_close(original[0], cloned[m], rtol=1e-5, atol=1e-6)
        single.zero_grad(set_to_none=True)
        shared.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(single(graph)[0, train.nodes], train.labels).backward()
        torch.nn.functional.cross_entropy(shared(graph)[:, train.nodes].reshape(-1, spec['native']['classes']),
            train.labels.repeat(4)).backward()
        # Body gradients must match the equal-member mean loss in eval mode.
        compared = 0
        native_parameters = dict(single.models[0].named_parameters())
        for name, parameter in shared.boundary.core.named_parameters():
            expected = native_parameters[name].grad
            if expected is not None or parameter.grad is not None:
                core.require(expected is not None and parameter.grad is not None, 'Body gradient availability mismatch')
                torch.testing.assert_close(expected, parameter.grad, rtol=1e-4, atol=1e-6)
                compared += 1
        boundaries = [('stem', 'lin1'), ('head', 'lin3')] if spec['backbone'] == 'polyformer_mono' else [
            ('stem', 'lin_in'), ('local_head', 'pred_local'), ('global_head', 'pred_global')]
        for boundary, native_name in boundaries:
            parameter = getattr(shared.boundary, boundary).weight
            expected = native_parameters[native_name+'.weight'].grad
            if expected is not None or parameter.grad is not None:
                core.require(expected is not None and parameter.grad is not None, 'Boundary weight gradient availability mismatch')
                torch.testing.assert_close(expected, parameter.grad, rtol=1e-4, atol=1e-6)
                compared += 1
        rows.append(dict(global_stage=stage, max_absolute_logit_difference=float((cloned-original).abs().max()),
            common_body_gradient_tensors_compared=compared, full_graph=True))
    del single, shared
    return dict(rtol_logits=1e-5, atol_logits=1e-6, rtol_gradients=1e-4, atol_gradients=1e-6,
        checks=rows, native_vs_identity_boundary_passed=True, shortened_fit_report_eligible=False)


def fit_body(args, admission, out, admission_record, started):
    core.check_implementation(admission['implementation_sha256'])
    core.require(admission['execution_authorized'] is True and admission['teacher_protocol'] == protocol(),
        'Complete exact prospective teacher admission required')
    qualify = args.command == 'teacher-qualify'
    if not qualify:
        gate = core.read_json(core.verified(admission['qualification_receipt']))
        core.require(gate['schema'] == 'modern-teacher-qualification-v1' and gate['passed'] is True and
            gate['backbone'] == admission['backbone'] and gate['family'] == admission['family'] and
            gate['environment'] == admission['environment'] and
            gate['implementation_sha256'] == admission['implementation_sha256'], 'Matching qualification required')
    np, torch, environment = core.runtime(admission['device'])
    core.require(environment == admission['environment'], 'Runtime fingerprint differs')
    import modern_teacher_adapter as adapter
    core.require(adapter.PROTOCOL == protocol(), 'Adapter/protocol custody mismatch')
    role_path = core.verified(admission['role_freeze'])
    roles, cell = core.read_json(role_path), role_path.parent
    core.require(roles['role_derivation_version'] == 'derived_roles_v2' and
        roles['preparation_driver_sha256'] in (PARENT_PREPARE_SHA, admission['implementation_sha256']['correction_screen_driver.py']),
        'Only immutable derived_roles_v2 allocation is accepted')
    core.verify_tree(cell, roles['payload'])
    core.require(roles['seed'] in adapter.SEEDS and roles['source_split_index'] == adapter.SEEDS.index(roles['seed']),
        'Fixed seed/split pairing')
    core.require(set(admission['source_labels']) == {'train','validation'} and
        admission['roles_frozen_before_label_extraction'] is True and admission['prior_exposure_disclosure'] and
        admission['independent_of_stage1_outcomes'] is True, 'Teacher accepts exactly two source label packs')
    manifest = core.read_json(core.verified(roles['graph_input']))
    core.require(admission['graph'] == roles['graph'] == manifest['graph'], 'Wrong graph')
    spec = adapter.specification(admission['backbone'], admission['family'], admission['config'], roles['seed'])
    core.require(manifest['num_classes'] == spec['native']['classes'], 'Wrong classes')
    x_np, edges_np = core.canonical_graph(np, manifest)
    x, edges = torch.from_numpy(x_np.copy()).to(admission['device']), torch.from_numpy(edges_np.copy()).to(admission['device'])
    binding = dict(graph_input=roles['graph_input'], implementation_sha256=admission['implementation_sha256'])
    graph, prep_cost = core.measured(torch, admission['device'], lambda:
        adapter.prepare_graph(x, edges, admission['backbone'], environment, binding))
    source = {}
    for key in ('train','validation'):
        nodes = np.load(cell/f'{key}_nodes.npy', allow_pickle=False)
        source[key] = core.load_source(np, torch, admission['source_labels'][key], nodes,
            admission['device'], manifest['num_classes'])
    parity, parity_cost = core.measured(torch, admission['device'], lambda:
        qualification_parity(torch, adapter, spec, graph, source['train'])) if qualify else (None, None)
    (model, selection), fit_cost = core.measured(torch, admission['device'], lambda:
        adapter.fit_teacher(spec, graph, source['train'], source['validation'], qualify, out))
    with torch.no_grad():
        logits, inference_cost = core.measured(torch, admission['device'], lambda: model(graph).detach())
        views = adapter.saved_probability_views(logits)
    replay_cost = None
    if qualify:
        # Replay constructor/load and explicit stage flag; qualification checkpoint
        # remains report-ineligible even when these prospective checks pass.
        def replay_check():
            checkpoint = torch.load(out/'teacher_checkpoint.pt', map_location=admission['device'], weights_only=True)
            replay = adapter.TeacherFamily(spec).to(admission['device'])
            replay.load_state_dict(checkpoint['state'])
            replay.set_global_stage(checkpoint['global_stage'])
            replay.eval()
            with torch.no_grad():
                replay_logits = replay(graph)
            torch.testing.assert_close(logits, replay_logits, rtol=1e-5, atol=1e-6)
            return float((logits-replay_logits).abs().max())
        difference, replay_cost = core.measured(torch, admission['device'], replay_check)
        parity['selected_checkpoint_replay_max_absolute_logit_difference'] = difference
    np.save(out/'teacher_member_logits.npy', logits.cpu().numpy(), allow_pickle=False)
    for name, values in views.items():
        np.save(out/f'{name}_point_probabilities.npy', values.cpu().numpy(), allow_pickle=False)
    from aligned_score_correction import randomized_aps
    uniforms = torch.from_numpy(np.load(cell/'aps_uniforms.npy', allow_pickle=False)).to(admission['device'])
    with torch.no_grad():
        control_scores = randomized_aps(views['primary'], uniforms)
    np.save(out/'primary_aps_scores.npy', control_scores.cpu().numpy(), allow_pickle=False)
    del control_scores, uniforms
    costs = dict(preprocessing=prep_cost, fit=fit_cost, selected_full_inference=inference_cost,
        qualification_parity=parity_cost, qualification_checkpoint_replay=replay_cost,
        member_trajectories=spec['members'], independent_width='native same width per member',
        all_attempts_charged=True, matched_compute_claim=False)
    core.write_json(out/'COSTS.json', costs, exclusive=True)
    core.check_implementation(admission['implementation_sha256'])
    core.verified(admission_record)
    core.verified(admission['role_freeze'])
    if qualify:
        core.write_json(out/'QUALIFICATION_RECEIPT.json', dict(schema='modern-teacher-qualification-v1',
            passed=True, backbone=spec['backbone'], family=spec['family'], environment=environment,
            implementation_sha256=admission['implementation_sha256'], parity=parity,
            updates_completed=selection['updates_completed'], global_stage=selection['global_stage'],
            labels_read=['train','validation'], final_labels_read=False, report_eligible=False,
            scope='Full native parity/mean common-body gradient check; full-graph finite training and stage restoration'), exclusive=True)
    core.write_json(out/'FIT_WALL_RECEIPT.json', dict(seconds=time.perf_counter()-started,
        scope='Residual inclusive admission/input loading, preprocessing, construction, parity if qualified, updates, selection, saves and inference; excludes final freeze serialization',
        final_labels_read=False, report_eligible=not qualify), exclusive=True)
    core.write_json(out/'TEACHER_CELL_FREEZE.json', dict(schema='modern-teacher-cell-freeze-v1',
        admission=admission_record, teacher_protocol=protocol(), implementation_sha256=admission['implementation_sha256'],
        role_freeze=admission['role_freeze'], graph=admission['graph'], environment=environment,
        specification=spec, selection=selection, costs=costs, label_scope=['train','validation'],
        final_labels_read=False, report_eligible=not qualify, payload=core.tree_records(out)), exclusive=True)


def fit_command(args):
    record = core.descriptor(args.admission)
    admission = core.read_json(args.admission)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    core.write_json(out/'ATTEMPT_STARTED.json', dict(admission=record, command=args.command,
        timestamp=time.time(), automatic_retry=False), exclusive=True)
    try:
        fit_body(args, admission, out, record, started)
    except Exception as error:
        core.write_json(out/'FAILED_ATTEMPT.json', dict(error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), seconds=time.perf_counter()-started,
            measured_cost=getattr(error, 'screen_cost', None),
            automatic_retry=False, final_labels_read=False), exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('teacher-fit','teacher-qualify'):
        command = sub.add_parser(name)
        command.add_argument('--admission', required=True)
        command.add_argument('--output', required=True)
    command = sub.add_parser('select-config')
    command.add_argument('--request', required=True)
    command.add_argument('--output', required=True)
    command = sub.add_parser('select-study')
    command.add_argument('--request', required=True)
    command.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'select-config':
        selection_body(args)
    elif args.command == 'select-study':
        study_selection(args)
    else:
        fit_command(args)


if __name__ == '__main__':
    main()
