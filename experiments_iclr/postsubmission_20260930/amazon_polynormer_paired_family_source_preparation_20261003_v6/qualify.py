"""Bounded full-Amazon engineering qualification; never scientific fit weights."""
from copy import deepcopy
import time
import common as c
import native_training as n
import study
import retention


def gradient_gate(actual, expected, name):
    import torch
    if expected is None or actual is None:
        c.require(actual is None and expected is None, 'Active/dormant gradient mismatch: ' + name)
        return 0.0
    c.require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32 and
              torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item() and
              torch.allclose(actual, expected, rtol=1e-4, atol=1e-6),
              'Fixed transformed-gradient parity failed: ' + name)
    return float((actual - expected).abs().max().item())


def initialization_pairing(device):
    import torch
    caller_rng = n.rng(device)
    started = time.perf_counter()
    row = {'split': 0, 'block_seed': 17, 'kind': 'native_independent', 'member': 0, 'seed': 17}
    try:
        native, _, native_rng = n.build(row, device)
        family, optimizer, family_rng = n.build(dict(row, kind='gnnm_boundary_4', member=None), device)
        c.require(n.exact(native_rng['after_device_reset'], family_rng['after_device_reset']),
                  'Native member0/GNNM common initial reset RNG pairing differs')
        reference = dict(native.named_parameters())
        for name, p in family.core.named_parameters():
            c.require(n.exact(p, reference[name]), 'Native member0/GNNM initial common body differs: ' + name)
        for native_name, family_name in (('lin_in', 'stem'), ('pred_local', 'local_head'), ('pred_global', 'global_head')):
            source = getattr(native, native_name)
            owner = getattr(family, family_name)
            c.require(n.exact(source.weight, owner.weight) and
                      n.exact(source.bias.detach().expand(4, -1).clone(), owner.B), 'Initial native W/private B pairing differs')
            c.require(torch.equal(owner.S, torch.ones_like(owner.S)), 'Initial S is not1')
            if family_name == 'stem':
                c.require(((owner.R == 1) | (owner.R == -1)).all().item(), 'Stem R is not Rademacher')
            else:
                c.require(torch.equal(owner.R, torch.ones_like(owner.R)), 'Initial head R is not1')
        n.optimizer_names(family, optimizer)
        return {'native_member0_and_GNNM_common_seed': 17, 'initial_common_body_W_bias_bitwise': True,
                'device_reset_RNG_pairing_bitwise': True, 'Rademacher_stem_and_identity_head_factors': True,
                'seconds': time.perf_counter() - started, 'memory': n.memory(device)}
    finally:
        n.restore_rng(caller_rng, device)


def instrument(model, records):
    """Read-only detached hooks; record actual local scores and private global axes."""
    import torch
    q = n.core(model)
    hooks = []
    for layer, conv in enumerate(q.local_convs):
        def local_hook(module, args, output, layer=layer):
            n.finite(output, 'Local hidden nonfinite')
            c.require(tuple(output.shape) == (24492, 512), 'Member local node axis mixed')
            records.append({'kind': 'local_hidden', 'layer': layer, 'shape': list(output.shape)})
        def score_hook(module, inputs, output, layer=layer):
            n.finite(output, 'GAT scores nonfinite')
            c.require(output.ndim == 2 and output.shape[1] == 2, 'Member GAT heads invalid')
            records.append({'kind': 'GAT_attention_scores', 'layer': layer, 'shape': list(output.shape)})
        hooks.append(conv.register_forward_hook(local_hook))
        hooks.append(conv.register_edge_update_forward_hook(score_hook))
    def global_hook(module, args):
        with torch.no_grad():
            x = args[0]
            c.require(tuple(x.shape) == (24492, 512) and module.num_layers == 1,
                      'Global private full-node input invalid')
            k = torch.sigmoid(module.k_lins[0](x)).view(24492, 256, 2)
            qv = k if module.qk_shared else torch.sigmoid(module.q_lins[0](x)).view(24492, 256, 2)
            v = module.v_lins[0](x).view(24492, 256, 2)
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            k_sum = torch.einsum('ndh -> dh', k)
            numerator = torch.einsum('ndh, dmh -> nmh', qv, kv)
            denominator = torch.einsum('ndh, dh -> nh', qv, k_sum).unsqueeze(1)
            values = {'Q': qv, 'K': k, 'V': v, 'KV': kv, 'K_sum': k_sum,
                      'numerator': numerator, 'denominator': denominator}
            for label, value in values.items():
                n.finite(value, 'Private global ' + label + ' nonfinite')
            c.require((denominator > 0).all().item(), 'Private global denominator nonpositive')
            records.append({'kind': 'private_global_axes', 'shapes': {k: list(v.shape) for k, v in values.items()}})
    hooks.append(q.global_attn.register_forward_pre_hook(global_hook))
    return hooks


def materialized_member(pristine, family, member):
    import torch
    model = deepcopy(pristine)
    for native_name, family_name in (('lin_in', 'stem'), ('pred_local', 'local_head'),
                                     ('pred_global', 'global_head')):
        destination = getattr(model, native_name)
        owner = getattr(family, family_name)
        with torch.no_grad():
            destination.weight.copy_(owner.S[member].unsqueeze(1) * owner.weight * owner.R[member].unsqueeze(0))
            destination.bias.copy_(owner.B[member])
    n.set_stage(model, n.stage(family))
    model.eval()
    return model


def parity(x, edge, roles, device, *, identity, global_stage):
    import torch
    before = n.rng(device)
    started = time.perf_counter()
    row = {'split': 0, 'block_seed': 17, 'kind': 'native_independent', 'member': 0, 'seed': 17}
    pristine, unused_optimizer, _ = n.build(row, device)
    del unused_optimizer
    _, boundary = n.neural()
    family = boundary.PolynormerBoundaryFamily(deepcopy(pristine), members=4)
    if identity:
        boundary.set_boundary_identity_(family)
    n.set_stage(family, global_stage)
    family.eval()
    c.require(all(isinstance(getattr(family.core, k), torch.nn.Identity)
                  for k in ('lin_in', 'pred_local', 'pred_global')), 'Dormant boundary owners remain')
    c.require(not any(k.startswith(('core.lin_in.', 'core.pred_local.', 'core.pred_global.'))
                      for k, _ in family.named_parameters()), 'Dormant boundary parameter registration remains')
    records = []
    hooks = instrument(family, records)
    try:
        family.zero_grad(set_to_none=True)
        z = n.forward(family, x, edge)
        n.loss_for(z, roles).backward()
        n.check_gradients(family)
        references = []
        maxima = []
        for member in range(4):
            reference = materialized_member(pristine, family, member)
            reference.zero_grad(set_to_none=True)
            rz = n.forward(reference, x, edge)
            maxima.append(n.logit_gate(z[member], rz[0]))
            n.loss_for(rz, roles).backward()
            n.check_gradients(reference)
            references.append(dict(reference.named_parameters()))
            # Retain only native parameter/gradient owners, not their activation graph.
            del rz, reference
        gradient_max = 0.0
        compared = 0
        native_boundaries = ('lin_in.', 'pred_local.', 'pred_global.')
        for name, parameter in family.core.named_parameters():
            c.require(not name.startswith(native_boundaries), 'Boundary owner not transferred')
            values = [r[name].grad for r in references]
            c.require(all(v is None for v in values) or all(v is not None for v in values),
                      'Native member activity differs')
            expected = None if values[0] is None else torch.stack(values).mean(0)
            gradient_max = max(gradient_max, gradient_gate(parameter.grad, expected, 'core.' + name))
            compared += 1
        for native_name, family_name in (('lin_in', 'stem'), ('pred_local', 'local_head'),
                                         ('pred_global', 'global_head')):
            owner = getattr(family, family_name)
            gradients = [r[native_name + '.weight'].grad for r in references]
            if all(g is None for g in gradients):
                for label in ('weight', 'R', 'S', 'B'):
                    gradient_gate(getattr(owner, label).grad, None, family_name + '.' + label)
                continue
            c.require(all(g is not None for g in gradients), 'Boundary reference gradient missing')
            transformed = [g * owner.S[m].detach().unsqueeze(1) * owner.R[m].detach().unsqueeze(0)
                           for m, g in enumerate(gradients)]
            gradient_max = max(gradient_max, gradient_gate(owner.weight.grad, torch.stack(transformed).mean(0),
                                                          family_name + '.weight'))
            for member, g in enumerate(gradients):
                r_expected = (g * owner.S[member].detach().unsqueeze(1) * owner.weight.detach()).sum(0) / 4
                s_expected = (g * owner.weight.detach() * owner.R[member].detach().unsqueeze(0)).sum(1) / 4
                b_expected = references[member][native_name + '.bias'].grad / 4
                for label, expected in (('R', r_expected), ('S', s_expected), ('B', b_expected)):
                    gradient_max = max(gradient_max, gradient_gate(getattr(owner, label).grad[member], expected,
                                                                  family_name + '.' + label + str(member)))
                    compared += 1
        c.require(sum(r['kind'] == 'local_hidden' for r in records) == 40 and
                  sum(r['kind'] == 'GAT_attention_scores' for r in records) == 40 and
                  sum(r['kind'] == 'private_global_axes' for r in records) == (4 if global_stage else 0),
                  'Complete four-member private trajectory instrumentation differs')
        return {'identity_boundary': identity, 'global_stage': global_stage, 'members': 4,
                'raw_logit_maxima': maxima, 'gradient_max_absolute': gradient_max,
                'gradient_groups_compared': compared, 'private_trajectory_shapes': records,
                'complete_member_eval_forwards': 8, 'complete_member_eval_gradient_trajectories': 8,
                'seconds': time.perf_counter() - started, 'memory': n.memory(device)}
    finally:
        for hook in hooks:
            hook.remove()
        n.restore_rng(before, device)


def selector_fixtures():
    # Literal protocol fixtures; no data, scores, tuning or early-stop rule.
    best = -1
    chosen = None
    events = ((1, False, 2), (2, False, 2), (3, True, 1), (4, True, 3), (5, True, 3))
    selected = []
    for update, global_stage, count in events:
        if count > best:
            best = count
            chosen = (update, global_stage)
            selected.append(update)
    c.require(selected == [1, 4] and chosen == (4, True), 'Strict earliest-tie cross-stage selector fixture failed')
    best = -1
    chosen = None
    for update, global_stage, count in ((1, False, 2), (2, False, 2), (3, True, 1)):
        if count > best:
            best, chosen = count, (update, global_stage)
    c.require(chosen == (1, False), 'Selected-local-final selector fixture failed')
    return {'first_actual_update_eligible': True, 'earliest_tie': True,
            'selector_persists_across_stages': True, 'selected_local_final_allowed': True}


def replay_scratch(model, optimizer, checkpoint, row, x, edge, roles, device, bindings):
    caller_rng = n.rng(device)
    scratch_model, scratch_optimizer = deepcopy((model, optimizer))
    try:
        n.restore(scratch_model, scratch_optimizer, n.load_image(checkpoint, bindings), device)
        return n.portable_replay(scratch_model, scratch_optimizer, checkpoint, row, x, edge, roles, device, bindings)
    finally:
        n.restore_rng(caller_rng, device)


def bounded_form(row, a, admission_record, output, x, edge, roles, preprocessing):
    device = a['device']
    form = output / c.fit_id(row)
    form.mkdir(exist_ok=False)
    start = time.perf_counter()
    model, optimizer, construction = n.build(row, device)
    bindings = {'source': a['source'], 'admission': admission_record, 'probe_row': row,
                'runtime_receipt': a['runtime_receipt'], 'consumer_release': a['consumer_release'],
                'preprocessing': preprocessing, 'report_eligible': False}
    best = -1
    selected = None
    updates = []
    checkpoint_io = []
    for local_update in (1, 2):
        n.synchronize(device)
        before = time.perf_counter()
        loss = n.train_step(model, optimizer, x, edge, roles)
        logits = n.eval_logits(model, x, edge)
        count = n.correct_count(logits, roles)
        n.synchronize(device)
        updates.append({'actual_update': local_update, 'stage': 'local', 'fit_CE': float(loss),
                        'val_correct': count, 'seconds': time.perf_counter() - before,
                        'memory': study.caps(a, device)})
        if count > best:
            selection = {'actual_update': local_update, 'stage_epoch': local_update, 'global': False,
                         'val_correct': count, 'val_count': len(roles['val'])}
            selected, io = n.snapshot_write(form / ('local_%d.pt' % local_update),
                       model, optimizer, bindings, selection, construction, logits, device)
            checkpoint_io.append(dict(io, checkpoint=selected))
            best = count
        del logits
    local_image = n.load_image(selected, bindings)
    local_replay = replay_scratch(model, optimizer, selected, row, x, edge, roles, device, bindings)
    # A real further scratch update makes model and Adam rollback nontrivial.
    caller_rng = n.rng(device)
    scratch_model, scratch_optimizer = deepcopy((model, optimizer))
    try:
        n.train_step(scratch_model, scratch_optimizer, x, edge, roles)
        changed = n.state(scratch_model, scratch_optimizer, device)
        c.require(not n.exact(changed['model'], local_image['state']['model']) and
                  not n.exact(changed['optimizer'], local_image['state']['optimizer']),
                  'Scratch update did not make model/Adam rollback nontrivial')
        n.restore_model_adam(scratch_model, scratch_optimizer, local_image['state'])
        rolled_back = n.state(scratch_model, scratch_optimizer, device)
        c.require(n.exact(rolled_back['model'], local_image['state']['model']) and
                  n.exact(rolled_back['optimizer'], local_image['state']['optimizer']),
                  'Scratch rollback failed full model/Adam/counter byte equality')
        c.require(n.stage(scratch_model) is False, 'Local scratch stage changed')
    finally:
        n.restore_rng(caller_rng, device)
        del scratch_model, scratch_optimizer
    transition = n.transition(model, optimizer, local_image, device)
    transition['qualification_local_updates'] = 2
    n.synchronize(device)
    before = time.perf_counter()
    loss = n.train_step(model, optimizer, x, edge, roles)
    logits = n.eval_logits(model, x, edge)
    count = n.correct_count(logits, roles)
    n.synchronize(device)
    updates.append({'actual_update': 3, 'stage': 'global', 'fit_CE': float(loss), 'val_correct': count,
                    'seconds': time.perf_counter() - before, 'memory': study.caps(a, device)})
    global_selection = {'actual_update': 3, 'stage_epoch': 1, 'global': True,
                        'val_correct': count, 'val_count': len(roles['val'])}
    global_image, io = n.snapshot_write(form / 'global_probe.pt',
                 model, optimizer, bindings, global_selection, construction, logits, device)
    checkpoint_io.append(dict(io, checkpoint=global_image))
    global_replay = n.portable_replay(model, optimizer, global_image, row, x, edge, roles, device, bindings)
    overall_selected = global_image if count > best else selected
    # Explicit selected-local restoration after actual global work, regardless of utility.
    n.restore(model, optimizer, local_image, device)
    local_logits = n.eval_logits(model, x, edge)
    c.require(n.stage(model) is False and n.correct_count(local_logits, roles) == best,
              'Saved selected local stage/VAL failed after global work')
    n.logit_gate(local_image['selected_raw_logits'], local_logits.cpu())
    byte_counts = n.state_bytes(model, optimizer)
    # Fresh V6 output only: remove an owned, superseded audit copy while proving
    # canonical model/Adam/gradients/modes/stage/RNG and original probes survive.
    before_retirement = n.state(model, optimizer, device)
    audit_copy_started = time.perf_counter()
    spare = n.save_image(form / 'checkpoints' / 'retirement_probe.pt', local_image)
    audit_copy_seconds = time.perf_counter() - audit_copy_started
    audit = [{'checkpoint': spare, 'selection': local_image['selection'],
              'checkpoint_io': {'serialized_bytes': spare['bytes']}, 'binary_retention': {'status': 'retained'}}]
    journal = form / 'RETIREMENT_PROBE.jsonl'
    with journal.open('x') as stream:
        retired = retention.retire(form, audit, (selected, global_image), global_image, stream, bindings)
    c.require(len(retired) == 1 and not c.confined(spare['path']).exists() and
              n.exact(before_retirement, n.state(model, optimizer, device)),
              'Retirement changed canonical live state or failed owned-binary removal')
    c.verify(selected); c.verify(global_image)
    retirement_probe = {'retired_copy': spare, 'journal': c.record(journal), 'receipt': retired[0],
                        'audit_copy_write_seconds': audit_copy_seconds,
                        'live_model_Adam_grad_modes_stage_and_RNG_bitwise_unchanged': True,
                        'selected_local_and_global_probe_still_available': True,
                        'scientific_selector_or_training_updates_added': 0}
    del before_retirement
    result = {'row': row, 'report_eligible': False, 'updates': updates,
              'selected_local_checkpoint': selected, 'global_probe_checkpoint': global_image,
              'overall_selected_reference': overall_selected, 'selector_best_local': best,
              'local_replay': local_replay, 'global_replay': global_replay,
              'checkpoint_io': checkpoint_io,
              'nontrivial_scratch_model_Adam_counter_rollback': True,
              'transition': transition, 'selected_local_final_restore_after_global': True,
              'charged_actual_train_updates': 8, 'complete_member_trajectory_updates':
                  8 * (4 if row['kind'] == 'gnnm_boundary_4' else 1),
              'state_bytes': byte_counts, 'body_seconds': time.perf_counter() - start,
              'retirement_probe': retirement_probe,
              'memory': study.caps(a, device)}
    c.write(form / 'FORM.json', result)
    return result


def run(a, admission_record, output):
    device = a['device']
    start = time.perf_counter()
    x, edge, blocks, preprocessing = c.load_data(a, device)
    roles = blocks[0]
    pairing = initialization_pairing(device)
    study.caps(a, device)
    c.write(output / 'INITIALIZATION_PAIRING.json', pairing)
    parity_records = []
    for identity in (True, False):
        for global_stage in (False, True):
            parity_records.append(parity(x, edge, roles, device, identity=identity, global_stage=global_stage))
            study.caps(a, device)
            c.write(output / ('PARITY_%s_%s.json' % ('identity' if identity else 'rademacher',
                                                     'global' if global_stage else 'local')), parity_records[-1])
    forms = []
    for row in c.schedule()[:5]:
        forms.append(bounded_form(row, a, admission_record, output, x, edge, roles, preprocessing))
    c.require(len(forms) == 5 and {f['row']['seed'] for f in forms if f['row']['kind'] == 'native_independent'} ==
              {17, 1026, 2035, 3044}, 'Complete block-zero form/seed qualification missing')
    c.preserve(a, admission_record)
    c.write(output / 'RESULT.json', {'schema': 'amazon_polynormer_qualification_v2', 'status': 'passed',
            'source': a['source'], 'runtime_receipt': a['runtime_receipt'], 'consumer_release': a['consumer_release'],
            'prior_attempt_registry': a['attempt_registry'],
            'report_eligible': False, 'actual_tested_block': {'split': 0, 'block_seed': 17},
            'reviewed_role_coverage': preprocessing['roles'], 'preprocessing': preprocessing,
            'initialization_pairing': pairing, 'parity': parity_records, 'forms': forms, 'selector_fixtures': selector_fixtures(),
            'cost': {'body_seconds': time.perf_counter() - start, 'charged_train_updates': 40,
                     'charged_complete_member_training_trajectory_updates': 64,
                     'parity_complete_member_eval_forwards': 32,
                     'parity_complete_member_eval_gradient_trajectories': 32, 'memory': study.caps(a, device)},
            'full_study_authorized': False, 'prior_failures': c.record(c.PACKET / 'PRESERVED_FAILURES.json')})
