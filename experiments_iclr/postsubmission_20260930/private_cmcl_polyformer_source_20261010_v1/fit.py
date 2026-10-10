"""Inactive full representative fit/reconstruction. No loader, CLI or campaign."""
import gc
import json
from pathlib import Path
from .caps import CLOSED
from .native import PHASE, require
from .session import Session
from .plan import MAXIMUM_EPOCHS, PATIENCE


def write(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(PHASE), 'Output remains in project phase')
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def fit(session, folder, *, numerical_qualification=False, caps=CLOSED):
    caps.require('source_bound', 'model', 'data', 'runtime')
    require(session.caps == caps, 'Exact reviewed session capability object')
    if not numerical_qualification:
        caps.require('scientific')
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(PHASE) and not (folder/'SELECTED_STATE.pt').exists(), 'Fresh root-owned selected-state slot')
    torch = session.torch
    best, bad, selected_epoch, history = -1., 0, -1, []
    checkpoint = folder/'SELECTED_STATE.pt'
    for epoch in range(1 if numerical_qualification else MAXIMUM_EPOCHS):
        update = session.step(numerical_qualification=numerical_qualification)
        scores, outputs = session.evaluate()
        improved = scores['VALID']['accuracy'] > best
        if improved:
            best, bad, selected_epoch = scores['VALID']['accuracy'], 0, epoch
            saved = session.snapshot(epoch, scores, outputs)
            with session.costs.measure('selected_checkpoint_serialization') as cost:
                temporary = checkpoint.with_name(checkpoint.name+'.tmp')
                torch.save(saved, temporary); temporary.replace(checkpoint)
                cost['serialized_bytes'] = checkpoint.stat().st_size
            del saved
        else:
            bad += 1
        history.append(dict(epoch=epoch, selected=improved, scores=scores, update=update))
        write(folder/'HISTORY.json', history)
        if bad == PATIENCE:
            break
    require(selected_epoch >= 0 and session.counters['updates'] == session.counters['optimizer_steps'], 'Complete genuine updates and selected state')
    storage = dict(total_parameters=sum(p.numel() for p in session.parameters),
        shared_parameters=sum(p.numel() for p in session.shared),
        private_factor_parameters=sum(p.numel() for p in session.private),
        parameter_bytes=sum(p.numel()*p.element_size() for p in session.parameters),
        checkpoint_bytes=checkpoint.stat().st_size,
        peak_CUDA_allocated_bytes=None if session.device.type != 'cuda' else torch.cuda.max_memory_allocated(session.device),
        peak_CUDA_reserved_bytes=None if session.device.type != 'cuda' else torch.cuda.max_memory_reserved(session.device))
    result = dict(selected_epoch=selected_epoch, selected_VALID_accuracy=best, history=history,
        counters=dict(session.counters), storage=storage, qualification_only=numerical_qualification,
        selector='strict complete VALID mean-probability accuracy; earliest ties; native2000 maximum,250 consecutive nonimprovements',
        macro_F1_NLL_and_member_competence_are_reported_not_used_for_checkpoint_shopping=True,
        qualification_or_scientific_admission_granted_here=False)
    write(folder/'FIT_RECORD.json', result)
    return result


def reconstruct(saved, inputs, device, costs, caps=CLOSED):
    caps.require('source_bound', 'model', 'data', 'runtime')
    # Caller should release the fitted session first; this is a full fresh native
    # constructor, not an in-place selected-score reread.
    session = Session(saved['condition']['name'], saved['seed'], inputs, device, costs, saved['identity'], caps)
    torch, state = session.torch, session.state
    caller = state.capture_rng(session.np, torch)
    try:
        session.restore(saved)
        scores, outputs = session.evaluate()
        drift = {}
        for role in ('TRAIN', 'VALID'):
            old, new = saved['outputs'][role], outputs[role]
            require(old['ids'] == new['ids'] and torch.equal(old['truth'], new['truth']), 'Exact reconstruction roles/targets')
            drift[role] = dict(max_abs_member_logit=float((old['member_logits']-new['member_logits']).abs().max().item()),
                max_abs_pool_log_probability=float((old['pool_log_probabilities']-new['pool_log_probabilities']).abs().max().item()),
                prediction_changes=int((old['prediction'] != new['prediction']).sum().item()))
            require(drift[role]['max_abs_member_logit'] <= .001 and drift[role]['max_abs_pool_log_probability'] <= .001
                    and drift[role]['prediction_changes'] == 0, 'Finite tolerance-based full selected replay; no bitwise floating-output requirement')
        return dict(scores=scores, drift=drift, exact_state_optimizer_RNG_restore=True,
                    finite_output_tolerance=.001, bitwise_output_equality_claim=False)
    finally:
        state.restore_rng(session.np, torch, caller)
        require(state.exact(torch, state.capture_rng(session.np, torch), caller), 'Fresh replay preserves caller streams')
        session = None
        gc.collect()


def load_and_reconstruct(checkpoint, inputs, device, costs, caps=CLOSED):
    caps.require('source_bound', 'model', 'data', 'runtime')
    import torch
    path = Path(checkpoint).resolve(strict=True)
    require(path.is_relative_to(PHASE) and path.is_file(), 'Owned project checkpoint only')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    return reconstruct(saved, inputs, device, costs, caps)
