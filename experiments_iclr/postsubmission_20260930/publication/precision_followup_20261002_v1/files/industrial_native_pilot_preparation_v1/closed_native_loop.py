"""UNEXECUTED native fit entry. No raw GraphDataset/GraphTask or test labels.

Caller must supply a reviewed acquisition/preprocess/runtime admission and the
exact pinned native Model + make_parameter_groups factory. All native fits use
full-graph forward, two-logit CE, strict validation AP,1000steps/patience100.
This source contains no CLI that can launch a scientific run during preparation.
"""
from __future__ import annotations
import copy


def fit_native_closed(model,graph,x_num,x_other,train_ids,train_y,val_ids,val_y,
                      config,make_parameter_groups,admission):
    import torch
    import numpy as np
    import delu
    from sklearn.metrics import average_precision_score
    if admission.get('phase')!='native_baseline_fit' or admission.get('source_only') is not False:
        raise ValueError('Scientific admission missing')
    if admission.get('worker_mount_excludes_raw_archive_and_targets') is not True:
        raise ValueError('Closed label mounts required')
    if config['n_steps']!=1000 or config['patience']!=100 or config.get('amp_dtype') is not None:
        raise ValueError('Frozen native update/precision policy changed')
    if len(train_ids)!=len(train_y) or len(val_ids)!=len(val_y):raise ValueError('Compact source role shapes differ')
    if set(train_ids.detach().cpu().tolist()) & set(val_ids.detach().cpu().tolist()):raise ValueError('Source roles overlap')
    # The orchestration constructs/seeds the model before this function. It must
    # use delu.random.seed(config['seed']) exactly as the native entry does.
    optimizer=torch.optim.AdamW(make_parameter_groups(model),**{k:v for k,v in config['optimizer'].items() if k!='type'})
    stopper=delu.tools.EarlyStopping(config['patience'],mode='max')
    best=None;best_ap=None;warm50=None;log=[]
    def snapshot(step):
        return {'step':step,'model':copy.deepcopy(model.state_dict()),
                'optimizer':copy.deepcopy(optimizer.state_dict()),'random_state':delu.random.get_state()}
    for step in range(1,config['n_steps']+1):
        model.train();optimizer.zero_grad()
        logits=model(graph,x_num,x_other).squeeze(-1).float()
        if logits.ndim!=2 or logits.shape[1]!=2:raise ValueError('Native binary model needs two logits')
        loss=torch.nn.functional.cross_entropy(logits[train_ids],train_y)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite train loss; preserve failed cell')
        loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():
            all_logits=model(graph,x_num,x_other).squeeze(-1).float()
            val_prob=all_logits[val_ids].softmax(-1)[:,1].cpu().numpy()
        if not np.isfinite(val_prob).all():raise ValueError('Nonfinite validation probabilities')
        ap=float(average_precision_score(val_y.cpu().numpy(),val_prob))
        if best_ap is None or ap>best_ap:
            best_ap=ap;best=snapshot(step)
        if step==50:warm50=snapshot(step)
        log.append({'step':step,'train_ce':float(loss.detach()),'validation_ap':ap})
        stopper.update(ap)
        if stopper.should_stop():break
    if warm50 is None or best is None:raise ValueError('Required fixed warm donor not produced')
    return {'best':best,'warm50':warm50,'best_validation_ap':best_ap,'trace':log,
            'test_label_access':False,'selection':'earliest_strict_validation_AP_best'}
