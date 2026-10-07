"""Prepared per-session facade for unchanged public WikiCS Session/replay.

No runner or scientific admission. Root must bind fullgraph targets/data and
qualify the actual original train-step and selector dispatch before release.
"""
import numpy as np
from context_positive_masks import sampled_weights
from context_alignment_objective import context_alignment


class ContextObjectivesFacade:
    def __init__(self, session, complete_train_labels, masks, mode):
        if session.task != 'wikics' or session.model.members not in (1,4):
            raise ValueError('native WikiCS M1/M4 only')
        if mode not in ('route','common','cycle'):
            raise ValueError('explicit context mode required')
        if session.model.members == 1 and mode != 'common':
            raise ValueError('matched single must use Qbar')
        self.session, self.mode = session, mode
        original = session.core['objectives']
        self.own_supervision = original.own_supervision
        self.complete_labels = complete_train_labels.to(session.device).clone()
        if self.complete_labels.ndim!=1 or self.complete_labels.dtype!=session.torch.long or len(self.complete_labels) != 580:
            raise ValueError('complete original580 TRAIN labels required')
        size = min(len(self.complete_labels), session.config['contrastive']['max_objects'])
        # Exact same device/operator as the pinned original Session and replay.
        self.index = session.torch.linspace(0,len(self.complete_labels)-1,steps=size,
                                            device=session.device).long()
        self.rows = self.index.cpu().numpy()
        self.masks = np.array(masks,copy=True)
        if self.masks.shape != (4,580,580):
            raise ValueError('complete frozen TRAIN-context relation required')
        labels = self.complete_labels.cpu().numpy()
        if np.any(self.masks & (labels[:,None] != labels[None,:])[None]):
            raise ValueError('wrong-class positive forbidden')
        self.masks.setflags(write=False)
        dtype=next(session.model.parameters()).dtype
        # The original deterministic auxiliary panel is fixed. Cache its
        # frozen targets once, rather than recreate/copy megabytes each update.
        if mode=='cycle':
            route=sampled_weights(self.masks,self.rows,'route')
            self.weight_cache=tuple(session.torch.as_tensor(
                np.broadcast_to(route[i:i+1],route.shape).copy(),device=session.device,dtype=dtype)
                for i in range(4))
        else:
            weights=sampled_weights(self.masks,self.rows,mode)
            if session.model.members==1:
                weights=weights[:1]
            self.weight_cache=session.torch.as_tensor(weights,device=session.device,dtype=dtype)
        self.alignment_source_calls = self.residual_source_calls = 0

    def alignment_loss(self, a, b, labels, task, temperature=.2, identities=None):
        if task != 'wikics' or temperature != .2 or identities is not None:
            raise ValueError('original WikiCS target/view contract required')
        if not self.session.torch.equal(labels,self.complete_labels[self.index]):
            raise ValueError('original deterministic auxiliary TRAIN order required')
        weights=self.weight_cache[self.session.steps%4] if self.mode=='cycle' else self.weight_cache
        self.alignment_source_calls+=1
        return context_alignment(a,b,weights,temperature)

    def residual_member_contrast(self,a,b,labels,temperature=.2):
        if temperature!=.2:
            raise ValueError('fixed original temperature required')
        # Source Session/replay still calls this facade; no repulsion is added.
        return a.sum()*0


def install(session, complete_train_labels, masks, mode):
    if not session.model.contrastive:
        raise ValueError('construct original unit/single/untied contrastive arm first')
    facade=ContextObjectivesFacade(session,complete_train_labels,masks,mode)
    session.core=dict(session.core,objectives=facade)
    session.config['contrastive'].update(alignment_weight=.05,residual_weight=0.,temperature=.2)
    return facade
