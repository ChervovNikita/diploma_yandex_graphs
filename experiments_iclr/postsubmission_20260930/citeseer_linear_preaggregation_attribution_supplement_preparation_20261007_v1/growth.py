"""Paired linear-message attribution control; original Growth-v2 unchanged."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import MethodType

PHASE=Path(__file__).resolve().parents[1]
BASE=PHASE/'citeseer_nonlinear_preaggregation_growth_pilot_source_20261007_v2/growth.py'
BASE_SHA='681b7f6e7d29eaaf0e9c29d761d46558f5404e5186904376a3c20a99b5ec1d36'
CONDITIONS=('no_growth','linear_graph_growth')
PAIRED_BASES=None

def original():
    if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Original Growth primitive changed')
    name='paired_original_growth_v2'
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=BASE.resolve():raise ValueError('Original Growth primitive shadowed')
        return module
    spec=importlib.util.spec_from_file_location(name,BASE);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def calibrate(*args,**kwargs):return original().calibrate(*args,**kwargs)

def bases(torch,encoder,h,g,support,required=None):
    # Pay the unchanged graph calibration/filter/SVD work. Use the exact paired
    # saved graph bank as the declared matched initialization, never a redraw.
    calculated,record=original().bases(torch,encoder,h,g,support,required='graph')
    if PAIRED_BASES is None:raise ValueError('Root-bound original graph basis bank required')
    rel=Path(PAIRED_BASES['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Phase-relative basis bank required')
    path=(PHASE/rel).resolve(strict=True)
    if not path.is_relative_to(PHASE) or hashlib.sha256(path.read_bytes()).hexdigest()!=PAIRED_BASES['sha256']:raise ValueError('Paired graph bank changed')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(saved,dict) or 'graph' not in saved:raise ValueError('Original graph bank missing')
    values=saved['graph']
    if tuple(values.shape)!=(4,2,256) or values.dtype!=torch.float32 or not bool(torch.isfinite(values).all()):raise ValueError('Original graph bank geometry differs')
    identity=torch.eye(2,dtype=values.dtype)
    if not torch.allclose(values@values.transpose(1,2),identity.expand(4,-1,-1),rtol=1e-5,atol=1e-5):raise ValueError('Paired graph Stiefel rows invalid')
    projectors=values.transpose(1,2)@values
    distances=[float((projectors[a]-projectors[b]).norm()/2.) for a in range(4) for b in range(a)]
    if min(distances)<=original().PROJECTOR_MIN:raise ValueError('Paired graph output projectors collapsed')
    current=calculated['graph'].detach().cpu()
    record.update(paired_graph_bases=PAIRED_BASES,exact_paired_graph_bank_used=True,
        graph_basis_recalculation_max_abs=float((current-values).abs().max()),
        basis_policy='Exact saved original same-seed TRAIN graph bank; unchanged recalculation paid and recorded; no redraw/rescue.',
        paired_graph_projector_distances=distances)
    return {'graph':values.to(device=h.device,dtype=h.dtype)},record

def make(torch,native,heads,models,initializer,warm,condition,seed,factor_seed,device,basis):
    if condition not in CONDITIONS:raise ValueError('Exact supplemental linear condition required')
    original_condition='no_growth' if condition=='no_growth' else'graph_growth'
    model,partition=original().make(torch,native,heads,models,initializer,warm,original_condition,seed,factor_seed,device,basis)
    if condition=='no_growth':return model,partition
    def forward(self,h,support,conv,members):
        incoming=torch.cat([self.V[m] for m in members],dim=1)
        packed=h@incoming
        propagated=conv(packed,support)
        groups=propagated.split(self.B.shape[1],dim=1)
        return [group@self.B[m] for group,m in zip(groups,members)]
    # Only this new model's message module changes; torch and original modules
    # are untouched. Parameter identities/names/ownership remain identical.
    model.growth.forward=MethodType(forward,model.growth)
    partition['growth'].update(condition=condition,message_activation='linear H@V; tanh omitted only',
        paired_original_graph_basis=True,new_parameters=4096,
        attribution='Private low-rank linear channel mixing before endpoint products can change member-specific bilinear link metrics. No nonlinear-neighbourhood-moment necessity claim if this explains the gain.',
        predictor_overall_still_nonlinear=True,sparse_growth_P_action_retained=True)
    return model,partition
