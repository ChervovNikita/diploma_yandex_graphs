"""Only new snapshot identity hooks for the existing Wiki12 collect_cell API."""
import hashlib
import math
from pathlib import Path

ARCHIVE_SHA = '80d30ed946a1408f861a773952b174713c487765c75e95b8e25456b30478eae6'
PANEL_SHA = '9b742aa62fdd93496c2604edc8198b988a20f7f2ba83922a0f1a16597b9860bd'
CONDITIONS = ('shared_common','shared_route','shared_route_permuted')


def require(value,message):
    if not value:
        raise ValueError(message)


class Adapter:
    def __init__(self,dispatch):
        self.dispatch = dispatch

    def configured_recipe(self,public,condition):
        require(condition in CONDITIONS,'Exact Stage1 shared condition')
        return self.dispatch.configured_recipe(public,condition)

    def identity(self,condition):
        require(condition in CONDITIONS,'Exact Stage1 shared condition')
        return dict(control_id=condition,underlying_session_arm='be_unit_contrastive')


def restore(torch,model,saved,record,config,specification,source_pins,public):
    """Retain original selected global flag, epoch and saved member RNG streams."""
    require(isinstance(saved,dict) and saved.get('config')==config,'Original context config identity')
    run = saved['run']; condition = record['condition']
    require(condition in CONDITIONS and run['task']=='wikics' and run['arm']==condition
        and run['seed']==record['seed'] and run['underlying_session_arm']=='be_unit_contrastive'
        and run['TEST_scoring'] is False and run['driver_sha256']==source_pins['context_driver_sha256'],
        'Original selected context driver/method/seed identity')
    require(run['checkpoint_policy']==dict(method=condition,underlying_arm='be_unit_contrastive',
        own_selected_four=False,objective_separable=True),'Original context shared joint-selection policy')
    require(run['explicit_context_objective']==config['context_target_method']
        and run['verified_frozen_targets']['target_archive_sha256']==ARCHIVE_SHA
        and run['verified_frozen_targets']['original_panel_sha256']==PANEL_SHA
        and run['verified_frozen_targets']['target_relations_regenerated'] is False,
        'Original verified family target identity')
    require(run['data']['train_npz_sha256']==source_pins['train']['sha256']
        and run['data']['valid_npz_sha256']==source_pins['development']['sha256'], 'Original data roles')
    expected_core = {name:hashlib.sha256((public.ROOT/'core'/name).read_bytes()).hexdigest()
                     for name in ('factors.py','models.py','objectives.py','selection.py')}
    require(run['core']==expected_core
        and run['native']['polynormer_model_sha256']==source_pins['polynormer']['sha256'],
        'Original selected core/native functions')
    require(type(saved.get('global')) is bool and type(saved.get('epoch')) is int
        and 1<=saved['epoch']<=1100
        and saved.get('checkpoint_kind')=='strict-first-maximum complete VALID joint snapshot',
        'Original selected stage/epoch/checkpoint kind')
    require(model.members==4 and not model.independent,'Original shared M4 predictor')
    model.load_state_dict(saved['model'],strict=True); model.set_global(saved['global'])
    streams = saved['streams']
    require(len(streams)==4 and all(set(s)=={'cpu','cuda'} and
        all(t.device.type=='cpu' and t.dtype==torch.uint8 for t in s.values()) for s in streams),
        'Original CPU byte member RNG streams')
    require(math.isfinite(saved['selected_VALID']) and len(saved['member_VALID'])==4
        and all(math.isfinite(x) for x in saved['member_VALID']),'Original selected scalar metadata')
    model.eval()
    return streams,dict(epoch=saved['epoch'],global_mode=saved['global'],
        stored_accuracy=saved['selected_VALID'],stored_member_accuracy=saved['member_VALID'],
        method_identity=condition,reselection=False,bitwise_parity_claimed=False,
        frozen_target_archive_sha256=ARCHIVE_SHA)
