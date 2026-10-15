"""Small resource selection API for the approved all-fit smoke/full scheduler.

No model imports, remote access, authority gates, launches or scientific claims.
The scheduler owns live capacity, physical GPU/host ownership and slot locks.
"""
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CATALOGUE=HERE/'RESERVATIONS.json'
CATALOGUE_SHA256='7f13ee3071049d0c2ea446661fc6e63b6e47efa55be5634b7e33699801884349'
FIELDS=('vram_bytes','ram_bytes','disk_bytes','cpu_slots','thread_count','profile')
WORKERS=('allocation/gpu0','server51/gpu0','server51/gpu1')


@lru_cache(maxsize=1)
def _catalogue():
    raw=CATALOGUE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CATALOGUE_SHA256:raise ValueError('Resource catalogue bytes differ')
    value=json.loads(raw)
    if value.get('schema')!='all-fit-resource-selection-v1':raise ValueError('Wrong resource catalogue')
    return value


def _one(values,name):
    values=[v for v in values if v is not None]
    if not values or any(v!=values[0]for v in values):raise ValueError('Missing/inconsistent '+name)
    return values[0]


def reserve_for(task, job=None):
    """Return six JSON-safe resource fields for an exact deployed release task.

    Task keys: kind,cell_id,scientific_method_id,family (worker optional).
    Job fallback: recipe_id/recipe.id,scientific_method_id/procedure,family.
    If a native fixed config is supplied, it must equal the finite source preset.
    No seed/controller/canonical ID changes alter the reservation.
    """
    if not isinstance(task,Mapping)or(job is not None and not isinstance(job,Mapping)):
        raise TypeError('Task and optional actual descriptor must be mappings')
    job={}if job is None else job
    if task.get('worker')is not None and task['worker']not in WORKERS:
        raise ValueError('Unrecognized campaign worker')
    recipe=job.get('recipe',{})
    if not isinstance(recipe,Mapping):raise ValueError('Native recipe must be a mapping')
    preset=_one((task.get('cell_id'),job.get('recipe_id'),recipe.get('id')),'native preset')
    table=_catalogue()
    if preset not in table['presets']:raise ValueError('Preset outside new-dataset finite campaign')
    definition=table['presets'][preset]
    if 'native_fixed_config'in recipe and recipe['native_fixed_config']!=definition['native_fixed_config']:
        raise ValueError('Native shape/settings changed: existing reservation no longer applies')
    kind=task.get('kind','fit')
    if kind=='aggregate_I4_exports':
        if job:raise ValueError('I4 export consumes selected logits, not a new native descriptor')
        group='export'
    elif kind=='fit':
        # scientific_method_id is canonical; procedure may be an explicit native
        # frontend alias, so it is only a fallback when no canonical ID exists.
        canonical=[task.get('scientific_method_id'),job.get('scientific_method_id')]
        method=_one(canonical if any(v is not None for v in canonical)else(task.get('procedure'),job.get('procedure')),'scientific method')
        family=_one((task.get('family'),job.get('family'),definition['family']),'task family')
        if method not in table['methods']or family not in table['methods'][method]:
            raise ValueError('Method/family outside finite source resource scope')
        composition=job.get('normalization_composition')or{}
        if composition and composition.get('policy')!='private_native_running_state_shared_affine_v1':
            raise ValueError('Unknown normalization composition')
        if 'torch_threads'in job and job['torch_threads']!=2:
            raise ValueError('Expected the native two-thread execution envelope')
        group='native'if method=='native_single'else'shared'
    else:raise ValueError('Unexpected task kind')
    profile=definition['profiles'][group]
    return dict(table['profiles'][profile])


def can_pair(left,right):
    """Static compatibility only; scheduler still checks same GPU/live capacity.

    Both budgets must be unchanged known reservations. Combined peak budgets
    are at most64GiB VRAM/48GiB RAM. Heavy69GiB Physics shared tasks stay serial.
    This permits capacity coexistence; it promises no throughput gain.
    """
    table=_catalogue();limits=table['pair_limits']
    try:
        for value in(left,right):
            known=table['profiles'][value['profile']]
            if any(type(value[k])is not type(known[k])or value[k]!=known[k]for k in FIELDS):return False
            if value['vram_bytes']>limits['heavy_serial_vram_threshold_bytes']:return False
        return(left['vram_bytes']+right['vram_bytes']<=limits['maximum_combined_vram_bytes']and
               left['ram_bytes']+right['ram_bytes']<=limits['maximum_combined_ram_bytes'])
    except(KeyError,TypeError):return False


if __name__=='__main__':
    table=_catalogue()
    print(json.dumps({'schema':table['schema'],'presets':len(table['presets']),
        'methods':len(table['methods']),'profiles':len(table['profiles']),
        'catalogue_sha256':CATALOGUE_SHA256,'new_native_runs':0},sort_keys=True))
