"""Derive a small self-contained runtime catalogue from reviewed finite mapping."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
E=HERE.parent.parent
FROZEN=E/'qualification/campaign_resources_v1'
SOURCE_SHA='dd52cd53f04f2e03bfe6c3b5f4cdc56d5f7714d2895db92bca5669cd5916459d'
FIELDS=('vram_bytes','ram_bytes','disk_bytes','cpu_slots','thread_count','profile')


def read(path,sha=None):
    raw=path.read_bytes()
    if sha is not None and hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Reviewed input bytes changed')
    return json.loads(raw)


def build():
    old=read(FROZEN/'review_snapshot_v1/RESOURCE_CLASS_PROPOSALS.json',SOURCE_SHA)
    examples=read(FROZEN/'ACTUAL_CLASS_INPUTS.json')['hosts']['allocation']['preset_examples']
    presets={};methods={};profiles={}
    for row in old['classes']:
        s=row['signature'];cell=s['cell_id'];kind=s['kind'];method=s['scientific_method_id']
        budget={k:row['resources'][k]for k in FIELDS};profile=budget['profile']
        if profile in profiles and profiles[profile]!=budget:raise ValueError('Host/controller budget divergence')
        profiles[profile]=budget
        native=method=='native_single';group='export'if kind!='fit'else'native'if native else'shared'
        record=presets.setdefault(cell,{'family':examples[cell]['family'],
            'native_fixed_config':examples[cell]['native_fixed_config'],'profiles':{}})
        if group in record['profiles']and record['profiles'][group]!=profile:raise ValueError('Ambiguous profile')
        record['profiles'][group]=profile
        if kind=='fit':
            families=methods.setdefault(method,[])
            if s['family']not in families:families.append(s['family'])
    for v in methods.values():v.sort()
    return {'schema':'all-fit-resource-selection-v1','source_basis':{
        'reviewed_resource_manifest_sha256':'6e6fd5acae6d37eb5803b869ff3a8f14d37bad190d9c618777550f553eae8bca',
        'resource_class_sha256':SOURCE_SHA,
        'independent_resource_review_sha256':'68a4f2b17afbf5b26c8e2183ac7ffb92977c68e2c8b7abbd1706cc9e127b1fde',
        'scope':'Reviewed existing mature source observations, finite component contracts and exact all-shuffle molecule size bounds; conservative admission reservations, not measured target-fit peaks'},
        'presets':presets,'methods':methods,'profiles':profiles,
        'pair_limits':{'maximum_owned_children_per_GPU':2,'maximum_combined_vram_bytes':64*1024**3,
            'maximum_combined_ram_bytes':48*1024**3,'heavy_serial_vram_threshold_bytes':64*1024**3},
        'smoke_policy':'Scheduler runs1child/GPU initially; reserve same envelope as training; no TEST or scientific score from2iterations',
        'original_paper_dataset_reruns':False,'new_native_models_or_updates':0}


if __name__=='__main__':
    result=build();path=HERE/'RESERVATIONS.json'
    with path.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    raw=path.read_bytes();print(json.dumps({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
        'presets':len(result['presets']),'methods':len(result['methods']),'profiles':len(result['profiles'])}))
