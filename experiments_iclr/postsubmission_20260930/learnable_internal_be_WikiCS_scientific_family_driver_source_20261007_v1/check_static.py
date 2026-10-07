"""Stdlib-only source/roster/disabled-release checks; no numerical fixture."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SUITE=ROOT.parent/'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
QUALIFIER=ROOT.parent/'learnable_internal_be_resource_qualifier_source_20261007_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pins=[(SUITE,'76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'),
        (QUALIFIER,'3668747a8e27ab7eaa3754f56744697d6a71cb1e2e3a05284175362e64833f50')]
    for folder,pin in pins:
        if sha(folder/'MANIFEST.json')!=pin:raise ValueError('Immutable source seal')
        for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
            p=folder/row['path']
            if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Immutable source bytes')
    parsed=[]
    for p in ROOT.glob('*.py'):ast.parse(p.read_text());parsed.append(p.name)
    config=json.loads((SUITE/'configs/wikics.json').read_text())
    adoption=json.loads((ROOT/'ADOPTION_PROSPECTIVE.json').read_text())
    expected=[{'arm':a,'seed':s,'cell':a+'_'+str(s)} for s in config['pilot_seeds'] for a in config['arms']]
    if adoption['cells']!=expected or len(expected)!=24 or adoption['epochs_each']!=1100 or adoption['total_planned_scientific_epochs']!=26400:
        raise ValueError('Exact full family roster/horizon')
    if adoption['root_adopted'] is not False or adoption['scientific_execution_enabled'] is not False or adoption['resource_checkpoints_reused_for_fits'] is not False:
        raise ValueError('Preparation/constructor boundary')
    launch=json.loads((ROOT/'LAUNCH_TEMPLATE_DISABLED.json').read_text())
    for key in ('root_scientific_fit_authorized','root_resource_execution_authorized','fixed_family_adopted','root_driver_source_approved','TEST_access','automatic_retry','predictive_opening_authorized'):
        if launch.get(key) is not False:raise ValueError('Disabled release')
    if launch['adoption']['sha256']!=sha(ROOT/'ADOPTION_PROSPECTIVE.json'):raise ValueError('Exact prospective binding')
    report={'schema':'internal-be-WikiCS-family-driver-static-check-v1','static_checks_passed':True,'python_AST_parsed':sorted(parsed),
        'source_v4_and_Qv1_full_seals_unchanged':True,'declared_cells':24,'full_scientific_epochs_each':1100,
        'root_launch_disabled':True,'framework_imports':0,'model_or_data_loading':False,'GPU_remote_staging_or_fits':False,
        'numerical_reference_or_fixture_gate':False,'scientific_results_present':False}
    (ROOT/'STATIC_CHECKS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'static_checks_passed':True,'declared_cells':24,'launch_disabled':True,'framework_imports':0}))


if __name__=='__main__':main()
