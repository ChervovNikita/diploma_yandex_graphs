"""Parse/binding/default checks only; no numerical tests or outcome reads."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
NATIVE_ANALYSIS_SHA='07be5b87f903a2a702c10a57ffe9865a0887865280ef4454326750f9c5e8d61b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    sources=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in sources['files']:
        path=(HERE.parent/row['path']).resolve(strict=True)
        assert path.is_relative_to(HERE.parent) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    predecessor=HERE.parent/'sehgnn_IMDB_complete_cohort_quality_readout_preparation_20261009_v1'
    seal=json.loads((predecessor/'SEAL.json').read_text())
    assert sha(predecessor/'SEAL.json')==sources['predecessor_seal_sha256']
    assert sha(predecessor/'MANIFEST.json')==seal['manifest_sha256']
    for row in json.loads((predecessor/'MANIFEST.json').read_text())['files']:
        path=predecessor/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    assert sha(HERE/'analysis.py')==sha(predecessor/'analysis.py')==NATIVE_ANALYSIS_SHA
    calls=[]
    for path in HERE.glob('*.py'):
        tree=ast.parse(path.read_text())
        if path.name=='RUN.py':
            main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
            main_calls=[node for node in ast.walk(main) if isinstance(node,ast.Call)]
            torch_line=next(node.lineno for node in main_calls if isinstance(node.func,ast.Attribute)
                and node.func.attr=='import_module' and ast.literal_eval(node.args[0])=='torch')
            for name,count in (('entry_owner',2),('family_owner',2),('custody',1),('launch_owner',1)):
                matched=[node for node in main_calls if isinstance(node.func,ast.Name) and node.func.id==name]
                assert len(matched)==count and all(node.lineno<torch_line for node in matched)
            costs=next(node for node in ast.walk(main) if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id=='original_costs' for target in node.targets))
            assert costs.lineno<torch_line
            aliases={}
            for node in ast.walk(main):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name) and node.value.value.id=='valid':
                    target=node.targets[0]
                    if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='valid':
                        aliases[ast.literal_eval(target.slice)]=ast.literal_eval(node.value.slice)
            assert aliases=={'micro_F1':'served_micro_F1','macro_F1':'served_macro_F1'}
            text=path.read_text()
            assert "valid=dict(result['fresh_selected']['fresh_scores']['VALID'])" in text
            assert 'selected_checkpoint_digests_verified_before_transfer' not in text
            assert text.index("assert bound(row)==Path(cell)/name")<text.index("torch=importlib.import_module('torch')")
            assert "path.is_relative_to(HERE.parent)" in text and 'stream.read(1048576)' in text
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                    calls.append(name)
                    assert name not in {'train','fit','fit_body','make_model','evaluate','collect_selected_logits','runtime','Popen','kill','sleep','check_output'}
                    if name=='load' and isinstance(node.func,ast.Attribute):
                        assert {item.arg:ast.literal_eval(item.value) for item in node.keywords}=={'map_location':'cpu','weights_only':True}
    assert calls.count('compare_independent_controls')==calls.count('compare_selected')==1
    release=json.loads((HERE/'RELEASE.disabled.json').read_text())
    assert release['enabled'] is release['root_source_review_approved'] is release['root_readout_approved'] is False
    assert release['artifact_mode']=='resident_original_files' and release['transfer_scope']=='small_analysis_reports_only'
    for key in ('shared_fit_release','reference_fit_release','native_fit_release','shared_entry_report','reference_entry_report'):
        assert release[key]=={'path':None,'bytes':None,'sha256':None}
    inventory=json.loads((HERE/'ORIGIN_INVENTORY_TEMPLATE_DISABLED.json').read_text())
    assert inventory['artifact_mode']=='resident_original_files'
    assert inventory['selected_checkpoint_digests_verified_at_original_locations_before_readout'] is False
    assert inventory['selected_checkpoints']=={} and inventory['root_authorized'] is False
    assert 'selected_checkpoint_digests_verified_before_transfer' not in inventory
    for path in HERE.glob('*.json'):
        json.loads(path.read_text())
    if (HERE/'SEAL.json').exists():
        own_seal=json.loads((HERE/'SEAL.json').read_text())
        assert own_seal['runtime_disabled'] is True and sha(HERE/'MANIFEST.json')==own_seal['manifest_sha256']
        for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
            path=HERE/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    default=subprocess.run([sys.executable,'-B',str(HERE/'RUN.py')],capture_output=True,text=True,check=True)
    assert json.loads(default.stdout)=={'inactive':True,'outcomes_or_providers_read':False} and not default.stderr
    return dict(status='passed',parse_bindings_and_disabled_default_only=True,existing_analysis_functions_reused=True,
        sealed_V1_preserved=True,thin_analysis_byte_identical_to_V1=True,native_saved_served_aliases_preserve_original_fields=True,
        original_owner_release_custody_cost_and_checkpoint_checks_precede_arrays=True,resident_original_files_contract_inactive=True,
        no_model_fit_server_GPU_query_or_numerical_test_suite=True,scientific_outcomes_or_actual_receipts_read=False,
        source_verification_does_not_establish_quality=True)


if __name__=='__main__':
    result=check();target=HERE/'STATIC_VERIFICATION.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
