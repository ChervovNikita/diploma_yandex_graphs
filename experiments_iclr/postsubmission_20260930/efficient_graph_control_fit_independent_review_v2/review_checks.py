"""Independent bounded stdlib review. Scientific bodies are never imported."""
from __future__ import annotations
import ast
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
TARGET=HERE.parent/'efficient_graph_control_fit_preparation_v2'
V1=HERE.parent/'efficient_graph_control_fit_preparation_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(folder):
    manifest=json.loads((folder/'MANIFEST.json').read_text())
    inspected=[]
    for item in manifest['files']:
        path=folder/item['path']
        assert path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        inspected.append(dict(packet=folder.name,**item))
    return inspected


def run_checks():
    inspected=verified(TARGET)+verified(V1)
    seal=json.loads((TARGET/'SEAL.json').read_text())
    assert seal['manifest_sha256']==sha(TARGET/'MANIFEST.json')
    assert seal['repair_report_sha256']==sha(TARGET/'REPAIR_REPORT.json')
    assert seal['source_checks_sha256']==sha(TARGET/'SOURCE_CHECKS.json')
    python=[]
    for path in sorted(TARGET.rglob('*.py')):
        tree=ast.parse(path.read_text(),filename=str(path));compile(tree,str(path),'exec')
        python.append(dict(path=str(path.relative_to(TARGET)),sha256=sha(path)))
    def stripped(folder,filename):
        tree=ast.parse((folder/filename).read_text())
        for node in tree.body:
            if filename=='control_protocol.py' and isinstance(node,ast.FunctionDef) and node.name=='validate_admission':
                node.body=[ast.Pass()]
            if filename=='fit_controls.py' and isinstance(node,ast.ClassDef) and node.name=='_Recorder':
                for method in node.body:
                    if isinstance(method,ast.FunctionDef) and method.name=='charge':method.body=[ast.Pass()]
        return ast.dump(tree,include_attributes=False)
    assert stripped(TARGET,'control_protocol.py')==stripped(V1,'control_protocol.py')
    assert stripped(TARGET,'fit_controls.py')==stripped(V1,'fit_controls.py')
    # Read/evaluate only the protocol's stdlib metadata and pure validator.
    protocol_tree=ast.parse((TARGET/'control_protocol.py').read_text())
    allowed={'__future__','copy','hashlib','json','re','dataclasses'}
    for node in protocol_tree.body:
        if isinstance(node,ast.Import):assert all(n.name.split('.')[0] in allowed for n in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in allowed
    protocol=runpy.run_path(str(TARGET/'control_protocol.py'))
    original=runpy.run_path(str(V1/'control_protocol.py'))
    for backend in ['vmap','sequential']:
        assert protocol['fixed_plan'](packed_backend=backend)==original['fixed_plan'](packed_backend=backend)
    plans=protocol['fixed_plan'](packed_backend='vmap')
    assert len(plans)==21
    identity_fields=['packet_manifest_sha256','prepared_graph_sha256','train_pack_sha256','validation_pack_sha256']
    cases=[]
    for spec in plans:
        request=dict(schema='efficient-graph-control-root-admission-v1',scientific_decision='frozen_admitted',
            mode='full',decision_id='SYNTHETIC-NOT-ADMISSION',attempt_id='SYNTHETIC-NOT-RUN',
            protocol_sha256=protocol['PROTOCOL_SHA256'],spec_sha256=protocol['digest'](spec),
            packet_manifest_sha256=sha(TARGET/'MANIFEST.json'),prepared_graph_sha256='2'*64,
            train_pack_sha256='3'*64,validation_pack_sha256='4'*64,
            prior_graph_init_closure={'closed':True,'receipt_sha256':'5'*64},
            runtime_binding=dict(torch_version='SYNTHETIC',pyg_version='SYNTHETIC',device='cpu',environment_sha256='6'*64))
        request['root_qualification']=dict(passed=True,spec_sha256=request['spec_sha256'],
            runtime_binding=copy.deepcopy(request['runtime_binding']),receipt_sha256='7'*64,
            passed_checks=protocol['qualification_checks'](spec),
            **{k:request[k] for k in identity_fields})
        validate=lambda r:protocol['validate_admission'](spec,r,mode='full',manifest_sha256=sha(TARGET/'MANIFEST.json'))
        assert validate(request)==request
        for key in identity_fields:
            for kind in ['wrong_well_formed','missing','malformed']:
                modified=copy.deepcopy(request)
                if kind=='missing':del modified['root_qualification'][key]
                else:modified['root_qualification'][key]='8'*64 if kind=='wrong_well_formed' else 'invalid'
                try:validate(modified)
                except ValueError:pass
                else:raise AssertionError('Qualification identity incorrectly accepted')
                cases.append(dict(spec_sha256=request['spec_sha256'],field=key,case=kind,rejected=True))
        modified=copy.deepcopy(request);modified['runtime_binding']['environment_sha256']='9'*64
        try:validate(modified)
        except ValueError:pass
        else:raise AssertionError('Old qualification accepted for a changed admission environment')
    # Extract just the actual repaired contextmanager into a stdlib namespace.
    tree=ast.parse((TARGET/'fit_controls.py').read_text())
    recorder=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_Recorder')
    charge=next(n for n in recorder.body if isinstance(n,ast.FunctionDef) and n.name=='charge')
    namespace={'contextlib':contextlib,'time':time}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[charge],type_ignores=[])),
                 '<independently-extracted-recorder-charge>','exec'),namespace)
    rows=[]
    scenarios=[('success',None,None,None),('body_only',ValueError('synthetic body'),None,None),
        ('end_only',None,None,RuntimeError('synthetic end sync')),
        ('dual',ValueError('synthetic body'),None,RuntimeError('synthetic end sync')),
        ('interruption_dual',KeyboardInterrupt('synthetic interruption'),None,RuntimeError('synthetic end sync')),
        ('begin_only',None,RuntimeError('synthetic begin sync'),None)]
    for name,body_error,begin_error,end_error in scenarios:
        holder=SimpleNamespace(intervals=[],active_stage='SYNTHETIC')
        calls=[];entered=[]
        def sync():
            calls.append(True)
            selected=begin_error if len(calls)==1 else end_error
            if selected is not None:raise selected
        holder.synchronize=sync
        caught=None
        try:
            with namespace['charge'](holder,'SYNTHETIC'):
                entered.append(True)
                if body_error is not None:raise body_error
        except BaseException as error:caught=error
        expected=begin_error if begin_error is not None else end_error if end_error is not None else body_error
        assert caught is expected
        if begin_error is not None:assert len(calls)==1 and not entered and not holder.intervals
        else:
            assert len(calls)==2 and entered==[True] and len(holder.intervals)==1
            event=holder.intervals[0]
            assert event['body_error']==(f'{type(body_error).__name__}: {body_error}' if body_error else None)
            assert event['synchronization_error']==(f'{type(end_error).__name__}: {end_error}' if end_error else None)
            assert event['wall_seconds']>=0 and event['cpu_seconds']>=0
        if end_error is not None and body_error is not None:
            assert caught.__cause__ is body_error and caught.__context__ is body_error
        rows.append(dict(case=name,expected_exception_identity=True,intervals=holder.intervals))
    forbidden={'torch','torch_geometric','numpy','scipy','pandas','dgl','sklearn'}
    assert not any(name.split('.')[0] in forbidden for name in sys.modules)
    return dict(schema='independent-efficient-control-v2-source-review-checks-v1',status='PASSED_BOUNDED_STDLIB_CHECKS',
        target_manifest_sha256=sha(TARGET/'MANIFEST.json'),target_seal_sha256=sha(TARGET/'SEAL.json'),
        verified_target_payload_files=len(json.loads((TARGET/'MANIFEST.json').read_text())['files']),
        verified_v1_payload_files=len(json.loads((V1/'MANIFEST.json').read_text())['files']),
        Python_AST_compile=python,inspected_hashes=inspected,
        semantic_AST_unchanged_outside_repairs=True,both_fixed_plans_unchanged=True,
        accepted_synthetic_full_cells=len(plans),qualification_identity_rejections=cases,
        environment_mismatch_rejections=len(plans),actual_charge_method_mock_cases=rows,
        scientific_imports=False,scientific_runtime_qualification=False,
        data_labels_checkpoints_or_original_scores_accessed=False,SSH_network_GPU_execution=False,
        scope='Independent inspection of v2 repairs; not full numerical qualification or new-host admission')


if __name__=='__main__':
    result=run_checks()
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['status','verified_target_payload_files','verified_v1_payload_files',
        'accepted_synthetic_full_cells','environment_mismatch_rejections','target_manifest_sha256']}))
