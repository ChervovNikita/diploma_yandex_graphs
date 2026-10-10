"""Separate disabled activation of the unchanged fixed18 V3 candidate driver."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SERVER_PHASE=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def route():
    assert socket.gethostname()=='anogena-2-0' and PHASE==SERVER_PHASE
    assert HERE.name=='typed_context_candidate_science_activation_source_20261010_v1'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(row):
    path=Path(row['path']).resolve(strict=True)
    assert path.is_relative_to(PHASE) and path.is_file()
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    return path


def read(row):return json.loads(bound(row).read_text())


def descriptor(path):
    path=Path(path).resolve(strict=True);assert path.is_relative_to(PHASE)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def frozen():
    manifest=HERE/'SOURCE_MANIFEST.json';value=json.loads(manifest.read_text())
    assert value['source_only'] is True and value['enabled'] is False
    for row in value['files']:
        path=(HERE/row['path']).resolve(strict=True)
        assert path.is_relative_to(HERE) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    b=json.loads((HERE/'BINDINGS.json').read_text())
    for key in ('method_source_manifest','method_source_seal','method_source_protocol','independent_reference_source_manifest',
                'independent_reference_source_seal','independent_reference_source_protocol','independent_reference_activation_manifest'):bound(b[key])
    qrelease=read(b['qualification_release']);terminal=read(b['qualification_terminal']);report=read(b['qualification_report'])
    assert terminal['exit_code']==0 and terminal['stop_reason'] is None and not terminal['cleanup_errors']
    assert terminal['child_reaped'] is True and terminal['original_pid_absent'] is True and terminal['owned_CUDA_absence_verified'] is True
    fields=b['frozen_release_fields']
    assert report['status']=='complete' and report['complete'] is True and report['mode']=='qualification' and report['qualification_passed'] is True
    assert report['release']==b['qualification_release'] and report['source_seal_sha256']==b['method_source_seal']['sha256']
    assert report['protocol_sha256']==b['method_source_protocol']['sha256'] and report['conditions']==fields['conditions'] and len(report['runs'])==6
    assert {r['condition'] for r in report['runs']}==set(fields['conditions'])
    assert all(r['complete'] is True and r['qualification_passed'] is True and r['status']=='complete' and r['seed_spec']==fields['seed_specs'][0]
               and r['counters']['epochs']==1 and r['fresh_selected']['exact_model_optimizer_scaler_buffers_RNG_restore'] is True for r in report['runs'])
    for key in ('TEST_file_access','TEST_membership_known','TEST_truth'):assert report[key] is False
    for key in ('native_qualification_receipt','schema_receipt','roles','runtime_environment','expected_runtime_versions','expected_math_flags'):
        assert qrelease[key]==fields[key]
    reference=read(b['independent_reference_release']);launch=read(b['independent_reference_launch'])
    assert reference['action']=='fit_independently_selected_references' and reference['enabled'] is True and reference['scientific_execution_approved'] is True
    assert reference['activation_source_manifest_sha256']==b['independent_reference_activation_manifest']['sha256']
    assert reference['families']==['ordinary_native_independent4','contextual_native_independent4'] and reference['seed_specs']==fields['seed_specs']
    for key in ('candidate_qualification_receipt','native_qualification_receipt','roles','development_files','runtime_environment','expected_math_flags','expected_runtime_versions','source_seal_sha256','protocol_sha256'):
        assert reference[key]==fields[key]
    assert launch['release_sha256']==b['independent_reference_release']['sha256'] and launch['activation_source_manifest_sha256']==reference['activation_source_manifest_sha256']
    assert launch['normal_host_execution'] is True and launch['automatic_retry'] is False
    return b,sha(manifest)


def approvals(b,manifest_sha,root_review,owner_review):
    review=read(root_review);owner=read(owner_review)
    assert review['approved'] is True and review['scientific_fit_approved'] is True and review['method_source_modified'] is False
    assert review['activation_source_manifest_sha256']==manifest_sha and review['successful_candidate_qualification_verified'] is True
    assert review['all18_fixed_conditions_roles_approved'] is True and review['independent_reference_protocol_approved'] is True
    assert review['comparison_requires_both_whole_families_successfully_closed'] is True and review['normal_host_execution'] is True
    assert review['source_seal_sha256']==b['method_source_seal']['sha256'] and review['protocol_sha256']==b['method_source_protocol']['sha256']
    assert owner['approved'] is True and owner['activation_source_manifest_sha256']==manifest_sha
    assert owner['owner_sha256']==sha(HERE/'OWNED_FIT.py') and owner['entry_sha256']==sha(HERE/'ENTRY.py')
    assert owner['finite_bounds']==b['finite_bounds'] and owner['method_unchanged'] is True and owner['normal_host_execution'] is True and owner['automatic_retry'] is False


def admitted(spec,b,manifest_sha):
    assert spec['activation_source_manifest_sha256']==manifest_sha and spec['action']=='fit_complete_native_context_comparison'
    for key in ('enabled','root_source_review_approved','data_scope_approved','provider_runtime_approved','scientific_execution_approved','normal_host_execution','comparison_requires_both_whole_families_successfully_closed'):
        assert spec[key] is True
    assert spec['capabilities']=={k:True for k in ('source_bound','model','data','runtime','scientific')}
    assert spec['automatic_retry'] is False and spec['finite_bounds']==b['finite_bounds']
    assert all(spec[k]==v for k,v in b['frozen_release_fields'].items())
    assert spec['candidate_qualification_terminal']==b['qualification_terminal'] and spec['candidate_qualification_release']==b['qualification_release']
    approvals(b,manifest_sha,spec['root_review'],spec['owner_review'])
    policy=json.loads((HERE/'INDEPENDENT_REFERENCE_PROTOCOL_TEMPLATE_DISABLED.json').read_text());policy.update(enabled=True,root_source_review_approved=True)
    assert Path(spec['independent_reference_protocol']['path'])==HERE/'INDEPENDENT_REFERENCE_PROTOCOL.json'
    assert read(spec['independent_reference_protocol'])==policy
    output=Path(spec['output_directory']).resolve(strict=False)
    assert output.is_relative_to(PHASE) and not output.is_relative_to(HERE) and not output.exists()
    return output


def main():
    route();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args();release=HERE/'RELEASE.json';assert sha(release)==args.release_sha256
    spec=json.loads(release.read_text());b,manifest_sha=frozen();admitted(spec,b,manifest_sha)
    sys.path.insert(0,str(PHASE));package='typed_label_context_factor_source_prototype_20261010_v3'
    caps_module=importlib.import_module(package+'.caps');driver=importlib.import_module(package+'.driver')
    caps=caps_module.Caps(source_bound=True,model=True,data=True,runtime=True,scientific=True,root_review_sha256=spec['root_review']['sha256'])
    result=driver.execute(release,caps=caps)
    assert result['complete'] is True and result['mode']=='scientific_development_comparison' and result['qualification_passed'] is False
    assert len(result['runs'])==18 and all(r['complete'] and r['status']=='complete' and not r['qualification_passed'] for r in result['runs'])
    assert {(r['condition'],r['seed_spec']['role_seed']) for r in result['runs']}=={(c,s['role_seed']) for c in spec['conditions'] for s in spec['seed_specs']}
    print(json.dumps(dict(complete=True,fixed_candidate_fits=18,comparison_waits_for_both_whole_families=True,accuracy_claim=False)),flush=True)


if __name__=='__main__':main()
