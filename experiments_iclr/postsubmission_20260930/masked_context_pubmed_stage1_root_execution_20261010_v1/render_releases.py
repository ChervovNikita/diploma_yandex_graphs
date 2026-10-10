"""Create exact nine scientific releases after root/source/runtime adoption."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
SOURCE = PHASE/'combination_masked_context_pubmed_stage1_source_20261010_v1'
PREP = PHASE/'masked_context_pubmed_allocation_preparation_20261010_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(path): return dict(path=str(path), sha256=sha(path))
def write(path, value):
    assert not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert HERE.is_relative_to(PHASE) and Path.cwd().resolve() == REPO
    assert sha(SOURCE/'SOURCE_MANIFEST.json') == '4ea24b47715777fee4f00b2c0d96a57657249c24d7d0541039645f4e3dd4aea9'
    assert (HERE/'OWNER_REVIEW.json').exists()
    owner_review=json.loads((HERE/'OWNER_REVIEW.json').read_text())
    assert owner_review['approved'] and owner_review['owner_sha256'] == sha(HERE/'queue.py')
    bindings=json.loads((SOURCE/'SOURCE_BINDINGS.json').read_text())
    qualifications={c:bound(PREP/folder/'COMPLETE.json') for c,folder in (
        ('shared4_core','qualification_V3b'),('shared4_own','qualification_shared4_own'),('independent4_native','qualification_independent4_native'))}
    terminals={c:bound(PREP/name) for c,name in (
        ('shared4_core','QUALIFICATION_V3b_TERMINAL.json'),('shared4_own','QUALIFICATION_shared4_own_TERMINAL.json'),('independent4_native','QUALIFICATION_independent4_native_TERMINAL.json'))}
    review=HERE/'ROOT_SOURCE_REVIEW.json'
    delta=PHASE/'masked_context_v2_v3_exporter_v4_assessment_20261010_v1/DELTA_REPORT.md'
    adoption=dict(schema='masked-context-stage1-root-adoption-v1',enabled=True,source_review_approved=True,
        source_delta_assessment_approved=True,complete_nine_roster_frozen=True,
        source_manifest_sha256=sha(SOURCE/'SOURCE_MANIFEST.json'),prototype_manifest_sha256=bindings['prototype_manifest']['sha256'],
        roster_sha256=sha(SOURCE/'ROSTER.json'),protocol_sha256=sha(SOURCE/'PROTOCOL.json'),
        source_review=bound(review),source_delta_assessment=bound(delta),qualifications=qualifications,
        qualification_terminals=terminals,no_TEST_scoring=True,all_nine_before_comparison=True,
        originally_reported_scores_unchanged=True)
    write(HERE/'ROOT_ADOPTION.json',adoption)
    readiness=dict(GPU_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip(),normal_host_execution=True)
    assert int(readiness['GPU_inventory'].split(',')[1]) >= 32768
    write(HERE/'READINESS.json',readiness)
    runtime=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
    records=[]
    for seed in (9101,9203,9307):
        for condition in ('shared4_own','shared4_core','independent4_native'):
            record_id=f'seed{seed}__{condition}'
            spec=json.loads((SOURCE/'releases_disabled'/(record_id+'.json')).read_text())
            for flag in ('enabled','root_science_authorized','source_review_approved','source_delta_assessment_approved',
                'runtime_qualified','complete_graph_data_custody_verified','VALID_custody_verified',
                'external_hard_bound_confirmed','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed','complete_nine_roster_frozen'):
                spec[flag]=True
            contract=json.loads((SOURCE/'FINITE_OWNER_CONTRACT_TEMPLATE_DISABLED.json').read_text())
            contract.update(enabled=True,record_id=record_id,separate_process_group=True,direct_wait_required=True,
                resource_caps_enforced=True,output_and_log_caps_enforced=True,root_admission_pending=False,
                owner_source=bound(HERE/'queue.py'),owner_review=bound(HERE/'OWNER_REVIEW.json'),no_owner_implementation_added=False)
            contract_file=HERE/'contracts'/(record_id+'.json');write(contract_file,contract)
            spec.update(source_manifest_sha256=sha(SOURCE/'SOURCE_MANIFEST.json'),root_admission=bound(HERE/'ROOT_ADOPTION.json'),
                split_custody=bound(PREP/'DATA_V4_ABSOLUTE_EXPORTER_CUSTODY.json'),
                train_bundle=bound(PREP/'data_v4/TRAIN_ONLY.npz'),valid_bundle=bound(PREP/'data_v4/VALID_ONLY.npz'),
                validation_custody=bound(HERE/'VALID_CUSTODY.json'),external_owner_release=bound(contract_file),
                resource_readiness_evidence=bound(HERE/'READINESS.json'),
                runtime=dict(python=bound(runtime),PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages')),
                output=str(HERE/'cells'/record_id))
            release_file=HERE/'releases'/(record_id+'.json');write(release_file,spec)
            records.append(dict(record_id=record_id,release=str(release_file.relative_to(PHASE)),release_sha256=sha(release_file)))
    plan=dict(enabled=True,automatic_retry=False,owner_sha256=sha(HERE/'queue.py'),
        train_source=str((SOURCE/'train.py').relative_to(PHASE)),limits=spec['limits'],records=records)
    write(HERE/'OWNER_PLAN.json',plan)
    print(json.dumps(dict(nine_releases_rendered=True,owner_plan_sha256=sha(HERE/'OWNER_PLAN.json'),science_not_started=True)))


if __name__ == '__main__': main()
