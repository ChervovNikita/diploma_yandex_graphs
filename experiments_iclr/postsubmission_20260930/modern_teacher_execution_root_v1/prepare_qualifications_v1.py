"""Freeze six full-graph qualification attempts before any numerical result."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
REMOTE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
ROOT = PHASE / 'modern_teacher_execution_root_v1'
PACKET = 'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v1'

def read(path):
    return json.loads((PHASE/path).read_text())

def descriptor(path):
    data = (PHASE/path).read_bytes()
    return dict(path=REMOTE+'/'+path, sha256=hashlib.sha256(data).hexdigest())

def write(path, value):
    with (PHASE/path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    for path in ('modern_teacher_execution_root_v1/qualification_entry_v2.py',
                 'modern_teacher_execution_root_v1/full_parity_checks_v1.py'):
        ast.parse((PHASE/path).read_text(), filename=path)
    base = read('modern_teacher_execution_root_v1/ENVIRONMENT_REQUEST_v1.json')
    prior = read('modern_teacher_execution_root_v1/ENVIRONMENT_ALLOCATION_v1.json')
    environment = read('modern_teacher_execution_root_v1/environment_run01/ENVIRONMENT.json')
    allocation_rel = 'modern_teacher_execution_root_v1/QUALIFICATION_ALLOCATION_v1.json'
    allocation = dict(schema='gnnm-modern-full-graph-qualification-allocation-v1',
        created_UTC=datetime.now(timezone.utc).isoformat(),
        prior_allocation=descriptor('modern_teacher_execution_root_v1/ENVIRONMENT_ALLOCATION_v1.json'),
        additional_phase_seconds=3600, additional_diagnostic_seconds=3600,
        additional_disk_bytes=4*1024**3,
        conservative_phase_bound_after_reservation=prior['conservative_phase_bound_after_reservation']+3600,
        phase_cap_seconds=137200,
        conservative_diagnostic_bound_after_reservation=prior['conservative_diagnostic_bound_after_reservation']+3600,
        diagnostic_cap_seconds=104160,
        conservative_disk_bound_after_reservation=prior['conservative_disk_bound_after_reservation']+4*1024**3,
        disk_cap_bytes=prior['disk_cap_bytes'], old_reservations_reclaimed=False,
        actual_GPU_time_claim=False, root_admitted=True,
        full_fit_admitted=False, heldout_scoring_admitted=False,
        rationale='Six complete-graph numerical qualifications at unchanged tolerances. Three Squirrel or four Photo updates are runtime checks and cannot become scientific fit results.')
    for kind in ('phase','diagnostic','disk'):
        cap_key=kind+'_cap_'+('bytes' if kind=='disk' else 'seconds')
        assert allocation['conservative_'+kind+'_bound_after_reservation'] < allocation[cap_key]
    write(allocation_rel, allocation)
    protected=[row for row in base['protected_files'] if not row['path'].endswith('/qualification_entry_v1.py')]
    protected += [descriptor(path) for path in (
        'modern_teacher_execution_root_v1/qualification_entry_v2.py',
        'modern_teacher_execution_root_v1/full_parity_checks_v1.py', allocation_rel)]
    template=read(PACKET+'/templates/TEACHER_ADMISSION_TEMPLATE.json')
    attempts=[]
    for backbone,graph in (('polyformer_mono','Squirrel'),('polynormer_r','Photo')):
        cell=read('coordinate_conformal_execution_root_v1/acquisition_run02/'+graph+'/LABEL_PACK_MANIFEST.json')['cells'][0]
        assert cell['seed']==17 and cell['source_split_index']==0 and cell['roles_frozen_before_label_extraction']
        for family in template['teacher_protocol']['families']:
            identity=backbone+'__'+family+'__config0_seed17_run01'
            admission_rel='modern_teacher_execution_root_v1/'+identity+'_ADMISSION.json'
            request_rel='modern_teacher_execution_root_v1/'+identity+'_REQUEST.json'
            admission=json.loads(json.dumps(template))
            admission.update(execution_authorized=True, environment=environment,
                device='cuda:0', graph=graph, backbone=backbone, family=family, config=0,
                role_freeze=cell['role_freeze'],
                source_labels={key:cell['source_labels'][key] for key in ('train','validation')},
                roles_frozen_before_label_extraction=True,
                prior_exposure_disclosure='Public benchmarks and prior GNNM findings are known. Photo had earlier coordinate pilot train/validation exposure, retained as STAGE1_NO_GO. No fitted pilot outputs are reused. This is the new derived_roles_v2 supervision protocol, not unseen-data confirmation or reproduction of published paper scores. Final label packs are not supplied.',
                independent_of_stage1_outcomes=True,
                qualification_only=True, report_eligible=False)
            write(admission_rel,admission)
            request=json.loads(json.dumps(base))
            request.update(action='qualification',
                entry_script=REMOTE+'/modern_teacher_execution_root_v1/qualification_entry_v2.py',
                teacher_admission=descriptor(admission_rel),
                protected_files=protected+[descriptor(admission_rel)],
                output=REMOTE+'/modern_teacher_execution_root_v1/'+identity,
                allocation=descriptor(allocation_rel), whole_cap_seconds=600)
            write(request_rel,request)
            attempts.append(dict(identity=identity, request=descriptor(request_rel),
                admission=descriptor(admission_rel), report_eligible=False,
                final_labels_supplied=False, numerical_result_at_registration=None,
                outer='modern_teacher_execution_root_v1/'+identity+'_bound',
                inner='modern_teacher_execution_root_v1/'+identity+'_supervisor',
                launch_receipt='protocols/MODERN_QUALIFICATION_LAUNCH_'+identity+'.json'))
    write('modern_teacher_execution_root_v1/QUALIFICATION_ATTEMPT_REGISTRY_v1.json',
        dict(schema='gnnm-modern-qualification-attempt-registry-v1',
             created_UTC=datetime.now(timezone.utc).isoformat(), source_manifest=base['source_manifest'],
             allocation=descriptor(allocation_rel), attempts=attempts,
             all_attempts_retained=True, automatic_retry=False, scientific_fit_admission=False,
             scope='Full graph / native blocks / fixed tolerances. Qualification only.'))
    print(json.dumps(dict(registered=len(attempts),allocation=allocation_rel)))

if __name__=='__main__':
    main()
