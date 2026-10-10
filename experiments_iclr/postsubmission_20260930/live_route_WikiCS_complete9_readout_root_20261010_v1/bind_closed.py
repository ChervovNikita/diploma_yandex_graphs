"""Bind actual closed-nine metadata before the one selected-state readout."""
from pathlib import Path
import hashlib,json,os,socket,subprocess

R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
S=P/'live_route_WikiCS_complete9_readout_source_20261010_v1'
A=P/'live_route_WikiCS_complete9_root_activation_20261010_v1'
F=P/'live_route_WikiCS_complete9_supervision_root_20261010_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def record(p):return dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p))
def save(p,value):
    with p.open('x') as handle:handle.write(json.dumps(value,indent=2)+'\n')

def main():
    assert socket.gethostname()=='peptide' and Path.cwd()==R
    inventory=['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==inventory
    launch=read(A/'LAUNCH.json');assert not Path('/proc',str(launch['PID'])).exists()
    assert not (F/'FAMILY_FAILURE.json').exists()
    plan=read(A/'OWNER_PLAN.json');family=read(F/'FAMILY_COMPLETE.json');pins=read(S/'SOURCE_BINDINGS.json')
    assert family['complete'] and family['all9_directly_waited'] and family['all9_owned_absence_verified'] and family['TEST_access'] is False
    assert len(plan['records'])==9 and family['records']==[r['cell_id'] for r in plan['records']]
    cells=[]
    for row in plan['records']:
        release_path=P/row['release']['path'];assert sha(release_path)==row['release']['sha256']
        fit=read(release_path);output=P/fit['output'];complete_path=output/'COMPLETE.json';complete=read(complete_path)
        exit_path=F/'logs'/(row['cell_id']+'.EXIT.json');terminal=read(exit_path)
        physical_path=F/'logs'/(row['cell_id']+'.PHYSICAL_TERMINAL.json');physical=read(physical_path)
        assert complete['complete'] and complete['epochs']==complete['steps']==1100 and complete['TEST_scoring'] is False
        assert complete['root_cell_release_sha256']==row['release']['sha256']
        assert terminal['exit_code']==0 and terminal['terminal_wait_observed'] and terminal['reason'] is None and not terminal['signals_sent']
        assert physical['owned_PID_absent'] and physical['owned_PID_no_CUDA_rows'] and physical['terminal_wait_observed']
        assert terminal['raw_identity_observation']['argv']==row['argv']
        assert not (output/'FAILURE.json').exists()
        cells.append(dict(cell_id=row['cell_id'],kind=row['kind'],seed=row['seed'],fit_release=row['release'],completion=record(complete_path),owner_exit=record(exit_path),physical_terminal=record(physical_path),preflight=record(F/'logs'/(row['cell_id']+'.PREFLIGHT.json')),trace=record(output/'VALID_TRACE.json'),selected=dict(path=str((output/'selected.pt').relative_to(P)),sha256=complete['selected_sha256'])))
    # Every original fit is closed before any selected-checkpoint readout is admitted.
    source=record(S/'readout.py');runtime=pins['runtime'];uuid=inventory[1]
    limits=dict(minimum_fresh_GPU_free_bytes=72*1024**3,resource_wait_seconds=60,telemetry_timeout_seconds=5,poll_interval_seconds=5,owned_tree_RSS_cap_bytes=32*1024**3,owned_tree_GPU_memory_cap_bytes=68*1024**3,own_fit_output_cap_bytes=256*1024**2,combined_child_log_cap_bytes=8*1024**2)
    output=str((H/'actual_readout_v1').relative_to(P));release_path=H/'RELEASE.json';supervision_path=H/'EXTERNAL_SUPERVISION.json'
    supervision=dict(enabled=True,root_owns_finite_launch_and_actual_costs=True,existing_run_fit_helper=pins['run_fit_helper'],existing_ownership_helper=pins['ownership_helper'],physical_gpu_uuid=uuid,entry_program=source,argv_prefix=[runtime['python'],'-B',str(S/'readout.py')],release_argument_path=str(release_path),output=output,active_seconds=600,hard_seconds=610,cleanup_seconds=10,resource_limits=limits)
    save(supervision_path,supervision)
    release=read(S/'RELEASE_TEMPLATE_DISABLED.json')
    release.update(enabled=True,root_execution_authorized=True,source_review_approved=True,whole9_closure_approved=True,existing_owner_binding_approved=True,physical_gpu_uuid=uuid,readout_manifest_sha256=sha(S/'MANIFEST.json'),external_supervision=record(supervision_path),fit_owner_plan=record(A/'OWNER_PLAN.json'),family_complete=record(F/'FAMILY_COMPLETE.json'),cells=cells,output=output)
    save(release_path,release)
    argv=[runtime['python'],'-B',str(S/'readout.py'),'--release',str(release_path),'--release-sha256',sha(release_path)]
    own=dict(enabled=True,automatic_retry=False,physical_gpu_uuid=uuid,existing_run_fit_helper=pins['run_fit_helper'],existing_ownership_helper=pins['ownership_helper'],entry_program=source,source_manifest_sha256=sha(S/'MANIFEST.json'),release=record(release_path),argv=argv,active_seconds=600,hard_seconds=610,resource_limits=limits,supervision_output=str((H/'supervision_v1').relative_to(P)),owner_program=record(H/'invoke_existing_owner.py'))
    save(H/'OWNER_PLAN.json',own)
    print(json.dumps(dict(whole9_closed=True,checkpoint_payload_opened=False,release=release,external_supervision=supervision,owner_plan=own,family_complete=family)))

if __name__=='__main__':main()
