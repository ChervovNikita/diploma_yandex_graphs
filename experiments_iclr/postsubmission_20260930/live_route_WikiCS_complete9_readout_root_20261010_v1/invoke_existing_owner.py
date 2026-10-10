"""One selected-readout child through the already reviewed run_fit owner."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,socket,subprocess,time

R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    assert socket.gethostname()=='peptide' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
    plan_path=H/'OWNER_PLAN.json';assert sha(plan_path)==args.plan_sha256
    plan=json.loads(plan_path.read_text());assert plan['enabled'] and not plan['automatic_retry']
    def bound(row):
        q=(P/row['path']).resolve(strict=True);assert q.is_relative_to(P) and sha(q)==row['sha256'];return q
    source=bound(plan['existing_run_fit_helper']);helper_path=bound(plan['existing_ownership_helper'])
    spec=importlib.util.spec_from_file_location('_complete9_readout_existing_owner',source)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner);assert owner.HELPER==helper_path
    helper,context=owner.lane(plan['physical_gpu_uuid'])
    context.SOURCE=bound(plan['entry_program']).parent;context.SOURCE_SHA=plan['source_manifest_sha256']
    release=bound(plan['release']);cfg=json.loads(release.read_text());output=P/cfg['output'];assert not output.exists()
    root=P/plan['supervision_output'];root.mkdir(exist_ok=False);(root/'logs').mkdir()
    started=time.monotonic()
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['physical_gpu_uuid'],PYTHONPATH='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');environment.pop('PYTHONHOME',None)
    entry=dict(cell_id='selected_readout9',job_relative=str(release.relative_to(P)),job_sha256=plan['release']['sha256'],hard_seconds=plan['active_seconds'],argv=plan['argv'])
    receipt=owner.run_fit(helper,root,entry,{'resource_limits':plan['resource_limits']},environment,output,context)
    child=receipt['raw_identity_observation'];assert child is not None
    current=helper.identity(child['PID']);absent=current is None or current['start_ticks']!=child['start_ticks']
    rows=helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],5)
    no_cuda=not any(len(x)>=2 and x[1].strip()==str(child['PID']) for x in (r.split(',') for r in rows))
    physical=dict(UTC=helper.now(),child_PID=child['PID'],child_start_ticks=child['start_ticks'],physical_gpu_uuid=plan['physical_gpu_uuid'],owned_PID_absent=absent,owned_PID_no_CUDA_rows=no_cuda,terminal_wait_observed=receipt['terminal_wait_observed'])
    owner.write(root/'logs/selected_readout9.PHYSICAL_TERMINAL.json',physical)
    success=receipt['exit_code']==0 and receipt['terminal_wait_observed'] and receipt['reason'] is None and not receipt['signals_sent'] and absent and no_cuda
    owner.write(H/'OWNER_END.json',dict(UTC=helper.now(),success=success,seconds=time.monotonic()-started,child=child,exit_receipt=receipt,physical_terminal=physical,TEST_access=False,training_updates=0))
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
