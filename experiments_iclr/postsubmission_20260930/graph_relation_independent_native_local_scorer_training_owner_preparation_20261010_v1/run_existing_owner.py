"""Finite fixed lane invocation over the unchanged reviewed run_fit helper."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
from types import SimpleNamespace

R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
P=R/'experiments_iclr/postsubmission_20260930'
GPUS=['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
ROSTERS={'8ced':['6101_alphaF','6101_relationJ','6307_alphaF','6307_relationJ'],'a998':['6203_alphaF','6203_relationJ']}
WORK=dict(shadow_member_forwards=8800,replay_member_forwards=8800,output_cotangent_collections=2200,member_reverse_collections=17600,optimizer_bank_updates=1100,exact_member_RNG_endpoint_checks=1100)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def phase_file(relative):
 path=(P/relative).resolve(strict=True)
 assert path.is_relative_to(P) and path.is_file()
 return path
def bound(row):
 path=phase_file(row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];return path
def binding(path):return dict(path=str(path.relative_to(P)),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
 temporary=path.with_name(path.name+'.tmp')
 with temporary.open('x') as stream:
  json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 os.replace(temporary,path)
def physical():
 assert socket.gethostname()=='peptide' and Path.cwd().resolve()==R
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==GPUS


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--owner-plan',type=Path,required=True);parser.add_argument('--owner-plan-sha256',required=True)
 parser.add_argument('--lane',choices=('8ced','a998'),required=True);args=parser.parse_args()
 began=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF);children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
 physical();assert args.owner_plan.resolve().is_relative_to(P) and sha(args.owner_plan)==args.owner_plan_sha256
 plan=read(args.owner_plan)
 assert plan['enabled'] is plan['root_owner_execution_authorized'] is plan['source_review_approved'] is True
 assert plan['scientific_fits']==6 and plan['epochs']==1100 and plan['local_epochs']==100
 assert plan['VALID_scores_read'] is plan['TEST_access'] is plan['automatic_retry'] is False
 assert bound(plan['owner_adapter'])==Path(__file__).resolve()
 bound(plan['scientific_source_seal']);bound(plan['source_review']);bound(plan['protocol'])
 source=bound(plan['scientific_source_manifest'])
 for row in read(source)['files']:
  path=source.parent/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
 adoption=read(bound(plan['conditional_adoption']))
 assert adoption['root_adopted'] and adoption['whole_relation12_closed'] and adoption['frozen_primary_failed'] and adoption['frozen_placement_failed']
 anchors=read(bound(plan['copied_anchor_custody']))
 assert anchors['complete'] and anchors['all_original12_closed'] and len(anchors['records'])==6 and not anchors['checkpoint_reuse_for_training']
 for row in anchors['records']:
  bound(row['release']);bound(row['completion']);bound(row['actual_exit'])
 qualifications=read(bound(plan['qualification_custody']))
 assert qualifications['complete'] and {row['physical_gpu_uuid'] for row in qualifications['records']}==set(GPUS)
 for row in qualifications['records']:
  q=read(bound(row['qualified']));ex=read(bound(row['exit']));terminal=read(bound(row['physical_terminal']));complete=read(bound(row['owner_complete']))
  assert q['complete'] and q['physical_gpu_uuid']==row['physical_gpu_uuid'] and q['source_manifest_sha256']==plan['scientific_source_manifest']['sha256']
  assert q['policies']==['alphaF','relationJ'] and q['real_complete_TRAIN_updates']==4 and not q['VALID_scores_read'] and not q['TEST_access']
  assert complete['passed'] and ex['exit_code']==0 and ex['terminal_wait_observed'] and not ex['signals_sent'] and ex['reason'] is None
  assert terminal['owned_PID_absent'] and terminal['owned_PID_no_CUDA_rows'] and terminal['direct_wait_observed']
 lane=plan['lanes'][args.lane];assert [row['cell_id'] for row in lane['cells']]==ROSTERS[args.lane]
 gpu=GPUS[1 if args.lane=='8ced' else 0];assert lane['physical_gpu_uuid']==gpu
 root=P/lane['owner_output'];assert not root.exists() and root.parent.is_dir()
 root.mkdir(mode=0o700);(root/'logs').mkdir(mode=0o700)
 old_path=bound(plan['existing_run_fit_helper']);bound(plan['existing_ownership_helper'])
 spec=importlib.util.spec_from_file_location('_scorer_unchanged_run_fit',old_path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 helper,_=old.lane(gpu)
 context=SimpleNamespace(REPO=R,SOURCE=source.parent,SOURCE_SHA=plan['scientific_source_manifest']['sha256'],GPU_UUID=gpu,GPU_UUIDS=tuple(GPUS),phase_file=phase_file,physical_host=physical,sha=sha,write=write)
 write(root/'PARENT_OWNER.json',dict(identity=helper.identity(os.getpid()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner_plan=binding(args.owner_plan),lane=args.lane,detached_parent_OS_exit=None,direct_parent_wait_observed=False,numerical_execution_by_parent=False))
 deadline=began+lane['finite_lane_seconds'];records=[dict(cell_id=row['cell_id'],status='unlaunched',automatic_retry=False) for row in lane['cells']]
 failure=None
 try:
  for row,record in zip(lane['cells'],records):
   assert deadline-time.monotonic()>=plan['reserved_next_whole_cell_seconds']
   physical();release_path=bound(row);release=read(release_path);supervision=read(bound(release['external_supervision']))
   assert release['enabled'] and release['root_execution_authorized'] and release['physical_gpu_uuid']==gpu
   assert release['conditional_adoption']==plan['conditional_adoption'] and release['native_qualification']==next(q['qualified'] for q in qualifications['records'] if q['physical_gpu_uuid']==gpu)
   assert supervision['active_seconds']==plan['active_seconds_per_cell'] and supervision['cleanup_seconds']==10
   output=P/release['output'];assert not output.exists() and output.parent.is_dir()
   entry=dict(cell_id=row['cell_id'],job_relative=row['path'],job_sha256=row['sha256'],hard_seconds=plan['active_seconds_per_cell'],argv=[plan['runtime']['python'],'-B',str(bound(plan['train_program'])),'--release',str(release_path),'--release-sha256',row['sha256']])
   environment=dict(os.environ,**lane['env']);environment.pop('PYTHONHOME',None)
   record.update(status='attempted',release=row)
   write(root/'PROGRESS.json',dict(records=records,VALID_scores_read=False,TEST_access=False))
   receipt=old.run_fit(helper,root,entry,{'resource_limits':plan['resource_limits']},environment,output,context)
   child=receipt['raw_identity_observation'];assert child is not None
   absent=helper.identity(child['PID']) is None
   cuda=helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],10)
   no_cuda=not any(len(parts)>1 and parts[1].strip()==str(child['PID']) for parts in (line.split(',') for line in cuda))
   terminal=dict(identity=child,child_absent=absent,child_no_CUDA_rows=no_cuda,actual_exit_code=receipt['exit_code'],direct_wait_observed=receipt['terminal_wait_observed'])
   write(root/'logs'/(row['cell_id']+'.PHYSICAL_TERMINAL.json'),terminal)
   record.update(status='terminal',actual_exit=receipt,physical_terminal=terminal)
   assert receipt['exit_code']==0 and receipt['terminal_wait_observed'] and receipt['reason'] is None and not receipt['signals_sent'] and receipt['signal_refusal'] is None and absent and no_cuda
   assert not (output/'FAILURE.json').exists()
   endpoint=read(output/'COMPLETE.json');descriptor=endpoint['graph_relation_credit']
   assert endpoint['complete'] and endpoint['epochs']==endpoint['steps']==1100 and endpoint['seed']==release['seed']
   assert descriptor['partition']['policy']==release['policy'] and descriptor['work']==WORK
   assert descriptor['validation_work']['evaluations']==1100 and descriptor['validation_work']['member_forwards']==4400
   assert endpoint['source_manifest_sha256']==plan['scientific_source_manifest']['sha256']
   record.update(status='complete',completion=binding(output/'COMPLETE.json'),selected_checkpoint=dict(path=str((output/'selected.pt').relative_to(P)),bytes=(output/'selected.pt').stat().st_size,sha256=endpoint['selected_sha256'],hash_authority='Original COMPLETE receipt'))
   write(root/'PROGRESS.json',dict(records=records,VALID_scores_read=False,TEST_access=False))
  write(root/'LANE_CLOSURE.json',dict(complete=True,lane=args.lane,records=records,VALID_scores_read=False,TEST_access=False,automatic_retry=False,detached_parent_OS_exit=None))
 except BaseException as error:
  failure=dict(type=type(error).__name__,error=str(error),partial_work_retained=True,automatic_retry=False)
  write(root/'FAILURE.json',dict(failure=failure,records=records,VALID_scores_read=False,TEST_access=False))
  raise
 finally:
  after=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
  write(root/'COST.json',dict(owner_inclusive_seconds=time.monotonic()-began,owner_CPU_user_seconds=after.ru_utime-before.ru_utime,owner_CPU_system_seconds=after.ru_stime-before.ru_stime,training_and_telemetry_children_CPU_user_seconds=children.ru_utime-children_before.ru_utime,training_and_telemetry_children_CPU_system_seconds=children.ru_stime-children_before.ru_stime,owner_peak_RSS_bytes=after.ru_maxrss*1024,separated_training_child_CPU_seconds=None,separated_admission_cleanup_mirroring_seconds=None,records=records,failure=failure,VALID_scores_read=False,TEST_access=False,automatic_retry=False,detached_parent_OS_exit=None))


if __name__=='__main__':main()
