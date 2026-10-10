"""Render the disabled six-cell plan after explicit root admission; launch nothing."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root-admitted',action='store_true');parser.add_argument('--plan',type=Path,required=True)
 parser.add_argument('--plan-sha256',required=True);parser.add_argument('--qualification-custody',type=Path,required=True)
 parser.add_argument('--qualification-custody-sha256',required=True);args=parser.parse_args()
 if not args.root_admitted:parser.error('Disabled until explicit root admission; no file or child created')
 R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');P=R/'experiments_iclr/postsubmission_20260930'
 GPUS=['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
 assert socket.gethostname()=='peptide' and Path.cwd().resolve()==R
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==GPUS
 def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
 def read(path):return json.loads(path.read_text())
 def bound(row):
  path=(P/row['path']).resolve(strict=True);assert path.is_relative_to(P) and path.is_file()
  assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];return path
 def binding(path):return dict(path=str(path.relative_to(P)),bytes=path.stat().st_size,sha256=sha(path))
 def write(path,value):
  with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 assert args.plan.resolve().is_relative_to(P) and sha(args.plan)==args.plan_sha256
 plan=read(args.plan);assert plan['enabled'] is False and plan['root_owner_execution_authorized'] is False
 assert bound(plan['admission_renderer'])==Path(__file__).resolve()
 bound(plan['owner_adapter']);bound(plan['scientific_source_manifest']);bound(plan['scientific_source_seal']);bound(plan['source_review']);bound(plan['protocol'])
 assert args.qualification_custody.resolve().is_relative_to(P) and sha(args.qualification_custody)==args.qualification_custody_sha256
 qualifications=read(args.qualification_custody)
 assert qualifications['complete'] and len(qualifications['records'])==2 and {row['physical_gpu_uuid'] for row in qualifications['records']}==set(GPUS)
 for row in qualifications['records']:
  qualified=read(bound(row['qualified']));exit_receipt=read(bound(row['exit']));terminal=read(bound(row['physical_terminal']));owner=read(bound(row['owner_complete']));bound(row['owner_cost'])
  assert qualified['complete'] and qualified['source_manifest_sha256']==plan['scientific_source_manifest']['sha256'] and qualified['physical_gpu_uuid']==row['physical_gpu_uuid']
  assert qualified['policies']==['alphaF','relationJ'] and qualified['real_complete_TRAIN_updates']==4 and not qualified['VALID_scores_read'] and not qualified['TEST_access']
  assert owner['passed'] and owner['QUALIFIED_sha256']==row['qualified']['sha256']
  assert exit_receipt['exit_code']==0 and exit_receipt['terminal_wait_observed'] and exit_receipt['reason'] is None and not exit_receipt['signals_sent']
  assert terminal['owned_PID_absent'] and terminal['owned_PID_no_CUDA_rows'] and terminal['direct_wait_observed']
 activation=P/plan['activation'];output_root=P/plan['scientific_output'];owner_root=P/plan['supervision_output']
 assert not activation.exists() and not output_root.exists() and not owner_root.exists()
 anchors=read(bound(plan['copied_anchor_custody']));assert anchors['complete'] and anchors['all_original12_closed'] and len(anchors['records'])==6
 for row in anchors['records']:bound(row['release']);bound(row['completion']);bound(row['actual_exit'])
 adoption=read(bound(plan['conditional_adoption']));assert adoption['whole_relation12_closed'] and adoption['frozen_primary_failed'] and adoption['frozen_placement_failed']
 activation.mkdir(mode=0o700);(activation/'cells').mkdir();(activation/'supervision').mkdir();output_root.mkdir(mode=0o700);owner_root.mkdir(mode=0o700)
 for seed in (6101,6203,6307):(output_root/('seed'+str(seed))).mkdir()
 adoption.update(root_adopted=True,copied_six_anchor_custody=plan['copied_anchor_custody'],qualification_custody=binding(args.qualification_custody))
 adoption_path=activation/'CONDITIONAL_ADOPTION.json';write(adoption_path,adoption)
 actual=copy.deepcopy(plan);actual.update(enabled=True,root_owner_execution_authorized=True,source_review_approved=True,qualification_custody=binding(args.qualification_custody),conditional_adoption=binding(adoption_path),admitted_disabled_plan_sha256=args.plan_sha256,scientific_fits_launched_by_renderer=0)
 for lane in actual['lanes'].values():
  gpu=lane['physical_gpu_uuid'];qualified=next(row['qualified'] for row in qualifications['records'] if row['physical_gpu_uuid']==gpu)
  new_cells=[]
  for cell in lane['cells']:
   preview=read(bound(cell));supervision=read(bound(preview['external_supervision']))
   supervision.update(enabled=True,root_owns_finite_launch_and_actual_costs=True)
   supervision_path=activation/'supervision'/(cell['cell_id']+'.json');write(supervision_path,supervision)
   preview.update(enabled=True,root_execution_authorized=True,source_review_approved=True,native_qualification_approved=True,native_qualification=qualified,conditional_adoption=binding(adoption_path),external_supervision=binding(supervision_path))
   release_path=activation/'cells'/(cell['cell_id']+'.json');write(release_path,preview)
   new_cells.append(dict(binding(release_path),cell_id=cell['cell_id']))
  lane['cells']=new_cells
 owner_plan=activation/'OWNER_PLAN.json';write(owner_plan,actual)
 print(json.dumps(dict(owner_plan=binding(owner_plan),scientific_children_launched=0,qualification_custody=binding(args.qualification_custody)),indent=2))


if __name__=='__main__':main()
