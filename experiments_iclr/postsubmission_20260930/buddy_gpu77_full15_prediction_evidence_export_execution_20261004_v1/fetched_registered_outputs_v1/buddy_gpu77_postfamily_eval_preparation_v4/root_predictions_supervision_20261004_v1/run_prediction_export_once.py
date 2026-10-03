
from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,hashlib,subprocess,time,resource
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
HERE=REPO/'experiments_iclr/postsubmission_20260930/buddy_gpu77_postfamily_eval_preparation_v4'
OUT=HERE/'root_predictions_supervision_20261004_v1'
ADMISSION=HERE/'ROOT_PREDICTION_EXPORT_ADMISSION.json'
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python'
EXPECTED_ADMISSION_SHA='3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
def identity(pid):
 text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
             argv=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x],cwd=os.readlink(f'/proc/{pid}/cwd'))
def save(name,value):
 with (OUT/name).open('x') as h:json.dump(value,h,indent=2);h.write('\n')
assert Path.cwd().resolve()==REPO and OUT.resolve().is_relative_to(REPO)
assert hashlib.sha256(ADMISSION.read_bytes()).hexdigest()==EXPECTED_ADMISSION_SHA
assert not (HERE/'root_predictions_v1').exists()
start=time.monotonic();began=datetime.now(timezone.utc).isoformat()
save('RUNNER_STARTED.json',dict(schema='buddy77_owned_prediction_export_runner_started_v1',UTC=began,identity=identity(os.getpid()),
                              ROOT_PREDICTION_EXPORT_ADMISSION_sha256=EXPECTED_ADMISSION_SHA,stage='export',export_once=True,official_TEST_access_authorized=True))
argv=[PYTHON,'-B',str(HERE/'export_predictions77.py'),'--execute','--admission',str(ADMISSION),
      '--gpu','GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998']
with (OUT/'CHILD.stdout.log').open('x') as stdout,(OUT/'CHILD.stderr.log').open('x') as stderr:
 child=subprocess.Popen(argv,cwd=REPO,stdout=stdout,stderr=stderr,start_new_session=True)
 save('CHILD_STARTED.json',dict(schema='buddy77_owned_prediction_export_child_started_v1',UTC=datetime.now(timezone.utc).isoformat(),
      identity=identity(child.pid),argv=argv,export_once=True,official_TEST_access_authorized=True,no_retry=True))
 code=child.wait()
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
save('PHYSICAL_TERMINAL.json',dict(schema='buddy77_owned_prediction_export_physical_terminal_v1',start_UTC=began,
 terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=code,argv=argv,
 physical_wrapper_wall_seconds=time.monotonic()-start,owned_descendant_user_seconds=usage.ru_utime,
 owned_descendant_system_seconds=usage.ru_stime,owned_descendant_peak_RSS_bytes=usage.ru_maxrss*1024,
 peak_RSS_scope='Linux wait4/RUSAGE_CHILDREN maximum for the owned export descendants; current Torch preflight is included.',
 overlapping_internal_export_intervals_not_added=True,terminal_write_tail_measured=False,
 ROOT_PREDICTION_EXPORT_ADMISSION_sha256=EXPECTED_ADMISSION_SHA,export_once=True,no_retry=True,official_TEST_access_authorized=True,
 other_jobs_stopped=False,namespace_isolation_used=False))
