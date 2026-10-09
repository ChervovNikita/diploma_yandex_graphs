from pathlib import Path
import socket,subprocess,json
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
output=P/'geometry_only_core_scientific18_execution_root_20261009_v1'
activation=P/'geometry_only_core_scientific18_activation_root_20261009_v1'
assert not output.exists() and not (activation/'LAUNCH.json').exists()
print(json.dumps(dict(original18_output_absent=True,original18_launch_absent=True,scientific_fits_started=0,comparative_outcomes_opened=False,head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip())))
