"""One read-only singleton closure observation; no predictive-value export."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ident(pid):
    try:
        fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(') ',1)[1].split()
        return {'PID':pid,'start_ticks':int(fields[19]),'state':fields[0],'ppid':int(fields[1])}
    except FileNotFoundError:return None


def observe(root,expected,pid,ticks):
    owner=ident(pid)
    if owner is not None and owner['start_ticks']!=ticks:raise ValueError('Original owner PID reused')
    complete=root/'owner/COMPLETE.json';failed=root/'owner/FAILURE.json'
    out={'owner':owner,'owner_complete_present':complete.is_file(),'owner_failure_present':failed.is_file(),'cells':{}}
    if complete.is_file():
        closed=json.loads(complete.read_text())
        out['owner_completion']={'path':str(complete.relative_to(PHASE)),'sha256':sha(complete),
            'completed_cell_ids':[x.get('cell_id') for x in closed.get('completed',[])],
            'fits':closed.get('fits'),'TEST_access':closed.get('TEST_access')}
    for cell in expected:
        folder=root/'runs'/cell;freeze=folder/'FREEZE.json'
        state={'freeze_present':freeze.is_file(),'failure_present':(folder/'FAILURE.json').is_file()}
        if freeze.is_file():state['freeze_sha256']=sha(freeze)
        progress=folder/'PROGRESS.json'
        if progress.is_file():
            row=json.loads(progress.read_text())
            state['progress']={k:row[k] for k in ('cycle','counters','condition','native_seed','complete_target') if k in row}
        out['cells'][cell]=state
    out['remaining_without_freeze']=[c for c,s in out['cells'].items() if not s['freeze_present']]
    return out


assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
os.chdir(REPO);assert Path.cwd().resolve()==REPO
init_conditions=('random_signs_all','tabm_first_normal','warm_identity','graph_covariance','feature_covariance','single','independent_warm4')
growth_conditions=('graph_growth','unfiltered_growth','unfiltered_top8_partition','capable_single_rank8','independent_graph_growth4')
original=PHASE/'citeseer_gnnm_initialization_pilot_execution_root_20261006_v2'
prior=original/'b0_fit_owner/BLOCK_FREEZE.json'
b0={'block_freeze_present':prior.is_file(),'cells':{}}
if prior.is_file():
    row=json.loads(prior.read_text())
    b0.update(block=row.get('block'),physical_postfits=row.get('physical_postfits'),selected_block_complete=row.get('selected_block_complete'),TEST_access=row.get('TEST_access'),block_freeze_sha256=sha(prior))
for condition in init_conditions:
    cell='b0_'+condition;folder=original/'runs'/cell
    b0['cells'][cell]={'freeze_present':(folder/'FREEZE.json').is_file(),'failure_present':(folder/'FAILURE.json').is_file()}
initialization=observe(PHASE/'citeseer_initialization_remaining_flat_execution_root_20261007_v1',
    ['b'+str(s)+'_'+c for s in (1,2) for c in init_conditions],495028,6013629570)
initialization['preserved_b0']=b0
initialization['full21_metadata_closed']=bool(initialization['owner_complete_present'] and not initialization['owner_failure_present'] and initialization['owner'] is None and not initialization['remaining_without_freeze'] and b0.get('selected_block_complete') is True and b0.get('physical_postfits')==7)
growth=observe(PHASE/'citeseer_growth_full15_flat_execution_root_20261007_v1',
    ['b'+str(s)+'_'+c for s in (0,1,2) for c in growth_conditions],495546,6013797508)
growth['full15_metadata_closed']=bool(growth['owner_complete_present'] and not growth['owner_failure_present'] and growth['owner'] is None and not growth['remaining_without_freeze'])
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'physical_gpu_uuid':GPU,
    'initialization':initialization,'growth':growth,'predictive_values_requested':False,'TEST_access':False,'jobs_launched_or_signalled':False}))
