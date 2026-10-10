"""Invoke only the existing finite own_one for one whole-family readout."""
from pathlib import Path
import hashlib,importlib.util,json,socket,subprocess,sys
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=P/'private_hop_credit_pubmed_fullfit_source_20261010_v1'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
plan_path=H/'COMPARISON_OWNER_PLAN.json'
assert hashlib.sha256(plan_path.read_bytes()).hexdigest()==sys.argv[1]
plan=json.loads(plan_path.read_text());assert plan['enabled'] and not plan['automatic_retry']
assert plan['purpose']=='comparison' and len(plan['records'])==1
source=P/'pubmed_factor1_controls_source_20261010_v1/queue.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='2a6c9ac1f867fb947533083ef1f6bc21503a25c45a1d198d0bba0a325ef7fa70'
spec=importlib.util.spec_from_file_location('_existing_full18_comparison_owner',source)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner);owner.HERE=H
folder=H/'comparison';folder.mkdir(exist_ok=True)
good=owner.own_one(plan['records'][0],plan)
owner.write(folder/('FAMILY_COMPLETE.json' if good else 'FAMILY_FAILURE.json'),dict(complete=good,record=plan['records'][0]['record_id'],all_records_directly_waited=True,automatic_retry=False,source_manifest_sha256=plan['source_manifest_sha256']))
raise SystemExit(0 if good else 1)
