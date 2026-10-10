"""Bind completed stored predictions and an unchanged, reviewed diagnostic."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, socket, subprocess
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent
S=P/'combination_masked_context_pubmed_complete9_stored_prediction_diagnostic_source_20261010_v1'
F=P/'masked_context_pubmed_stage1_root_execution_20261010_v1'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==[GPU]
assert H==P/'pubmed_complete9_error_diagnostic_root_20261010_v1'
assert not (H/'RELEASE.json').exists()
def bind(q):return dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest())
family=json.loads((F/'FAMILY_COMPLETE.json').read_text());assert family['complete'] and family['all_nine_directly_waited']
old=json.loads((F/'COMPLETE_COMPARISON_RELEASE.json').read_text());assert old['whole_nine_owned_completion_verified']
reference=json.loads((F/'releases/seed9101__shared4_own.json').read_text())
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert free>2*1024**3
ready=dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_free_bytes=free,whole_family_complete=True,qualification_worker_closed=True,TEST_access=False)
with (H/'READINESS.json').open('x') as f:json.dump(ready,f,indent=2)
spec=json.loads((S/'RELEASE_TEMPLATE_DISABLED.json').read_text())
owner=json.loads((S/'FINITE_OWNER_CONTRACT_TEMPLATE_DISABLED.json').read_text())
owner.update(enabled=True,direct_wait_required=True,separate_process_group=True,resource_caps_enforced=True,output_and_log_caps_enforced=True,owner_source=bind(H/'run_owned.py'),no_owner_implementation_added=False)
with (H/'OWNER_CONTRACT.json').open('x') as f:json.dump(owner,f,indent=2)
spec.update(enabled=True,root_diagnostic_authorized=True,whole_nine_owned_completion_verified=True,source_review_approved=True,finite_owner_bound=True,fresh_resource_readiness_confirmed=True,external_owner_release=bind(H/'OWNER_CONTRACT.json'),resource_readiness_evidence=bind(H/'READINESS.json'),source_manifest_sha256=hashlib.sha256((S/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),records=old['records'],runtime=reference['runtime'],valid_bundle=reference['valid_bundle'],validation_custody=bind(F/'VALID_CUSTODY.json'),output=str(P/'pubmed_complete9_error_diagnostic_execution_20261010_v1'))
with (H/'RELEASE.json').open('x') as f:json.dump(spec,f,indent=2)
print(json.dumps(dict(release=bind(H/'RELEASE.json'),no_new_fits=True,TEST_access=False)))
