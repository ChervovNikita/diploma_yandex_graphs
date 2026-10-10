"""Prepare full-input independent references using unchanged qualified V3 inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, socket, subprocess

P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=Path(__file__).resolve().parent
S=P/'typed_context_individually_selected_native_references_source_20261010_v1'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert H==P/'typed_context_reference_qualification_root_20261010_v1'
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def bind(q):return dict(path=str(q),sha256=sha(q),bytes=q.stat().st_size)
def write(q,v):
    with q.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

assert sha(S/'MANIFEST.json')=='a9a12566097f799879f4efb66de4cf947f205762bd1c3af601397f89d30309f4'
assert sha(S/'SEAL.json')=='1c8087e4450bc8343f6a50497ecf7829505e0cf398b441614613d98cbcec108a'
base=json.loads((P/'typed_context_allocation_qualification_preparation_20261010_v2/RELEASE.json').read_text())
spec=json.loads((S/'RELEASE_TEMPLATE_DISABLED.json').read_text())
for key in ('cuda_device_uuid','development_files','device','expected_math_flags','expected_runtime_versions','input_root','native_qualification_receipt','python_executable','python_version','resource_budget','roles','runtime_environment','schema_receipt','seed_specs'):
    spec[key]=base[key]
spec.update(action='qualify_independently_selected_references',enabled=True,root_source_review_approved=True,data_scope_approved=True,provider_runtime_approved=True,capabilities=base['capabilities'],reference_protocol_sha256=sha(S/'PROTOCOL.json'),reference_source_seal_sha256=sha(S/'SEAL.json'),candidate_qualification_receipt=bind(P/'typed_context_allocation_qualification_execution_20261010_v2/COHORT_REPORT.json'),output_directory=str(P/'typed_context_reference_qualification_execution_20261010_v1'))
review=dict(schema='typed-context-reference-root-review-v1',approved=True,scientific_reference_fits_approved=False,method_source_modified=False,reference_protocol_sha256=sha(S/'PROTOCOL.json'),reference_source_seal_sha256=sha(S/'SEAL.json'),UTC=datetime.now(timezone.utc).isoformat(),checks=['fresh complete native constructors','four independent starts and selectors','width32 same-information reference','complement target exclusion','selected state replay and exact prediction checks','raw context/body pooling order','private scaler and dropout ownership'],review_type='author source review; not manuscript review')
write(H/'ROOT_REVIEW.json',review);spec['root_review']=bind(H/'ROOT_REVIEW.json')
write(H/'OWNER_REVIEW.json',dict(approved=True,entry=bind(H/'ENTRY.py'),owner=bind(H/'OWNED_QUALIFY.py'),method_unchanged=True,source_only=True,finite_bounds=dict(active_seconds=3600,cleanup_seconds=10,host_RSS_bytes=32*1024**3,GPU_bytes=24*1024**3,output_bytes=32*1024**3,log_bytes=8*1024**2),automatic_retry=False))
write(H/'RELEASE.json',spec)
print(json.dumps(dict(release=bind(H/'RELEASE.json'),qualification_only=True,scientific=False)))
