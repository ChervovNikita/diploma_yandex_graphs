"""UNEXECUTED trusted outer preflight and CPU native-import supervisor."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

bundle = json.loads(sys.argv[1])
run_id = bundle['run_id']
if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', run_id):
    raise ValueError('Fresh simple identity required')
for name, text in bundle['files'].items():
    if name not in ('proven_cpu_policy.py','native_import_worker.py','root_outer.py','IMPORT_CONTRACT.json'):
        raise ValueError('Unexpected transmitted source name')
    if hashlib.sha256(text.encode()).hexdigest() != bundle['pins'][name]:
        raise ValueError('Transmitted source pin differs')
contract = json.loads(bundle['files']['IMPORT_CONTRACT.json'])
if contract.get('public_device_read_grant') != {'path': '/dev/urandom', 'object_type': 'character_device', 'major': 1, 'minor': 9, 'allowed_access_fs': 4, 'read_file_only': True, 'write_file_granted': False, 'inherited_entropy_FD': False}:
    raise ValueError('Only the fixed read-only OS entropy device is admitted')
phase = Path(contract['phase'])
if phase.resolve(strict=True) != phase:
    raise ValueError('Actual fixed project phase required')
inventory_argv = ['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits']
inventory = subprocess.run(inventory_argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                           timeout=10,check=False,env={'PATH':'/usr/bin'})
uuids = [line.strip() for line in inventory.stdout.splitlines() if line.strip()]
authorized = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
if inventory.returncode != 0 or uuids != [authorized]:
    raise RuntimeError('Inventory identity mismatch before fixture writes')
root = phase / contract['run_container'] / run_id
cursor = phase
for part in root.relative_to(phase).parts:
    cursor /= part
    if cursor.is_symlink():
        raise ValueError('Symlink in new run destination')
if root.exists():
    raise ValueError('Single-use fresh run required')
root.parent.mkdir(parents=True, exist_ok=True)
root.mkdir(mode=0o700)
for name in ('trusted_source','outputs','cache','tmp','tmp/home','excluded_fixture'):
    (root / name).mkdir(mode=0o700)
(root / 'excluded_fixture/blocked.txt').write_bytes(b'excluded synthetic fixture\n')
for name, text in bundle['files'].items():
    (root / 'trusted_source' / name).write_text(text)
host = {name:os.readlink('/proc/self/ns/'+name) for name in ('user','mnt','pid','net','ipc','uts')}
mountinfo = Path('/proc/self/mountinfo').read_text()
receipt = {'schema':'root-CPU-native-import-qualification-v6',
           'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'run_root':str(root),'source_pins':bundle['pins'],'wrapper_sha256':bundle['wrapper_sha256'],
           'host_namespaces':host,'host_mountinfo_before':mountinfo,'stages':[],
           'inventory_preflight':{'argv':inventory_argv,'uuids':uuids,'exit_code':inventory.returncode,
                                  'before_fixture_writes':True,'inventory_only':True},
           'dataset_loading_requested':False,'model_construction_or_fit_requested':False,
           'GPU_compute_requested':False,'automatic_retry':False,
           'explicit_policy_difference_from_v5':True,
           'public_device_read_grant':contract['public_device_read_grant']}
started = time.monotonic()
deadline = started + contract['preflight_cap_seconds']
child = None
def outer_timeout(signum, frame):
    raise TimeoutError('Whole trusted outer/worker diagnostic wall cap exceeded')
signal.signal(signal.SIGALRM, outer_timeout)
signal.alarm(contract['preflight_cap_seconds'] + contract['worker_cap_seconds'] + 10)
def save():
    (root / 'NATIVE_IMPORT_CAPABILITY.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
def stage(name, **values):
    receipt['stages'].append({'name':name,'seconds':time.monotonic()-started,**values})
    save()
def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('Trusted metadata/library integrity preflight cap exceeded')
            block = stream.read(8*1024*1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()
def actual_directory(path):
    path = Path(path)
    if not path.is_dir() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError('Actual dedicated readonly directory required: '+str(path))
    return path
save()
try:
    image = actual_directory(contract['runtime_image'])
    image_manifest = Path(contract['runtime_manifest'])
    if hash_file(image_manifest) != contract['runtime_manifest_sha256']:
        raise RuntimeError('Runtime manifest pin differs')
    image_metadata = json.loads(image_manifest.read_text())
    if image_metadata['schema'] != 'candidate-allowlist-runtime-image-v2':
        raise RuntimeError('Runtime manifest schema differs')
    readonly_dirs = [actual_directory(p) for p in contract['image_readonly_directories']]
    image_records = {}
    for item in image_metadata['payload']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or item['path'] in image_records:
            raise RuntimeError('Runtime manifest relative path differs')
        path = image / relative
        if any(path == p or p in path.parents for p in readonly_dirs):
            image_records[item['path']] = item
    actual_names = set()
    regular_count = symlink_count = 0
    for prefix in readonly_dirs:
        for path in prefix.rglob('*'):
            if time.monotonic() > deadline:
                raise TimeoutError('Image inventory preflight cap exceeded')
            if path.is_symlink():
                name = path.relative_to(image).as_posix(); actual_names.add(name)
                item = image_records.get(name)
                if item is None or item.get('symlink') != os.readlink(path):
                    raise RuntimeError('Image symlink differs: '+name)
                resolved = path.resolve(strict=True)
                if not any(resolved == p or p in resolved.parents for p in readonly_dirs):
                    raise RuntimeError('Allowed library symlink leaves readonly image roots: '+name)
                symlink_count += 1
            elif path.is_file():
                name = path.relative_to(image).as_posix(); actual_names.add(name)
                item = image_records.get(name)
                if item is None or item.get('bytes') != path.stat().st_size or item.get('sha256') != hash_file(path):
                    raise RuntimeError('Image library bytes differ: '+name)
                regular_count += 1
            elif not path.is_dir():
                raise RuntimeError('Special file in readonly runtime')
    if actual_names != set(image_records):
        raise RuntimeError('Missing/undeclared image files in exposed roots')
    host_files = []
    for item in contract['host_readonly_library_files']:
        path = Path(item['path']).resolve(strict=True)
        if not path.is_file() or path.stat().st_size != item['bytes'] or hash_file(path) != item['sha256']:
            raise RuntimeError('Explicit host loader/library pin differs: '+str(path))
        host_files.append(str(path))
    stage('EXPOSED_LIBRARY_BYTES_VERIFIED', image_manifest_sha256=contract['runtime_manifest_sha256'],
          readonly_image_roots=[str(p) for p in readonly_dirs], regular_files=regular_count,
          symlinks=symlink_count, individually_pinned_host_library_files=host_files)
    source = actual_directory(contract['native_source'])
    if hash_file(source/'MANIFEST.json') != contract['native_manifest_sha256'] or hash_file(source/'SEAL.json') != contract['native_seal_sha256']:
        raise RuntimeError('Native source manifest/seal pin differs')
    source_manifest = json.loads((source/'MANIFEST.json').read_text())
    expected = {item['path'] for item in source_manifest['payload']} | {'MANIFEST.json','SEAL.json'}
    actual = set()
    for path in source.rglob('*'):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise RuntimeError('Native source contains symlink/special file')
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    if actual != expected:
        raise RuntimeError('Native source has missing/undeclared files')
    for item in source_manifest['payload']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Native source relative path differs')
        path = source/rel
        if path.stat().st_size != item['bytes'] or hash_file(path) != item['sha256']:
            raise RuntimeError('Native source byte pin differs')
    readonly_dirs.append(source)
    stage('SEALED_NATIVE_SOURCE_VERIFIED',manifest_sha256=contract['native_manifest_sha256'],seal_sha256=contract['native_seal_sha256'])
    public = actual_directory(contract['public_inputs'])
    actual = set()
    for path in public.rglob('*'):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise RuntimeError('Public bundle contains symlink/special file')
        if path.is_file():
            actual.add(path.relative_to(public).as_posix())
    if actual != set(contract['public_files']):
        raise RuntimeError('Public-only file names differ; labels/raw input excluded')
    role = public/contract['public_manifest_name']
    role_sha = hash_file(role)
    metadata = json.loads(role.read_text())
    if (metadata.get('schema') != 'tolokers2-public-role-freeze-v1'
        or metadata.get('targets_member_opened') is not False or metadata.get('test_labels') != 'CLOSED'
        or metadata.get('source_archive_sha256') != contract['source_archive_sha256']
        or metadata.get('role_policy_sha256') != contract['role_policy_sha256']):
        raise RuntimeError('Public-only manifest metadata differs')
    public_descriptors = {contract['public_manifest_name']:{'sha256':role_sha,'bytes':role.stat().st_size}}
    for name in contract['public_files']:
        if name == contract['public_manifest_name']:
            continue
        item = metadata['public_members'][Path(name).name]
        if (public/name).stat().st_size != item['bytes'] or not re.fullmatch('[0-9a-f]{64}',item['sha256']):
            raise RuntimeError('Declared public member metadata differs')
        public_descriptors[name] = {'sha256':item['sha256'],'bytes':item['bytes'],'member_contents_hashed_or_decoded':False}
    stage('PUBLIC_MANIFEST_METADATA_BOUND_NO_MEMBER_LOADING', public_manifest_sha256=role_sha,
          public_descriptors=public_descriptors, public_member_contents_read=False)
    readonly_files = [str(image_manifest),*host_files,*[str(public/name) for name in contract['public_files']]]
    bindings = {'schema':'CPU-native-import-run-bindings-v1',
                'runtime_manifest_sha256':contract['runtime_manifest_sha256'],
                'native_manifest_sha256':contract['native_manifest_sha256'],'native_seal_sha256':contract['native_seal_sha256'],
                'public_manifest_sha256':role_sha,'public_descriptors':public_descriptors,
                'public_device_read_grant':contract['public_device_read_grant'],
                'readonly_directories':[str(p) for p in readonly_dirs],'readonly_files':readonly_files,
                'directory_listing_only':[str(public),str(public/'tolokers-2')],
                'writable_directories':[str(root/name) for name in ('outputs','cache','tmp')]}
    binding_path = root/'RUN_BINDINGS.json'
    binding_path.write_text(json.dumps(bindings,indent=2,allow_nan=False)+'\n')
    binding_sha = hash_file(binding_path)
    argv = ['/usr/bin/unshare','--user','--map-root-user','--mount','--propagation','unchanged',
            '--pid','--fork','--net','--ipc','--uts','--kill-child',contract['python_executable'],
            '-I','-S','-B',str(root/'trusted_source/native_import_worker.py'),str(root),
            json.dumps(host),json.dumps(bundle['pins']),binding_sha]
    environment = {'PATH':'/usr/bin','HOME':str(root/'tmp/home'),'TMPDIR':str(root/'tmp'),
                   'XDG_CACHE_HOME':str(root/'cache'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
                   'LD_LIBRARY_PATH':':'.join(contract['ld_library_path']),'DGLBACKEND':'pytorch',
                   'DGL_DOWNLOAD_DIR':str(root/'cache/dgl'),'TORCH_HOME':str(root/'cache/torch'),
                   'CUDA_VISIBLE_DEVICES':'','CUDA_CACHE_PATH':str(root/'cache/cuda'),
                   'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    receipt.update(argv=argv,run_bindings=bindings,run_bindings_sha256=binding_sha,
                   preflight_cap_seconds=contract['preflight_cap_seconds'],worker_cap_seconds=contract['worker_cap_seconds'])
    stdin_fd = os.open('/dev/null',os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        value = os.fstat(stdin_fd); flags = fcntl.fcntl(stdin_fd,fcntl.F_GETFL)
        if not stat.S_ISCHR(value.st_mode) or value.st_rdev != os.makedev(1,3) or flags & os.O_ACCMODE != os.O_RDONLY:
            raise RuntimeError('Explicit readonly /dev/null stdin preflight differs')
        child = subprocess.Popen(argv,stdin=stdin_fd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,close_fds=True,start_new_session=True,env=environment,cwd=root)
    finally:
        os.close(stdin_fd)
    stage('RESTRICTING_WORKER_STARTED',child_pid=child.pid,stdin_readonly_verified=True)
    try:
        stdout,stderr = child.communicate(timeout=contract['worker_cap_seconds']); receipt['timed_out']=False
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        stdout,stderr = child.communicate(); receipt['timed_out']=True
    (root/'worker.stdout').write_text(stdout); (root/'worker.stderr').write_text(stderr)
    receipt.update(exit_code=child.returncode,stdout=stdout,stderr=stderr)
    try: receipt['worker_result']=json.loads(stdout)
    except ValueError: receipt['worker_result']=None
    passed = child.returncode == 0 and not receipt['timed_out'] and isinstance(receipt['worker_result'],dict) and receipt['worker_result'].get('status') == 'PASSED_CPU_NATIVE_IMPORTS_ONLY'
    receipt['status'] = 'PASSED_CPU_NATIVE_IMPORT_QUALIFICATION_ONLY' if passed else 'REFUSED_OR_FAILED_QUALIFICATION'
except BaseException as error:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGKILL)
        stdout,stderr = child.communicate(timeout=10)
        (root/'worker.stdout').write_text(stdout); (root/'worker.stderr').write_text(stderr)
        receipt.update(exit_code=child.returncode,stdout=stdout,stderr=stderr,worker_terminated_on_outer_error=True)
    receipt.update(status='REFUSED_OR_FAILED_QUALIFICATION',error_type=type(error).__name__,error=str(error))
signal.alarm(0)
receipt['host_namespaces_after'] = {name:os.readlink('/proc/self/ns/'+name) for name in host}
receipt['host_mountinfo_after'] = Path('/proc/self/mountinfo').read_text()
receipt['host_namespaces_and_mountinfo_unchanged'] = receipt['host_namespaces_after'] == host and receipt['host_mountinfo_after'] == mountinfo
if not receipt['host_namespaces_and_mountinfo_unchanged']:
    receipt['status'] = 'REFUSED_OR_FAILED_QUALIFICATION'
receipt['seconds'] = time.monotonic()-started
save()
print(json.dumps(receipt,sort_keys=True,allow_nan=False))
raise SystemExit(0 if receipt['status'] == 'PASSED_CPU_NATIVE_IMPORT_QUALIFICATION_ONLY' else 1)
