"""One bounded CPU public converter/full ordered audited-role equality check."""
import hashlib,json,os,resource,socket,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PHASE=HERE.parent;REPO=PHASE.parents[1]
PUBLIC=PHASE/'portable_internal_be_public_interface_20261007_v2'
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1048576),b''):digest.update(block)
    return digest.hexdigest()
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
started=time.monotonic();stage='startup';os.umask(0o077)
try:
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).strip().splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert Path.cwd()==REPO and Path(sys.executable).resolve()==PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    config=json.loads((HERE/'TASK.json').read_text());task=config['task']
    assert sha(PUBLIC/'MANIFEST.json')==config['public_source_manifest_sha256']
    for row in json.loads((PUBLIC/'MANIFEST.json').read_text())['files']:
        path=PUBLIC/row['path'];assert path.is_relative_to(PUBLIC) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    review_path=PHASE/config['author_review']['path'];assert sha(review_path)==config['author_review']['sha256']
    review=json.loads(review_path.read_text());assert review['approved'] and review['task']==task
    binding=review['data_manifest'];authority_path=PHASE/binding['path'];assert sha(authority_path)==binding['sha256']
    authority=json.loads(authority_path.read_text());assert authority['task']==task
    raw=PHASE/config['raw_path'];assert raw.is_file()
    sys.path.insert(0,str(PUBLIC))
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from data_interface import _data
    assert not torch.cuda.is_initialized()
    stage='public_conversion'
    with (HERE/'export.log').open('x') as log:
        result=subprocess.run([str(PYTHON),'-B',str(PUBLIC/'export.py'),task,config['raw_flag'],str(raw),'--output',str(HERE/'roles')],cwd=REPO,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Public converter exit '+str(result.returncode))
    conversion=json.loads((HERE/'roles/DATA.json').read_text());assert conversion['complete'] and conversion['source_bytes_matched_public_pins'] and not conversion['TEST_scoring']
    stage='complete_ordered_audited_author_role_equality';fields={}
    for role in ('train','valid'):
        reference=(PHASE/authority['payloads'][role]['path']).resolve();assert reference.is_relative_to(PHASE) and sha(reference)==authority['payloads'][role]['sha256']
        keys=_data().ROLE_KEYS[task][role]
        original=_data().load_npz(reference,keys);converted=_data().load_npz(HERE/'roles'/(role+'.npz'),keys);fields[role]={}
        for key in keys:
            a,b=original[key],converted[key]
            assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),role+'/'+key
            fields[role][key]={'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.numel()*a.element_size(),'complete_ordered_equal':True,'raw_tensor_sha256':hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()}
    assert not torch.cuda.is_initialized()
    usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt={'schema':'internal-BE-public-task-CPU-conversion-equality-v1','task':task,'complete':True,
             'source_manifest_sha256':config['public_source_manifest_sha256'],'author_role_manifest':binding,
             'author_review':config['author_review'],'public_conversion_source_bytes_matched':True,
             'complete_ordered_role_equality':fields,'public_roles':conversion['roles'],
             'conversion_origin':conversion['origin'],'inclusive_worker_seconds':time.monotonic()-started,
             'worker_CPU_user_seconds':usage.ru_utime,'worker_CPU_system_seconds':usage.ru_stime,
             'export_CPU_user_seconds':children.ru_utime,'export_CPU_system_seconds':children.ru_stime,
             'peak_worker_RSS_bytes':usage.ru_maxrss*1024,'peak_export_RSS_bytes':children.ru_maxrss*1024,
             'output_storage_bytes':sum(path.stat().st_size for path in (HERE/'roles').iterdir() if path.is_file()),
             'cuda_initialized':False,'models_constructed':0,'scientific_fits':0,'TEST_scoring':False,
             'automatic_retry':False,'GPU_or_other_jobs_modified':False,'binaries_transferred':False,
             'task_GPU_path_qualified':False}
    write('CPU_CHECK.json',receipt)
    print(json.dumps({'task':task,'complete':True,'full_ordered_equality':True,'models_or_GPU_work':False}))
except BaseException as error:
    write('FAILURE.json',{'complete':False,'stage':stage,'error_type':type(error).__name__,'error':str(error),
                        'seconds':time.monotonic()-started,'automatic_retry':False,'TEST_scoring':False})
    raise
