"""One complete public-graph/B-label native resource check; no training fit."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
HELPER = PHASE / 'learnability_responsibility_native_synthetic_execution_root_20261005_v1/run_once.py'
SYNTHETIC = PHASE / 'learnability_responsibility_native_synthetic_execution_root_20261005_v2/RESULT.json'
SYNTHETIC_SHA = '95ba1b70251123294c593b7fbe4bb2775939e278e29c897d103ba6b2de8c4f9b'
ACCESSOR = PHASE / 'amazon_learnability_responsibility_engineering_accessor_20261005_v1'
ACCESSOR_MANIFEST = '69262fafa1a9d0f439e696f0e94d4c40b5489461417e5facfd573c03f7677f33'
ACCESSOR_SHA = '9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85'


def main():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--qualifier-manifest', required=True)
    parser.add_argument('--qualifier-sha', required=True)
    args = parser.parse_args()
    assert not (HERE / 'EXECUTION_RELEASE.json').exists()
    assert hashlib.sha256(SYNTHETIC.read_bytes()).hexdigest() == SYNTHETIC_SHA
    prior = json.loads(SYNTHETIC.read_text())
    assert prior['status'] == 'PASS_SYNTHETIC_ENGINEERING_ONLY'
    qualifier = PHASE / 'learnability_responsibility_native_numerical_worker_preparation_20261005_v3'
    assert hashlib.sha256((qualifier / 'qualify.py').read_bytes()).hexdigest() == args.qualifier_sha
    packets = {qualifier: args.qualifier_manifest, ACCESSOR: ACCESSOR_MANIFEST}
    files = []
    for packet, pin in packets.items():
        assert hashlib.sha256((packet / 'MANIFEST.json').read_bytes()).hexdigest() == pin
        for row in json.loads((packet / 'MANIFEST.json').read_text())['files']:
            data = (packet / row['path']).read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        for path in sorted(packet.iterdir()):
            assert path.is_file() and not path.is_symlink()
            data = path.read_bytes()
            files.append({'path': str(path.relative_to(PHASE)), 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest(),
                          'data': base64.b64encode(data).decode()})
    spec = importlib.util.spec_from_file_location('full_native_transport', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    remote = helper.REMOTE
    old = "worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'"
    new = "worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py'"
    assert remote.count(old) == 1
    remote = remote.replace(old, new)
    original_argv = "argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','synthetic','--source-root',str(phase),'--output',str(out/'output')]"
    preparation = r'''
synthetic=phase/'learnability_responsibility_native_synthetic_execution_root_20261005_v2/output/RESULT.json'
assert hashlib.sha256(synthetic.read_bytes()).hexdigest()==request['synthetic_sha256']
assert json.loads(synthetic.read_text())['status']=='PASS_SYNTHETIC_ENGINEERING_ONLY'
accessor=phase/'amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py'
assert hashlib.sha256(accessor.read_bytes()).hexdigest()==request['accessor_sha256']
memory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
(out/'GPU_MEMORY_BEFORE.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'record':memory})+'\n')
projection_code="""from pathlib import Path
import hashlib,importlib.util,json,sys
phase=Path(sys.argv[1]);source=Path(sys.argv[2]);destination=Path(sys.argv[3])
assert hashlib.sha256(source.read_bytes()).hexdigest()==sys.argv[4]
sys.path.insert(0,str(phase.parent.parent/'.venv/lib/python3.11/site-packages'))
spec=importlib.util.spec_from_file_location('engineering_train_custodian',source)
accessor=importlib.util.module_from_spec(spec);spec.loader.exec_module(accessor)
assert accessor.SOURCE_RELEASED is True
print(json.dumps(accessor.prepare_train_roles(phase,destination)))
"""
preparation_start=time.monotonic()
projection=subprocess.run([str(runtime),'-B','-c',projection_code,str(phase),str(accessor),str(out/'roles'),request['accessor_sha256']],cwd=repo,capture_output=True,text=True,timeout=120)
preparation_receipt={'UTC':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-preparation_start,'exit_code':projection.returncode,'stdout':projection.stdout,'stderr':projection.stderr,'custodian_joint_TRAIN_including_A_decoded':True,'predictive_metrics_or_fits':False}
(out/'PROJECTION_RECEIPT.json').write_text(json.dumps(preparation_receipt,indent=2)+'\n')
print(json.dumps({'stage':'projection','receipt':preparation_receipt}),flush=True)
assert projection.returncode==0,'Fixed TRAIN projection failed; no reseeding or fallback'
argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','full','--device','cuda:0','--source-root',str(phase),'--output',str(out/'output'),'--synthetic-pass',str(synthetic),'--accessor',str(accessor),'--accessor-sha',request['accessor_sha256'],'--public-b-dir',str(out/'roles/public_b')]
'''
    assert remote.count(original_argv) == 1
    remote = remote.replace(original_argv, preparation)
    release = {'UTC': datetime.now(timezone.utc).isoformat(),
               'purpose': 'One actual full-graph float32 public episode, recompute and resource check',
               'worker_sha256': args.qualifier_sha, 'synthetic_sha256': SYNTHETIC_SHA,
               'accessor_sha256': ACCESSOR_SHA,
               'remote_wrapper_sha256': hashlib.sha256(remote.encode()).hexdigest(),
               'original_coarse_failure_preserved': True,
               'full_graph_directional_fd': False,
               'fits_or_persistent_parameter_updates_or_A_scoring': False,
               'custodian_joint_TRAIN_label_decode_disclosed': True}
    (HERE / 'EXECUTION_RELEASE.json').write_text(json.dumps(release, indent=2) + '\n')
    request = {'files': files, 'packet_manifests': {p.name: h for p, h in packets.items()},
               'worker_sha256': args.qualifier_sha, 'run_name': HERE.name, 'release': release,
               'synthetic_sha256': SYNTHETIC_SHA, 'accessor_sha256': ACCESSOR_SHA}
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20', helper.LOGIN,
               'cd ' + shlex.quote(helper.REPO) + ' && exec /usr/bin/python3 -I -S -B -c ' + shlex.quote(remote)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    child.stdin.write(json.dumps(request))
    child.stdin.close()
    transcript = []
    for line in child.stdout:
        transcript.append(line)
        item = json.loads(line)
        if item['stage'] == 'projection':
            (HERE / 'PROJECTION_RECEIPT.json').write_text(json.dumps(item['receipt'], indent=2) + '\n')
            print(json.dumps({'stage': 'projection', 'exit_code': item['receipt']['exit_code']}), flush=True)
        elif item['stage'] == 'launched':
            (HERE / 'LAUNCH_RECEIPT.json').write_text(json.dumps(item['owned_process'], indent=2) + '\n')
            print(json.dumps({'stage': 'launched', 'PID': item['owned_process']['PID'],
                              'start_ticks': item['owned_process']['start_ticks']}), flush=True)
        else:
            (HERE / 'TERMINAL.json').write_text(json.dumps(item['terminal'], indent=2) + '\n')
            if item['result'] is not None:
                (HERE / 'RESULT.json').write_text(json.dumps(item['result'], indent=2) + '\n')
            print(json.dumps({'stage': 'terminal', 'exit_code': item['terminal']['exit_code'],
                              'status': item['result'].get('status') if item['result'] else None,
                              'error': item['result'].get('error') if item['result'] else None}), flush=True)
    stderr = child.stderr.read()
    exit_code = child.wait()
    (HERE / 'TRANSPORT_RECEIPT.json').write_text(json.dumps({
        'UTC': datetime.now(timezone.utc).isoformat(), 'exit_code': exit_code,
        'stdout_bytes': len(''.join(transcript).encode()),
        'stdout_sha256': hashlib.sha256(''.join(transcript).encode()).hexdigest(),
        'stderr': stderr}, indent=2) + '\n')
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
