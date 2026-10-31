"""Copy and hash only two frozen installed PyG source files; no dataset access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--supervisor', type=Path, required=True)
    args = parser.parse_args()
    phase = Path(__file__).resolve().parents[1]
    repo = phase.parents[1]
    request = args.request.resolve()
    supervisor = args.supervisor.resolve()
    assert request.is_relative_to(phase) and supervisor.is_relative_to(phase)
    bound = Path(os.environ['GNNM_BOUND_START_JSON']).resolve()
    assert bound.is_relative_to(phase)
    assert json.loads(bound.read_text())['root_request']['sha256'] == sha(request)
    assert (supervisor / 'environment.json').is_file()
    assert Path.cwd().resolve() == repo and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert uuid == 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    declaration = json.loads(request.read_text())
    assert declaration['operation'] == 'copy_two_installed_PyG_modules_only'
    assert declaration['data_or_model_execution_admitted'] is False
    output = phase / declaration['output_relative']
    assert output.resolve().is_relative_to(phase) and not output.exists()
    output.mkdir()
    records = []
    expected = {
        'datasets/coauthor.py': '3666bd68b6a90c4d52ee5082e3459acd065b049f27310577249651722f85dd74',
        'io/npz.py': 'cf84847e11ef7ee003b6b65b374b29ed545c804675efc63dc63d17b0aee6ac26',
    }
    for relative, digest in expected.items():
        path = repo / '.venv/lib/python3.11/site-packages/torch_geometric' / relative
        assert path.resolve().is_relative_to(repo / '.venv') and path.is_file()
        data = path.read_bytes()
        target = output / relative.replace('/', '_')
        target.write_bytes(data)
        records.append({'installed_path': str(path), 'snapshot': target.name,
                        'bytes': len(data), 'sha256': sha(target), 'expected_sha256': digest,
                        'matches_expected': sha(target) == digest})
    result = {'request_sha256': sha(request), 'operation': declaration['operation'],
              'gpu_uuid': uuid, 'CUDA_hidden': True, 'data_or_model_execution': False,
              'records': records, 'complete': all(x['matches_expected'] for x in records)}
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['complete'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
