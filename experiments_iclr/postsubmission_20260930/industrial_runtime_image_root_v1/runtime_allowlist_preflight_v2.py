"""Read dedicated interpreter/library metadata; no model/data import or GPU work."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

BASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')


def main():
    prefix = Path(sys.base_prefix).resolve()
    environment = BASE / 'envs/industrial_native_py312_v1'
    assert prefix.is_relative_to(BASE / 'envs/managed_python')
    assert Path(sys.prefix).resolve() == environment
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,minor_number', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    assert len(gpu) == 1 and gpu[0].startswith('GPU-44039938-fd82-41d2-fefd-de71514e2fac,')
    roots = [prefix / 'bin/python3.12']
    packages = environment / 'lib/python3.12/site-packages'
    roots += [packages / name for name in ['torch/lib/libtorch_global_deps.so',
              'torch/lib/libtorch_python.so', 'dgl/libdgl.so']]
    roots += sorted((prefix / 'lib/python3.12/lib-dynload').glob('*.so'))
    cache = subprocess.run([shutil.which('ldconfig') or '/sbin/ldconfig', '-p'],
                           capture_output=True, text=True, check=True).stdout
    for library in ['libcuda.so.1', 'libnvidia-ml.so.1', 'libnvidia-ptxjitcompiler.so.1']:
        matches = [line.split('=>')[-1].strip() for line in cache.splitlines()
                   if line.strip().startswith(library + ' ') and 'x86-64' in line]
        assert len(matches) == 1, (library, matches)
        roots.append(Path(matches[0]))
    records, libraries, unresolved = [], set(), []
    for path in roots:
        assert path.is_file(), path
        result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, timeout=8)
        records.append({'object': str(path), 'exit_code': result.returncode,
                        'stdout': result.stdout, 'stderr': result.stderr})
        for line in result.stdout.splitlines():
            if 'not found' in line:
                unresolved.append({'object': str(path), 'line': line.strip()})
            for name in re.findall(r'(?<!\S)(/(?:lib|lib64|usr/lib|usr/lib64|usr/local/nvidia/lib)[^\s()]+)', line):
                target = Path(name)
                libraries.update([str(target), str(target.resolve())])
        if not path.is_relative_to(BASE):
            libraries.update([str(path), str(path.resolve())])
    print(json.dumps({'python_version': sys.version.split()[0], 'python_prefix': str(prefix),
        'candidate_environment': str(environment), 'gpu_route': gpu[0],
        'system_library_inputs': sorted(libraries), 'ldd_records': records,
        'unresolved_dependencies': unresolved, 'project_free_bytes': shutil.disk_usage(BASE).free,
        'model_imported': False, 'dataset_accessed': False, 'GPU_compute_performed': False}))


if __name__ == '__main__':
    main()
