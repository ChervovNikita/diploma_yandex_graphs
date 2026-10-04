"""Run the unchanged read-only observer under its required server interpreter.

The first observer failed before reading runtime state: the relay interpreter
differs from the pinned numerical interpreter. No scientific run is relaunched.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
CLIENT = ROOT / 'root_client_preparation_20261004_v1'
sys.path.insert(0, str(CLIENT))
spec = importlib.util.spec_from_file_location('owned_observer_original_v1', CLIENT / 'monitor_owned_queue.py')
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
original_transport = observer.transport
PINNED_PYTHON = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'


def pinned_observation(identity, code):
    command = [PINNED_PYTHON, '-B', '-c', code]
    wrapper = 'import subprocess,json\n'
    wrapper += 'r=subprocess.run(' + repr(command) + ',capture_output=True,text=True,timeout=60)\n'
    wrapper += 'assert r.returncode==0,dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)\n'
    wrapper += 'print(r.stdout.strip())\n'
    return original_transport(identity + '_pinned_interpreter_v2', wrapper)


observer.transport = pinned_observation
if __name__ == '__main__':
    observer.main()
