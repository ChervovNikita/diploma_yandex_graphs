"""Run the unchanged synthetic numerical suite against native-reference v2."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
previous = PHASE / 'graph_view_source_numerical_execution_root_20261004_v1/run_cpu_checks.py'
spec = importlib.util.spec_from_file_location('native_reference_retained_cpu_transport', previous)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
client.HERE = HERE
client.SOURCE = PHASE / 'accuracy_first_native_gnnm_reference_source_preparation_20261004_v2'
client.MANIFEST = 'fbe1aec5eafb029fce9c9d42ebf9a5b94b98cfbc2ce9881502f0e3fc8e192ca6'
client.PROTOCOL = '9da2f5ebed5660d09b77ef4a620caa50bda81e4dc8ee999444aefc8f87e58524'
client.REMOTE = client.REMOTE.replace(
    'graph_view_source_numerical_execution_root_20261004_v1', HERE.name).replace(
    'accuracy_first_graph_view_source_preparation_20261004_v1', client.SOURCE.name)
client.REMOTE = client.REMOTE.replace(
    "PYTHONDONTWRITEBYTECODE='1')", "PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root))")

if __name__ == '__main__':
    client.main()
