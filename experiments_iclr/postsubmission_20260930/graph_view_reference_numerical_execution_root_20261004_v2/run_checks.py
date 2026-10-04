"""Run the preserved numerical suite against the repaired v2 reference bytes."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
previous = PHASE / 'graph_view_reference_numerical_execution_root_20261004_v1/run_checks.py'
spec = importlib.util.spec_from_file_location('ordinary_reference_retained_cpu_transport', previous)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
client.HERE = HERE
client.SOURCE = PHASE / 'accuracy_first_graph_view_reference_source_preparation_20261004_v2'
client.MANIFEST = '3a054568e54baed6ee6526256b6d37fa5f82b449257d7983c1ef54eace8236e0'
client.PROTOCOL = '05d13ba716857c70a399cf1b308dc9ed1d76596afbb72067d98cf3e97a107546'

if __name__ == '__main__':
    client.main()
