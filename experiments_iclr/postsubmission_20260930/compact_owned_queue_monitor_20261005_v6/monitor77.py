"""Sixth bounded observation of existing owned queues; no outcome access."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('observer_v1', HERE.parent / 'compact_owned_queue_monitor_20261005_v1/monitor77.py')
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
observer.transport.HERE = HERE
with (HERE / 'REMOTE_CODE.py.txt').open('x') as handle:
    handle.write(observer.CODE)
result = observer.transport.run('compact_owned_77_queues_20261005_v6', observer.CODE)
with (HERE / 'OBSERVATION.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(dict(UTC=result['UTC'],
    collab_completed=result['collab']['progress']['completed_cells'],
    collab_fit=result['collab'].get('current_fit'),
    collab_queue_live=result['collab']['queue_handle'] is not None,
    collab_child_live=result['collab'].get('current_child') is not None,
    ddi_completed=len(result['ddi']['progress']['completed_cells']),
    ddi_current=result['ddi']['progress']['current_cell'],
    ddi_elapsed=result['ddi']['progress']['elapsed_cell_seconds'],
    ddi_queue_live=result['ddi']['queue_handle'] is not None,
    ddi_child_live=result['ddi'].get('current_child') is not None,
    metadata_only=True)))
