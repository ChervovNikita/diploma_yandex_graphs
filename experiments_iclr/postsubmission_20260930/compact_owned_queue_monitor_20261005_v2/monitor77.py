"""Successor observation; reuse the exact metadata-only observer operation."""
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
result = observer.transport.run('compact_owned_77_queues_20261005_v2', observer.CODE)
with (HERE / 'OBSERVATION.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(dict(UTC=result['UTC'],collab=result['collab'],ddi=result['ddi'],metadata_only=True)))
