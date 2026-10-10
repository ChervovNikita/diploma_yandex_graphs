"""Prospective whole-scope cost recorder, available only after runtime authority."""
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import sys
import time
from .caps import CLOSED
from .native import PHASE, require


class Costs:
    def __init__(self, folder, torch, device, caps=CLOSED):
        caps.require('source_bound', 'runtime')
        self.folder = Path(folder).resolve(strict=True)
        require(self.folder.is_relative_to(PHASE) and self.folder.is_dir(), 'Existing root-owned project output only')
        self.torch, self.device, self.events = torch, device, []

    @contextmanager
    def measure(self, scope):
        torch = self.torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start, before = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
        row = dict(scope=scope, status='started')
        try:
            yield row
            row['status'] = 'complete'
        except BaseException as error:
            row.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            after = resource.getrusage(resource.RUSAGE_SELF)
            row.update(seconds=time.perf_counter()-start,
                       CPU_user_seconds=after.ru_utime-before.ru_utime,
                       CPU_system_seconds=after.ru_stime-before.ru_stime,
                       cumulative_process_RSS_peak_bytes=int(after.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)))
            self.events.append(row)
            with (self.folder/'COST_EVENTS.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
