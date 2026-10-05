"""Exact reviewed v3 ownership/terminal helpers; GPU is set to root-selected physical UUID."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time
GPU=None

class IncompleteExitObservation(RuntimeError):
    """Same recorded process/group, but empty cmdline cannot authorize signals."""

def now():
    return datetime.now(timezone.utc).isoformat()

def identity(pid):
    directory = Path('/proc') / str(pid)
    def stat():
        raw = (directory / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'PID': pid, 'start_ticks': int(fields[19]), 'state': fields[0],
                'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'sid': int(fields[3])}
    # cmdline becomes empty during exit. A second stat observation identifies
    # confirmed zombies, while start/group changes remain identity failures.
    for attempt in range(3):
        try:
            before = stat()
            argv = [s.decode() for s in (directory / 'cmdline').read_bytes().split(b'\0') if s]
            after = stat()
        except FileNotFoundError:
            return None
        if any(before[key] != after[key] for key in ('start_ticks', 'pgid', 'sid')):
            raise RuntimeError('Process identity changed during observation; no signal authorized')
        after['argv'] = argv
        after['observation_complete'] = bool(argv or after['state'] == 'Z')
        if after['state'] == 'Z' or argv or attempt == 2:
            return after
        time.sleep(.01)

def owned_tree(owner):
    leader = identity(owner['PID'])
    if leader is None or leader['state'] == 'Z':
        return []
    if any(leader[key] != owner[key] for key in ('start_ticks', 'pgid', 'sid')):
        raise RuntimeError('Owned child identity changed; no signal authorized')
    if not leader['observation_complete']:
        return [leader]  # Explicit incomplete exit observation, never signal identity.
    if leader['argv'] != owner['argv']:
        raise RuntimeError('Owned child argv changed; no signal authorized')
    rows, pending = [], [leader]
    while pending:
        row = pending.pop()
        rows.append(row)
        children = Path('/proc') / str(row['PID']) / 'task' / str(row['PID']) / 'children'
        try:
            pids = children.read_text().split()
        except FileNotFoundError:
            continue
        for token in pids:
            child = identity(int(token))
            if child is None:
                continue
            if child['ppid'] != row['PID'] or child['pgid'] != owner['PID'] or child['sid'] != owner['PID']:
                raise RuntimeError('Owned descendant escaped declared child group')
            pending.append(child)
    return rows

def kill_owned(process, owner, reason):
    live = identity(owner['PID'])
    if live is None or live['state'] == 'Z' or process.poll() is not None:
        return None
    if any(live[key] != owner[key] for key in ('start_ticks', 'pgid', 'sid')):
        raise RuntimeError('Identity changed; refusing group termination')
    if not live['observation_complete'] or not live['argv']:
        raise IncompleteExitObservation('Empty cmdline exit observation; no signal authorized')
    if live['argv'] != owner['argv']:
        raise RuntimeError('Argv changed; refusing group termination')
    if live['pgid'] != live['sid'] or live['pgid'] != process.pid:
        raise RuntimeError('Fresh scientific child session no longer matches')
    os.killpg(process.pid, signal.SIGKILL)
    return {'UTC': now(), 'PID': process.pid, 'start_ticks': owner['start_ticks'],
            'signal': 'SIGKILL', 'reason': reason, 'owned_group_only': True}

def observe_terminal(process, owner, started, limits, signals, reason, signal_refusal):
    """Only the direct Popen wait/poll can provide this child's exit status."""
    observed = False
    remaining = max(.01, limits['external_hard_seconds_per_cell'] - (time.monotonic() - started))
    try:
        process.wait(timeout=remaining)
        observed = True
    except subprocess.TimeoutExpired:
        reason = 'external_hard_wall_bound'
        try:
            sent = kill_owned(process, owner, reason) if owner is not None else None
            if sent: signals.append(sent)
        except Exception as refusal:
            signal_refusal = type(refusal).__name__ + ': ' + str(refusal)
        if process.poll() is not None:
            observed = True
        elif signals:
            try:
                process.wait(timeout=5)
                observed = True
            except subprocess.TimeoutExpired:
                pass
    return reason, signal_refusal, observed

def query(arguments, timeout):
    return subprocess.check_output(['nvidia-smi', *arguments], text=True,
                                   timeout=timeout).strip().splitlines()

def resources(rows, limits, remaining):
    rss = 0
    for row in rows:
        try:
            status = (Path('/proc') / str(row['PID']) / 'status').read_text()
        except FileNotFoundError:
            continue
        rss += sum(int(line.split()[1]) * 1024 for line in status.splitlines()
                   if line.startswith('VmRSS:'))
    pids = {row['PID'] for row in rows}
    gpu = {}
    timeout = max(.05, min(limits['telemetry_timeout_seconds'], remaining))
    for line in query(['--query-compute-apps=gpu_uuid,pid,used_memory',
                       '--format=csv,noheader,nounits'], timeout):
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 3 and parts[0] == GPU and parts[1].isdigit() and int(parts[1]) in pids:
            if not parts[2].isdigit():
                raise RuntimeError('Owned GPU memory telemetry unavailable')
            gpu[int(parts[1])] = int(parts[2]) * 1024 ** 2
    return rss, sum(gpu.values())
