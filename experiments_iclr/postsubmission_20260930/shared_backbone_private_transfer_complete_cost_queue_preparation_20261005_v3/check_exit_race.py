"""Scoped stdlib fault injection; no real process, signal, model or server."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cost_queue_exit_fixture', HERE / 'queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)  # This supervisor imports only standard library.

OWNER = {'PID': 123, 'start_ticks': 100, 'pgid': 123, 'sid': 123,
         'ppid': 99, 'state': 'R', 'argv': ['python', 'run.py']}
results = []


def stat(state='R', start=100, group=123):
    fields = ['0'] * 22
    fields[0:4] = [state, '99', str(group), str(group)]
    fields[19] = str(start)
    return '123 (worker) ' + ' '.join(fields)


def snapshot(stats, commands):
    stats, commands = iter(stats), iter(commands)
    class ScriptedPath:
        def __init__(self, value): self.value = str(value)
        def __truediv__(self, value): return ScriptedPath(self.value + '/' + str(value))
        def read_text(self): return next(stats)
        def read_bytes(self): return b'\0'.join(s.encode() for s in next(commands))
    with mock.patch.object(queue, 'Path', ScriptedPath), mock.patch.object(queue.time, 'sleep'):
        return queue.identity(123)


def checked(name, function):
    function()
    results.append({'case': name, 'passed': True})


def incomplete_live():
    row = snapshot([stat()] * 6, [[], [], []])
    assert row['start_ticks'] == 100 and row['argv'] == [] and row['observation_complete'] is False
    with mock.patch.object(queue, 'identity', return_value=row):
        assert queue.owned_tree(OWNER) == [row]


def confirmed_zombie():
    row = snapshot([stat(), stat('Z')], [[]])
    assert row['state'] == 'Z' and row['observation_complete'] is True
    with mock.patch.object(queue, 'identity', return_value=row):
        assert queue.owned_tree(OWNER) == []


def changed_start():
    try: snapshot([stat(), stat(start=101)], [['python', 'run.py']])
    except RuntimeError: return
    raise AssertionError('Changed start ticks were accepted')


def changed_group():
    try: snapshot([stat(), stat(group=124)], [['python', 'run.py']])
    except RuntimeError: return
    raise AssertionError('Changed group was accepted')


class FakeProcess:
    pid = 123
    def __init__(self, waits): self.waits, self.returncode = iter(waits), None
    def poll(self): return self.returncode
    def wait(self, timeout=None):
        action = next(self.waits)
        if isinstance(action, Exception): raise action
        self.returncode = action
        return action


def signal_refused(row):
    process = FakeProcess([])
    with mock.patch.object(queue, 'identity', return_value=row), mock.patch.object(queue.os, 'killpg') as kill:
        try: queue.kill_owned(process, OWNER, 'external_hard_wall_bound')
        except RuntimeError: pass
        else: raise AssertionError('Incomplete or changed identity authorized signal')
        kill.assert_not_called()


def changed_argv():
    row = snapshot([stat(), stat()], [['different', 'command']])
    with mock.patch.object(queue, 'identity', return_value=row):
        try: queue.owned_tree(OWNER)
        except RuntimeError: pass
        else: raise AssertionError('Changed nonempty argv accepted')
    signal_refused(row)


def empty_does_not_signal():
    signal_refused(snapshot([stat()] * 6, [[], [], []]))


def authoritative_terminal(code):
    process, signals = FakeProcess([code]), []
    with mock.patch.object(queue.os, 'killpg') as kill:
        reason, refusal, observed = queue.observe_terminal(process, OWNER, queue.time.monotonic(),
            {'external_hard_seconds_per_cell': 720}, signals, None, None)
        assert observed and process.returncode == code and reason is None and refusal is None and not signals
        kill.assert_not_called()


def incomplete_timeout_remains_unknown():
    row = snapshot([stat()] * 6, [[], [], []])
    process = FakeProcess([subprocess.TimeoutExpired('fake', .01)])
    signals = []
    with mock.patch.object(queue, 'identity', return_value=row), mock.patch.object(queue.os, 'killpg') as kill:
        reason, refusal, observed = queue.observe_terminal(process, OWNER, queue.time.monotonic() - 721,
            {'external_hard_seconds_per_cell': 720}, signals, None, None)
        assert reason == 'external_hard_wall_bound' and refusal.startswith('IncompleteExitObservation:')
        assert not observed and process.returncode is None and not signals
        kill.assert_not_called()


def strong_owned_cap_signal():
    row = {**OWNER, 'observation_complete': True}
    process, signals = FakeProcess([subprocess.TimeoutExpired('fake', .01), -9]), []
    with mock.patch.object(queue, 'identity', return_value=row), mock.patch.object(queue.os, 'killpg') as kill:
        reason, refusal, observed = queue.observe_terminal(process, OWNER, queue.time.monotonic() - 721,
            {'external_hard_seconds_per_cell': 720}, signals, None, None)
        assert reason == 'external_hard_wall_bound' and refusal is None and observed and process.returncode == -9
        kill.assert_called_once_with(123, queue.signal.SIGKILL)
        assert len(signals) == 1 and signals[0]['owned_group_only'] is True


for name, function in [('live_stat_empty_cmdline_remains_incomplete', incomplete_live),
                       ('torn_live_to_confirmed_zombie', confirmed_zombie),
                       ('start_tick_change_rejected', changed_start), ('group_change_rejected', changed_group),
                       ('nonempty_changed_argv_rejected', changed_argv), ('incomplete_never_signals', empty_does_not_signal),
                       ('terminal_zero_requires_wait', lambda: authoritative_terminal(0)),
                       ('terminal_nonzero_preserved', lambda: authoritative_terminal(3)),
                       ('incomplete_timeout_exit_unknown', incomplete_timeout_remains_unknown),
                       ('cap_signals_only_strong_owned_identity', strong_owned_cap_signal)]:
    checked(name, function)

record = {'scope': 'Mocked proc/terminal-wait fault injection for concrete exit race only',
          'queue_sha256': hashlib.sha256((HERE / 'queue.py').read_bytes()).hexdigest(),
          'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'cases': results, 'passed': True, 'real_processes_spawned': 0, 'real_signals_sent': 0,
          'server_access': False, 'numerical_imports_models_gradients_fits': False,
          'synthetic_exit_codes_are_unit_fixture_values_only': True}
with (HERE / 'FAULT_INJECTION_RESULT.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps({'passed': True, 'mocked_cases': len(results)}))
