"""Launch a queue in tmux, inspect durable status, or request graceful stop."""
import argparse
import shlex
import subprocess
from common import *


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('start', 'status', 'stop'))
    p.add_argument('--worker', choices=WORKERS, required=True)
    p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True)
    p.add_argument('--session'); p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    config, _ = environment(a.worker)
    require(pin(a.plan)['sha256'] == a.plan_sha256, 'Plan hash differs')
    plan = read(a.plan); require(plan['worker'] == a.worker, 'Wrong plan worker')
    root = Path(plan['state_root']); root.mkdir(parents=True, exist_ok=True)
    if a.command == 'status':
        print(json.dumps(read(root / 'STATUS.json') if (root / 'STATUS.json').exists() else {'phase': 'not_started'}, indent=2))
        return
    if a.command == 'stop':
        write(root / 'STOP', {'requested': time.time(), 'mode': 'finish current children, stop new launches'})
        print(json.dumps({'graceful_stop': str(root / 'STOP')})); return
    source_manifest()
    session = a.session or 'gnnm-allfit-' + a.worker.replace('/', '-') + '-20261016'
    require(all(char.isalnum() or char in '-_' for char in session), 'Simple tmux session name required')
    exists = subprocess.run(['tmux', 'has-session', '-t', session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists:
        panes = subprocess.check_output(['tmux', 'list-panes', '-t', session, '-F', '#{pane_dead}'], text=True).splitlines()
        require(panes and all(value == '1' for value in panes), 'Session already has a live pane; inspect its durable STATUS')
        subprocess.check_call(['tmux', 'kill-session', '-t', session])
    if (root / 'STOP').exists():
        (root / 'STOP').rename(root / ('STOP-acknowledged-' + str(time.time_ns()) + '.json'))
    command = [str(Path(config['project_root']) / '.venv/bin/python'), '-B', '-u', str(HERE / 'worker.py'),
        '--worker', a.worker, '--plan', str(Path(a.plan).resolve()), '--plan-sha256', a.plan_sha256]
    if a.retry_failed:
        command.append('--retry-failed')
    # Install remain-on-exit before sending the actual queue command to the shell.
    subprocess.check_call(['tmux', 'new-session', '-d', '-s', session, '-c', config['project_root']])
    subprocess.check_call(['tmux', 'set-option', '-t', session, 'remain-on-exit', 'on'])
    log = root / ('OWNER-' + str(time.time_ns()) + '.log')
    script = 'exec ' + shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.check_call(['tmux', 'send-keys', '-t', session, '-l', script])
    subprocess.check_call(['tmux', 'send-keys', '-t', session, 'Enter'])
    result = {'session': session, 'worker': a.worker, 'plan': pin(a.plan), 'source_manifest': plan['source_manifest'],
        'owner_log': str(log), 'status': str(root / 'STATUS.json'), 'smoke_summary': str(root / 'SMOKE_SUMMARY.json'),
        'global_barrier': plan['global_smoke_pass']}
    write(root / 'TMUX_LAUNCH.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
