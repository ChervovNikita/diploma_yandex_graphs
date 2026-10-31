"""Own one native invocation; leases survive supervisor or guardian loss."""
import argparse
import signal
import resource
import subprocess
from common import *


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--invocation', required=True); p.add_argument('--invocation-sha256', required=True)
    p.add_argument('--lease-fds', required=True)
    a = p.parse_args()
    receipt = pin(a.invocation)
    require(receipt['sha256'] == a.invocation_sha256, 'Invocation bytes changed')
    request = read(a.invocation)
    directory = Path(a.invocation).parent
    fds = tuple(int(fd) for fd in a.lease_fds.split(','))
    # All flock descriptors are inherited by the numerical process as well.
    # A scheduler/guardian crash therefore cannot release a busy physical slot.
    for fd in fds:
        os.fstat(fd)
    started = time.time()
    write(directory / 'GUARDIAN.json', {'identity': identity(), 'started': started, 'invocation': receipt})
    child = None
    signalled = []
    def stop(signum, frame):
        signalled.append(signum)
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    code = None; error = None
    try:
        with (directory / 'native.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(request['argv'], cwd=request['project_root'], env=dict(os.environ, **request['environment']),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=fds)
            write(directory / 'CHILD.json', {'identity': identity(child.pid), 'argv': request['argv'], 'started': started})
            code = child.wait()
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); code = child.wait(); signalled.append(signal.SIGKILL)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result_present = (Path(request['output']) / 'RESULT.json').exists()
    terminal = {'schema': 'all-fit-child-terminal-v1', 'invocation': receipt, 'exit_code': code,
        'started': started, 'finished': time.time(), 'wall_seconds': time.time() - started,
        'signals_forwarded': signalled, 'forced_kill': signal.SIGKILL in signalled,
        'failure': error, 'result_present': result_present,
        'child_cpu_user_seconds': usage.ru_utime, 'child_cpu_system_seconds': usage.ru_stime,
        'child_peak_rss_platform_units': usage.ru_maxrss,
        'unknown_partial_work': code is None or not result_present, 'prior_partial_work_preserved': True}
    write(directory / 'TERMINAL.json', terminal)
    return 0 if code == 0 and error is None else 1


if __name__ == '__main__':
    sys.exit(main())
