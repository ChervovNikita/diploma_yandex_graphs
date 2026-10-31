"""Allocation route identity before any project operation."""
import socket,subprocess
def allocation_guard():
    if socket.gethostname()!='anogena-2-0':raise ValueError('Excluded wrong scientific hostname')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Excluded wrong sole GPU UUID')
