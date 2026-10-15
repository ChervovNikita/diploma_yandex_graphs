"""Host-wide permanent custody and atomic startup/running resource reservations.

The same flock protects GPU0/GPU1 admission, Popen and identity commit. A
supervisor crash leaves a reservation in place. No timeout or missing PID can
free it. Reap/terminal custody or explicit reviewed reconciliation is required.
"""
from __future__ import annotations
import os
from pathlib import Path
import time
import uuid
from custody import digest, locked, read, require, same_or_create, write
from ownership import claim_host, claim_task


def admissible(resources, gpu_uuid, active, probe, budget):
    """Pure measured-free-plus-unconsumed-reservation admission, all host workers."""
    if probe['untracked_own_gpu_pids']:
        return False, 'untracked_owned_GPU_process'
    gpu = [r for r in active if r['gpu_uuid'] == gpu_uuid]
    if len(gpu) >= 2:
        return False, 'two_slot_limit'
    if gpu and (resources['max_concurrent_owned_children'] == 1 or any(r['resources']['max_concurrent_owned_children'] == 1 for r in gpu)):
        return False, 'serial_only_resource_profile'
    if any(r['resources']['profile'] not in resources['qualified_peer_profiles'] or resources['profile'] not in r['resources']['qualified_peer_profiles'] for r in gpu):
        return False, 'pair_profile_not_qualified'
    def observed(r, name):
        pid = (r.get('child_identity') or {}).get('pid')
        return probe[name].get(pid, 0)
    remaining_gpu = sum(max(0, r['resources']['vram_bytes'] - observed(r, 'allocated_vram_by_pid')) for r in gpu)
    remaining_ram = sum(max(0, r['resources']['ram_bytes'] - observed(r, 'rss_by_pid')) for r in active)
    for needed, available, reason in (
        (resources['cpu_slots'] + sum(r['resources']['cpu_slots'] for r in active), budget['host_cpu_slots'], 'host_shared_CPU_reservation'),
        (resources['vram_bytes'] + remaining_gpu + budget['gpu_headroom_bytes'], probe['free_vram_by_uuid'][gpu_uuid], 'VRAM_reservation'),
        (resources['ram_bytes'] + remaining_ram + budget['host_ram_headroom_bytes'], probe['available_ram_bytes'], 'host_shared_RAM_reservation'),
        (resources['disk_bytes'] + sum(r['resources']['disk_bytes'] for r in active) + budget['disk_headroom_bytes'], probe['free_disk_bytes'], 'host_shared_disk_reservation'),
    ):
        if needed > available:
            return False, reason
    return True, 'admitted'


class Ledger:
    def __init__(self, root, ownership, host):
        self.root = Path(root)
        self.ownership = ownership
        self.owner_hash = digest(ownership)
        self.host = host
        self.lock = self.root / 'HOST.lock'
        self.ticket = None

    def establish(self):
        self.root.mkdir(parents=True, exist_ok=True)
        require(self.root.resolve() == self.root, 'Canonical host ledger path cannot be aliased')
        require(not any((self.root / name).is_symlink() for name in ('HOST.lock', 'OWNER.json', 'STATE.json')), 'Ledger authority cannot use symlink aliases')
        with locked(self.lock):
            claim_host(self.root, self.ownership, self.host)
            (self.root / 'archive').mkdir(exist_ok=True)
            (self.root / 'stops').mkdir(exist_ok=True)
            state = self.root / 'STATE.json'
            if not state.exists():
                require(not any((self.root / 'archive').iterdir()) and not (self.root / 'claims').exists(), 'Missing ledger with retained custody; review required')
                self.save({'schema': 'portable-host-reservations-v2', 'ownership_sha256': self.owner_hash,
                           'host': self.host, 'generation': 0, 'active': {}})
            self.load()

    def load(self):
        envelope = read(self.root / 'STATE.json')
        state = envelope['state']
        require(envelope['sha256'] == digest(state), 'Host ledger corruption')
        require(state['ownership_sha256'] == self.owner_hash and state['host'] == self.host, 'Host ledger owner mismatch')
        require(state['schema'] == 'portable-host-reservations-v2' and isinstance(state['active'], dict), 'Invalid host ledger')
        claim_host(self.root, self.ownership, self.host)
        for token, row in state['active'].items():
            require(row['token'] == token and row['state'] in ('startup', 'running'), 'Invalid active reservation')
            require(row['assignment']['worker'].split('/')[0] == self.host, 'Foreign host reservation')
            require(row['assignment'] in self.ownership['assignments'], 'Reservation assignment differs from immutable owner')
            claim_path = self.root / 'claims' / (digest(row['assignment']['id']) + '.json')
            require(read(claim_path) == row['permanent_claim'], 'Active reservation lost permanent claim custody')
        return state

    def save(self, state):
        write(self.root / 'STATE.json', {'state': state, 'sha256': digest(state)}, replace=True)

    def stop(self, reason):
        # Independent of ledger parsing: a corrupt ledger can still halt peers.
        def persist():
            folder = self.root / 'stops'
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / (str(time.time_ns()) + '-' + uuid.uuid4().hex + '.json'),
                  {'schema': 'portable-host-local-STOP-v2', 'ownership_sha256': self.owner_hash,
                   'host': self.host, 'reason': reason, 'effect': 'halt_new_admissions_and_drain_own_children'})
        if self.ticket is not None:
            persist()  # This process already holds the shared admission lock.
        else:
            with locked(self.lock):
                persist()

    def reserve(self, assignment, bound_hash, output, gpu_uuid, resources, budget, probe_fn, identity_fn, supervisor_identity, baseline_stops=()):
        require(self.ticket is None, 'Outstanding startup transaction')
        context = locked(self.lock)
        context.__enter__()
        try:
            state = self.load()
            require(not ({str(p) for p in (self.root / 'stops').glob('*.json')} - set(baseline_stops)), 'Host STOP received')
            require(assignment in self.ownership['assignments'] and assignment['worker'].split('/')[0] == self.host, 'Task has another immutable owner')
            active = list(state['active'].values())
            for row in active:
                require(row['state'] == 'running' and row.get('child_identity'), 'Uncertain startup reservation; review required')
                require(identity_fn(row['supervisor_identity']) == 'same', 'Orphan supervisor; reservation retained pending review')
                # A live custodian may be between waitpid and terminal collection.
                # Missing child retains its entire reservation until that custodian releases it.
                require(identity_fn(row['child_identity']) in ('same', 'missing'), 'Reused or uncertain child PID; review required')
            require(all(r['assignment']['id'] != assignment['id'] for r in active), 'Task already reserved')
            probe = probe_fn(active)
            ok, reason = admissible(resources, gpu_uuid, active, probe, budget)
            if not ok:
                context.__exit__(None, None, None)
                return False, reason
            permanent_claim = claim_task(self.root, self.owner_hash, assignment, bound_hash, output)
            token = uuid.uuid4().hex
            row = {'token': token, 'assignment': assignment, 'permanent_claim': permanent_claim, 'gpu_uuid': gpu_uuid, 'resources': resources,
                   'admission_probe': probe,
                   'state': 'startup', 'supervisor_pid': os.getpid(), 'supervisor_identity': supervisor_identity, 'child_identity': None,
                   'events': [{'event': 'startup_reserved', 'UTC_unix': time.time()}]}
            state['active'][token] = row
            state['generation'] += 1
            self.save(state)
            self.ticket = {'context': context, 'state': state, 'token': token, 'child_created': False}
            return True, token
        except BaseException:
            context.__exit__(None, None, None)
            raise

    def commit(self, identity):
        ticket = self.ticket
        require(ticket is not None, 'No startup reservation')
        ticket['child_created'] = True
        try:
            require(not identity.get('identity_unavailable'), 'Child identity unavailable; preserve startup reservation')
            row = ticket['state']['active'][ticket['token']]
            row['state'] = 'running'
            row['child_identity'] = identity
            row['events'].append({'event': 'child_identity_committed', 'identity': identity, 'UTC_unix': time.time()})
            ticket['state']['generation'] += 1
            self.save(ticket['state'])
            return ticket['token']
        finally:
            ticket['context'].__exit__(None, None, None)
            self.ticket = None

    def cancel_unspawned(self, reason):
        """Only the holder before Popen is allowed to declare no child was created."""
        ticket = self.ticket
        if ticket is None:
            return
        try:
            require(not ticket['child_created'], 'Created child cannot be cancelled as unspawned')
            row = ticket['state']['active'].pop(ticket['token'])
            row['events'].append({'event': 'definite_no_child', 'reason': reason, 'UTC_unix': time.time()})
            row['state'] = 'launch_failed_no_child'
            write(self.root / 'archive' / (ticket['token'] + '.json'), row)
            ticket['state']['generation'] += 1
            self.save(ticket['state'])
        finally:
            ticket['context'].__exit__(None, None, None)
            self.ticket = None

    def reaped(self, token, terminal_pin):
        """Caller must have reaped its own Popen and durably collected its terminal."""
        with locked(self.lock):
            state = self.load()
            row = state['active'][token]
            require(row['supervisor_pid'] == os.getpid(), 'Only owning supervisor may release an actual reap')
            row['state'] = 'reaped'
            row['events'].append({'event': 'reaped_and_collected', 'terminal': terminal_pin, 'UTC_unix': time.time()})
            same_or_create(self.root / 'archive' / (token + '.json'), row)
            del state['active'][token]
            state['generation'] += 1
            self.save(state)

    def has_peer_activity(self, worker):
        with locked(self.lock):
            state = self.load()
            return any(r['assignment']['worker'] != worker for r in state['active'].values())

    def reconcile(self, approval_pin, verify, identity_fn):
        """Explicit dead-custodian disposition; never signal or adopt any PID."""
        verify(approval_pin)
        approval = read(approval_pin['path'])
        require(approval['schema'] == 'portable-reviewed-ledger-reconciliation-v2'
                and approval['ownership_sha256'] == self.owner_hash and approval['host'] == self.host
                and approval['reviewed_no_live_owned_child'] is True, 'Exact reviewed orphan disposition required')
        verify(approval['root_review_evidence'])
        verify(approval['terminal_or_retained_cost_evidence'])
        with locked(self.lock):
            state = self.load()
            token = approval['token']
            row = state['active'][token]
            require(approval['reservation_sha256'] == digest(row), 'Review must bind exact retained reservation')
            require(identity_fn(row['supervisor_identity']) in ('missing', 'different'), 'Live or uncertain custodian remains')
            if row.get('child_identity'):
                require(identity_fn(row['child_identity']) in ('missing', 'different'), 'Live or uncertain child remains')
            require(approval['unknown_partial_costs_preserved'] is True, 'Unknown costs and every attempt must remain')
            archive = {'reservation': row, 'disposition': approval_pin, 'state': 'reviewed_closed_without_retry'}
            same_or_create(self.root / 'archive' / (token + '-reviewed.json'), archive)
            del state['active'][token]
            state['generation'] += 1
            self.save(state)
