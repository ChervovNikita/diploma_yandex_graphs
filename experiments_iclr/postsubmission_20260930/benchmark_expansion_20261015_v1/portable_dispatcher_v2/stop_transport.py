"""Pinned monotonic STOP files; transport may happen whenever a route is available.

No network I/O, heartbeat, coordinator, automatic retry or clearing. Received
acknowledgment means the host has durably stopped new admissions. It does not
claim that its current children have exited. Undelivered STOP cannot stop an
offline host. Existing training does not depend on this transport being online.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from custody import digest, locked, pinned, read, require, same_or_create, write


def issue(outbox, ownership, issuer, targets, reason, authority):
    require(issuer == ownership['stop_issuer'] and reason.strip(), 'Pinned issuer and explicit reason required')
    require(targets and set(targets).issubset(ownership['hosts']), 'Explicit target hosts required')
    pinned(authority['path'], authority['sha256'])
    outbox = Path(outbox)
    outbox.mkdir(parents=True, exist_ok=True)
    with locked(outbox / 'OUTBOX.lock'):
        previous = [read(p) for p in outbox.glob('STOP-*.json')]
        require(all(v['ownership_sha256'] == digest(ownership) and v['issuer'] == issuer for v in previous), 'Outbox belongs to another ownership/issuer')
        sequence = max((v['sequence'] for v in previous), default=0) + 1
        value = {'schema': 'portable-global-STOP-v2', 'ownership_sha256': digest(ownership),
                 'issuer': issuer, 'sequence': sequence, 'targets': sorted(set(targets)), 'reason': reason,
                 'authority': authority, 'UTC_unix': time.time(),
                 'effect': 'halt_new_admissions_and_drain_own_children'}
        path = outbox / f'STOP-{sequence:08d}.json'
        write(path, value)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def deliver(command_path, expected_sha256, ledger, ownership, host):
    command = pinned(command_path, expected_sha256)
    require(command['schema'] == 'portable-global-STOP-v2' and command['ownership_sha256'] == digest(ownership), 'STOP belongs to another ownership')
    require(command['issuer'] == ownership['stop_issuer'] and host in command['targets'], 'STOP issuer/target mismatch')
    require(type(command['sequence']) is int and command['sequence'] > 0, 'Positive monotonic sequence required')
    require(command['effect'] == 'halt_new_admissions_and_drain_own_children' and command['reason'], 'Explicit STOP effect/reason required')
    require(len(command['authority']['sha256']) == 64, 'Pinned issuer authority required')
    root = Path(ledger)
    with locked(root / 'HOST.lock'):
        owner = read(root / 'OWNER.json')
        require(owner['ownership_sha256'] == digest(ownership) and owner['host'] == host, 'Receiver host custody mismatch')
        inbox = root / 'inbox'
        inbox.mkdir(exist_ok=True)
        existing = list(inbox.glob('STOP-*.json'))
        seq = command['sequence']
        target = inbox / f'STOP-{seq:08d}.json'
        retained = {'command': command, 'command_sha256': expected_sha256}
        if target.exists():
            require(read(target) == retained, 'Conflicting STOP sequence replay')
        else:
            require(seq > max((read(p)['command']['sequence'] for p in existing), default=0), 'Stale STOP sequence')
            write(target, retained)
        # Admission checks under this same lock; crash before ACK is safely redeliverable.
        stop_path = root / 'stops' / f'delivered-{seq:08d}.json'
        stop_path.parent.mkdir(exist_ok=True)
        same_or_create(stop_path, retained)
        ack_path = inbox / f'ACK-{host}-{seq:08d}.json'
        ack = {'schema': 'portable-STOP-received-ack-v2', 'host': host, 'sequence': seq,
               'ownership_sha256': digest(ownership), 'command_sha256': expected_sha256,
               'received_and_durable': True, 'all_children_drained': 'not_claimed',
               'effect': 'halt_new_admissions_and_drain_own_children'}
        same_or_create(ack_path, ack)
        return {'ack': ack, 'path': str(ack_path.resolve()), 'sha256': hashlib.sha256(ack_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    create = sub.add_parser('issue')
    create.add_argument('--outbox', required=True)
    create.add_argument('--issuer', required=True)
    create.add_argument('--target', action='append', required=True)
    create.add_argument('--reason', required=True)
    create.add_argument('--authority', required=True)
    create.add_argument('--authority-sha256', required=True)
    receive = sub.add_parser('deliver')
    receive.add_argument('--command', required=True)
    receive.add_argument('--command-sha256', required=True)
    receive.add_argument('--worker', required=True, choices=('allocation/gpu0', 'server77/gpu0', 'server77/gpu1'))
    for p in (create, receive):
        p.add_argument('--ownership', required=True)
        p.add_argument('--ownership-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'deliver':
        from host_binding import load_config, config_path, guard
        import os
        config, config_pin = load_config(config_path(args.worker))
        require(config['host_enabled'], 'Host disabled before probes')
        os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
        guard(config)
    ownership = pinned(args.ownership, args.ownership_sha256)
    if args.action == 'issue':
        result = issue(args.outbox, ownership, args.issuer, args.target, args.reason,
                       {'path': args.authority, 'sha256': args.authority_sha256})
    else:
        host = args.worker.split('/')[0]
        ledger = Path(config['project_root']) / config['phase_relative'] / '.portable_campaign_host_v2'
        require(ownership['hosts'][host]['project_root'] == config['project_root'] and ownership['hosts'][host]['phase_relative'] == config['phase_relative'], 'STOP host root/phase mismatch')
        from host_ledger import Ledger
        Ledger(ledger, ownership, host).establish()
        result = deliver(args.command, args.command_sha256, ledger, ownership, host)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
