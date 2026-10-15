"""Same-lock, fail-closed transition from terminal TRAIN to final scoring.

Original OWNER.json, claims, STOPs, archives and STATE bytes are preserved.
Final reservations use the actual portable Ledger implementation. Its old
training loader refuses the successor STATE ownership hash; it cannot admit
another training child after handoff. No independent racing ledger exists.
"""
import json
from pathlib import Path
from .common import digest, pin, require


def ledger_class(portable_ledger, custody, claim_original_host):
    class FinalPhaseLedger(portable_ledger):
        def __init__(self, root, ownership, host, *, training_ownership, handoff, verify_terminal_seal):
            super().__init__(root, ownership, host)
            self.training_ownership = training_ownership
            self.handoff = handoff
            self.verify_terminal_seal = verify_terminal_seal

        def establish(self):
            require(self.root.is_dir() and self.root.resolve() == self.root, 'Original canonical training ledger required')
            require(not any((self.root / name).is_symlink() for name in ('HOST.lock', 'OWNER.json', 'STATE.json', 'FINAL_OWNER.json', 'PHASE_HANDOFF.json')), 'Phase authority cannot use symlink aliases')
            with custody.locked(self.lock):
                claim_original_host(self.root, self.training_ownership, self.host)
                self.verify_terminal_seal()
                final_owner = {'schema': 'portable-final-phase-owner-v1', 'host': self.host,
                    'training_ownership_sha256': digest(self.training_ownership), 'final_ownership_sha256': self.owner_hash,
                    'handoff': self.handoff, 'original_OWNER': pin(self.root / 'OWNER.json')}
                marker = self.root / 'FINAL_OWNER.json'
                if marker.exists():
                    require(custody.read(marker) == final_owner, 'Final phase owner cannot change')
                    self.load()
                    return
                original = self.root / 'STATE.json'
                exact = original.read_text()
                envelope = json.loads(exact)
                state = envelope['state']
                require(envelope['sha256'] == digest(state) and state['schema'] == 'portable-host-reservations-v2'
                        and state['ownership_sha256'] == digest(self.training_ownership) and state['host'] == self.host
                        and state['active'] == {}, 'All TRAIN reservations must be durably reaped before phase handoff')
                archive = {'schema': 'preserved-training-ledger-at-final-handoff-v1',
                    'origin': pin(original), 'exact_json_text': exact, 'handoff': self.handoff}
                custody.same_or_create(self.root / 'TRAINING_STATE_AT_HANDOFF.json', archive)
                # Marker precedes STATE. A crash here blocks the final loader
                # pending review. The old loader does not inspect the marker:
                # its inventory is already terminal and no new TRAIN resume is
                # authorized by the seal. After STATE replacement its old owner
                # check also refuses. Do not claim an earlier training block.
                custody.same_or_create(marker, final_owner)
                custody.same_or_create(self.root / 'PHASE_HANDOFF.json', {'schema': 'portable-sealed-final-phase-handoff-v1',
                    'owner': final_owner, 'old_STATE_archive': pin(self.root / 'TRAINING_STATE_AT_HANDOFF.json'),
                    'same_HOST_lock': str(self.lock), 'original_owner_preserved': True})
                self.save({'schema': 'portable-host-reservations-v2', 'ownership_sha256': self.owner_hash,
                    'host': self.host, 'generation': 0, 'active': {}, 'phase': 'final_selected_TEST',
                    'phase_handoff': pin(self.root / 'PHASE_HANDOFF.json')})
                self.load()

        def load(self):
            claim_original_host(self.root, self.training_ownership, self.host)
            marker = custody.read(self.root / 'FINAL_OWNER.json')
            require(marker['final_ownership_sha256'] == self.owner_hash and marker['handoff'] == self.handoff, 'Changed final phase owner/handoff')
            envelope = custody.read(self.root / 'STATE.json')
            state = envelope['state']
            require(envelope['sha256'] == digest(state) and state['ownership_sha256'] == self.owner_hash
                    and state['host'] == self.host and state['schema'] == 'portable-host-reservations-v2'
                    and state['phase'] == 'final_selected_TEST'
                    and state['phase_handoff'] == pin(self.root / 'PHASE_HANDOFF.json')
                    and isinstance(state['active'], dict), 'Final host ledger corrupt or handoff incomplete')
            for token, row in state['active'].items():
                require(row['token'] == token and row['state'] in ('startup', 'running'), 'Invalid final reservation')
                require(row['assignment'] in self.ownership['assignments']
                        and row['assignment']['worker'].split('/')[0] == self.host, 'Foreign final assignment')
                require(custody.read(self.root / 'claims' / (digest(row['assignment']['id']) + '.json')) == row['permanent_claim'], 'Final reservation lost permanent claim')
            return state

    return FinalPhaseLedger
