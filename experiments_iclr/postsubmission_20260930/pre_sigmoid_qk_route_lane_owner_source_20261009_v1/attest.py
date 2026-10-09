"""After parent exit, attest actual local custody absence without quality reads."""
import argparse
from common import PHASE, absence, config, read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--authorized', action='store_true')
    args = parser.parse_args()
    cfg, pins, custody, routing, route, route_pins, cells = config(args.release, args.release_sha256, args.authorized)
    output = PHASE / cfg['output_relative']
    close = read(output / 'CLOSURE.json')
    require(close['complete'] is True and close['clean_owned_terminal'] is True
            and close['release_sha256'] == args.release_sha256 and close['scores_read'] is False
            and close['execution_source_commit'] == cfg['execution_source_commit'], 'Successful exact lane closure')
    identities = [close['parent']] + [item for exit in close['exits'] for item in exit['witnessed_owned_members']]
    groups = [close['parent']] + [exit['child'] for exit in close['exits']]
    absence(custody, identities, groups)
    require(len(close['exits']) == (12 if cfg['mode'] == 'seed_block' else 1)
            and all(exit['reaped'] and exit['group_absent'] and exit['no_owned_CUDA'] and exit['exit_code'] == 0 for exit in close['exits']),
            'Every actual child cleanly reaped')
    if cells: require(set(close['cells']) == {c['key'] for c in cells}, 'All twelve lane cells only')
    receipt = dict(close, actual_owned_absence_attested=True, owned_identities=identities, owned_groups=groups,
                   closure_sha256=sha(output / 'CLOSURE.json'), scores_read=False, comparative_opening_authorized=False)
    write(output / 'ABSENCE_RECEIPT.json', receipt, True)
    print(str(output / 'ABSENCE_RECEIPT.json'))


if __name__ == '__main__': main()
