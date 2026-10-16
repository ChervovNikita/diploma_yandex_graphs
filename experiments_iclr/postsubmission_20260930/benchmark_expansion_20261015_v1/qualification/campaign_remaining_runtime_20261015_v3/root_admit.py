"""Explicit root-review admission materializer; does not launch, wait or reserve GPU."""
import argparse
from witness_support import E, HERE, guard, inside, own_source, read, receipt, require


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=('bound_observer_molbace', 'buddy_selected_VALID'))
    parser.add_argument('--reviewed-source-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--gru-peer', required=True, choices=('present', 'absent'))
    args = parser.parse_args()
    own_source(args.reviewed_source_sha256)
    output, target = inside(args.output), inside(args.admission)
    require(output.is_relative_to(E/'qualification') and not output.exists(), 'Fresh own qualification output required')
    require(target.is_relative_to(E/'qualification') and not target.exists(), 'Fresh root admission file required')
    value = read(HERE/('ROOT_ADMISSION_TEMPLATE_'+args.case+'.json'))
    value.update(status='ADMITTED', root_reviewed_source=True, source_manifest_sha256=args.reviewed_source_sha256,
        INPUTS_sha256=receipt(HERE/'INPUTS.json')['sha256'], output=str(output))
    if args.gru_peer == 'absent':
        value['peers'] = []
    # Runtime supervisor rechecks presence, exact start ticks, resources, and held slot.
    import json
    with target.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(receipt(target)))


if __name__ == '__main__':
    main()
