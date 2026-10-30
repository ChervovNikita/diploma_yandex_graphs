"""Reviewed plaintext native launcher with the additive host/reporting authority."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
sys.path.insert(0, str(HERE))
from host_extension import guard, pin, require


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-dir')
    parser.add_argument('--repair-source-sha256', required=True)
    parser.add_argument('--record-dir', required=True)
    parser.add_argument('native_argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native = args.native_argv[1:] if args.native_argv[:1] == ['--'] else args.native_argv
    require(Path(native[0]).resolve() == E / 'server51_campaign_port_v6/reporting/runtime_entry.py',
            'Only the frozen allocation reporting invocation may be redirected')
    sys.path.insert(0, str(E / 'campaign_reporting_repair_20261024_v2'))
    from repair_native import source
    from reporting_patch import install
    source(args.repair_source_sha256)
    repair = install()
    # Parent reporting functions are imported without changing their bytes.
    # New reporting_entry differs only in the explicit host authority.
    actual = [str(HERE / 'reporting_entry.py'), *native[1:]]
    import json
    (Path(args.record_dir) / 'ADDITIVE_HOST_PRODUCER.json').write_text(json.dumps({
        'source': pin(HERE / 'MANIFEST.json'), 'reporting_repair': repair,
        'frozen_reporting_argv': native, 'actual_reporting_argv': actual,
        'native_scientific_arguments_changed': False}, indent=2) + '\n')
    sys.path.insert(0, str(E / 'campaign_named_processes_20261021_v1'))
    from launch_named import run_native
    run_native(actual, args.dependency_dir)


if __name__ == '__main__':
    main()
