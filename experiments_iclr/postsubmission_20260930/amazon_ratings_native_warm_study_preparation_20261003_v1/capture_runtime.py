"""Root-run metadata capture; no model/data/quality access or job launching."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import common as c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = c.confined(args.output)
    c.require(not output.is_relative_to(c.PACKET) and not output.is_relative_to(c.V4),
              'Runtime receipt must be outside sealed sources')
    source = c.verify_sources()
    _, value = c.configure_runtime(args.device)
    c.write(output, {'schema': 'amazon_native_warm_runtime_v1', 'UTC': c.utc(),
                         'packet_manifest': source, 'runtime': value,
                         'models_data_labels_or_quality_accessed': False,
                         'declared_file_scope_only_not_transitive_dependency_closure': True})


if __name__ == '__main__':
    main()
