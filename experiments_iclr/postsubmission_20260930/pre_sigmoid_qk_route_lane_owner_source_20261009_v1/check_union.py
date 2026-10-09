"""Metadata-only check of three externally authenticated twelve-cell receipts."""
import argparse
from common import ASSIGNMENTS, PHASE, bound, module, read, require, sha, sources, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--receipt-bindings', required=True)
    parser.add_argument('--receipt-bindings-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and sha(args.receipt_bindings) == args.receipt_bindings_sha256,
            'Externally authenticated exact metadata files')
    pins, custody, routing = sources()
    plan = read(args.plan)
    require(plan['seeds'] == list(ASSIGNMENTS.values()) and plan['route_ids'] == list(ASSIGNMENTS)
            and plan['route_adapter_manifest_sha256'] == pins['routing']['manifest_sha256'], 'Fixed reviewed assignments')
    rows = read(args.receipt_bindings)
    require(set(rows) == set(ASSIGNMENTS), 'One root-authenticated actual absence receipt per route')
    receipts = {route: read(bound(row)) for route, row in rows.items()}
    queue = module(PHASE / pins['routing']['directory'] / 'queue_plan.py', '_lane_unchanged_union_check')
    result = queue.verify_union(plan=plan, plan_sha256=args.plan_sha256, receipts=receipts)
    write(args.output, dict(result, receipt_bindings_sha256=args.receipt_bindings_sha256), True)
    print(args.output)


if __name__ == '__main__': main()
