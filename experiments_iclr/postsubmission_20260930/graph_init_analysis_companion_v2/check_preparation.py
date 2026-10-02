"""Stdlib engineering checks; no experiment outcomes are read."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main():
    source = HERE / 'analyze_complete_report.py'
    ast.parse(source.read_text())
    namespace = runpy.run_path(str(source), run_name='source_preparation_only')
    summaries = namespace['summaries']
    checks = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    constant = summaries([-1., -1., -1.])
    check('constant synthetic differences retain the degenerate interval',
          constant['paired_mean'] == constant['paired_t95_low'] == constant['paired_t95_high'] == -1.)
    check('three same-sign pairs cannot attain two-sided p below .25', constant['sign_reference_p'] == .25)
    mixed = summaries([-1., 0., 1.])
    check('synthetic zero-excluding mixed signs', mixed['sign_reference_p'] == 1. and mixed['zero_pairs'] == 1)
    varying = summaries([-3., -2., -1.])
    half = 4.302652729696142 / math.sqrt(3.)
    check('df2 t interval agrees with independently supplied quantile',
          abs(varying['paired_t95_low'] - (-2. - half)) < 1e-9 and
          abs(varying['paired_t95_high'] - (-2. + half)) < 1e-9)
    for values in ([float('nan'), 0., 1.], [1., 2.], [True, 0., 1.]):
        try:
            summaries(values)
        except ValueError:
            checks.append('invalid synthetic paired vector rejected: ' + repr(values))
        else:
            raise AssertionError('invalid summary accepted')
    for value in ('/outside-project/example', str(HERE / '..' / '..' / '..' / 'outside-project')):
        try:
            namespace['confined'](value)
        except ValueError:
            checks.append('out-of-phase path rejected before opening: ' + value)
        else:
            raise AssertionError('unconfined path accepted')
    registry_path = namespace['REGISTRY']
    check('immutable registered metadata digest', namespace['sha'](registry_path) == namespace['REGISTRY_SHA'])
    registry = namespace['read'](registry_path)
    contexts = {namespace['object_hash'](c) for c in registry['contexts']}
    check('actual native registry context hashing matches every attempt',
          len(contexts) == 6 and len(registry['attempts']) == 72 and
          all(a['context_sha256'] in contexts for a in registry['attempts']))
    check('actual registry phase-arm counts',
          sum(a['phase'] == 'fit' for a in registry['attempts']) == 30 and
          sum(a['phase'] == 'initialize' for a in registry['attempts']) == 30)
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'schema': 'analysis-companion-source-preparation-v2',
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'checks': checks,
               'scientific_outcomes_opened': False, 'labels_checkpoints_logits_opened': False,
               'summary_values_scope': 'synthetic engineering fixtures, never paper evidence',
               'full_closed_cohort_and_admitted_report_execution_tested': False}
    output = HERE / 'PREPARATION_CHECKS.json'
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'checks_passed': len(checks), 'receipt': str(output), 'outcomes_opened': False}))


if __name__ == '__main__':
    main()
