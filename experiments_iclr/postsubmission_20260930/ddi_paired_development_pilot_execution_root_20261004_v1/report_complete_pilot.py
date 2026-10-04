"""Explicit final report after all12 physical fits; enumerate all eight flips.

This is separate from metadata monitoring. It reads predictive outputs only
when explicitly invoked after full queue completion. Never run during staging.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode=True
from execution_common import ROOT, PHASE, save, sha, source_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    source_inventory()
    terminal=json.loads((ROOT/'QUEUE_TERMINAL.json').read_text())
    plan=json.loads((ROOT/'PLAN.json').read_text())
    assert terminal['status']=='COMPLETE_PHYSICAL_12CELL_QUEUE'
    assert terminal['completed_cells']==plan['queue'] and len(terminal['terminals'])==12
    assert all(row['physical_exit_code']==0 and row['direct_child_reaped'] is True
               and row['physical_session_closed'] is True and row['stop'] is None and not row['errors']
               for row in terminal['terminals'])
    pilot=PHASE/plan['candidate_packet']
    sys.path.insert(1,str(pilot))
    import compare_pilot
    assert Path(compare_pilot.__file__).resolve()==pilot/'compare_pilot.py'
    # This fixed complete-family call performs all scientific/paired eligibility checks.
    result=compare_pilot.compare(ROOT/'fits',pilot)
    for contrast in result['contrasts']:
        differences=contrast['seed_differences']
        contrast['root_supplement_all_eight_sign_flips']=[
            dict(seed_order=[0,1,2],sign_pattern=list(signs),
                 signed_mean_Hits20_fraction=statistics.mean(sign*value for sign,value in zip(signs,differences)),
                 signed_mean_percentage_points=100*statistics.mean(sign*value for sign,value in zip(signs,differences)))
            for signs in itertools.product((-1,1),repeat=3)]
    result['root_reporting_supplement']=dict(
        nonblocking_review_finding='PILOT-N01',sealed_comparer_emitted_sign_patterns=False,
        root_report_enumerates_all_eight_patterns_and_means=True,
        scientific_source_unchanged=True,pilot_manifest_sha256=plan['candidate_manifest_sha256'],
        queue_terminal_sha256=sha(ROOT/'QUEUE_TERMINAL.json'),predictive_values_read=True,
        full500_qualification=False,interpretation='Descriptive seed variation on one graph; no significance or acceptance claim.')
    sys.path.insert(1,str(PHASE/'hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2'))
    from runtime_paths import require_project_output
    output=require_project_output(args.output_dir,ROOT);output.mkdir(parents=True,exist_ok=False)
    save(output/'ROOT_COMPLETE_PILOT_REPORT.json',result)
    print(json.dumps(dict(status='COMPLETE_FAMILY_REPORT_WITH_EIGHT_SIGN_PATTERNS',
                         development_decision=result['development_decision'],output=str(output)),indent=2))


if __name__=='__main__':
    main()
