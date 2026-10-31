"""Reject executing this candidate before root-owned admission and explicit GO."""


def require_admitted_plan(plan, studies):
    if plan.get('scientific_launch_enabled') is not True:
        raise ValueError('Local common-wrapper preparation candidate has scientific launch disabled')
    if plan.get('scientific_roster_approved') is not True or plan.get('resource_admission_status') != 'qualified':
        raise ValueError('Common-wrapper scientific roster/resource admission is pending')
    for row in studies.values():
        study = row.get('spec', row)
        source = study['source']
        if source.get('source_commit_status') != 'root_verified' or source.get('git_commit', '').startswith('PENDING_'):
            raise ValueError('Root-owned verified source commit binding is pending')
