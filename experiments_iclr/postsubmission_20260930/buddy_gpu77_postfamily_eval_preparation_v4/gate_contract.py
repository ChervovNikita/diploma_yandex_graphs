"""Stdlib-only receipt gates; no tensor, archive member or GPU access."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def matches(actual, expected, label):
    require(isinstance(actual, dict) and all(actual.get(key) is value if type(value) is bool else actual.get(key) == value
                                           for key, value in expected.items()),
            label + ' differs or is incomplete')


def archive_path_gate(declared_relative, actual_staging_relative):
    require(declared_relative == actual_staging_relative,
            'Actual77 archive path differs from its recorded staging command; no source-host fallback')


def terminal_gate(receipt, context, argv, resource_sha, admission_sha):
    matches(receipt, dict(schema='buddy77-family-launch-receipt-v2', exit_code=0,
                          argv=argv, resource_receipt_sha256=resource_sha, admission_sha256=admission_sha,
                          family_cells=15, optimizer_fits=24, other_jobs_stopped=False, test_access=False,
                          prior_partial_fits_excluded=True,
                          predecessor_resource_is_runtime_observation_not_confinement_proof=True,
                          runtime_environment_sha256=context['runtime_environment_sha256']),
            'Successful terminal family receipt')
    require(type(receipt.get('exit_code')) is int, 'Terminal exit code must be an integer')
    previous, current = dict(receipt.get('context', {})), dict(context)
    previous.pop('preflight_seconds', None)
    current.pop('preflight_seconds', None)
    require(previous == current, 'Terminal family source/cache/runtime context changed')


def complete_cells_gate(rows, arms, seeds, epochs):
    expected = {(arm, seed) for arm in arms for seed in seeds}
    require(len(rows) == 15 and len(expected) == 15, 'All fifteen complete cells are required')
    require(all(type(row.get('seed')) is int for row in rows), 'Cell seed type differs')
    require({(row.get('arm'), row.get('seed')) for row in rows} == expected,
            'Missing, duplicate or foreign arm/seed cell')
    for row in rows:
        matches(row, dict(status='training_complete', epochs_completed=epochs,
                          optimizer_fits=4 if row['arm'] == 'independent4' else 1,
                          test_loaded_or_scored=False), 'Full training cell')
        require(type(row.get('epochs_completed')) is int and type(row.get('optimizer_fits')) is int,
                'Epoch/optimizer count type differs')
    require(sum(row['optimizer_fits'] for row in rows) == 24, 'All twenty-four optimizer fits required')


def evaluation_admission_gate(admission, identity):
    runtime_mode_gate(admission)
    matches(admission, dict(identity, schema='buddy77-postfamily-evaluation-admission-v2', decision='admitted',
                            family_cells=15, optimizer_fits=24, epochs_per_cell=100,
                            all_selected_checkpoints_and_ledgers_reviewed=True,
                            production_family_lock_reviewed=True, official_test_access_authorized=True,
                            evaluation_once=True, other_jobs_stopped=False, prior_partial_fits_excluded=True),
            'Separate prospective root evaluation admission')
    require(isinstance(admission.get('root_evaluation_cost_decision'), str) and
            bool(admission['root_evaluation_cost_decision'].strip()), 'Root evaluation-cost decision missing')


def audit_admission_gate(admission, identity):
    runtime_mode_gate(admission)
    matches(admission, dict(identity, schema='buddy77-postfamily-lock-audit-admission-v1', decision='admitted',
                            family_cells=15, optimizer_fits=24, epochs_per_cell=100,
                            full_terminal_and_all_training_ledgers_reviewed=True,
                            audit_once=True, official_test_access_authorized=False, other_jobs_stopped=False,
                            prior_partial_fits_excluded=True),
            'Separate prospective root production lock-audit admission')
    require(isinstance(admission.get('root_audit_cost_decision'), str) and
            bool(admission['root_audit_cost_decision'].strip()), 'Root lock-audit cost decision missing')


def runtime_mode_gate(admission):
    mode = admission.get('runtime_boundary_mode', 'environment_and_explicit_repo_paths_only')
    require(mode in {'environment_and_explicit_repo_paths_only', 'external_namespace_profile_bound'},
            'Unknown declared runtime mode')
    if mode == 'external_namespace_profile_bound':
        require(admission.get('execution_guarded') is True, 'External namespace mode lacks its guard declaration')
    else:
        require(admission.get('execution_guarded') is None or admission.get('execution_guarded') is False,
                'Ordinary runtime must not claim namespace enforcement')
    return mode
