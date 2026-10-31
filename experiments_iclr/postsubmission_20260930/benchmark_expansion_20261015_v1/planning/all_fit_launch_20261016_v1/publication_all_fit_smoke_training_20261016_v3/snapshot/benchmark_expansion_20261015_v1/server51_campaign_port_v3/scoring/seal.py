"""Offline metadata sealing after every frozen training task is terminal.

No checkpoint/data reads, model library, network coordinator or new approval.
The same reviewed campaign GO specifies this deterministic derived stage.
"""
from pathlib import Path
from .contracts import digest, inside, require, verify

TERMINAL = {'complete', 'no_selected_state', 'failed', 'launch_failed',
            'blocked_dependency_no_selected_state', 'blocked_dependency_failed'}


def build_training_seal(training_release_pin, scoring_roster_pin, state_pins, root):
    release = verify(training_release_pin, root, json_metadata=True)
    roster = verify(scoring_roster_pin, root, json_metadata=True)
    require(roster['schema'] == 'frozen-final-selected-scoring-roster-v2', 'Exact frozen final scoring roster required')
    tasks = {task['id']: task for task in release['tasks']}
    expected = {task.get('training_fit_id', task.get('training_I4_id')) for task in roster['tasks']}
    require(set(tasks) == set(state_pins) == expected and len(tasks) == 6730,
            'Seal every frozen campaign outcome before any final TEST serving')
    outcomes = {}
    for task_id, task in tasks.items():
        expected_state = inside(Path(release['output_root']) / 'control/tasks' / digest(task_id) / 'STATE.json', root)
        require(state_pins[task_id]['path'] == str(expected_state), 'State is not the exact original campaign task')
        state = verify(state_pins[task_id], root, json_metadata=True)
        require(state.get('status') in TERMINAL, 'Pending/running/interrupted work cannot be sealed or silently dropped')
        terminal = verify(state['attempt_terminal'], root, json_metadata=True)
        require(terminal.get('id') == task_id and terminal.get('status') == state['status'], 'Task terminal authority differs')
        raw_pin = terminal.get('live_native_result_receipt')
        raw = None if raw_pin is None else verify(raw_pin, root, json_metadata=True)
        if raw_pin is not None:
            require(raw_pin['path'] == str(inside(Path(release['output_root']) / task['output_relative'] / 'RESULT.json', root)),
                    'Native result is outside its original output namespace')
        if state['status'] == 'complete':
            require(raw is not None and raw.get('status') == 'complete', 'Complete terminal lacks complete native result')
        selected = None
        if task['kind'] == 'fit' and state['status'] == 'complete':
            require(raw['job_id'] == task_id and raw['job_hash'] == task['runtime_job_hash'], 'Selected fit identity differs')
            selected = {'checkpoint': raw['selected_checkpoint'], 'epoch': raw['selected_epoch'],
                        'VALID_export': raw['selected_VALID_export'], 'runtime_binding': raw['runtime_binding']}
        outcomes[task_id] = {'status': state['status'], 'state': state_pins[task_id],
                             'terminal': state['attempt_terminal'], 'training_result': raw_pin,
                             'selected_state': selected, 'scientific_settings_reselected': False}
    return {'schema': 'all-campaign-outcomes-and-selections-seal-v2', 'training_release': training_release_pin,
            'scoring_roster': scoring_roster_pin, 'all_6730_training_tasks_terminal': True,
            'outcomes': outcomes, 'TEST_accessed': False, 'human_action_claimed': False,
            'selection_rule': 'each_original_own_VALID_selector_only', 'retry_or_arm_changes_after_seal': False,
            'distribution': 'copy this immutable manifest and required receipts to each worker; no live coordinator'}
