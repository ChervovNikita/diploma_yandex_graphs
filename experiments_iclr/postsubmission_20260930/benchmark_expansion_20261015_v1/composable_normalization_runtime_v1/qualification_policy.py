"""One-epoch BUDDY engineering controller; original source optimizer stays intact."""
import copy
from dataclasses import replace
from pathlib import Path
from runtime_bridge import load, require

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
FINITE = {'family': 'link', 'track': 'bounded_engineering', 'maximum_epochs': 1, 'minimum_epochs': 1,
          'patience': 1, 'metric': 'hits@50', 'direction': 'max', 'ties': 'first',
          'evaluate_TRAIN_each_epoch': False, 'evaluation_mode': 'VALID_only_disclosed_adaptation'}


def original(root):
    helper = load(EXPANSION / 'portable_dispatcher_v3/runtime_link/engineering_link_policy.py',
                  '_composed_qualification_original_policy_loader')
    return helper.base_policy(root)


def create_policy(job, root=None):
    require(job.get('family') == 'link' and job.get('engineering_only') is True and job.get('TEST_access') is False
            and job['recipe'].get('architecture') == 'BUDDY' and job.get('seed') == 1
            and job.get('procedure') == 'interaction_input_conditionalJ' and job['policy'] == FINITE,
            'Only the declared one-epoch nonzero-seed BUDDY lifecycle is admitted')
    root = Path(root or job['recipe']['_repository_root']).resolve()
    parent = copy.deepcopy(job)
    parent['policy'] = copy.deepcopy(job['normalization_engineering_parent_policy'])
    require(parent['policy']['track'] == 'matched_extended', 'Retain the exact native matched parent policy')
    policy = original(root).create_policy(parent, root)
    return replace(policy, schema='bounded-composed-BUDDY-engineering-policy-v1', track='bounded_engineering',
                   maximum_epochs=1, minimum_epochs=1, patience=1, source_role='finite_owned_normalization_lifecycle',
                   source_protocol_equivalent=False, source_RNG_trajectory_equivalent=False,
                   fidelity_limitations=[*policy.fidelity_limitations,
                       'One complete epoch for ownership/state/serving qualification; no accuracy or convergence inference'])


def create_selector(policy):
    return original(EXPANSION.parent.parent).create_selector(policy)
