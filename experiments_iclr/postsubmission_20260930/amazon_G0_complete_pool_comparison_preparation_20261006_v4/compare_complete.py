"""Disabled integrated serving/export/A comparison; no fitting or selection.

The original metric/gate/error-flow math is imported by exact source pin.
Only a separately reviewed, immutable root scoring scope admits this caller.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import socket
import sys
import traceback

SOURCE_RELEASED = False
ARMS = ('live', 'uniform', 'margins', 'graph_free', 'permuted', 'stop_q')
G0 = ('initial', *ARMS)
REFERENCES = ('own', 'own_pool', 'ENS4', 'SINGLE', 'CMCL')
CMCL_ARM = 'CMCL_exact_KL_M4_K3_beta075'
CMCL_FIT_OWNER = {'path':'cmcl_graph_common400_H16_owned_fit_preparation_20261006_v2/run_cmcl_H16.py',
    'bytes':34185,'sha256':'35ba09752e94bb5f26540a7559ce227057910f7a5a47fb5c4169d0a45d382b77'}
CMCL_FIT_CALLABLE = {'path':'cmcl_graph_common400_H16_owned_fit_preparation_20261006_v2/cmcl_H16_released.py',
    'bytes':14261,'sha256':'239ee8048808a8d53263d384829dbb9569418702ad6b6d8e39bd1aa71510b774'}
CMCL_FIT_COUNTS = {'native_value_callbacks':64,'native_replay_callbacks':64,'small_logit_grad_APIs':16,
    'native_parameter_VJP_APIs':64,'simultaneous_SGD_attempts':16,'shared_SGD_maps':16,
    'private_SGD_row_maps':64,'completed_updates':16,'serving_callbacks':8}
CMCL_FIT_BILL = {'callable_native_callbacks':136,'serialization_replay_callbacks':4,'owner_native_callbacks':140}
DEVICE = 'cuda:0'
ROLE_IDENTITY = {
    'public_graph_sha256':'19757299bcfd9e493e9ceae9e73248753ab1e773ccc1f6b57c6fadf8c843310f',
    'roles_sha256':'9cab6f2cf24dbecee59a2179e860d78b9b7b995adf0bb5347ca16dce86f32c97',
    'native_edge_logical_sha256':'229a8a787ef9120a4d7a1dcdd6481b973619a1c7f44a4abe9ed69d05c256e550',
    'public_b_manifest_sha256':'da03b6c14615da934443c9b8f36b68f9d643e2b23e04ebe17dab7dacf630df55'}
# Written from the sealed existing-source inventory, never from payloads.
PINS = {'accessor': {'bytes': 19074,
              'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/train_only_accessor.py',
              'sha256': '9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85'},
 'bindings': {'bytes': 15706,
              'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/SOURCE_BINDINGS.json',
              'sha256': 'a324b647f5bad14ececa90a370a742435af427f4b2f7fef488d687306a1f00df'},
 'boundary': {'bytes': 6797,
              'path': 'amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py',
              'sha256': '699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac'},
 'flow': {'bytes': 9451,
          'path': 'corrective_error_flow_common_opponent_preparation_20261006_v3/error_flow.py',
          'sha256': 'f145aabcc83c7aacd45f2c6bde7acfc079e5adbb938beaaaa83d9a7a8a39b1e9'},
 'metrics': {'bytes': 13400,
             'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/held_a_evaluator.py',
             'sha256': '3b1a3cd0b4c36e1bbd4e1601fbd71f9faf868dff408ea4c3937b7c5eefd6a000'},
 'native': {'bytes': 7129,
            'path': 'amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py',
            'sha256': '9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8'},
 'native_reference': {'bytes': 37438,
                      'path': 'amazon_native_single_independent4_allocation_flag_only_release_prospective_admission_20261006_v2/native_reference.py',
                      'sha256': '56ca97c68437d59a74f98733b9399a770bbbce08b67322648f70fe0c82fb3ffc'},
 'ordinary_training': {'bytes': 40364,
                       'path': 'amazon_ordinary_shared_bank_two_gpu_flag_only_release_prospective_admission_20261006_v1/run_one.py',
                       'sha256': '369f03496fd3f597da29df3dd1271747ebb8a889f3dcad8ed4e026d06cc676be'},
 'process': {'bytes': 25151,
             'path': 'amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3/run_scientific.py',
             'sha256': '65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb'},
 'queue': {'bytes': 3912,
           'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/QUEUE.json',
           'sha256': 'f6206a4a8eef663d763ea72950d074e5de810ba4a6b10f6d86c5c7d40bb78464'},
 'worker': {'bytes': 41360,
            'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/six_arm_worker.py',
            'sha256': 'ad28d1c8a05a168aeadb6825183ef8d1d1a6cffea3b488c5a33aa3c90318c8f3'}}
SCOPE_TEMPLATE = {'CMCL': {'RESULT': {'bytes': 2503,
            'path': 'cmcl_graph_common400_H16_owned_fit_execution_root_20261006_v2/fit/RESULT.json',
            'sha256': '989052c51acbace9f960d12cb621d70881cf2ef8dd96c0c5415d8793301b93fc'},
 'ROOT_SCOPE': {'bytes': 9619,
                'path': 'cmcl_graph_common400_H16_owned_fit_activation_root_20261006_v2/ROOT_SCOPE.json',
                'sha256': 'd8d0a9cb54473e9b13faf6b4dc80d8720bb998371246bb5e74e2e6414abcfff9'},
 'endpoint': {'bytes': 36535686,
              'path': 'cmcl_graph_common400_H16_owned_fit_execution_root_20261006_v2/fit/endpoint.pt',
              'sha256': 'b4bf082b4f9a9fcc36396be3d2908114c71960d304226d9e0c776545acedbbe6'},
 'terminal': {'bytes': 3906,
              'path': 'cmcl_graph_common400_H16_owned_fit_execution_root_20261006_v2/TERMINAL.json',
              'sha256': '06f0b3c5bdae83540f72e05ce69f60c75d09b0a072951ec19223bb9e9c85b63b'}},
 'GPU_UUID': 'GPU-44039938-fd82-41d2-fefd-de71514e2fac',
 'VALID_TEST_access': False,
 'all_training_terminal_closed': False,
 'automatic_retry': False,
 'caller_review': {'bytes': None, 'path': None, 'sha256': None},
 'caller_source_review_approved': False,
 'candidate_complete': {'bytes': 19491,
                        'path': 'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/COMPLETE.json',
                        'sha256': '5a85d8736fc683d2671b70eccb9698c9bb7385fbcac3af0c948f922e2a842579'},
 'candidate_terminal': {'bytes': 3020,
                        'path': 'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/monitor_20261006T021633Z/TERMINAL.json',
                        'sha256': '751e1d71334abd29a51c56fa649104eaef6e1422ef0308ef2309ba41a5eb3cd8'},
 'custody': {'bytes': None,
             'path': 'learnability_responsibility_native_full_execution_root_20261005_v1/roles/CUSTODY.json',
             'sha256': None},
 'exclusive_evaluation_window': False,
 'executor': {'bytes': None,
              'path': 'amazon_G0_complete_pool_comparison_preparation_20261006_v4/compare_complete.py',
              'sha256': None},
 'external_owned_supervision_required': True,
 'external_watchdog_seconds': None,
 'fits_authorized': False,
 'host': 'anogena-2-0',
 'native': {'RESULT': {'bytes': 5666,
                       'path': 'allocation_native_single_independent4_scientific_execution_root_20261006_v1/cohort/RESULT.json',
                       'sha256': '683c9b89755a878987a3e55aab0362666824c66000a0d863c014627f3fbb50f3'},
            'RUN': {'bytes': None,
                    'path': 'allocation_native_single_independent4_scientific_execution_root_20261006_v1/cohort/RUN.json',
                    'sha256': None},
            'endpoints': {'0': {'bytes': 109543470,
                                'path': 'allocation_native_single_independent4_scientific_execution_root_20261006_v1/cohort/member0_SR2300.pt',
                                'sha256': '470ba4db675f4c9e997c306fb4ffd61befe3e91fa288b9d53a25e68cb06dbfdc'},
                          '1': {'bytes': 109543342,
                                'path': 'allocation_native_single_independent4_scientific_execution_root_20261006_v1/cohort/member1_SR2300.pt',
                                'sha256': '5f8774155dce56014923d8011ec9918a38b9a2be4cc9ebf659344153cf288139'},
                          '2': {'bytes': 109543214,
                                'path': 'allocation_native_single_independent4_scientific_execution_root_20261006_v1/cohort/member2_SR2300.pt',
                                'sha256': '6536c7327049585e4f77499d3c2db9b04fbbc5d6da06dccfbe54658c45b64eb0'},
                          '3': {'bytes': 109543150,
                                'path': 'allocation_native_single_independent4_scientific_execution_root_20261006_v1/cohort/member3_SR2300.pt',
                                'sha256': '08d8f9808d28a4c0d0b82543b67180b1d3939479364574c718fb90bf30962166'}},
            'supervisor': {'bytes': 12119,
                           'path': 'allocation_native_single_independent4_scientific_execution_root_20261006_v1/supervisor/RESULT.json',
                           'sha256': 'a432e2d79e870b6231739348e1f9c20fc20995cba348fcd69b01687fbb15ee1a'}},
 'ordinary': {'own': {'RESULT': {'bytes': None,
                                 'path': 'amazon_ordinary_shared_bank_two_gpu_scientific_execution_root_20261006_v1/own/RESULT.json',
                                 'sha256': None},
                      'RUN': {'bytes': None,
                              'path': 'amazon_ordinary_shared_bank_two_gpu_scientific_execution_root_20261006_v1/own/RUN.json',
                              'sha256': None},
                      'endpoint': {'bytes': None,
                                   'path': 'amazon_ordinary_shared_bank_two_gpu_scientific_execution_root_20261006_v1/own/own_2300.pt',
                                   'sha256': None}},
              'own_pool': {'RESULT': {'bytes': None,
                                      'path': 'amazon_ordinary_shared_bank_two_gpu_scientific_execution_root_20261006_v1/own_pool/RESULT.json',
                                      'sha256': None},
                           'RUN': {'bytes': None,
                                   'path': 'amazon_ordinary_shared_bank_two_gpu_scientific_execution_root_20261006_v1/own_pool/RUN.json',
                                   'sha256': None},
                           'endpoint': {'bytes': None,
                                        'path': 'amazon_ordinary_shared_bank_two_gpu_scientific_execution_root_20261006_v1/own_pool/own_pool_2300.pt',
                                        'sha256': None}}},
 'ordinary_pair_terminal': {'bytes': None,
                            'path': 'amazon_ordinary_shared_bank_two_gpu_scientific_execution_root_20261006_v1/pair_supervisor/RESULT.json',
                            'sha256': None},
 'origin_run': {'bytes': 6402,
                'path': 'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/RUN.json',
                'sha256': '2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5'},
 'output_relative': 'amazon_G0_complete_pool_comparison_execution_root_20261006_v4',
 'python_executable': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python',
 'python_resolved': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.tools/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11',
 'repository': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs',
 'resource_limits': {'max_cuda_allocated_bytes': None,
                     'max_cuda_reserved_bytes': None,
                     'max_elapsed_seconds': None,
                     'max_process_rss_bytes': None},
 'root_scoring_authorized': False,
 'runtime_identity_expected': {'GPU_capability': [8, 0],
                               'GPU_name': 'NVIDIA A100-SXM4-80GB',
                               'PyG_module_path': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages/torch_geometric/__init__.py',
                               'PyG_version': '2.7.0',
                               'cudnn_version': 8700,
                               'extension_versions': {'pyg-lib': None,
                                                      'torch-scatter': None,
                                                      'torch-sparse': None},
                               'torch_CUDA_version': '11.8',
                               'torch_module_path': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages/torch/__init__.py',
                               'torch_version': '2.1.2+cu118'},
 'schema': 'root_existing_G0_complete_pool_comparison_scope_v4',
 'selection_authorized': False,
 'site_packages': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages',
 'sources': {'accessor': {'bytes': 19074,
                          'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/train_only_accessor.py',
                          'sha256': '9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85'},
             'bindings': {'bytes': 15706,
                          'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/SOURCE_BINDINGS.json',
                          'sha256': 'a324b647f5bad14ececa90a370a742435af427f4b2f7fef488d687306a1f00df'},
             'boundary': {'bytes': 6797,
                          'path': 'amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py',
                          'sha256': '699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac'},
             'flow': {'bytes': 9451,
                      'path': 'corrective_error_flow_common_opponent_preparation_20261006_v3/error_flow.py',
                      'sha256': 'f145aabcc83c7aacd45f2c6bde7acfc079e5adbb938beaaaa83d9a7a8a39b1e9'},
             'metrics': {'bytes': 13400,
                         'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/held_a_evaluator.py',
                         'sha256': '3b1a3cd0b4c36e1bbd4e1601fbd71f9faf868dff408ea4c3937b7c5eefd6a000'},
             'native': {'bytes': 7129,
                        'path': 'amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py',
                        'sha256': '9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8'},
             'native_reference': {'bytes': 37438,
                                  'path': 'amazon_native_single_independent4_allocation_flag_only_release_prospective_admission_20261006_v2/native_reference.py',
                                  'sha256': '56ca97c68437d59a74f98733b9399a770bbbce08b67322648f70fe0c82fb3ffc'},
             'ordinary_training': {'bytes': 40364,
                                   'path': 'amazon_ordinary_shared_bank_two_gpu_flag_only_release_prospective_admission_20261006_v1/run_one.py',
                                   'sha256': '369f03496fd3f597da29df3dd1271747ebb8a889f3dcad8ed4e026d06cc676be'},
             'process': {'bytes': 25151,
                         'path': 'amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3/run_scientific.py',
                         'sha256': '65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb'},
             'queue': {'bytes': 3912,
                       'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/QUEUE.json',
                       'sha256': 'f6206a4a8eef663d763ea72950d074e5de810ba4a6b10f6d86c5c7d40bb78464'},
             'worker': {'bytes': 41360,
                        'path': 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/six_arm_worker.py',
                        'sha256': 'ad28d1c8a05a168aeadb6825183ef8d1d1a6cffea3b488c5a33aa3c90318c8f3'}},
 'strict_backend_expected': {'deterministic_algorithms_enabled': True,
                             'deterministic_algorithms_warn_only': False,
                             'deterministic_debug_mode': 2,
                             'flags': {'cuda_matmul': {'allow_bf16_reduced_precision_reduction': True,
                                                       'allow_fp16_reduced_precision_reduction': True,
                                                       'allow_tf32': False},
                                       'cudnn': {'allow_tf32': True,
                                                 'benchmark': False,
                                                 'deterministic': False,
                                                 'enabled': True}}},
 'torch_module_path': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages/torch/__init__.py'}


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def bound(root, row):
    relative = Path(row['path'])
    require(relative.parts and not relative.is_absolute() and '..' not in relative.parts, 'In-phase descriptor required')
    path = root / relative
    require(path.is_file() and path.resolve().is_relative_to(root), 'Descriptor escapes phase')
    for item in (path, *path.parents):
        if item == root.parent: break
        require(not item.is_symlink(), 'Symlink descriptor refused')
    require(path.stat().st_mode & 0o222 == 0 and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'Immutable descriptor differs: ' + str(relative))
    return path


def same_artifact(a, b):
    return all(a[k] == b[k] for k in ('bytes', 'sha256'))


def load(root, sources, key, names):
    row = sources[key]
    require(same_artifact(row, PINS[key]), 'Exact reused source pin required: ' + key)
    path = bound(root, row); name = '_G0_complete_comparison_' + key
    require(name not in sys.modules, 'Fresh source namespace required')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; names.append(name)
    spec.loader.exec_module(module)
    return module


def artifact(root, parent, row):
    path = parent / row['path']
    return bound(root, dict(row, path=str(path.relative_to(root))))


def authenticate(args):
    root = Path(args.source_root).absolute()
    require(root.resolve() == root and not root.is_symlink() and sys.dont_write_bytecode
            and not any(n == 'torch' or n.startswith('torch.') for n in sys.modules), 'Fresh resolved phase/-B/pre-Torch required')
    scope_path = bound(root, {'path':str(Path(args.scope).absolute().relative_to(root)),
                             'bytes':Path(args.scope).stat().st_size, 'sha256':args.scope_sha256})
    scope = read(scope_path)
    require(scope['schema'] == 'root_existing_G0_complete_pool_comparison_scope_v4'
            and all(scope[k] is True for k in ('root_scoring_authorized', 'caller_source_review_approved',
                'all_training_terminal_closed', 'exclusive_evaluation_window', 'external_owned_supervision_required'))
            and all(scope[k] is False for k in ('fits_authorized', 'selection_authorized', 'automatic_retry', 'VALID_TEST_access')),
            'Explicit reviewed scoring-only root scope required')
    require(socket.gethostname() == scope['host'] and Path.cwd().resolve() == Path(scope['repository']).resolve()
            and root.parent.parent == Path(scope['repository']).resolve()
            and str(Path(sys.executable).absolute()) == scope['python_executable']
            and str(Path(sys.executable).resolve()) == scope['python_resolved']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == scope['GPU_UUID'] and ',' not in scope['GPU_UUID'], 'Exact host/runtime/sole GPU required')
    require(bound(root, scope['executor']).resolve() == Path(__file__).resolve(), 'Exact caller source required')
    bound(root, scope['caller_review'])
    require(set(scope['sources']) == set(PINS), 'Complete existing-source binding map required')
    for key, row in scope['sources'].items():
        require(same_artifact(row, PINS[key]), 'Changed reused source'); bound(root, row)
    caps = scope['resource_limits']
    require(set(caps) == {'max_elapsed_seconds','max_process_rss_bytes','max_cuda_allocated_bytes','max_cuda_reserved_bytes'}
            and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in caps.values())
            and scope['external_watchdog_seconds'] > caps['max_elapsed_seconds'], 'Root-frozen evaluation resources/watchdog required')
    complete_path = bound(root, scope['candidate_complete']); complete = read(complete_path)
    require(complete['schema'] == 'amazon_G0_seven_states_complete_v1' and complete['warm_role'] == 'W'
            and complete['warm_updates'] == 400 and complete['episodes_per_arm'] == 16
            and complete['arms_in_order'] == list(ARMS) and set(complete['endpoints']) == set(ARMS)
            and complete['A_labels_received'] is False and complete['A_scoring_performed'] is False, 'Original seven complete A-closed states required')
    terminal = read(bound(root, scope['candidate_terminal']))
    require(terminal['exit_code'] == 0 and terminal['watchdog_fired'] is False
            and terminal['A_scoring'] is False and terminal['VALID_TEST_access'] is False, 'Actual candidate terminal required')
    recipe = complete['recipe']
    require(recipe['bindings_sha256'] == scope['sources']['bindings']['sha256']
            and recipe['queue_sha256'] == scope['sources']['queue']['sha256']
             and recipe['worker_source_sha256'] == scope['sources']['worker']['sha256'], 'Original candidate recipe differs')
    require(terminal['owned_child']['worker_sha256'] == recipe['worker_source_sha256']
            and any(a['path'].endswith('/COMPLETE.json') and same_artifact(a,scope['candidate_complete'])
                for a in terminal['artifacts']), 'Candidate terminal does not join this COMPLETE')
    states = {n:artifact(root, complete_path.parent, row) for n,row in {'initial':complete['common'], **complete['endpoints']}.items()}
    artifact(root, complete_path.parent, complete['initial_response'])
    origin = read(bound(root, scope['origin_run']))
    require(origin['recipe'] == recipe and origin['A_labels_received'] is False and origin['A_scoring_performed'] is False, 'Common origin exposure differs')
    identity = {k:recipe[k] for k in ('public_graph_sha256','roles_sha256','native_edge_logical_sha256','public_b_manifest_sha256')}
    require(identity == ROLE_IDENTITY, 'Original fixed public graph/role projection required')
    pair = read(bound(root, scope['ordinary_pair_terminal']))
    require(pair['status'] == 'TWO_GPU_ORDINARY_PAIR_COMPLETE_A_CLOSED' and pair['comparison_complete'] is True
            and pair['A_scoring'] is False and pair['VALID_TEST_access'] is False and not pair['cleanup_errors']
            and set(pair['outcomes']) == {'own','own_pool'} and pair['launch_attempts'] == {'own':1,'own_pool':1}
            and len(pair['children']) == 2 and all(c['wait4_closed'] is True and c['child_exit_code'] == 0
                and c['external_timeout'] is False for c in pair['children']), 'Both genuine closed ordinary outcomes required')
    ordinary_runs = {}
    for kind in ('own','own_pool'):
        rows = scope['ordinary'][kind]; outcome = pair['outcomes'][kind]
        require(all(outcome[k] == rows[k] for k in ('RESULT','RUN','endpoint')), 'Ordinary terminal/output join differs')
        result = read(bound(root, rows['RESULT'])); run = read(bound(root, rows['RUN']))
        require(result['status'] == 'ONE_ASSIGNED_ORDINARY_REFERENCE_COMPLETE_A_CLOSED'
                and result['assigned_objective_complete'] is True and result['assigned_objective'] == kind
                and result['A_scoring'] is False and result['VALID_TEST_access'] is False and not result['restoration_errors']
                and same_artifact(result['endpoints'][kind], rows['endpoint']), 'Final ordinary result/exposure differs')
        require(run['input_identity'] == identity and run['A_labels_received'] is False and run['A_scoring'] is False
                and run['VALID_TEST_access'] is False and run['updates_per_objective'] == 2300
                and run['assigned_objective'] == kind and same_artifact(run['common_checkpoint'], complete['common'])
                and same_artifact(run['common_origin_run'], scope['origin_run'])
                and run['source_sha256'] == scope['sources']['ordinary_training']['sha256'], 'Ordinary W/S/R/A role exposure or source differs')
        states[kind] = bound(root, rows['endpoint']); ordinary_runs[kind] = run
    native_result = read(bound(root, scope['native']['RESULT'])); native_run = read(bound(root, scope['native']['RUN']))
    supervised = read(bound(root, scope['native']['supervisor']))
    require(supervised['status'] == 'FRESH_NATIVE_SINGLE_ENS4_COHORT_COMPLETE_A_CLOSED'
            and supervised['worker_result'] == scope['native']['RESULT'] and supervised['wait4_closed'] is True
            and supervised['child_exit_code'] == 0 and supervised['external_whole_child_resource_closure_verified'] is True
            and not supervised['cleanup_errors'], 'Native actual whole-child terminal join required')
    require(native_result['status'] == 'FRESH_NATIVE_SINGLE_INDEPENDENT4_COMPLETE_A_CLOSED'
            and native_result['all_four_complete'] is True and native_result['comparison_complete'] is True
            and native_result['A_scoring'] is False and native_result['VALID_TEST_access'] is False
            and native_result['old_checkpoint_inputs'] is False and native_result['all_W400_frozen_before_SR_decode'] is True
            and native_result['public_and_W_inputs_unchanged'] is True and native_result['public_and_SR_inputs_unchanged'] is True
            and not native_result['restoration_errors'] and set(native_result['endpoints']) == {'0','1','2','3'}, 'Native cohort role exposure is not admissible')
    require(native_run['input_identity'] == identity and native_run['A_labels_received'] is False
            and native_run['A_scoring'] is False and native_run['VALID_TEST_access'] is False
            and native_run['W_local_updates'] == native_run['W_global_updates'] == 200
            and native_run['SR_global_updates'] == 2300 and native_run['S_R_received_only_after_all_W400_freeze'] is True
            and native_run['new_independent_W_acquisitions'] == 4 and native_run['copied_shared_history'] is False
            and native_run['member_seeds'] == [17,1026,2035,3044]
            and native_run['single_alias_member'] == 0 and native_run['source_sha256'] == scope['sources']['native_reference']['sha256'], 'Native recipe/exposure/source differs')
    for m in range(4):
        row = scope['native']['endpoints'][str(m)]
        require(same_artifact(row, native_result['endpoints'][str(m)]), 'Native endpoint join differs')
        states['native'+str(m)] = bound(root, row)
    require(set(scope['native']['endpoints']) == {'0','1','2','3'}, 'Exactly four native endpoints required')
    cmcl_rows = scope['CMCL']
    cmcl_scope = read(bound(root,cmcl_rows['ROOT_SCOPE']))
    cmcl_result_path = bound(root,cmcl_rows['RESULT']); cmcl_result = read(cmcl_result_path)
    cmcl_terminal = read(bound(root,cmcl_rows['terminal']))
    require(cmcl_scope['schema'] == 'root_owned_CMCL_H16_fit_scope_v1'
            and all(cmcl_scope[k] is True for k in ('root_owned_fit_authorized','owner_source_review_approved','fixed_before_fit','exclusive_GPU_window'))
            and cmcl_scope['A_VALID_TEST_scoring'] is False and cmcl_scope['automatic_retry'] is False
            and same_artifact(cmcl_scope['executor'],CMCL_FIT_OWNER)
            and same_artifact(cmcl_scope['sources']['h16'],CMCL_FIT_CALLABLE)
            and cmcl_scope['planned_native_callback_bill'] == CMCL_FIT_BILL, 'Actual reviewed CMCL fit scope/source required')
    bound(root,cmcl_scope['executor']); bound(root,cmcl_scope['sources']['h16']); bound(root,cmcl_scope['owner_review'])
    context = cmcl_scope['context_admission']
    require(context['schema'] == 'root_common400_CMCL_H16_context_admission_v1' and context['arm'] == CMCL_ARM
            and all(context[k] is True for k in ('root_fit_authorized','fixed_before_fit','caller_source_review_approved',
                'CPU_fixture_PASS_verified','new_native_CMCL_first_order_PARITY_PASS_verified','native_context_verified','one_fixed_outcome_no_retry'))
            and same_artifact(context['common400'],complete['common']) and same_artifact(context['origin_run'],scope['origin_run'])
            and context['H'] == 16 and context['M_K_beta'] == [4,3,0.75] and context['eta_core_private'] == [0.001,0.01]
            and context['A_VALID_TEST_scoring'] is False and context['optional_feature_sharing'] is False, 'Same W400 CMCL recipe/roles required')
    require(cmcl_terminal['status'] == 'CMCL_H16_FIT_COMPLETE_OWNED_WHOLE_CHILD_CLOSED'
            and cmcl_terminal['owned_fit_resource_closure_complete'] is True and cmcl_terminal['launch_attempts'] == 1
            and not cmcl_terminal['cleanup_errors'] and cmcl_terminal['A_VALID_TEST_scoring'] is False
            and cmcl_terminal['automatic_retry'] is False and cmcl_terminal['scope_sha256'] == cmcl_rows['ROOT_SCOPE']['sha256']
            and same_artifact(cmcl_terminal['worker_result'],cmcl_rows['RESULT'])
            and len(cmcl_terminal['children']) == 1 and cmcl_terminal['children'][0]['wait4_closed'] is True
            and cmcl_terminal['children'][0]['child_exit_code'] == 0 and cmcl_terminal['children'][0]['external_timeout'] is False,
            'Actual CMCL terminal/worker/reaping closure required')
    require(cmcl_result['status'] == 'CMCL_H16_COMPLETE_A_CLOSED_RESOURCE_ONLY'
            and cmcl_result['scope_sha256'] == cmcl_rows['ROOT_SCOPE']['sha256'] and cmcl_result['counts'] == CMCL_FIT_COUNTS
            and cmcl_result['last_completed_update'] == 16 and cmcl_result['model_fits'] == cmcl_result['fit_invocations'] == 1
            and cmcl_result['native_callback_attempts'] == 136 and cmcl_result['serialization_replay_callback_attempts'] == 4
            and cmcl_result['owner_native_callback_attempts'] == 140 and cmcl_result['planned_native_callback_bill'] == CMCL_FIT_BILL
            and cmcl_result['A_VALID_TEST_scoring'] is False and cmcl_result['optional_feature_sharing'] is False
            and cmcl_result['initial_family_state_exact'] is True
            and not cmcl_result['restoration_errors'] and not cmcl_result['callback_finalization_errors']
            and not cmcl_result['custody_errors'] and set(cmcl_result['custody_checks']) ==
                {'original_functional_state_and_inputs','original_native_custody','original_common_file'}
            and all(cmcl_result['custody_checks'].values())
            and all(cmcl_result['serialization_checks'][k] is True for k in
                ('endpoint_family_state_exact','fresh_native_family','installed_family_state_exact',
                 'all_four_complete_native_logits_match','mean_softmax_probabilities_match')),
            'Complete sixteen-update CMCL saved replay/custody evidence required')
    require(same_artifact(cmcl_result['artifacts']['endpoint'],cmcl_rows['endpoint']), 'CMCL endpoint/result join differs')
    states['CMCL'] = bound(root,cmcl_rows['endpoint'])
    custody_path = bound(root, scope['custody']); custody = read(custody_path)
    require(custody['schema'] == 'amazon_G0_TRAIN_label_custody_v2'
            and custody['public_b_manifest']['sha256'] == identity['public_b_manifest_sha256']
            and custody['A_labels']['path'] == 'evaluator_a/A_LABELS.npz', 'Original A/public role custody required')
    artifact(root, custody_path.parent, custody['public_b_manifest'])
    # A_LABELS is deliberately neither hashed nor opened here.
    return root, scope, complete, states, ordinary_runs, native_run, custody_path, custody


def run(args, context, output, save, receipt):
    root,scope,complete,states,ordinary_runs,native_run,custody_path,custody = context
    names=[]; torch=None; backend=None; old_threads=None; rng=None; old_path=list(sys.path)
    old_python=random.getstate(); old_env=('CUBLAS_WORKSPACE_CONFIG' in os.environ,os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    old_signal=signal.getsignal(signal.SIGALRM); old_timer=signal.getitimer(signal.ITIMER_REAL); body_error=None
    require(old_timer == (0.0,0.0), 'Fresh evaluation child timer required')
    def resources():
        row={'elapsed_seconds':time.monotonic()-STARTED,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             'cuda_peak_allocated_bytes':0,'cuda_peak_reserved_bytes':0}
        if torch is not None and torch.cuda.is_initialized():
            row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(DEVICE))
        return row
    def limits():
        row=resources(); receipt['whole_process_resources']=row
        require(all(row[k] <= scope['resource_limits'][c] for k,c in [('elapsed_seconds','max_elapsed_seconds'),('process_peak_rss_bytes','max_process_rss_bytes'),
            ('cuda_peak_allocated_bytes','max_cuda_allocated_bytes'),('cuda_peak_reserved_bytes','max_cuda_reserved_bytes')]), 'Evaluation whole-child cap exceeded')
    try:
        def expired(number,frame): raise TimeoutError('Root-frozen evaluation deadline exceeded; no retry')
        signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,max(0.001,scope['resource_limits']['max_elapsed_seconds']-(time.monotonic()-STARTED)))
        modules={k:load(root,scope['sources'],k,names) for k in ('worker','metrics','flow','native_reference','accessor','process')}
        require(modules['metrics'].SOURCE_RELEASED is False and modules['flow'].SOURCE_RELEASED is False, 'Preserved source libraries required')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; sys.path.insert(0,scope['site_packages'])
        import torch
        import numpy as np
        require(str(Path(torch.__file__).resolve()) == scope['torch_module_path'] and not torch.cuda.is_initialized(), 'Pinned pre-CUDA Torch required')
        process=modules['process']; process.torch=torch; backend=process.backend_snapshot(); old_threads=torch.get_num_threads()
        torch.use_deterministic_algorithms(True,warn_only=False); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(process.backend_snapshot() == scope['strict_backend_expected'], 'Qualified strict backend differs')
        require(process.runtime_identity() == scope['runtime_identity_expected'] and torch.cuda.device_count() == 1, 'Qualified native runtime differs')
        rng=(np.random.get_state(),torch.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()])
        native=load(root,scope['sources'],'native',names); boundary=load(root,scope['sources'],'boundary',names)
        worker,metrics,flow,nref,accessor=(modules[k] for k in ('worker','metrics','flow','native_reference','accessor'))
        require(custody['visibility'] == accessor.VISIBILITY, 'Original custody visibility declaration differs')
        data=accessor.load_public_b(root,custody_path.parent/'public_b',device=DEVICE)
        recipe=complete['recipe']; provenance=data['provenance']
        require(all(provenance[k]['sha256'] == recipe[v] for k,v in [('public_b_manifest','public_b_manifest_sha256'),('roles','roles_sha256'),('public_graph','public_graph_sha256')])
                and provenance['preprocessing']['edge_logical_sha256'] == recipe['native_edge_logical_sha256'], 'Original public graph and role IDs differ')
        ids=data['A_ids']; require(ids.numel() == 2449 and data['W_ids'].numel() == 4898
            and data['inner_indices'].numel() == 2449 and data['query_indices'].numel() == 2450, 'Original role cardinalities required')
        require(not bool(torch.isin(ids,torch.cat((data['W_ids'],data['inner_indices'],data['query_indices']))).any()), 'Held A overlaps training exposure')
        family,_=worker._fresh_family(native,boundary,DEVICE); expected=family.state_dict()
        def tensor_bank(values,expected):
            require(set(values) == set(expected), 'Complete native parameter/buffer bank required')
            require(all(isinstance(v,torch.Tensor) and v.dtype == e.dtype and v.shape == e.shape and v.device.type == 'cpu'
                and bool(torch.isfinite(v).all()) for k,v in values.items() for e in [expected[k]]), 'State tensor schema/finite values differ')
        def family_values(name):
            image=torch.load(states[name],map_location='cpu',weights_only=True)
            if name in G0:
                require(image['schema'] == 'amazon_G0_frozen_state_v1' and image['id'] == name and image['recipe'] == recipe
                    and image['warm_role'] == 'W' and image['warm_updates'] == 400 and image['episodes'] == (0 if name == 'initial' else 16)
                    and image['global_stage'] is True and image['eval_mode'] is True
                    and (name == 'initial' or image['common_state'] == complete['common']), 'Exact original G0 state required')
            elif name == 'CMCL':
                require(image['schema'] == 'common400_CMCL_H16_frozen_state_v1' and image['arm'] == CMCL_ARM
                    and image['completed_updates'] == image['H'] == 16 and image['M_K_beta'] == [4,3,0.75]
                    and image['eta_core_private'] == [0.001,0.01] and image['seed'] == 17 and image['split'] == 0
                    and same_artifact(image['common400'],complete['common']) and same_artifact(image['origin_run'],scope['origin_run'])
                    and image['recipe'] == recipe and image['global_stage'] is True and image['eval_mode'] is True
                    and image['A_VALID_TEST_scoring'] is False and image['W_loss_during_continuation'] is False
                    and image['fit_scope_sha256'] == scope['CMCL']['ROOT_SCOPE']['sha256'], 'Genuine closed CMCL H16 endpoint required')
            else:
                # Producer PINS has five tuple pairs; RUN.json encodes those pairs as lists.
                native_pins = image['recipe']['native_source_pins']
                require(type(native_pins) is dict and set(native_pins) == {'scientific_helpers','native','boundary','accessor','process_helpers'}
                    and all(type(v) is tuple and len(v) == 2 and all(type(x) is str for x in v) for v in native_pins.values()),
                    'Exact ordinary producer source-pin tuple pairs required')
                recipe_json = dict(image['recipe'], native_source_pins={k:list(v) for k,v in native_pins.items()})
                require(image['schema'] == 'amazon_fixed_ordinary_reference_state_v1' and image['objective'] == name
                    and image['update'] == 2300 and image['competence_endpoint'] is True and image['diagnostic_prefix_only'] is False
                    and recipe_json == ordinary_runs[name] and image['global_stage'] is True and image['eval_mode'] is True, 'Genuine final ordinary endpoint required')
            tensor_bank(image['family_state'],expected); return image['family_state']
        def native_values(member,model):
            image=torch.load(states['native'+str(member)],map_location='cpu',weights_only=True)
            # The native producer stores one tuple pair; its RUN.json stores a list.
            source_pin = image['recipe']['native_source_pin']
            require(type(source_pin) is tuple and len(source_pin) == 2 and all(type(x) is str for x in source_pin),
                'Exact native producer source-pin tuple pair required')
            recipe_json = dict(image['recipe'], native_source_pin=list(source_pin))
            require(image['schema'] == 'fresh_native_W400_SR2300_reference_state_v1' and image['member'] == member
                and image['seed'] == nref.MEMBER_SEEDS[member] and image['kind'] == 'SR2300' and recipe_json == native_run
                and image['W_updates'] == 400 and image['SR_updates'] == 2300 and image['absolute_update'] == 2700
                and image['last_fixed_horizon'] is True and image['A_labels_received'] is False and image['VALID_TEST_access'] is False
                and image['state']['global_stage'] is True and image['state']['eval_mode'] is True, 'Native full-horizon/exposure state differs')
            tensor_bank(image['state']['model'],model.state_dict()); return image['state']['model']
        # Complete semantic validation before any serving, and before A label access.
        for name in (*G0,'own','own_pool','CMCL'): family_values(name); limits()
        for m in range(4):
            model,optimizer,_=nref.fresh(native,worker,nref.MEMBER_SEEDS[m],DEVICE)
            native_values(m,model); del model,optimizer; limits()
        predictions={}; rows={}; receipt['completed_served_member_forwards']=0
        def retain(name,logits,members,pool):
            value={'A_ids':ids.cpu().clone(),'native_FP32_logits':logits.cpu().clone(),
                'served_FP32_member_probabilities':members.cpu().clone(),'served_FP32_pool_probabilities':pool.cpu().clone()}
            rows[name]=worker._save_state(output/(name+'_A_predictions.pt'),value); predictions[name]=value; save(); limits()
        for name in (*G0,'own','own_pool','CMCL'):
            family.load_state_dict(family_values(name),strict=True); family.set_global_stage(True); family.eval(); limits()
            with torch.no_grad():
                z=family(data['features'],data['edge_index']); require(z.dtype == torch.float32 and z.shape == (4,24492,5) and bool(torch.isfinite(z).all()), 'Full native family serving failed')
                za=z.index_select(1,ids); p=torch.softmax(za,dim=-1); pool=p.mean(dim=0)
            receipt['completed_served_member_forwards']+=4; retain(name,za,p,pool); del z,za,p,pool
        del family,expected
        logits=[]
        for m in range(4):
            model,optimizer,_=nref.fresh(native,worker,nref.MEMBER_SEEDS[m],DEVICE)
            model.load_state_dict(native_values(m,model),strict=True); model._global=True; model.eval(); limits()
            with torch.no_grad():
                z=model(data['features'],data['edge_index']); require(z.dtype == torch.float32 and z.shape == (24492,5) and bool(torch.isfinite(z).all()), 'Complete native reference serving failed')
            logits.append(z); receipt['completed_served_member_forwards']+=1; del model,optimizer
        served=nref.served_probabilities(logits); stack=torch.stack(logits); members=torch.stack([torch.softmax(z,dim=-1) for z in logits])
        retain('ENS4',stack.index_select(1,ids),members.index_select(1,ids),served['ENS4'].index_select(0,ids))
        retain('SINGLE',logits[0].index_select(0,ids),members[0].index_select(0,ids),served['SINGLE'].index_select(0,ids))
        del logits,stack,members,served
        require(len(rows) == 12 and set(rows) == set(G0+REFERENCES) and receipt['completed_served_member_forwards'] == 44, 'All complete predictions before A required')
        worker._write(output/'PREDICTIONS_COMPLETE.json',{'predictions':rows,'A_labels_opened':False,'served_member_forwards':44,
            'role_exposure_equal_by_exact_public_role_hashes':True,'states':scope['candidate_complete'],'ordinary_pair':scope['ordinary_pair_terminal'],'native':scope['native']['RESULT'],'CMCL':scope['CMCL'],'prediction_artifacts':12})
        for row in scope['sources'].values(): bound(root,row)
        for name,path in states.items():
            require(path.stat().st_mode & 0o222 == 0, 'State no longer immutable')
        # First A label artifact hash/read occurs only after every prediction is frozen.
        a_path=artifact(root,custody_path.parent,custody['A_labels'])
        labels=torch.tensor(accessor._read_compact(np,a_path,ids.cpu().tolist()).tolist(),dtype=torch.long)
        require(labels.shape == (2449,), 'Complete original A population required'); receipt['A_labels_opened']=True; save()
        scores={n:metrics._score(v['native_FP32_logits'],v['served_FP32_member_probabilities'],v['served_FP32_pool_probabilities'],labels)
            for n,v in predictions.items() if n != 'SINGLE'}
        single=predictions['SINGLE']; single_log=torch.log_softmax(single['native_FP32_logits'].double(),dim=-1)
        scores['SINGLE']={'pool':metrics._metrics(single['served_FP32_pool_probabilities'],single_log,labels),
            'class_cells':[{'class':c,'n':int((labels==c).sum()),'pool':metrics._metrics(
                single['served_FP32_pool_probabilities'][labels==c],single_log[labels==c],labels[labels==c])} for c in range(5)]}
        original_scores={n:scores[n] for n in G0}; gate=metrics._gate(original_scores)
        reference_differences={n:{'accuracy':scores['live']['pool']['accuracy']-scores[n]['pool']['accuracy'],
            'NLL':scores[n]['pool']['NLL']-scores['live']['pool']['NLL'],'Brier':scores[n]['pool']['Brier']-scores['live']['pool']['Brier']} for n in REFERENCES}
        worker._write(output/'SCORES.json',{'schema':'existing_G0_complete_pool_comparison_scores_v1','scores':scores,'original_seven_state_gate':gate,
            'live_minus_reference_improvement':reference_differences,'reference_acceptance_threshold_added':False,'predictions':rows,'A_labels':custody['A_labels'],
            'no_fitting_or_selection':True,'different_training_horizons_disclosed':{'G0':16,'CMCL_H16_SGD':16,'ordinary_SR':2300,'native_W':400,'native_SR':2300},
            'optimizer_mismatch_disclosed':{'CMCL':'simultaneous SGD core/private0.001/0.01,H16',
                'ordinary_and_native':'Adam,S/R2300','equivalent_optimization_or_compute_claimed':False}})
        edge=data['edge_index']; edge=edge[:,edge[0]!=edge[1]]; unique=torch.unique(edge[0]*24492+edge[1])
        degrees=torch.bincount(unique//24492,minlength=24492).index_select(0,ids).cpu().tolist()
        # Preserve the existing initial-to-each-G0-endpoint descriptive comparisons.
        flows={n:flow._analyze(predictions['initial'],predictions[n],labels.tolist(),ids.cpu().tolist(),degrees) for n in ARMS}
        worker._write(output/'ERROR_FLOW.json',{'schema':'existing_G0_six_fixed_error_flows_v1','initial':'initial','endpoints':flows,
            'degree':'complete public unique undirected nonself degree','new_gate_or_selector':False})
        receipt.update(original_G0_metric_gate_pass=gate['fixed_metric_gate_pass'],evaluation_outputs_complete=True,model_fits=0,persistent_updates=0)
        limits()
    except BaseException as error:
        body_error=error; receipt['body_error']={'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}; raise
    finally:
        errors=receipt.setdefault('restoration_errors',[])
        def restore(label,call):
            try: call()
            except BaseException: errors.append({'stage':label,'traceback':traceback.format_exc()})
        restore('cancel_timer',lambda:signal.setitimer(signal.ITIMER_REAL,0))
        if rng is not None:
            def restore_rng():
                np.random.set_state(rng[0]); torch.set_rng_state(rng[1]); torch.cuda.set_rng_state_all(rng[2])
            restore('numeric_RNG',restore_rng)
        if backend is not None: restore('backend',lambda:process.backend_restore(backend))
        if old_threads is not None: restore('intra_op_threads',lambda:torch.set_num_threads(old_threads))
        receipt['interop_restoration']='One-time fresh-child initialization; ends with child termination.'
        restore('python_rng',lambda:random.setstate(old_python)); restore('path',lambda:sys.path.__setitem__(slice(None),old_path))
        def environment():
            if old_env[0]: os.environ['CUBLAS_WORKSPACE_CONFIG']=old_env[1]
            else: os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
        restore('environment',environment)
        restore('signal_timer',lambda:(signal.signal(signal.SIGALRM,old_signal),signal.setitimer(signal.ITIMER_REAL,*old_timer)))
        for name in names: sys.modules.pop(name,None)
        restore('terminal_resources',limits); restore('receipt',save)
        if body_error is None: require(not errors,'Evaluation restoration/resource/publication failure')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--execute-authorized',action='store_true')
    for name in ('source-root','scope','scope-sha256','output'): parser.add_argument('--'+name)
    args=parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({'status':'DISABLED_G0_COMPLETE_POOL_COMPARISON','SOURCE_RELEASED':False})); return 0
    require(SOURCE_RELEASED is False and all((args.source_root,args.scope,args.scope_sha256,args.output)), 'Explicit root-admitted engineering scorer required')
    context=authenticate(args); root=context[0]; output=Path(args.output).absolute()
    require(output == root/context[1]['output_relative'] and output.resolve().is_relative_to(root) and not output.exists() and not output.is_symlink()
            and output.parent.is_dir() and output.parent.resolve().is_relative_to(root),'Fresh deliberate in-phase output required')
    output.mkdir(mode=0o700); identity=(output.stat().st_dev,output.stat().st_ino)
    receipt={'status':'RUNNING_COMPLETE_POOL_EVALUATION','scope_sha256':args.scope_sha256,'worker_sha256':sha(__file__),
             'A_labels_opened':False,'model_fits':0,'persistent_updates':0,'automatic_retry':False,'restoration_errors':[]}
    def save():
        require(not output.is_symlink() and (output.stat().st_dev,output.stat().st_ino) == identity,'Created output ownership differs')
        temporary=output/'RESULT.tmp'; require(not temporary.exists() and not temporary.is_symlink(),'Receipt temporary already exists')
        temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n'); os.replace(temporary,output/'RESULT.json')
    code=1
    try:
        save(); run(args,context,output,save,receipt); receipt['status']='COMPLETE_POOL_COMPARISON_SCORED_RESOURCE_ONLY'; code=0
    except BaseException as error:
        receipt.update(status='FAIL_COMPLETE_POOL_COMPARISON',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        save(); (output/'RESULT.json').chmod(0o444)
    print(json.dumps({'status':receipt['status'],'original_G0_metric_gate_pass':receipt.get('original_G0_metric_gate_pass'),'error':receipt.get('error')}))
    return code


if __name__ == '__main__': raise SystemExit(main())
