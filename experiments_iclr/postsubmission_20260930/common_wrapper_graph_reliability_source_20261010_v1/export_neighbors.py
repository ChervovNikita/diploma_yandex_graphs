"""Root-called one-pass native selected export; no fit, owner or retry loop."""
from pathlib import Path
import time

from support import (ARMS, ARCHIVE_KEYS, ATOL, RTOL, PHASE, closed_family,
                     load_contract, load_source, new_output, require, root_runtime, sha, write_json, write_progress)


def neighbor_operator(torch, edges, nodes):
    """Unique incoming nonself row mean; isolates use their own prediction."""
    edge = edges.detach().cpu()
    edge = torch.unique(edge[:, edge[0] != edge[1]], dim=1)
    source, target = edge[0], edge[1]
    degree = torch.bincount(target, minlength=nodes)
    isolate = torch.where(degree == 0)[0]
    indices = torch.cat((torch.stack((target, source)), torch.stack((isolate, isolate))), dim=1)
    weights = torch.cat((degree[target].to(torch.float64).reciprocal(),
                         torch.ones(len(isolate), dtype=torch.float64)))
    return torch.sparse_coo_tensor(indices, weights, (nodes, nodes)).coalesce(), len(isolate)


def run(support_path, output_path):
    """Only root executes this callable after source/support and cost admission."""
    root_runtime()
    contract = load_contract(support_path)
    contexts = [(row, *closed_family(row)) for row in contract['families']]
    output = new_output(output_path)
    write_json(output / 'SUPPORT.json', contract)
    import numpy as np
    import torch
    require(torch.__version__ == '2.1.2+cu118', 'Use the qualified native runtime')
    torch.set_num_threads(2)
    started, records, native_calls, native_trajectories = time.perf_counter(), [], 0, 0
    write_progress(output / 'PROGRESS.json', {'banks_completed': 0, 'selected_model_forward_calls': 0,
                                             'current_key': None, 'elapsed_seconds': 0.0})
    for row, cfg, index in contexts:
        source = load_source('_reliability_native_' + row['backbone'], row['family_source'])
        scratch = output / ('native_restore_' + row['backbone'])
        scratch.mkdir()
        family = source.Family(cfg, scratch, row['config_sha256'])
        nodes = len(family.data.x)
        require(nodes == 11701, 'Exact authorized full-node support')
        all_ids = torch.arange(nodes, device=family.device)
        P, isolates = neighbor_operator(torch, family.data.edge_index, nodes)
        graph_sha = sha(Path(cfg['train_npz']))
        for bank in row['banks']:
            bank_start = time.perf_counter()
            units = index[bank['arm'], bank['seed']]['fits']
            four_shared = bank['arm'] == 'shared4_unchanged'
            factorized = bank['arm'].startswith('factorized') or four_shared
            rows, states = [], []
            for unit, checkpoint_name in zip(units, bank['checkpoints']):
                checkpoint = Path(checkpoint_name)
                digest_before = sha(checkpoint)
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                require(set(state) == {'model', 'optimizer', 'streams', 'step', 'selection'}
                        and state['step'] == unit['selected_step'], 'Exact selected native state layout/step')
                model, unused_optimizer, unused_streams = family.make(
                    bank['seed'], unit['member'], 'baseline', factorized, 4 if four_shared else 1)
                del unused_optimizer, unused_streams
                model.load_state_dict(state['model'], strict=True)
                model.requires_grad_(False)
                model.eval()
                family.synchronize()
                forward_start = time.perf_counter()
                with torch.no_grad():
                    full = family.logits(model, state['streams'], four_shared, all_ids).detach().cpu()
                family.synchronize()
                forward_seconds = time.perf_counter() - forward_start
                require(full.dtype == torch.float32 and full.shape == (4 if four_shared else 1, 11701, 10)
                        and torch.isfinite(full).all(), 'Finite exact full-node native logit layout')
                require(sha(checkpoint) == digest_before, 'Selected checkpoint changed during export')
                states.append({'path': str(checkpoint), 'sha256': digest_before,
                               'selected_step': state['step'], 'forward_seconds': forward_seconds})
                rows.append(full)
                native_calls += 1
                native_trajectories += full.shape[0]
                del model, state
                if family.device.type == 'cuda':
                    torch.cuda.empty_cache()
            full = torch.cat(rows, dim=0)
            archive = Path(bank['archive'])
            require(sha(archive) == bank['archive_sha256'], 'Authoritative VALID archive changed')
            with np.load(archive, allow_pickle=False) as existing:
                require(set(existing.files) == ARCHIVE_KEYS, 'Original VALID archive schema')
                valid_ids = existing['ids'].copy()
                own_raw = existing['raw_logits'].copy()
            require(valid_ids.dtype == np.int64 and valid_ids.shape == (5274,)
                    and np.array_equal(valid_ids, family.valid_ids.cpu().numpy())
                    and own_raw.dtype == np.float32 and own_raw.shape == (bank['members'], 5274, 10),
                    'Exact VALID IDs and archive layout')
            exported_valid = full[:, torch.from_numpy(valid_ids)].to(torch.float64)
            authoritative = torch.from_numpy(own_raw).to(torch.float64)
            error = (exported_valid - authoritative).abs()
            scale = ATOL + RTOL * authoritative.abs()
            within = bool((error <= scale).all())
            discrepancies = {'atol': ATOL, 'rtol': RTOL, 'within_fixed_tolerance': within,
                             'max_abs_logit_difference': error.max().item(),
                             'max_tolerance_scaled_difference': (error / scale).max().item(),
                             'member_argmax_disagreements': exported_valid.argmax(-1).ne(authoritative.argmax(-1)).sum().item(),
                             'member_node_pairs': bank['members'] * 5274,
                             'quality_labels_used': False, 'bitwise_equivalence_required': False,
                             'replay_or_mismatch_investigation_loop': False}
            # The authoritative archive always supplies own-node probabilities.
            # Full-node exported probabilities supply ONLY neighbor context.
            probability = full.to(torch.float64).log_softmax(-1).exp()
            flat = probability.permute(1, 0, 2).reshape(nodes, -1)
            neighbor = torch.sparse.mm(P, flat).reshape(nodes, bank['members'], 10).permute(1, 0, 2)
            valid_neighbor = neighbor[:, torch.from_numpy(valid_ids)].contiguous()
            require(torch.isfinite(valid_neighbor).all() and (valid_neighbor >= 0).all()
                    and torch.allclose(valid_neighbor.sum(-1), torch.ones_like(valid_neighbor[..., 0]), atol=1e-12, rtol=1e-12),
                    'Finite normalized label-free neighbor summaries')
            cache_path = output / (bank['key'] + '.npz')
            np.savez_compressed(cache_path, ids=valid_ids, neighbor_probability_mean=valid_neighbor.numpy())
            record = {'key': bank['key'], 'backbone': row['backbone'], 'arm': bank['arm'], 'seed': bank['seed'],
                      'members': bank['members'], 'archive': str(archive), 'archive_sha256': bank['archive_sha256'],
                      'cache': str(cache_path), 'cache_sha256': sha(cache_path), 'cache_bytes': cache_path.stat().st_size,
                      'selected_states': states, 'support_train_container_sha256': graph_sha,
                      'isolates': isolates, 'discrepancies': discrepancies, 'seconds': time.perf_counter() - bank_start,
                      'own_VALID_authority': 'existing_selected_VALID_raw_logits',
                      'persisted_keys': ['ids', 'neighbor_probability_mean'], 'full_node_labels_or_TEST_quality_exported': False}
            write_json(output / (bank['key'] + '.json'), record)
            records.append(record)
            require(within, 'Single export discrepancy exceeds fixed tolerance; preserve artifacts and return to root, no replay loop')
            write_progress(output / 'PROGRESS.json', {'banks_completed': len(records),
                                                     'selected_model_forward_calls': native_calls,
                                                     'current_key': bank['key'],
                                                     'elapsed_seconds': time.perf_counter() - started})
        del family, P
    require(len(records) == 45 and native_calls == 99 and native_trajectories == 126, 'Complete fixed selected export')
    result = {'schema': 'common-wrapper-reliability-neighbors-v1', 'complete': True, 'banks': records,
              'support_sha256': sha(support_path), 'selected_model_forward_calls': native_calls,
              'native_fullgraph_member_trajectories': native_trajectories, 'new_base_fits': 0,
              'seconds': time.perf_counter() - started, 'TEST_access': False,
              'labels_exported': False, 'fullnode_probabilities_persisted': False,
              'interpretation': 'One selected-state support export; discrepancy tolerance is engineering, never a scientific utility gate.'}
    write_json(output / 'COMPLETE_EXPORT.json', result)
    return {'complete': True, 'banks': 45, 'selected_model_forward_calls': 99,
            'native_fullgraph_member_trajectories': 126, 'TEST_access': False}
