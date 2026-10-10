"""Disabled I4 own-selected bank assembly; no data decoder, score or forward call."""
import argparse
from pathlib import Path
import resource
import sys
import time

import common as g
import train as reference_train


def reconstruct_bank(state, device='cpu'):
    g.require(state['schema'] == 'matched-native-I4-own-selected-bank-v1'
              and state['evaluation_only'] is True and len(state['selected_bodies']) == 4,
              'Four independent selected whole native bodies')
    sessions = [reference_train.reconstruct_selected(row['state'], device) for row in state['selected_bodies']]
    specs = [row['state']['matched_reference']['spec'] for row in state['selected_bodies']]
    g.require([spec['member_index'] for spec in specs] == list(range(4))
              and len({spec['paired_seed'] for spec in specs}) == 1
              and all(spec['family'] == 'native_independent4' for spec in specs), 'Prescribed coherent bank roster')
    parameter_sets = [{id(parameter) for parameter in session.model.parameters()} for session in sessions]
    optimizer_sets = [{id(parameter) for group in session.optimizers[0].param_groups for parameter in group['params']}
                      for session in sessions]
    g.require(all(parameter_sets[i] == optimizer_sets[i] for i in range(4))
              and all(not parameter_sets[i] & parameter_sets[j] for i in range(4) for j in range(i+1, 4)),
              'Four wholly private native bodies/fresh private optimizer owners; no histories restored')
    return sessions


def bank_forward(sessions, batch):
    """Four actual own-selected native calls; fixed probability mean, no router."""
    g.require(len(sessions) == 4 and all(not session.model.training for session in sessions),
              'Exactly four serving-only selected native bodies')
    torch = sessions[0].torch
    with torch.no_grad():
        outputs = [session.forward(batch) for session in sessions]
        logits = torch.stack([row[0][0] for row in outputs], dim=0)
        g.require(logits.ndim == 3 and logits.shape[0] == 4 and logits.shape[-1] == 10
                  and torch.isfinite(logits).all(), 'Finite actual four-member native logits')
        member_log_probabilities = logits.log_softmax(-1)
        served_log_probabilities = torch.logsumexp(member_log_probabilities, dim=0) - logits.new_tensor(4.).log()
        return dict(member_logits=logits, served_probabilities=logits.softmax(-1).mean(0),
            member_log_probabilities=member_log_probabilities,
            served_log_probabilities=served_log_probabilities, model_forwards=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); began = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF)
    release, pins = g.admission(args.release, args.release_sha256, assembly=True)
    g.require(release['schema'] == 'matched-native-reference-I4-assembly-release-v1'
              and release['paired_seed'] in g.SEEDS and len(release['bodies']) == 4,
              'One fixed own-selected ordinary bank')
    closure = g.read(g.bound(release['whole_reference_terminal_custody']))
    g.require(closure['complete'] is True and closure['all15_native_bodies_closed'] is True
              and closure['all6_reference_groups_retained'] is True, 'Whole fresh reference acquisition closure')
    expected = 'combination_matched_native_reference_execution_root_20261010_v1/seed%d/native_independent4/selected_bank' % release['paired_seed']
    g.require(release['output'] == expected, 'Fixed independent bank output')
    output = g.inside(expected); g.require(not output.exists() and output.parent.is_dir(), 'Fresh assembly, no bank reselection')
    output.mkdir(mode=0o700); failure = None
    try:
        import torch
        selected = []
        for member, record in enumerate(release['bodies']):
            terminal = g.read(g.bound(record['terminal']))
            complete = g.read(g.bound(record['completion'])); g.bound(record['cost'])
            g.require(terminal['actual_direct_wait'] is True and terminal['actual_exit_code'] == 0
                      and terminal['child_absent'] is True and terminal['child_no_CUDA_rows'] is True,
                      'Own complete native body direct-wait/absence/CUDA custody')
            spec = dict(family='native_independent4', paired_seed=release['paired_seed'], member_index=member,
                        body_seed=release['paired_seed']+1009*member)
            g.require(complete['complete'] is True and complete['epochs'] == complete['steps'] == 1100
                      and complete['matched_reference']['spec'] == spec
                      and complete['matched_reference']['source_manifest_sha256'] == g.sha(g.HERE/'SOURCE_MANIFEST.json')
                      and complete['selected_sha256'] == record['selected_checkpoint']['sha256'],
                      'Exact complete own native trajectory, no shared-bank member splice')
            state = torch.load(g.bound(record['selected_checkpoint']), map_location='cpu', weights_only=False)
            g.require(state['matched_reference']['spec'] == spec and type(state['global']) is bool,
                      'Exact selected own member and saved mode')
            reference_train.reconstruct_selected(state, 'cpu')
            # Store only the selected predictor/config/source metadata; no Adam or live RNG histories.
            slim = {key: state[key] for key in ('model', 'epoch', 'global', 'run', 'config', 'matched_reference')}
            selected.append(dict(member_index=member, original_checkpoint=record['selected_checkpoint'], state=slim))
            del state
        bank = dict(schema='matched-native-I4-own-selected-bank-v1', evaluation_only=True,
            paired_seed=release['paired_seed'], selected_bodies=selected,
            serving='Arithmetic mean of4 actual native class-probability vectors',
            native_body_seeds=[release['paired_seed']+1009*m for m in range(4)],
            selected_epochs=[row['state']['epoch'] for row in selected],
            selected_member_modes=[row['state']['global'] for row in selected],
            source_manifest_sha256=g.sha(g.HERE/'SOURCE_MANIFEST.json'),
            selector='Four original own strict-first maxima; no pooled checkpoint selection',
            optimizer_or_RNG_history_restored=False, arbitrary_member_index_pairing=False)
        sessions = reconstruct_bank(bank, 'cpu'); del sessions
        torch.save(bank, output/'selected_bank.pt')
        g.write(output/'COMPLETE.json', dict(schema='matched-native-I4-bank-assembly-complete-v1', complete=True,
            paired_seed=release['paired_seed'], members=4, selected_checkpoint=g.binding(output/'selected_bank.pt'),
            selected_epochs=bank['selected_epochs'], selected_member_modes=bank['selected_member_modes'],
            source_manifest_sha256=bank['source_manifest_sha256'], model_forwards=0, TRAIN_updates=0,
            quality_scores_read=False, TEST_access=False, no_pooled_reselection=True))
    except BaseException as error:
        failure = dict(type=type(error).__name__, error=str(error), automatic_retry=False, partial_work_retained=True)
        g.write(output/'FAILURE.json', failure)
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF)
        g.write(output/'COST.json', dict(schema='matched-native-I4-assembly-cost-v1', inclusive_wall_seconds=time.monotonic()-began,
            CPU_user_seconds=after.ru_utime-before.ru_utime, CPU_system_seconds=after.ru_stime-before.ru_stime,
            peak_RSS_bytes=int(after.ru_maxrss*(1024 if sys.platform.startswith('linux') else 1)),
            model_forwards=0, TRAIN_updates=0, source_manifest_sha256=g.sha(g.HERE/'SOURCE_MANIFEST.json'),
            failure=failure, outer_terminal_and_transfer_cost=None, automatic_retry=False))


if __name__ == '__main__':
    main()
