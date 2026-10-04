"""Exact conditional-pattern bridge to the saved private-hop F4 HL-GNN.

The dependency is read from its sealed sibling packet. This module adds no
parameters, no learned count input and no serving override. Source only.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from f4_model import F4PrivateHopTargetModel
from native.utils import get_pos_neg_edges
from exact_conditional_loss import TrainPatternLabels, training_pattern_losses
from ddi_pattern_support import FullTrainTeacher, owned_seed, select_strata, tensor_sha


AUXILIARY = {
    'positive_queries': 32, 'negative_queries': 32, 'slot_chunk': 8192,
    'coefficient_in_mean_margin_units': 1.0,
    'mask': 'all_native_batch_positive_edge_identities_plus_selected_query_identities',
    'candidate_support': 'complete_visible_residual_minus_both_endpoints',
    'input_dropout': 'reuse_native_dropped_embedding_input',
    'scorer_dropout': 'owned_seed_in_fork_rng_with_checkpoint_preserve_rng_state',
    'reduction': 'half_positive_mean_plus_half_negative_mean_including_extreme_supports',
    'teacher': 'complete_TRAIN_observation_membership',
    'additional_parameters': 0, 'counts_in_served_scores': False,
}


def rng_sha(device):
    result = {'cpu': tensor_sha(torch.get_rng_state())}
    if device.type == 'cuda':
        result['cuda'] = tensor_sha(torch.cuda.get_rng_state(device))
    return result


class F4ConditionalPatternModel(F4PrivateHopTargetModel):
    def __init__(self, *args, arm, training_seed, stream_path, runtime_observer=None, **kwargs):
        if arm not in ('target_only', 'joint', 'separate'):
            raise ValueError('Declare target_only, joint or separate.')
        super().__init__(*args, **kwargs)
        self.arm, self.training_seed = arm, training_seed
        self.stream_path = Path(stream_path)
        self.epoch = 0
        self.teacher = None
        self.initial_parameter_sha256 = None
        self.runtime_observer = runtime_observer

    def _initial_parameter_sha(self):
        digest = hashlib.sha256()
        for prefix, module in (('encoder', self.encoder), ('predictor', self.predictor), ('embedding', self.emb)):
            for name, value in module.state_dict().items():
                digest.update(f'{prefix}/{name}/{tensor_sha(value)}'.encode())
        return digest.hexdigest()

    def _candidate_logits(self, h, queries, side):
        rows, nodes = side
        outputs = []
        for member, predictor in enumerate(self.predictor):
            chunks = []
            # Bind the head now so backward checkpoint recomputation cannot
            # capture a later member's Python loop variable.
            def score(x, y, head=predictor):
                return head(x, y).reshape(-1)
            for start in range(0, len(rows), AUXILIARY['slot_chunk']):
                part = slice(start, start + AUXILIARY['slot_chunk'])
                chunks.append(checkpoint(score, h[member, queries[rows[part]]], h[member, nodes[part]],
                                         use_reentrant=False, preserve_rng_state=True))
            outputs.append(torch.cat(chunks) if chunks else h[member, :0, 0])
        return torch.stack(outputs, dim=0)

    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        if (set(split_edge) != {'train', 'valid'} or set(split_edge['train']) != {'edge'}
                or num_neg != 3 or batch_size != 65536 or neg_sampler_name != 'global'):
            raise ValueError('The unchanged native missing-weight DDI target recipe is required.')
        self.encoder.train()
        self.predictor.train()
        self.epoch += 1
        if self.teacher is None:
            self.initial_parameter_sha256 = self._initial_parameter_sha()
            # Uses TRAIN only; no VALID split is consulted by the teacher.
            self.teacher = FullTrainTeacher.from_train(
                split_edge['train']['edge'].to(self.device), self.num_nodes, data.adj_t)
        pos_train_edge, neg_train_edge = get_pos_neg_edges(
            'train', split_edge, edge_index=data.edge_index, num_nodes=self.num_nodes,
            neg_sampler_name=neg_sampler_name, num_neg=num_neg)
        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)
        negative_sha = tensor_sha(neg_train_edge)
        total_loss = total_examples = 0
        with self.stream_path.open('a') as stream:
            for batch, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)):
                if self.runtime_observer is not None:
                    self.runtime_observer.start_update()
                self.optimizer.zero_grad()
                before_target_rng = rng_sha(self.device)
                x = self.create_input_feat(data)
                z, P = self.encoder.context(x, data.adj_t, data.edge_weight)
                # This is exactly the M4 forward branch in SharedPowerHLGNN.
                h = self.encoder.factored(z, P, storage='aggregates')
                pos_pairs = pos_train_edge[perm]
                neg_pairs = torch.reshape(neg_train_edge[perm], (-1, 2))
                route_losses = []
                for member, predictor in enumerate(self.predictor):
                    pos_out = predictor(h[member, pos_pairs[:, 0]], h[member, pos_pairs[:, 1]])
                    neg_out = predictor(h[member, neg_pairs[:, 0]], h[member, neg_pairs[:, 1]])
                    route_losses.append(self.calculate_loss(pos_out, neg_out, num_neg, margin=None))
                target = torch.stack(route_losses).mean()  # Native AUC SUM retained literally.
                after_target_rng = rng_sha(self.device)
                positive_positions, negative_positions = select_strata(
                    len(perm), num_neg, self.training_seed, self.epoch, batch)
                queries = torch.cat((pos_pairs[positive_positions.to(self.device)],
                                     neg_pairs[negative_positions.to(self.device)]), dim=0)
                positive_queries = len(positive_positions)
                receipt = {
                    'arm': self.arm, 'seed': self.training_seed, 'epoch': self.epoch, 'batch': batch,
                    'native_records': len(perm), 'native_record_ids_sha256': tensor_sha(perm),
                    'positive_positions': positive_positions.tolist(), 'negative_positions': negative_positions.tolist(),
                    'selected_query_sha256': tensor_sha(queries), 'before_target_rng': before_target_rng,
                    'after_target_rng': after_target_rng, 'auxiliary_parameter_count': 0,
                    'native_negative_draw_sha256': negative_sha,
                    'initial_parameter_sha256': self.initial_parameter_sha256,
                }
                loss = target
                if self.arm != 'target_only':
                    visible, hidden_keys = self.teacher.visible(pos_pairs, queries)
                    supports = visible.supports(queries)
                    bits = self.teacher.labels(queries, supports)
                    devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
                    auxiliary_seed = owned_seed(self.training_seed, self.epoch, batch, 'scorer-dropout')
                    # Additional scorer draws must not advance the native stream.
                    # Checkpoints restore their own captured dropout RNG on backward.
                    with torch.random.fork_rng(devices=devices):
                        torch.random.default_generator.manual_seed(auxiliary_seed)
                        if self.device.type == 'cuda':
                            with torch.cuda.device(self.device):
                                torch.cuda.manual_seed(auxiliary_seed)
                        _, visible_P = self.encoder.context(x, visible.adj_t, None, dropped_x=z)
                        masked_h = self.encoder.factored(z, visible_P, storage='aggregates')
                        left_logits = self._candidate_logits(masked_h, queries[:, 1], supports.left)
                        right_logits = self._candidate_logits(masked_h, queries[:, 0], supports.right)
                        values = training_pattern_losses(
                            left_logits, right_logits, supports.left[0], supports.right[0],
                            TrainPatternLabels(bits, 'complete_TRAIN_observation_membership'), len(queries))
                        key = 'J_K' if self.arm == 'joint' else 'J_K_sep'
                        auxiliary = 0.5 * (values[key][:positive_queries].mean()
                                           + values[key][positive_queries:].mean())
                        # L/(B*3) = mean native squared margin + lambda*A.
                        loss = target + len(perm) * num_neg * AUXILIARY['coefficient_in_mean_margin_units'] * auxiliary
                    receipt.update(
                        auxiliary_seed=auxiliary_seed, hidden_keys_sha256=tensor_sha(hidden_keys),
                        visible_keys_sha256=tensor_sha(visible.keys),
                        ordered_support_teacher_sha256=self.teacher.support_sha(queries, supports, bits),
                        support_counts=values['support_counts'], teacher_counts=values['teacher_counts'],
                        candidate_slots=len(bits), native_margin_terms=len(perm) * num_neg)
                if rng_sha(self.device) != after_target_rng:
                    raise RuntimeError('Auxiliary changed the native target RNG stream.')
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite combined loss; this cell is incomplete.')
                loss.backward()
                if rng_sha(self.device) != after_target_rng:
                    raise RuntimeError('Checkpoint backward changed the native target RNG stream.')
                if self.clip_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)
                self.optimizer.step()
                stream.write(json.dumps(receipt) + '\n')
                stream.flush()
                num_examples = pos_pairs.size(0)
                total_loss += loss.item() * num_examples
                total_examples += num_examples
                if self.runtime_observer is not None:
                    self.runtime_observer.finish_update(receipt)
        return total_loss / total_examples
