"""SOURCE ONLY: unexecuted evaluation adapters for selected, trained SAGE arms.

No canonical source/model/checkpoint/graph is imported or loaded by this module.
Adapters require an existing admitted model. They cannot train or change its
parameters. The canonical serial path remains the prospective reference.
"""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path


CANONICAL_PINS = {
    "coordinate_ensemble_source_v1/models.py": "a74a87dc26b7675a2d3fa0aaf0e7734b4786fa6c49411ef52f9bb21c0516dbc1",
    "propagation_cost_impl_v1/_pinned/frozen_models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
}


def verify_canonical_sources(postsubmission_root=None):
    root = Path(postsubmission_root) if postsubmission_root else Path(__file__).resolve().parents[2]
    receipt = {}
    for name, expected in CANONICAL_PINS.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Canonical source changed: {name}")
        receipt[name] = actual
    return receipt


def _runtime(admitted):
    if not admitted:
        raise PermissionError("This source packet has no model/runtime execution admission.")
    import torch
    import torch_geometric
    from torch import nn
    from torch_geometric.nn import SAGEConv
    if torch.__version__.split("+")[0] != "2.1.2" or torch_geometric.__version__ != "2.7.0":
        raise ValueError("Qualification contract is Torch2.1.2/PyG2.7.0.")
    return torch, nn, SAGEConv


def _check_eval(model, nn):
    if any(module.training for module in model.modules()):
        raise ValueError("Evaluation only: every submodule must already be in eval mode.")
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise ValueError("Semantic forward hooks require separate qualification.")


def _check_trunk(base, nn, SAGEConv):
    if base.model_name != "SAGE":
        raise ValueError("Only the selected homogeneous SAGE architecture is bound.")
    norms = [r.normalization for r in base.residual_modules] + [base.output_normalization]
    if any(not isinstance(norm, nn.LayerNorm) or len(norm.normalized_shape) != 1
           or not norm.elementwise_affine or norm.weight is None or norm.bias is None for norm in norms):
        raise ValueError("Affine feature-only LayerNorm must preserve its selected shape/eps.")
    if not isinstance(base.act, nn.GELU):
        raise ValueError("Selected native input activation must remain GELU.")
    for residual in base.residual_modules:
        conv = residual.module.conv
        if (type(conv) is not SAGEConv or conv.aggr != "mean" or conv.project
                or conv.normalize or not conv.root_weight or conv.node_dim != -2
                or conv.flow != "source_to_target" or conv.decomposed_layers != 1
                or conv.explain):
            raise ValueError("Only default mean/root SAGE with node_dim=-2 is bound.")
        if conv.lin_r.bias is not None:
            raise ValueError("Native SAGE root transform is bias-free.")
        if not isinstance(residual.module.feed_forward_module.act, nn.GELU):
            raise ValueError("Selected native FFN activation must remain GELU.")
        for name in ("_propagate_forward_pre_hooks", "_propagate_forward_hooks",
                     "_message_forward_pre_hooks", "_message_forward_hooks",
                     "_aggregate_forward_pre_hooks", "_aggregate_forward_hooks"):
            if getattr(conv, name, {}):
                raise ValueError("PyG propagation hooks require separate qualification.")


def _check_inputs(graph, x, torch):
    edges = graph.edge_index
    if type(edges) is not torch.Tensor or edges.layout != torch.strided or edges.dtype != torch.long:
        raise ValueError("Use the unchanged ordinary int64 COO edge_index tensor.")
    if edges.ndim != 2 or edges.shape[0] != 2 or x.ndim != 2 or not x.is_floating_point():
        raise ValueError("Expected edge_index[2,E] and input features[N,D].")
    if edges.device != x.device:
        raise ValueError("Graph and selected features must already share a device.")


class OriginalSAGEEval:
    """Same live selected parameters; M4 full private states through shared maps.

    member_chunk batches complete trajectories; stem_chunk bounds the private
    input-scaled [chunk,N,D] temporary independently. Both retain every member.
    Defaults batch four hidden trajectories but build input stems one at a time
    to avoid multiplying large raw-feature storage by four.
    """

    def __init__(self, model, *, member_chunk=4, stem_chunk=1,
                 runtime_admitted=False, postsubmission_root=None):
        self.torch, nn, sage = _runtime(runtime_admitted)
        self.source_receipt = verify_canonical_sources(postsubmission_root)
        if model.arm != "original" or model.members != 4:
            raise ValueError("Only the original four-member selected model is bound.")
        if member_chunk not in (1, 2, 4) or stem_chunk not in (1, 2, 4) or stem_chunk > member_chunk:
            raise ValueError("Use1/2/4 member chunks with stem_chunk<=member_chunk.")
        _check_eval(model, nn)
        _check_trunk(model.base, nn, sage)
        for block in (model.base.input_be_block, model.base.output_be_block):
            if block.W.bias is not None or block.R.shape[0] != 4 or block.S.shape[0] != 4 or block.B.shape[0] != 4:
                raise ValueError("Preserve native bias-free W and all four R/S/B rows.")
        self.model, self.nn = model, nn
        self.member_chunk, self.stem_chunk = member_chunk, stem_chunk

    def __call__(self, graph, x):
        torch, base = self.torch, self.model.base
        _check_eval(self.model, self.nn)
        _check_inputs(graph, x, torch)
        results = []
        with torch.inference_mode():
            for first in range(0, 4, self.member_chunk):
                last = min(first + self.member_chunk, 4)
                stems = []
                block = base.input_be_block
                for start in range(first, last, self.stem_chunk):
                    stop = min(start + self.stem_chunk, last)
                    hidden = block.W(x[None] * block.R[start:stop, None, :])
                    hidden = hidden * block.S[start:stop, None, :] + block.B[start:stop, None, :]
                    stems.append(base.act(base.dropout(hidden)))
                hidden = stems[0] if len(stems) == 1 else torch.cat(stems, dim=0)
                del stems
                for residual in base.residual_modules:
                    normalized = residual.normalization(hidden)
                    module, ff = residual.module, residual.module.feed_forward_module
                    # PyG node_dim=-2 addresses N in [member,N,H]. COO graph
                    # and all private hidden fields remain; no member mean.
                    message = module.conv(normalized, graph.edge_index)
                    branch = ff.linear_1(torch.cat([normalized, message], dim=-1))
                    branch = ff.act(ff.dropout_1(branch))
                    branch = ff.dropout_2(ff.linear_2(branch))
                    hidden = hidden + branch
                hidden = base.output_normalization(hidden)
                block = base.output_be_block
                output = block.W(hidden * block.R[first:last, None, :])
                output = output * block.S[first:last, None, :] + block.B[first:last, None, :]
                results.append(output)
            # Retain C even for C=1, as the canonical ensemble wrapper does.
            return results[0] if len(results) == 1 else torch.cat(results, dim=0)


def _stack_linear(modules, torch):
    weights = torch.stack([module.weight.detach() for module in modules])
    flags = [module.bias is not None for module in modules]
    if len(set(flags)) != 1:
        raise ValueError("Control members must have the same selected bias flags.")
    biases = torch.stack([module.bias.detach() for module in modules]) if flags[0] else None
    return weights, biases


def _private_linear(x, packed, torch, first, last):
    weight, bias = packed
    output = torch.bmm(x, weight[first:last].transpose(1, 2))
    return output if bias is None else output + bias[first:last, None, :]


class PackedControlEval:
    """Untied/head evaluation receives the same batching opportunity.

    Packing copies every private trained tensor once. Cold conversion time and
    its temporary original+packed peak count. After equivalence, release_source
    permits a serving representation with only packed private tensors; caller
    must release its original reference too. No warm-request repacking is needed.
    """

    def __init__(self, model, *, member_chunk=4, runtime_admitted=False, postsubmission_root=None):
        torch, nn, sage = _runtime(runtime_admitted)
        self.source_receipt = verify_canonical_sources(postsubmission_root)
        _check_eval(model, nn)
        if model.arm not in ("untied", "heads") or model.members != 4 or member_chunk not in (1, 2, 4):
            raise ValueError("Bound controls are four-member untied/heads with1/2/4 chunks.")
        self.model, self.torch, self.nn = model, torch, nn
        self.member_chunk, self.arm = member_chunk, model.arm
        self.signature = self._parameter_signature()
        self.blocks = []
        self.aggregations, self.activations = [], []
        if self.arm == "untied":
            bases = list(model.models)
            for base in bases:
                _check_trunk(base, nn, sage)
            if len({len(base.residual_modules) for base in bases}) != 1:
                raise ValueError("All selected control members need the same architecture.")
            self.input = _stack_linear([b.input_linear for b in bases], torch)
            self.output = _stack_linear([b.output_linear for b in bases], torch)
            for layer in range(len(bases[0].residual_modules)):
                residuals = [b.residual_modules[layer] for b in bases]
                norms = [r.normalization for r in residuals]
                if len({(n.normalized_shape, n.eps) for n in norms}) != 1:
                    raise ValueError("Private LayerNorm shapes/eps must agree.")
                self.blocks.append({"norm": self._pack_norm(norms),
                    "neighbor": _stack_linear([r.module.conv.lin_l for r in residuals], torch),
                    "root": _stack_linear([r.module.conv.lin_r for r in residuals], torch),
                    "first": _stack_linear([r.module.feed_forward_module.linear_1 for r in residuals], torch),
                    "second": _stack_linear([r.module.feed_forward_module.linear_2 for r in residuals], torch)})
                # Native mean message aggregation is parameter-free. Keep its
                # selected operator state without retaining unused lin_l/r W.
                self.aggregations.append(copy.deepcopy(residuals[0].module.conv.aggr_module))
                self.activations.append(copy.deepcopy(residuals[0].module.feed_forward_module.act))
            self.output_norm = self._pack_norm([b.output_normalization for b in bases])
            self.input_activation = copy.deepcopy(bases[0].act)
        else:
            _check_trunk(model.base, nn, sage)
            self.common_base = model.base
            self.common_signature = tuple((id(p), p.data_ptr(), p._version, p.device, p.dtype)
                                          for p in self.common_base.parameters())
            self.first = _stack_linear([h[0] for h in model.heads], torch)
            self.second = _stack_linear([h[3] for h in model.heads], torch)
            self.head_activation = copy.deepcopy(model.heads[0][1])

    def release_source(self):
        """After qualification, drop original private weights from this adapter.

        The caller owns any remaining source reference. Source/packed cache is
        immutable; a new checkpoint gets a new charged conversion. Heads keep
        their existing shared trunk once, not a duplicate trunk.
        """
        if self.model is not None and self._parameter_signature() != self.signature:
            raise ValueError("Source changed before releasing it.")
        self.model = None

    def _parameter_signature(self):
        return tuple((id(p), p.data_ptr(), p._version, p.device, p.dtype, tuple(p.shape))
                     for p in self.model.parameters())

    def _pack_norm(self, norms):
        if any(not n.elementwise_affine or n.bias is None for n in norms):
            raise ValueError("Original controls use complete private affine LayerNorm.")
        if len({(n.normalized_shape, n.eps) for n in norms}) != 1:
            raise ValueError("Control LayerNorm shape/epsilon mismatch.")
        return (norms[0].normalized_shape, norms[0].eps,
                self.torch.stack([n.weight.detach() for n in norms]),
                self.torch.stack([n.bias.detach() for n in norms]))

    def _norm(self, x, packed, first, last):
        shape, eps, weight, bias = packed
        normalized = self.torch.nn.functional.layer_norm(x, shape, None, None, eps)
        return normalized * weight[first:last, None, :] + bias[first:last, None, :]

    def __call__(self, graph, x):
        torch, model = self.torch, self.model
        if model is not None:
            _check_eval(model, self.nn)
        _check_inputs(graph, x, torch)
        if model is not None and self._parameter_signature() != self.signature:
            raise ValueError("Packed control is stale; repack and charge preparation.")
        outputs = []
        with torch.inference_mode():
            if self.arm == "heads":
                base = self.common_base
                _check_eval(base, self.nn)
                signature = tuple((id(p), p.data_ptr(), p._version, p.device, p.dtype)
                                  for p in base.parameters())
                if signature != self.common_signature:
                    raise ValueError("Selected common head trunk changed.")
                common = base.act(base.dropout(base.input_linear(x)))
                for residual in base.residual_modules:
                    common = residual(graph, common)
                common = base.output_normalization(common)
            for first in range(0, 4, self.member_chunk):
                last = min(first + self.member_chunk, 4)
                if self.arm == "heads":
                    hidden = _private_linear(common[None].expand(last-first, -1, -1), self.first, torch, first, last)
                    # Selected source heads use GELU then eval dropout.
                    hidden = self.head_activation(hidden)
                    outputs.append(_private_linear(hidden, self.second, torch, first, last))
                    continue
                hidden = _private_linear(x[None].expand(last-first, -1, -1), self.input, torch, first, last)
                hidden = self.input_activation(hidden)
                for layer, block in enumerate(self.blocks):
                    normalized = self._norm(hidden, block["norm"], first, last)
                    propagated = self.aggregations[layer](
                        normalized.index_select(-2, graph.edge_index[0]),
                        index=graph.edge_index[1], dim_size=x.shape[0], dim=-2)
                    message = (_private_linear(propagated, block["neighbor"], torch, first, last)
                               + _private_linear(normalized, block["root"], torch, first, last))
                    branch = _private_linear(torch.cat([normalized, message], dim=-1), block["first"], torch, first, last)
                    branch = self.activations[layer](branch)
                    hidden = hidden + _private_linear(branch, block["second"], torch, first, last)
                hidden = self._norm(hidden, self.output_norm, first, last)
                outputs.append(_private_linear(hidden, self.output, torch, first, last))
            return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


if __name__ == "__main__":
    raise SystemExit("Source only: no model loading, execution, training, or benchmarking CLI.")
