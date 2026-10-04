"""Full native ordered side arrays, exact grouping/padding and query chunks."""
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from vector_density import require, marginals, nll
from vector_head import potential


def validate_ragged(left,right,lo,ro,context,swapped):
    require(left.ndim == right.ndim == 1 and left.dtype == right.dtype == torch.float64
            and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), "Finite FP64 native side arrays required")
    require(lo.ndim == ro.ndim == 1 and len(lo) == len(ro) and len(lo) >= 1
            and lo.dtype == ro.dtype == torch.long and lo[-1] == len(left) and ro[-1] == len(right),
            "Complete side arrays/offsets required")
    require(context.shape == swapped.shape == (len(lo)-1,520)
            and context.dtype == swapped.dtype == torch.float32, "Frozen full context shapes required")
    require(left.device == right.device == lo.device == ro.device == context.device == swapped.device, "Ragged devices differ")


def partitions(left_offsets, right_offsets, *, workspace_cells=262144, max_queries=256):
    require(left_offsets.dtype == right_offsets.dtype == torch.long
            and left_offsets.ndim == right_offsets.ndim == 1 and len(left_offsets) == len(right_offsets) and len(left_offsets) >= 1
            and workspace_cells > 0 and max_queries > 0, "Valid offsets/workspace required")
    lo, ro = left_offsets.detach().cpu().tolist(), right_offsets.detach().cpu().tolist()
    require(lo[0] == ro[0] == 0 and all(a <= b for a,b in zip(lo,lo[1:]))
            and all(a <= b for a,b in zip(ro,ro[1:])), "Invalid ordered side offsets")
    sizes = [(lo[i+1]-lo[i], ro[i+1]-ro[i]) for i in range(len(lo)-1)]
    # Grouping has no learned sort or readout-order effect; results scatter back.
    order = sorted(range(len(sizes)), key=lambda i: (sum(sizes[i]), max(sizes[i]), sizes[i], i))
    chunks, current, a, b = [], [], 0, 0
    for i in order:
        na, nb = max(a, sizes[i][0]), max(b, sizes[i][1])
        cells = (len(current)+1) * ((na+1)**2 + (nb+1)**2 + (na+1)*(nb+1))
        if current and (len(current) == max_queries or cells > workspace_cells):
            chunks.append((current,a,b)); current, a, b = [], 0, 0
            na, nb = sizes[i]
        current.append(i); a, b = na, nb
    if current:
        chunks.append((current,a,b))
    require(sorted(i for c,a,b in chunks for i in c) == list(range(len(sizes))), "Query lost or repeated")
    # A single oversized query is preserved in its own chunk, never truncated.
    return chunks


def pack(flat, offsets, rows, width):
    lengths = offsets[rows+1] - offsets[rows]
    column = torch.arange(width, device=flat.device)[None, :]
    valid = column < lengths[:, None]
    indices = offsets[rows, None] + column
    if not len(flat):
        return flat.new_zeros((len(rows), width)), lengths, indices, valid
    padded = flat[indices.clamp(max=len(flat)-1)]
    return torch.where(valid, padded, torch.zeros_like(padded)), lengths, indices, valid


@torch.no_grad()
def unary_marginals(left, right, lo, ro, context, swapped, head, **chunk_options):
    validate_ragged(left,right,lo,ro,context,swapped)
    left, right, context, swapped = left.detach(), right.detach(), context.detach(), swapped.detach()
    mu_l, mu_r, logz = torch.empty_like(left), torch.empty_like(right), left.new_empty(len(lo)-1)
    chunks = partitions(lo, ro, **chunk_options)
    for chunk,a,b in chunks:
        rows = torch.tensor(chunk, device=left.device)
        l, nl, il, ml = pack(left, lo, rows,a)
        r, nr, ir, mr = pack(right, ro, rows,b)
        g = potential(head, context[rows], swapped[rows], nl, nr,padded_shape=(a,b))
        lm, rm, z = marginals(l, r, nl, nr, g)
        mu_l[il[ml]], mu_r[ir[mr]], logz[rows] = lm[ml], rm[mr], z
    return mu_l, mu_r, logz


def source_nll(left, right, lo, ro, context, swapped, head, zl, zr, *, recompute=True, **chunk_options):
    validate_ragged(left,right,lo,ro,context,swapped)
    require(zl.shape == left.shape and zr.shape == right.shape and zl.device == left.device and zr.device == right.device,
            "Labels must align with fixed native side arrays")
    result = left.new_zeros(len(lo)-1)
    for chunk,a,b in partitions(lo, ro, **chunk_options):
        rows = torch.tensor(chunk, device=left.device)
        l, nl, _, _ = pack(left, lo, rows,a)
        r, nr, _, _ = pack(right, ro, rows,b)
        labels_l, _, _, _ = pack(zl.detach(), lo, rows,a)
        labels_r, _, _, _ = pack(zr.detach(), ro, rows,b)
        def loss(left, right, context, swapped, nl, nr, zl, zr):
            return nll(left, right, nl, nr, potential(head, context, swapped, nl, nr,
                                                    padded_shape=(left.shape[1],right.shape[1])), zl, zr)
        inputs = (l, r, context[rows], swapped[rows], nl, nr, labels_l, labels_r)
        if recompute:
            with set_checkpoint_early_stop(False):
                value = checkpoint(loss, *inputs, use_reentrant=False, preserve_rng_state=True)
        else:
            value = loss(*inputs)
        result = result.index_copy(0, rows, value)
    return result
