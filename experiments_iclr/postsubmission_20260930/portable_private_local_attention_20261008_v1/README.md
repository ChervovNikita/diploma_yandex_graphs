# Initially copied private local attention

This inactive extension tests whether members learn more useful neighbor rankings when they can adjust local attention scores directly, while continuing to share the dense backbone matrices. Existing BE factors and route-specific value tensors remain. The final prediction is still the mean of four class-probability vectors. No accuracy, novelty or efficiency improvement is established.

`private_local_attention.py` registers standard Torch parametrizations for the native local GAT `att_src` and `att_dst` vectors. Each original vector becomes a four-row bank initialized by copying that vector. The native GAT forward, edges, softmax, dense projections, normalization, mixing and task heads are unchanged. Global Q/K/V BE factors are unchanged. No unused common scorer is added. Selection of the member row is explicit and restored after a forward or exception.

## Integration before fresh Adam

Use the pinned Polynormer and BE sources in SOURCE_BINDINGS.json. Install after their normal reset/factor initialization, **before constructing Adam** in a new prospective constructor. The existing public Session constructs Adam immediately after Ensemble; do not install into that completed Session. Retrofitting any old optimizer is unsupported. Active Session, source and checkpoints must remain untouched.

At the indicated location in a new constructor:

```python
from functools import partial
from private_local_attention import install_private_local_attention

# self.model has just been constructed with the original shared BE factory.
attention = install_private_local_attention(self.model.models[0].body, members=4)
original_member_forward = self.model.member_forward
self.model.member_forward = partial(attention.forward_member, original_member_forward)

# The original WikiCS optimizer recipe follows; build it only now.
self.optimizers = [torch.optim.Adam(self.model.models[0].parameters(),
    lr=self.config['training']['lr'], weight_decay=0., eps=1e-8)]
```

The captured original member forward retains BE factor routing and representation capture. Rebuild this constructor before loading a candidate state_dict. Scorer state keys now contain `parametrizations.att_*.original`. Do not load shared-anchor checkpoints into this differently shaped bank, reset it after installation, deepcopy an installed model, serialize the full Python model, cache parametrizations across members, or run concurrent member forwards.

## Engineering checks

`qualify_cpu.py --phase <project-research-folder>` uses only the exact pinned sources and a fabricated six-node CPU graph. It checks initial local/global and matched-dropout outputs; unchanged non-scorer parameter aliases; copied initialization without RNG consumption; native gradients into the selected scorer rows; summed initial scorer gradients; unchanged value projections and edge support; context restoration; fresh Adam's exact parameter membership; and strict state_dict reconstruction. One Adam is constructed and never stepped.

The parent ran all 11 checks successfully in the existing allocation CPU runtime (Torch 2.1.2, PyG 2.7.0), taking 7.78 seconds with zero updates. The receipt is `representative_scientific_continuation_root_20261008_v2/PRIVATE_ATTENTION_CPU01.json`. One additional constructor, with the real seven-layer/512-channel recipe, returned these actual counts without a graph forward:

| Prediction parameters | Original shared scorers | Private copied scorers |
| --- | ---: | ---: |
| Total | 7,659,284 | 7,680,788 |
| BE private factors | 122,112 | 122,112 |
| Private attention scorers | 0 | 28,672 |
| All private parameters | 122,112 | 150,784 |
| All shared parameters | 7,537,172 | 7,530,004 |

The change adds 21,504 stored prediction scalars; it is not a runtime or memory-saving claim. This is source qualification only. No local Torch/PyG installation, scientific run, capacity conclusion or quality evidence was produced.

## Scientific decision and controls

PLAN.json retains the single prospective COMMON/ROUTE-by-shared/private comparison: six compatible complete anchors and six possible new full fits. Training remains inactive until complete Context9 evidence and the parent's source/resource/quality decision. Incompatible or undocumented anchors must be flagged, not silently rerun. Whole-population repairs/harms, actual pooling, member competence and costs matter; attention disagreement alone does not.

Failure cases include temperature-only changes, nuisance attention, weaker members, shared values missing needed evidence, and equal COMMON/ROUTE gains that undermine the specific steering interpretation. GATv1 static-ranking limitations remain. The ordinary and objective-matched single/untied references are still required. Before claiming a new diversity-learning benefit, a competent **GNCL** actual member/pool-risk mixture is a necessary published control; its gradient must reach all intended parameters. The existing internal-only allocation is a different control. For a broader efficient-ensemble claim, qualify a **TabM-style graph adaptation** with its private task-head and initialization conventions. Neither tabular results nor altered graph recipes constitute a native published reproduction. No additional arm is admitted by this packet.

Closest prior also includes GAT/GATv2, attention-disagreement regularization, BotSCL/SupCon and graph-evidence diversity. This is an attributed, testable ownership intervention, not an established methodological gap.
