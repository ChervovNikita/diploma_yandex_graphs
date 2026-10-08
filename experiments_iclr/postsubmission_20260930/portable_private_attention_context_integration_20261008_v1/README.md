# Full-recipe private-attention context integration

This adds one constructor adapter and a thin CLI successor to the existing public context implementation. The proposed author comparison remains inactive until complete Context9 evidence and a separate source/resource/quality decision. The public API is callable: it runs one explicitly requested caller cell and requires no author account, private archive, approval file or token.

## What changes

`constructor_adapter.py` extracts the hash-pinned public Session constructor and inserts one call immediately before its original Adam factory. It compiles that constructor in a new subclass/namespace; the original class/module and all existing Sessions are untouched. CONSTRUCTOR.diff shows the insertion. Scorers are copied and the existing member forward is wrapped before Adam is created. No old optimizer is retrofitted.

Both the original shadow-forward path and the unchanged replay VJP path call the same wrapped member method. It supplies the local scorer-row context while the captured original method still selects BE factors and captures representations. Global Q/K/V factors, dense projections, views, target objective, Adam settings, streams, local transition, checkpoint selector and mean-probability serving remain unchanged.

The successor delegates the existing complete 1,100-epoch driver. It restricts methods to COMMON and ROUTE, consumes a caller target archive with an explicit hash, and adds constructor/hook/scorer/count metadata to RUN, selected snapshots, COMPLETE and cost receipts. It does not regenerate positive relations. The underlying driver/source identity remains recorded separately. No entire training framework or native third-party model is copied here.

## Caller target preparation and run

The existing public `prepare_targets` function can create the six logical arrays from the caller's complete TRAIN graph/features/labels, before constructing a model. For example, using this successor's source loader:

```python
from pathlib import Path
import numpy as np
import train as successor

driver = successor.load_driver(Path('../portable_context_steering_public_interface_20261008_v1'))
driver.load_public_interface(Path('../portable_internal_be_public_interface_20261007_v2'))
torch, device = driver.initialize_runtime('cpu')
train_roles, _, _ = driver.load_train_valid('wikics', Path('train.npz'), Path('valid.npz'))
targets = driver.prepare_targets(train_roles, torch, device)
np.savez('targets.npz', **targets.arrays)
print(successor.sha('targets.npz'))
```

Loading the development role in this example does not supply its labels to target generation. Keep the original source-custody evidence and freeze target/method/seed choices before comparison. The consumer verifies logical TRAIN identities, classes, panel and permutations; arbitrary caller files do not certify their generation history or official dataset custody.

```sh
python train.py --method shared_route --seed 8101 --device cuda:0 \
  --public-interface ../portable_internal_be_public_interface_20261007_v2 \
  --context-interface ../portable_context_steering_public_interface_20261008_v1 \
  --attention-interface ../portable_private_local_attention_20261008_v1 \
  --polynormer external/Polynormer/model.py \
  --train train.npz --valid valid.npz \
  --frozen-targets targets.npz --frozen-targets-sha256 <printed-digest> \
  --output runs/private_route_8101
```

There is no shorter-horizon, tuning, retry, resume or automatic next-run option. COMMON uses the same command with `--method shared_common`. Run identities have the distinct `public_private_attention_context__` prefix. Public caller runs do not join or replace the author's registered family. The author's separate release must enforce its exact original target/archive/source/initialization/selector bindings before reusing the six shared-attention anchors; absent compatibility is flagged rather than silently rerun.

## Selected-state reconstruction

Use `reconstruct_selected(public, hook_root, trusted_state, device, polynormer)` from `constructor_adapter.py` for serving. It rebuilds private banks and routing before strict state loading, checks the saved scorer/source/count contract, and restores the selected Python local/global flag. It loads no optimizer/RNG history and disables training on the returned serving-only Session. It neither reselects a checkpoint nor supports training resume. Rebuild from state_dict; full-model pickle, post-install reset/deepcopy and cross-member parametrization caching remain unsupported.

## Qualification and scientific scope

Static compilation and stdlib CLI help passed. `qualify_integration_cpu.py` is a separate engineering fixture for the existing allocation CPU runtime: 580 fabricated rows, five input features, width eight and two local layers; one synthetic replay transition; exact scorer/BE route observations on every shadow/replay call; original Adam membership; immutable target consumption; and selected-state reconstruction. Its reduced recipe is isolated in a test module copy and is not exposed by the CLI. It supplies no scientific quality evidence. No runtime installation or scientific data/results/server execution occurred during preparation.

The actual seven-layer/512-channel parameter count and initial-function hook qualification remain in the preceding hook packet: 7,659,284 total parameters become 7,680,788; BE factors remain 122,112; all private parameters become 150,784. This does not demonstrate a capacity separation, accuracy gain or efficiency improvement. The six prospective full fits and stronger GNCL/single/ordinary/appropriate efficient-ensemble controls remain separate scientific work. Generic private attention, contrastive alignment and graph diversity have established prior; novelty remains unresolved.
