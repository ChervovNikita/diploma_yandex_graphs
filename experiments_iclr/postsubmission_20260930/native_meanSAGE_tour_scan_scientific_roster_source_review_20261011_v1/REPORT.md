# Independent tour/scan scientific-roster source review

11 October 2026. Bounded source inspection of sealed `native_meanSAGE_tour_scan_preparation_source_20261011_v1/core.py` and its native dependency calls against the exact fixed representative experiment. **No concrete scientific-roster mismatch was identified in the inspected source.** This is source evidence; no model was constructed, numerical library imported, derivative fixture run, runtime qualified or scientific job launched. No manuscript review or acceptance verdict is supplied.

## Confirmed roster semantics

| Arm/family | Actual source construction and forward | Loss/ownership consequence |
| --- | --- | --- |
| P1/F1/I4/S4 | Calls the pinned native core's corresponding no-feedback family. P1/F1 have one route; I4 returns four separate M1 objects; S4 has one body with four native factor/bias rows. | The source equations and effective first-layer penalty remain intact. Baseline selected-state reuse still needs root's identity audit. |
| PV1 | Builds native P1 with one body and one bias row, one scanner, and logical views4. Every parameter route is0; tours are0/1/2/3. | All ordinary native biases/maps and scan parameters are tied. Four complete logits and own-view losses remain separate. A four-bias nonfactor body is not constructed. |
| V1 | Builds native F1 with one body, one factor/bias set and one scanner; parameter routes are0/0/0/0, tours0/1/2/3. | All factors, biases, slow maps and scanner are tied across views. The one effective native penalty is counted once. |
| WI4 | Builds native I4's four genuinely separate F1 models, then constructs a new scanner for each body m. That learner uses only tour m and local view/RNG index0. | Four complete disjoint native+scan parameter groups and four own optimizers. Each own objective has one full CE and one full own penalty. Own selectors/stops remain controller duties. Four total contexts, not four tours per body. |
| SW4 | Builds native S4: shared slow maps, four private native factor/bias rows, one shared scanner. Route m uses tour m. | Four own-route CE values averaged, plus the native mean effective penalty. One joint optimizer/selector. |
| C4 | Identical SW4 ownership and four forwards; every route's tour ID is0. | Private native factors/biases and separate own losses/RNG scopes are retained. Only the persistent tour assignment changes. |
| R4 | Identical SW4 ownership and tours; scanner first forms original `a=sigmoid(...)`, `q=(1-a)*u`, then requests reset-every-token. Backend returns gathered q and its occurrence mean. | Retained recurrence is removed without changing q to u or detaching q's gate/input derivatives. Bidirectional reset states both equal q, so their mean is q. Same3136 scan parameters. |

`Model.forward` calls native first/scan/last separately per view and stacks final logits. It never averages hidden states. Native source layers retain p=.4 sparse-value/hidden dropout, one shared dropped input for aggregate/skip, original adjacency/degrees, row clipping and first-only ReLU. Each complete view executes inside its own supplied persistent RNG scope; evaluation consumes no dropout RNG. The scanner introduces no additional dropout.

`own_objective` averages CE over the returned model's own logits and adds `model.regularizer()` exactly once. One-route independent models therefore receive a full own loss. PV1/V1's native one-row regularizer is full once, while SW4/C4/R4's native four-row regularizer already averages effective penalties. No pooled-probability training loss, scan penalty or separate scan optimizer appears.

`make_optimizers` constructs one optimizer per returned learner and checks exact disjoint parameter ownership. The supplied native `author_model.optimizer_class` implements the required TF1 Adam.001/beta.9,.999/epsilon-before-bias-correction with float32 beta powers. The caller must bind that exact class.

The factory's parameter totals match the frozen ten-arm roster: P1/F1/I4/S4 =538917/555885/2223540/606900; PV1/V1/WI4 =542053/559021/2236084; SW4/C4/R4 =610036 each. The scanner contributes three32×32 matrices and two32 biases,3136 parameters per actual learner. PV1/V1 inspections reject extra view factor/bias rows; independent inspections reject cross-learner parameter/storage aliases.

## Explicit caller boundaries

* Persistent TRAIN RNG states belong to logical view j for PV1/V1 and shared banks, but to each independent body m with local index0 for WI4/I4. The core requires a context; it does not validate RNG independence, construct stream managers or restore them. Interleaving first/last within a view preserves the specified draw law only when the caller supplies persistent independently owned scopes.
* I4/WI4 need four own selectors, stopping states and complete selected restores. PV1/V1/SW4/C4/R4 need one complete joint objective/selector/restore. The factory creates no training controller or checkpoint selector. It returns separate models/optimizers and the intended member/view count for those duties.
* Raw serving and inside-member temperature fitting are controller/reader contracts; this core returns all native final logits without a pool. PV1/V1 must retain four outputs at serving and CAL. The stronger independently trained16-view reference remains an explicit later gap rather than an added pilot arm.
* Wrapping changes state-dict prefixes to `native.*`, and no-scan shared forward order is interleaved by view. Existing baseline checkpoints/exports must retain their original identity/custody or be loaded under root's declared adapter audit. The source does not certify bitwise baseline replay by itself.
* R4 also changes residual amplitude/conditioning as the fixed spec acknowledges. A later SW4−R4 gain would support the declared intervention, not isolated causality of long-range information.

No repair is requested from this bounded inspection. Root's finite derivative/resource check and the implementation owner's controller remain separately required. The custom scan's numerical derivative correctness, actual persistent RNG behavior, source-state replay and useful prediction quality were not tested here.

READ_BINDINGS.json pins the exact source/spec/dependency bytes. SOURCE_REVIEW.json gives source locations and the confirmed construction table. Existing files and seals are unchanged; no literature/history sweep, payload, numerical import, job, reviewer or manuscript action occurred.
