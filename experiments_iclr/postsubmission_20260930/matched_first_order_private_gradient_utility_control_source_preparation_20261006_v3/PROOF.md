# Exact streamed product-rule credit

Let x collect shared theta and one original private row phi. At fixed raw-cost cotangent t, let g(x)=grad_phi ownCE(x) and h(x)=grad_phi sum_items(t*margin(x)). The utility scalar is u(x)=-eta_probe h(x)^T g(x). Its complete shared and private credit is

Du(x) = -eta_probe ((Dh(x))^T g(x) + (Dg(x))^T h(x)).

The successor computes both terms. The h branch differentiates h while treating g's value at the same original x as constant. The g branch differentiates g while treating h's value at that x as constant. The cotangent t is detached in both because this is a VJP after the complete original normalized allocation map; detaching t does not omit a path of this local VJP.

## Why two native forwards suffice

Replay 1 builds the original native forward. It computes g with create_graph=False and retain_graph=True, so only original native activations are retained for the following differentiable h partial. It copies detached g and h values, computes the h-branch mixed VJP, and returns only detached credits/values/restricted logits. The helper frame then ends before replay 2 begins.

Replay 2 starts fresh leaves from exactly the same saved theta and original phi, uses the same fixed eval-mode pure callback, checks exact equality of the restricted logits and own-gradient values, constructs differentiable g only, and computes the g-branch mixed VJP against detached h from replay 1. The shared and private credits are added out of place. No update or adaptation occurs between these replays. The original native port rejects a callback bound outside complete eval mode or changed stage/eval flags. The declared smooth synthetic callback is also deterministic and pure.

A third factor-value forward is unnecessary: both detached factors are already available from replay 1. At most one native graph and one differentiable private factor tree are scheduled in each utility-credit branch. This structural reduction does not bound derivative bytes or guarantee native memory success. The inherited direct main/query helper still schedules two native graphs.

Both product-rule terms remain live in the total. Neither detached-factor term alone is an accepted comparator. The existing monolithic synthetic oracle independently checks all shared/private credit coordinates, both factor terms, discriminating stop-g/stop-h controls, g=0 analytic behavior and the complete theta+/original-phi commitment. Its original fixture, formulas and tolerances must remain unchanged in a successor qualification. Floating-point summation order changes; no tolerance expansion is authorized.

## Native acceptance gate

The existing isolated native monolithic product-support oracle retains its full-coordinate reference semantics, both separately nontrivial shared branches and original FP32 tolerances. It must not be replaced by a replay decomposition that mirrors the streamed candidate. Root may defer this check until the authorized one-GPU 80GB allocation is free after the current correction. Any future larger engineering caps must be explicitly reviewed and prospectively frozen from the preserved failure evidence; no cap or invocation is changed here.
