Disabled MolHIV message-own source control
========================================

Status: source preparation only, unadopted, outside the frozen 18-fit family.
No training, numerical gradient fixture, model construction, data access,
runtime qualification, TEST scoring, server access, or launch was performed.

One question and one method
---------------------------
The distinct method is be_init__allocation_I_message_own. It starts from the
existing MolHIV be_init I policy at fixed lambda=.5. Only the supervised gradient
of the existing message_factors tensor becomes the two-view mean-own gradient.
All other internal factors receive J. Shared and boundary parameters receive
own supervision. The SAME .05 alignment gradient remains on ALL audited phi,
including message_factors. The control changes no model, graph operator,
initializer, parameter, factor, optimizer, strength, or seed roster.

The catalog entry is read from the exact original BACKBONE_AUDIT.json after
the unchanged auditor partitions the original Session. The sole extra_private
entry must resolve by object identity to the existing backbone.message_factors.
The full parameter path is obtained from the catalog rather than guessed.

Delegation and isolation
------------------------
The exact sealed PolicyAdapter.train_step is reused. Its raw Session is audited
first. Only that new adapter's delegate.session receives a small facade. The
facade delegates forward, serving, optimizer, model, and every other Session
operation to the original object. Assignment to steps is forwarded to the raw
Session. It exposes a private Torch view and a copied core mapping containing
an objectives view; raw Torch, global autograd, raw core, and raw Session are
unchanged. The existing alignment function is called once with unchanged args
and its original tensor is captured. No global Torch interception occurs.

The facade checks the exact ordered parameter identities and options of the
two pinned I reverse collections. It retains the second graph, checks that
collection's finite gradients, then collects own + .05 alignment on the sole
audited message tensor. Only that tensor's returned gradient is replaced.
The exact I delegate checks the resulting exhaustive finite assignment,
detaches gradients for assignment, executes one native Adam transition, and
advances the raw Session. Its scalar internal_loss remains J+.05A, a diagnostic
for the remaining internal factors; this mixed allocation claims no global
scalar objective. There is no upstream detach or parameter update between
reverse collections.

control.py delegates the exact allocation-controls run_complete, which loads
the unchanged public Session, data interface, full train.main, complete VALID
evaluation, strict-first joint selector, snapshots, and failure preservation.
Only a fresh in-memory allocation module receives the adapter factory, method
identity, and three-call completion contract. The original sources and active
18-fit roster are unchanged. RUN/PROGRESS/COMPLETE/FAILURE, VALID trace rows,
selected snapshots, config, and returned metadata retain the distinct identity.
Exact resume remains unsupported.

Actual work
-----------
Each update uses eight member/view forwards and one native Adam transition.
There are THREE real autograd.grad collections: own on shared/boundary;
J+.05A on the original full phi; own+.05A on message_factors. The original I
message gradient is computed then discarded. The additional collection is
charged in the delegate counter before it is called, so its returned charge
and every completion check are three. This is not equal compute with I and
there is no padding. Counts do not imply equal reverse cost or elapsed time.

Full unchanged MolHIV source recipe: hidden256, layers5, dropout.5, batch128,
lr.001, 100 epochs, 258 updates/epoch, 25,800 updates; 206,400 training member
forwards; 25,800 Adam transitions; 77,400 reverse collections. I has 51,600
reverse collections, so this control adds 25,800. Original complete VALID
evaluation remains once per epoch and is charged separately by the delegate.

Disabled callable and unresolved decisions
------------------------------------------
run_complete(train, valid, output, *, admission=None, seed=6101, device='cpu')
is disabled by default. Before any data/runtime/output access it requires an
explicit admission mapping: fixed18_closed=True, message_question_warranted=True,
admitted_by='root', and a nonempty decision_reference. These are caller assertions
of a later decision, not a machine-verified finding or authorization created by
this packet. Root decides admission only after fixed18 closes and supports this
mechanism question. No CLI, campaign, queue, or activation instructions are added.

Still pending: numerical/runtime gradient routing and retained-graph behavior;
role and complete-data runtime qualification; full fit; selected-readout
qualification; and any empirical result or scientific adoption. Source checks
are not evidence for these gates. Prior adapter runtime success does not qualify
this successor. source_identity() is a stdlib-only description, not admission.

Scope of this control
---------------------
Conditional I versus message-own can test whether assigning supervised J to the
existing message tensor contributes under matched architecture, initialization,
views, lambda, and phi alignment. It does not replace the full-population selected
readout analysis or establish graph-evidence diversity from pooled AUC alone.
The existing mechanism note is attributed in SOURCE_BINDINGS.json. No new
literature claim or experiment is introduced here.
