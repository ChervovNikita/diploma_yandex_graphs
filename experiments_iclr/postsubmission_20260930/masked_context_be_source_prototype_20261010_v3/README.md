# Masked-context prototype V3

The complete method, nine conditions, scalar objectives, gradient paths, native PolyFormer architecture, published optimizer, mask ownership and serving remain exactly those of V2. Method and native provider source bytes are unchanged. Science remains unconditionally disabled.

V3 adds one explicit qualification protocol, class_stratified_floor60_20_20, with identity PubMed-class-stratified-floor60-20-20 and split seed190111. The data custodian samples floor(.6*n_c) TRAIN and floor(.2*n_c) VALID identities for each class, and assigns the remainder to TEST using an independent fixed NumPy RandomState. For PubMed this yields11,829 TRAIN,3,942 VALID and3,946 TEST nodes. Labels are used only by the data custodian to construct the roles. The engineering worker receives only the four TRAIN arrays.

This change is motivated by data-construction evidence before any model was created. The published native split leaves VALID class counts[88,1880,1975] and TEST[72,1916,1957]. Equal training quotas nearly exhaust the smallest class. Preserve that population and its acquisition record, but do not use it as the primary representative screen. The new split preserves class proportions within each role. It is a declared protocol amendment with published model hyperparameters, not exact reproduction of the author split or unused confirmation.

Use the separate amended protocol in masked_context_pubmed_allocation_preparation_20261010_v1/STRATIFIED_PROTOCOL_AMENDMENT.md. Bind source, exporter and TRAIN custody before numerical qualification. No outcomes, accuracy improvement or novelty is established by this source change.
