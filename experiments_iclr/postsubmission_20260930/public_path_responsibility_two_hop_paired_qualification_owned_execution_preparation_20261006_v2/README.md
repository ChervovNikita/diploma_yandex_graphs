# Disabled ownership observation successor V2

The exact V1 helper and failed CMCL invocation are preserved and bound in BINDINGS.json. SOURCE.diff is the complete source delta.

identity() keeps process start-time changes fatal. Only unchanged-start pgid/sid changes between the two stat observations raise ObservationTransition. physical() records those observations and retries under its original single three-second deadline, with the original 20ms sleep. The first transition start tick is pinned across retries; a caller-supplied start tick must also match. A later start-time change is fatal. Observations accompany a stable returned identity or failure. Full argv/cwd/exe/parent/start/pgid/sid verification before signals remains unchanged.

SOURCE_RELEASED remains false. Constants, process supervisor, launch/watchdog/cleanup paths, scientific source and caps are unchanged. This packet is a stdlib AST/diff/hash preparation only. No source import, CLI, server access, staging, launch or retry was performed.

Root and joint source review must precede a separate explicit root authorization for one new CMCL engineering invocation, after authoritative closure of the preserved failed startup. The request template retains false approval flags and unbound manifest/review slots.
