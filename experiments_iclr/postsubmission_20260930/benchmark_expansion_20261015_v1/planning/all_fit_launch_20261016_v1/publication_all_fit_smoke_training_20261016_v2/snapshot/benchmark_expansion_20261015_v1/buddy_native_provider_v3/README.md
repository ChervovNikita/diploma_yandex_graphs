# Portable native BUDDY provider v3

Successor to frozen provider v2. The only behavioral change is typed receipt
path comparison for the two authorized repository roots. GRAPH/TRAIN/VALID
relative paths, bytes and SHA256 remain exact, and the native factory's actual
receipts stay unchanged. Fixed configuration, dependency version, static tensor
hashes, initialization, sampling, hydration, optimizer and checkpoint checks are
retained. Metadata records both canonical and actual paths explicitly.

STATIC_PROVIDER.json is byte-identical to v2. No historical paid output or source
is rewritten. This package alone grants no launch authority.
