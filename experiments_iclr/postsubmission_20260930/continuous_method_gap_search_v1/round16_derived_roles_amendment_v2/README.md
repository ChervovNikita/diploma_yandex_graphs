# Source-only correction driver v2

The new driver is `prototype/correction_screen_driver.py`, SHA256 `376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253`. `STATIC_SOURCE_RECEIPT.json` records unchanged adapter identities. Preserve their relative locations: driver implementation guards hash all three sibling files.

For root's future execution, replace the acquisition driver's correction-driver path and identity with this v2 path/hash, prepare into a fresh acquisition_run02 or equivalent, and build new admission manifests from the resulting v2 ROLE_FREEZE records. Fill all implementation hashes with these three sibling source identities. Do not alter any v1 manifest or failed run. Consult the original round5 prototype README for CLI interfaces; this packet changes only role derivation and adds explicit role-version/native-count/surplus provenance.

No execution has been performed here. Child verification consists of exact patch review, metadata arithmetic and source AST parsing. The parent owns subsequent preparation and scientific execution.
