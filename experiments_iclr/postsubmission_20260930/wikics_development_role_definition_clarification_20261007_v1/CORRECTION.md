# WikiCS development role definition

The fixed pilot's 5,274-node VALID role is the union of the official validation and stopping masks for split 0. Public converter `export.py` lines 44–45 states that union explicitly. The full ordered converter checks matched the original native role; this note changes no mask, fit, checkpoint choice or score.

Use **development accuracy on the union of official validation and stopping masks, split 0** in new reports. TRAIN has 580 nodes, and TEST remains a separate role. The source field name VALID remains for compatibility. A development score from this combined selection population is not directly comparable to a published WikiCS TEST score.

The same merged role selected checkpoints and supplies the planned diagnostic readouts. Preserve that selection optimism and conditional three-seed scope. Independent heldout claims require a separately frozen TEST or other unused-population scoring protocol. The submitted paper's original five-dataset scores remain unchanged.
