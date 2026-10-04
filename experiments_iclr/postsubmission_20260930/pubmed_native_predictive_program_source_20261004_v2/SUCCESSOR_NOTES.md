# Minimal supervisor correction

V1 and its independent review remain unchanged. V2 changes only the supervisor and source documentation/metadata. Native models, six seeds/fits, training, selector, state capture, score semantics, replay and limits remain byte-identical.

Resource preflight has a fixed maximum 15-second nvidia-smi timeout capped by remaining stage time. It publishes a failed preflight receipt without launching a child on failure, and checks the remaining stage budget before launch. A disappearing owned child returns from the signal helper without a signal; cleanup still attempts wait4 reaping and preserves kernel usage. A PID identity mismatch continues to refuse signalling. Cleanup timeout remains a failed attempt, never a completed fit.

The original source review is not represented as independent review of V2. A correction review and root release remain pending. No numerical execution, scientific fit or score access occurred in this preparation.
