# NCN interaction v2 native CPU gate — stopped

The first recipe reached its first pre-Adam guard and failed exact repeated
old-state gradient equality. No optimizer transition or VALID call occurred.
The other three cases never started; there was no retry.

Actual cost: one native construction, four native route forwards/eight task
calls, two extra gradient-probe collections plus two source block collections,
47.13 seconds and 9.42 GiB peak process RSS.

The failure does not establish a wrong objective or a harmless numerical
difference: the original guard did not record parameter/magnitude, so neither
interpretation is supported. No tolerance was changed or broad numerical
investigation started. The exact source, qualifier, resource admission, failure
and costs remain preserved. The source package's eight local algebra checks
remain a distinct result; no native lifecycle readiness is claimed.

No TEST, GPU update, scientific fit, accuracy result or full-campaign GO.
