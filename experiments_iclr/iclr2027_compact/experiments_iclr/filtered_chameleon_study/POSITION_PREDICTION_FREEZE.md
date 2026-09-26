# Filtered-Chameleon placement prediction before outcome access

The fixed split has 409 training nodes, below the exploratory 5,000-label
threshold noticed only after four earlier graphs were observed. Before
opening any filtered-Chameleon result, we predict a negative mean paired
test contrast (private-first minus private-last) over optimizer seeds 0, 1, 2
under the equal-parameter 300-epoch recipe. Individual seed signs are not
predicted. This is a prospectively written test of a post hoc pattern, not
evidence that training-set size causes a placement preference.

The secondary validation-choice analysis selects between the two partial
arms by higher mean selected validation accuracy over the three seeds,
smaller mean selected validation cross-entropy on an exact accuracy tie,
and private-last on a remaining exact tie. It reports both candidates,
the selected mean test accuracy, retrospective test regret to the better
candidate, paired seed differences, and the two-arm search cost. It cannot
be interpreted as an unbiased estimate of a universal placement rule.
