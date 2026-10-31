# Prospective primary uncertainty reporting

The MolHIV primary pilot contains all 30 fits: seeds 0/1/2, own-member and pooled training losses, and tied/first/middle/last/low-rank layouts at 10% labels. Training-only locks determine the recurrence-policy layout separately for each seed and the lookahead layout globally for each loss family. An abstaining policy aliases tied; aliases are reused fits, never extra replications.

The eight primary contrasts are each of recurrence-own, recurrence-pooled, lookahead-own and lookahead-pooled against its same-family tied and low-rank control. Report both contrasts for every method, including abstention and unfavorable outcomes. The inherited continuation gate requires a mean ROC-AUC improvement of at least 0.005 and a positive improvement in every seed against both controls. This is a practical screening rule. Three seeds cannot establish statistical significance under an exact two-sided sign-flip test: its minimum p-value is 0.25.

Report all paired seed differences, mean/SD, paired t 95% intervals and enumerated two-sided sign-flip p-values with Holm correction across the complete eight-contrast family. The interval and test assumptions must accompany results. Seeds measure optimization variation on one fixed split. They provide no uncertainty over graphs, splits or tasks. Identical aliases and correlated contrasts remain disclosed.

Selected-checkpoint ROC-AUC is the inherited primary endpoint; the maximum-AUC/minimum-BCE/earliest rule remains fixed. Report epoch100 contrasts as a fixed-budget diagnostic with the same layout identities. Validation average precision, BCE, Brier score and member disagreement can be descriptive analyses of saved logits; they must not replace the primary endpoint or select another method. No original-paper scores or new test scores are computed here.

The full-label and expanded mandatory baseline stages remain governed by the existing integration plan. Passing the primary practical gate authorizes their completion and later independent confirmation; it does not establish novelty, broad superiority or acceptance. This source packet reads no dataset or model artifacts and launches no job.
