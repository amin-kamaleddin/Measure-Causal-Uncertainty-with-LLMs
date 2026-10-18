"""Populate only numbers from real computations, and label causal assumptions."""
from pathlib import Path

def manuscript(result, folder):
    flow=result['flow'];primary=result['estimates'].get('aipw_demographic_country')
    raw=result['estimates']['unadjusted'];alt=result['estimates'].get('aipw_with_w8_depression')
    fmt=lambda x:f'{x:.3f}'
    text=f'''# Identification-aware causal conclusions under structural uncertainty in SHARE

**Exploratory manuscript draft, October 2026 — NOT SUBMISSION-READY.**
Results below were computed on supplied SHARE r9.0.0 ZIP files, but private aggregates require disclosure and PI review before dissemination. Title and novelty remain provisional.

## Abstract

**Background:** Large-language-model causal graphs may disagree, and disagreement does not identify causal effects. **Methods:** We linked regular SHARE Wave 8 social club involvement to Wave 9 EURO-D scores, inspected cohort retention and considered graph-conditional AIPW estimation and explicit abstention. **Results:** {flow['valid_wave8_activity']:,} Wave 8 respondents had a valid binary activity response; {flow['w9_valid_eurod_forward_order']:,} supplied valid Wave 9 EURO-D scores after time-order checks. The crude observed-outcome difference (participant minus nonparticipant) was {fmt(raw['effect'])} EURO-D points (country-cluster 95% interval {fmt(raw['ci_lower'])} to {fmt(raw['ci_upper'])}).
''' + (f'''The demographic/education/country-adjusted cross-fitted AIPW point estimate was {fmt(primary['effect'])} (conditional interval {fmt(primary['ci_lower'])} to {fmt(primary['ci_upper'])}; n={primary['n']:,}).''' if primary else 'Adjusted AIPW PENDING EXECUTION.') + f'''
**Conclusions:** These observational differences are NOT demonstrated causal treatment effects. Plausible unmeasured frailty and outcome selection graphs lack identification; the prespecified robust causal-sign procedure **abstained**.

## Design and data

Sources: SHARE-ERIC (2024) release 9.0.0 Waves 7, 8 and 9. Wave 8 activity `ac035d5` asks about any sports/social/other club during the preceding year, not a defined randomized strategy; EURO-D is the Wave 9 outcome (0–12, higher worse). Wave 8 fieldwork was interrupted by COVID-19 in March 2020; Wave 9 interviews began in autumn 2021 and continued to October 2022. W7 `cc733_` comes from retrospective history, not universally jointly observed with Wave 7 depression. No researcher may infer a treatment initiation date from the questionnaire.

## Participant flow and selection

{flow['wave8_ac_module']:,} Wave 8 AC records; {flow['valid_wave8_activity']:,} valid exposure (1: {flow['wave8_exposed']:,}; 0: {flow['wave8_unexposed']:,}); {flow['w9_cv_record_linked']:,} with W9 coverscreen record; {flow['w9_main_interview']:,} with a main interview; {flow['w9_known_deceased']:,} known deceased in W9 coverscreen; {flow['w9_valid_eurod_forward_order']:,} valid W9 EURO-D. Linked valid W7 childhood-financial category n={flow['w7_valid_childhood_financial']:,}, with simultaneous W7 regular interview EURO-D n={flow['w7_both_childhood_and_regular_depression']:,}.

## Estimation results

Unadjusted difference: {fmt(raw['effect'])} (interval {fmt(raw['ci_lower'])}, {fmt(raw['ci_upper'])}), n={raw['n']:,} observed Wave 9 scores. These intervals use country-cluster influence approximations, not full complex-survey repeated-sample inference.
''' + (f'''
Assumption-conditional AIPW with age, gender, ISCED education and country: {fmt(primary['effect'])} (interval {fmt(primary['ci_lower'])}, {fmt(primary['ci_upper'])}), n={primary['n']:,}. Fold count and nuisance specifications are in the private manifest. This does NOT correct unmeasured health/social propensity or survival selection.
''' if primary else '') + (f'''
**Post-exposure-sensitive alternative:** When W8 EURO-D was included despite ambiguous overlap, AIPW {fmt(alt['effect'])} (interval {fmt(alt['ci_lower'])}, {fmt(alt['ci_upper'])}); different adjustment potentially estimates a different causal quantity. See paired complete-case comparison in results JSON.
''' if alt else '') + f'''
## Structural uncertainty and identification

Verdict: **{result['structural_robustness']['decision']}**. Reason: {result['structural_robustness']['reason']}. Since supported scenarios include latent unmeasured confounding, a full effect is not identified. The additive sensitivity values [-1, 0, +1] are illustrative choices, not empirical bias bounds. Our rule reports no scientifically warranted robust sign under these premises.

## Limitations

No precise exposure strategy; COVID disruptions; W8 depression not pre-exposure; severe Wave 7 SHARELIFE vs regular respondent-branch mismatch; unmeasured confounding; complex design and absent longitudinal weights; Wave 9 death/attrition; W9 depression undefined for decedents; cross-country measurement and instrument variation; treatment versions. The AIPW estimate is a conditional modeling exercise, not causal proof. Synthetic benchmark mock graphs must not be described as actual LLM validation. Investigators should not report conventional inferential statistics as fully structural uncertainty-adjusted.

## Source citations to verify in bibliography

SHARE-ERIC (2024) Wave 7 DOI:10.6103/SHARE.w7.900; Wave 8 DOI:10.6103/SHARE.w8.900; Wave 9 DOI:10.6103/SHARE.w9.900. Börsch-Supan et al. (2013) DOI 10.1093/ije/dyt088. Aziz & Brookhart (2025) DOI 10.2147/CLEP.S550565. Li & Russo (2026) PMLR v337. Bibliographic and funder acknowledgements require SHARE compliance review.

## Missing for publication

PENDING PI APPROVAL: target intervention/estimand, graph support, plausible confounding bias shifts, attrition/death handling, W7 effect modification interpretation, formal design effect. PENDING EXECUTION: 3–4 real OpenRouter model comparisons and held-out model stability analysis. PENDING SCIENTIFIC VALIDATION: coverage under wider SCM mechanism suite and external calibration. PENDING DISCLOSURE APPROVAL: real-data aggregate release.
'''
    p=Path(folder)/'manuscript_computed_private.md';p.write_text(text)
    return p
