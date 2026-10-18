# Non-oracle algorithm and evaluation audit

## Information firewall

1. `challenge.generate_challenge(mechanism, n, seed)` simulates the full data-generating SCM and calculates coupled `Y(0), Y(1)` truth privately in the *scoring runner*.
2. `blind.analyze_blind(observable_frame, seed, bias_radius, public_card)` receives ONLY `X1, X2, A, Y, M, C, observed`; rows with `observed=0` are dropped *before any fit*. No `SCMData`, mechanism, causal true edges, true ATE or counterfactual outcomes enter it. The public card fixes the variable timing.
3. Four offline graph-source fixtures are identical for all replicates. One intentionally violates known time ordering and must be rejected. These sources **are not OpenRouter LLM outputs**. Three accepted but mutually non-equivalent graphs yield measured-variable adjustment sets. Graph output includes source provenance, reason for rejection, adjustment sets, interval and abstention reason.
4. `nonoracle.score_predictions` is the first function allowed to see true ATE and identifying status. Scoring that a hidden confounder exists is legal after decisions; selecting a graph based on that label is not.
5. `calibrate_margins_on_development` accepts strictly `method`, `decision`, `assertion_margin` (no truth). The heldout set is scored with those thresholds unchanged.

## Comparison and necessary caveats

Methods are unadjusted, fixed evidence-based single DAG (no LLM), naive all-observed adjustment, each of four mock graph sources, unvalidated directed majority graph, screened multiple-DAG interval envelope and strict latent-U abstention. All rely on cross-fitted AIPW or a difference in means. This is an algorithm/unit-test benchmark, NOT a fair evaluation of four real LLM families. The screened candidate-set may omit the truth or leave some hidden confounding unrecognized, and unconditional causal assertions can fail despite hard temporality checks. Scenarios with nonobserved outcomes challenge the missingness rule; they are not conflated with ordinary sampling uncertainty. Candidate DAGs are not estimated by causal discovery, and the fixed support rule cannot be transported to SHARE without scientific review.

Primary endpoint is selective false-sign risk (only among assertions), conditional on heldout achieved coverage; independent truth-known error in a graph not observed by the learner is not called identified. A risk of 0 with few assertions should never be called a performance win. For n=0 assertions risk is undefined. The locked margin threshold curve and available coverage give the abstention cost. The *strict U* arm is supposed to abstain on every task and is an impossibility/control diagnostic, not a useful benchmark leader.

**Known limitations:** Mechanism class complexity, source fixture quality, bias grid and even our operational definition of wrong sign influence results. The 5×12 challenge experiment cannot settle novelty or calibration; next gates must replace fixtures with real evidence-grounded LLM responses, add independently reviewed graphs and mechanism-family holdouts, use stronger heterogeneity/selection estimators, evaluate intervals beyond normal AIPW approximations and formally compare matched-coverage risk uncertainty. No real LLM API requests have been authorized.
