# Pre-request engineering amendment, 2026-10-18

First live launch stopped in source-hash verification with zero provider POSTs.
The worker guard omitted manifest-declared script/analysis-plan paths from its
read allowlist. Add only those exact committed source inputs before verification,
as the earlier pilot worker already does. No truth/data/credential permission,
scientific rule,task,split,estimator,threshold or scoring change. Regression now
runs a guarded pipeline with actual declared script and plan hashes. Preserve the
zero-call preparation directory and use a new r1 directory. No scores were opened.
