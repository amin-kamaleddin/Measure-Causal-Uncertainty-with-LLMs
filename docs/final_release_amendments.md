# Post-score reporting correction

The original experiment source commitment is preserved. Its reporting exporter did not handle rejected graph identification=None. The clean release corrects that post-score-only case; no graph admission, estimate, prediction, scoring rule, protocol or raw response is changed. The original reporting source is retained as executed_reporting_v1.py.txt. Scientific outputs remain bound to the original committed source and sealed prediction hashes.

Executed reporting SHA256: 752ccaf8287574649a51379f01727e9a9d507b91e0723d56daf285dc19db7c7a

Release reporting SHA256: 6253dcbd0d847d63fed934181e56f1c8289a8748b06d69e312d4fe9f3aad2f50

The original run's outer CLI may exit2 after successful scientific workers. A separate diagnostic repair script verifies the seal and unchanged prediction/score hashes, exports missing diagnostic tables and records exit0. This is not a new experiment or a paid retry.
