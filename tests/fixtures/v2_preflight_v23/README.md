# Exposed fictional v2.3 development fixtures

These files contain no SHARE records, provider responses, actual prices or secrets.
`public_tasks.json` contains nine content-addressed public cards, with versioned
credibility commitments. `private_truth.json` contains ten rational binary SCMs,
including an observationally equivalent pair sharing one identical public task.
The runner opens that truth file only after sealing the entire prediction batch.

`families.json` defines eight exposed development families and three **future
candidate definitions only**. No unseen heldout cards have been generated or
scored. The feedback/spillover proposals require new targets or schemas before
use. Labels, signatures and feature differences do not prove independent
scientific authorship or external custody.

`decisions.json` faithfully copies the 16 recommendations/gates from the supplied
PI decision matrix, retaining DECISION_REQUIRED everywhere. A review recommendation
or illustrative sign-off is not approval.

Regenerate these exposed fixtures from the repository root using:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/build_v23_fixtures.py
```

This deterministic authoring script has deliberately seen both public design and
private synthetic mechanisms. It is not a blinded author or generator of an
independent confirmatory set. Do not run it to rewrite historical result files.
Its source and the fixture contents are hashed in new run provenance. Fictional
pricing is reused unchanged from `../v2_preflight/fake_pricing.json`.
