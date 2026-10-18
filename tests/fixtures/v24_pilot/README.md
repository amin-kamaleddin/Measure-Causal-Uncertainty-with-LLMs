# v2.4 exposed synthetic pilot fixtures

`generation_plan.json` contains rational binary CPTs for the nine already exposed
v2.3 public cards. It has no scoring labels/effects/certificates or private family
IDs. Only the generator process may read it. It is not a public prompt packet.
The scorer separately opens the unchanged v2.3 truth fixture after verifying the
entire prediction seal. The predictor can read neither file.

The plan was authored from exposed synthetic SCMs with:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/build_v24_generation_plan.py
```

This authoring step is not an independent blinded design. It writes only the new
v2.4 generation fixture and never rewrites previous truth or results.

`mock_rules.json` fixes two public-only failure stresses: source C emits malformed
JSON on challenge variants, and source D is absent when recording is publicly
unknown. Other responses reuse the deterministic v2.3 evidence-based slot rules.
No model, network, credentials or real records are involved. Raw rejected text is
not propagated to reports; only its hash and fixed error status are retained.

Samples are drawn by inverse-CDF sampling from the exact full observable joint
law of each CPT model. Observationally equivalent private worlds produce
byte-identical samples under shared random draws. Hidden U is integrated out;
unrecorded Y and post-treatment measurements are masked before analysis export.
Future unseen-family definitions from v2.3 are not instantiated or scored here.
