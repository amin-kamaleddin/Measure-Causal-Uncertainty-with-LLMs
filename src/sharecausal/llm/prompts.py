"""PUBLIC-DOMAIN knowledge only. No real dataset statistics."""
import hashlib

PUBLIC_CONTEXT="""Research question: direction and identifiability of causal influence of
club/sport/social-group participation in the preceding 12 months at SHARE Wave 8 (A)
on depressive symptom EURO-D at Wave 9 (Y). Wave 8 depression D8 may overlap the
exposure period. B denotes age, sex, completed education and country; H denotes
retrospectively recalled childhood finances; U unmeasured pre-exposure health/
social inclination; S availability of W9 health outcome after death/attrition.
Wave 8 interviews were stopped in March 2020 due to COVID; Wave 9 interviews
occurred 2021-2022. Literature: Aziz & Brookhart (2025; DOI 10.2147/CLEP.S550565)
warned of LLM citation hallucination and DAG errors; Li & Russo (UAI 2026,
https://proceedings.mlr.press/v337/li26e.html) treat LLMs as imperfect experts.
There are NO empirical coefficients, treatment effects or participant records.
"""

def messages(variant='base'):
    if variant not in ('base','counterargument'):raise ValueError('Unknown prompt variant')
    task="""Return exactly the GraphAnswer structured JSON object. Every directed edge
needs roles, a causal temporal rationale, a *traceable* evidence reference OR
"unverified hypothesis", and honest uncertainty. Do not invent scientific
papers. Prefer abstention to unjustified causal identification. Question whether
D8 precedes A; do NOT use observed associations to construct DAGs."""
    if variant=='counterargument': task+=' Challenge each assumed edge and explicitly consider selection and U.'
    return [{'role':'system','content':'You are an epidemiologic DAG auditor, not a causal ground-truth oracle.'},
            {'role':'user','content':PUBLIC_CONTEXT+'\n'+task}]

def hash_messages(variant):
    import json
    return hashlib.sha256(json.dumps(messages(variant),sort_keys=True).encode()).hexdigest()
