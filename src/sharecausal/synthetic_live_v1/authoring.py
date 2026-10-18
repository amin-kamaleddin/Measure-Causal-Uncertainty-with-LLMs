"""One-team task authoring only. NEVER imported by elicitation/prediction.

New narratives and parameterizations; mechanism classes reuse the tested binary
SCM contract. These are exploratory cards, not independently held-out families.
"""
from pathlib import Path
from sharecausal.preflight.contracts import canonical, sha, make_evidence
from sharecausal.preflight_v23.packets import PublicTask, task_hash
from sharecausal.preflight_v23.certificates import BinaryDAG, effect


def node(name, parents, values):
    return dict(name=name, parents=list(parents), probability_one=values.split())


def build(destination):
    out = Path(destination); out.mkdir(parents=True, exist_ok=False); out.chmod(0o700)
    tasks=[]; worlds=[]; assignments=[]; plans={}
    roles={'U':('latent_hypothesis',0,False),'X':('baseline',0,True),
           'A':('treatment',1,True),'M':('mediator',2,True),
           'Y':('outcome',3,True),'C':('post_outcome',4,True)}
    def add(title, description, variables, specs, scms, family, *, randomized=False,
            selection=False, exclusion=False, pair=False):
        note=make_evidence(kind='general', source='', target='', relation='supported',
                           statement='Synthetic scenario evidence: '+description)
        evidence=[note]; rules=[]
        for a,b,relation,text in specs:
            e=make_evidence(kind='edge',source=a,target=b,relation=relation,statement=text)
            evidence.append(e)
            if relation=='ambiguous':
                rules.append(dict(evidence_id=e['evidence_id'],basis_evidence_id=note['evidence_id'],
                    mechanism='common_cause' if a=='U' else 'direct_channel',
                    rationale='This explicitly described physical route is a candidate hypothesis; the scenario does not establish its existence.',
                    status='candidate_stipulation_requires_PI_review'))
        if randomized:
            evidence.append(make_evidence(kind='assignment',source='A',target='Y',relation='supported',
                statement='Synthetic scenario evidence: an audited independent lottery assigns A before each run, independently of all exogenous inputs.'))
        if exclusion:
            evidence.append(make_evidence(kind='exclusion',source='A',target='Y',relation='supported',
                statement='Synthetic scenario evidence: physical separation excludes every direct and indirect A-to-Y causal path.'))
        card=dict(schema_version='public-world-2.2',title=title,
            variables=[dict(name=v,role=roles[v][0],time=roles[v][1],observed=roles[v][2]) for v in variables],
            estimand=dict(treatment='A',outcome='Y',contrast='E[Y(do(A=1))-Y(do(A=0))]',
                intervention='Set the described binary apparatus setting A to engaged versus bypass before the run.',
                population='All eligible fictional units in this apparatus scenario.',
                observation_target='Y_defined_for_all_eligible_units',
                consistency='stipulated_no_interference_and_single_treatment_version',
                positivity='both_interventions_possible_in_target'),
            assignment_description='Audited independent lottery.' if randomized else 'Operator assignment is incompletely documented; no independence is assumed.',
            assignment='randomized_documented' if randomized else 'assignment_unknown',
            selection_description='Later inspection C can govern recording of a defined Y; the response law is unknown.' if selection else 'Y is recorded for every eligible unit.',
            selection='selection_unknown' if selection else 'complete_outcomes',
            structural_exclusion=exclusion,evidence=evidence)
        card=dict(card_id='card-'+sha(card),**card)
        content=dict(schema_version='public-task-2.3',card=card,rubric_version='admissibility-2.3.1',credibility=rules)
        task=PublicTask.model_validate(dict(task_id='task-'+sha(content),**content)); h=task_hash(task)
        tasks.append(task.model_dump()); assignments.append(dict(task_hash=h,family_id=family,narrative_style=title))
        ids=['world-pilot-'+str(len(tasks))+'-'+str(i) for i in range(len(scms))]
        for wid,nodes in zip(ids,scms):
            scm=BinaryDAG.model_validate(dict(nodes=nodes))
            worlds.append(dict(schema_version='truth-2.3',world_id=wid,public_task_hash=h,
                scm=scm.model_dump(),population_effect_exact=str(effect(scm)),family_id=family,
                proof=dict(kind='observational_pair' if pair else 'public_exclusion' if exclusion else
                           'randomized_complete' if randomized else 'none',witness_ids=ids if pair else [])))
        plans[h]=dict(nodes=scms[0],recording='post_inspection_dependent' if selection else 'complete')
    U=node('U','','1/2'); A=node('A','','1/2'); X=node('X','','2/5')
    def e(a,b,text,relation='supported'): return a,b,relation,text
    add('Seed cabinet light timer',
        'A timer engages an illumination strip; Y flags germination at a fixed later inspection. The timer circuit is verified and units are isolated.',
        ['U','A','Y'],[e('A','Y','The verified illumination circuit changes the germination mechanism.')],
        [[U,A,node('Y','A','3/20 4/5')]],'direct_assignment',randomized=True)
    add('Cooling sleeve for resin coupons',
        'A sleeve changes heat transfer during curing. Y flags a later thermal defect. The sleeve coupling is documented without disclosing its magnitude or sign.',
        ['U','A','Y'],[e('A','Y','The sleeve has a verified thermal connection to the defect mechanism.')],
        [[U,node('A','','2/5'),node('Y','A','17/20 1/4')]],'direct_assignment',randomized=True)
    add('Culture cartridge nutrient latch',
        'A opens a nutrient latch, M records delivery before Y indicates colony detection. All influence from A traverses the measured delivery stage.',
        ['U','A','M','Y'],[e('A','M','The latch changes nutrient delivery.'),e('M','Y','Delivery changes the later detection mechanism.')],
        [[U,A,node('M','A','1/5 9/10'),node('Y','M','1/10 4/5')]],'mediated_assignment',randomized=True)
    add('Packaging line audit flag',
        'A changes sealing pressure and Y flags leakage. A later audit C responds to both setting and leakage. Every result is recorded regardless of C; C is a collider rather than a baseline covariate.',
        ['U','A','Y','C'],[e('A','Y','Pressure changes leakage.'),e('A','C','The setting affects later audit.'),e('Y','C','Leakage affects later audit.')],
        [[U,A,node('Y','A','3/4 1/5'),node('C','AY','1/10 2/5 3/5 9/10')]],'complete_collider',randomized=True)
    add('Mineral filter routing log',
        'Baseline mineral class X affects the operator setting A and later residue flag Y. A verified filter path also affects residue. Unrecorded causes are not ruled out.',
        ['U','X','A','Y'],[e('X','A','Mineral class influences routing.'),e('X','Y','Mineral class affects residue.'),e('A','Y','The filter path affects residue.')],
        [[U,X,node('A','X','1/4 3/4'),node('Y','XA','1/5 1/10 9/10 3/5')]],'measured_confounding')
    add('Ink viscosity nozzle selector',
        'A viscosity flag X is measured before a disconnected selector A. X affects selection and the later droplet defect Y. Physical isolation excludes all selector-to-defect paths despite their possible association.',
        ['U','X','A','Y'],[e('X','A','Viscosity influences selector choice.'),e('X','Y','Viscosity changes defects.'),e('A','Y','The disconnected selector cannot affect defects.','contradicted')],
        [[U,X,node('A','X','1/5 4/5'),node('Y','X','1/10 4/5')]],'public_exclusion',exclusion=True)
    add('Magnetic bearing vibration logger',
        'Latent wear U may influence drive setting A, defect Y, and a noisy measured vibration proxy X. A direct drive-to-defect route is plausible. X does not fully measure wear.',
        ['U','X','A','Y'],[e('U','X','Wear may change vibration.','ambiguous'),e('U','A','Wear may affect setting.','ambiguous'),e('U','Y','Wear may affect defects.','ambiguous'),e('A','Y','The drive may affect defects.','ambiguous')],
        [[U,node('X','U','1/5 3/4'),node('A','U','1/10 4/5'),node('Y','UA','1/10 3/10 7/10 9/10')]],'proxy_latent')
    add('Irrigation bed load switch',
        'Baseline soil load X strongly influences setting A. Both load and setting influence a later stress flag Y; their joint response may be nonlinear and heterogeneous. Rare assignment combinations remain possible.',
        ['U','X','A','Y'],[e('X','A','Soil load governs usual setting.'),e('X','Y','Load changes stress.'),e('A','Y','The valve changes stress.')],
        [[U,X,node('A','X','1/100 99/100'),node('Y','XA','1/10 3/4 4/5 1/5')]],'weak_heterogeneous')
    add('Battery load pulse bench',
        'The lottery can assign either pulse A, but one setting is rare. Baseline load X and pulse jointly alter failure Y with possible effect reversal across load. This is a binary heterogeneous response, not a smooth-function test.',
        ['U','X','A','Y'],[e('X','Y','Load affects the failure mechanism.'),e('A','Y','The pulse affects failure conditional on load.')],
        [[U,node('X','','1/2'),node('A','','1/25'),node('Y','XA','1/10 9/10 9/10 1/10')]],'weak_heterogeneous',randomized=True)
    for title,desc,prob in [
        ('Ceramic inspection archive','A firing option affects cracks Y; later inspection C depends on both firing and cracks, and only a selected subset is archived. Y exists even without an archive entry.','1/5 4/5'),
        ('Food pouch sample retention','A cooling option affects spoilage flag Y. Retention C occurs afterward and responds to both option and spoilage. A missing record does not mean an undefined spoilage outcome.','4/5 1/5')]:
        add(title,desc,['U','A','Y','C'],[e('A','Y','The option changes the outcome mechanism.'),e('A','C','The option affects later retention.'),e('Y','C','The outcome affects later retention.')],
            [[U,A,node('Y','A',prob),node('C','AY','1/10 3/5 2/5 9/10')]],'informative_recording',randomized=True,selection=True)
    add('Sealed humidity chamber wiring',
        'Only setting A and indicator Y are observed. A sealed humidity source U might drive both, or a direct A-to-Y connection might exist. Assignment and internal wiring are unknown; these descriptions do not distinguish the possibilities.',
        ['U','A','Y'],[e('A','Y','A direct connection may exist.','ambiguous'),e('U','A','Humidity may drive setting.','ambiguous'),e('U','Y','Humidity may drive indicator.','ambiguous')],
        [[U,A,node('Y','A','1/5 4/5')],[U,node('A','U','1/5 4/5'),node('Y','U','0 1')]],'compatible_worlds',pair=True)
    for name,data in [('public_tasks.json',tasks),('scoring_truth.json',worlds),
                      ('families.json',dict(assignments=assignments)),
                      ('generation_plan.json',dict(schema_version='generation-plan-2.4',plans=plans))]:
        path=out/name; path.write_bytes(canonical(data)); path.chmod(0o600)
    return dict(public_cards=len(tasks),world_probes=len(worlds),declared_families=len({a['family_id'] for a in assignments}))
