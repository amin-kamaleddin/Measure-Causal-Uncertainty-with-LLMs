"""One-team task authoring only. NEVER imported by elicitation/prediction.

Final exploratory expansion. Twelve independently specified scenario definitions,
two new narrative/parameter worlds each. Single-team development reuse disclosed.
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
    U=node('U','','1/2'); X=node('X','','1/2'); A=node('A','','1/2')
    def e(a,b,text,relation='supported'): return a,b,relation,text
    themes=[
      ('lottery_channel', ['Ultrasonic weld modulation','Photovoltaic cleaning gate']),
      ('measured_backdoor', ['Brine density desalinator','Polymer batch heat selector']),
      ('equivalence_witness', ['Sealed acoustic relay','Opaque optical relay']),
      ('physical_exclusion', ['Decoupled antenna indicator','Isolated fermentation dial']),
      ('temporal_mediation', ['Membrane staging valve','Enzyme activation cascade']),
      ('post_outcome_collider', ['Acoustic defect audit','Turbine resonance review']),
      ('selected_recording', ['Reactor retention archive','Optical assay retention']),
      ('weak_overlap', ['Rare ceramic pressure setting','Rare catalyst dosing setting']),
      ('effect_modification', ['Binary alloy response reversal','Binary substrate response reversal']),
      ('latent_proxy', ['Unmeasured abrasion proxy','Unmeasured contamination proxy']),
      ('uncertain_indirect_route', ['Disconnected routing telemetry','Disconnected cooling telemetry']),
      ('instrument_like_encouragement', ['Encouraged compressor setting','Encouraged sensor gain'])]
    for family,titles in themes:
      for j,title in enumerate(titles):
        positive=j==0
        y='1/8 7/8' if positive else '7/8 1/8'
        # Every card includes a genuinely unresolved physical hypothesis. It is
        # public evidence, not an annotation of which hidden world was drawn.
        if family=='lottery_channel':
          add(title,'A modulates an actuator before a later binary failure Y. X is an initial sensor; an unmeasured environmental input U may influence X. No response magnitude or direction is known.',
            ['U','X','A','Y'],[e('A','Y','The actuator is physically connected to the failure mechanism.'),e('U','X','An environmental input may alter the initial sensor.','ambiguous')],
            [[U,node('X','U','1/4 2/3'),node('A','','2/5'),node('Y','A',y)]],family,randomized=True)
        elif family=='measured_backdoor':
          add(title,'Initial material class X changes operator setting A and later failure Y. The actuator also affects failure. A residual unmeasured environment U could affect both setting and failure; measurement of X does not certify exchangeability.',
            ['U','X','A','Y'],[e('X','A','Material class changes setting.'),e('X','Y','Material class changes failure.'),e('A','Y','The actuator affects failure.'),e('U','A','Environment might affect setting.','ambiguous'),e('U','Y','Environment might affect failure.','ambiguous')],
            [[U,X,node('A','X','3/10 4/5'),node('Y','XA','1/10 3/5 2/5 9/10' if positive else '3/5 1/10 9/10 2/5')]],family)
        elif family=='equivalence_witness':
          add(title,'Only A and Y are measured. Sealed internal wiring could connect A to Y; alternatively a hidden input U could jointly drive them. No independent assignment or internal wiring inspection is available. Public evidence does not resolve these alternatives.',
            ['U','A','Y'],[e('A','Y','A direct channel may exist.','ambiguous'),e('U','A','A hidden input may drive A.','ambiguous'),e('U','Y','The same input may drive Y.','ambiguous')],
            [[U,A,node('Y','A','1/4 3/4' if positive else '3/4 1/4')],[U,node('A','U','1/4 3/4' if positive else '3/4 1/4'),node('Y','U','0 1')]],family,pair=True)
        elif family=='physical_exclusion':
          add(title,'The setting A is physically isolated from every direct and indirect path to failure Y. Initial material X affects the setting and failure. An unmeasured input U may affect the sensor X. Association need not vanish.',
            ['U','X','A','Y'],[e('X','A','Material class affects dial choice.'),e('X','Y','Material class changes failure.'),e('U','X','Environment may change material class.','ambiguous'),e('A','Y','Physical isolation forbids all A-to-Y paths.','contradicted')],
            [[U,node('X','U','1/3 3/4'),node('A','X','1/6 5/6'),node('Y','X',y)]],family,exclusion=True)
        elif family=='temporal_mediation':
          add(title,'Initial class X changes assignment A and final detection Y. A changes delivery M before Y; a direct A-to-Y channel is also physically possible but unverified. M follows A and is not a baseline adjustment variable.',
            ['U','X','A','M','Y'],[e('X','A','Class affects assignment.'),e('X','Y','Class affects detection.'),e('A','M','A changes delivery.'),e('M','Y','Delivery changes detection.'),e('A','Y','A bypass channel might affect detection.','ambiguous')],
            [[U,X,node('A','X','1/4 2/3'),node('M','A','1/6 5/6'),node('Y','XM','1/10 7/10 3/10 9/10' if positive else '7/10 1/10 9/10 3/10')]],family)
        elif family=='post_outcome_collider':
          add(title,'A changes failure Y. A later audit C responds to both setting and failure, but does not determine recording: all outcomes are recorded. An environmental input U might also influence the later audit. C cannot be a baseline adjustment.',
            ['U','A','Y','C'],[e('A','Y','The setting affects failure.'),e('A','C','Setting changes the audit.'),e('Y','C','Failure changes the audit.'),e('U','C','Environment may affect audit.','ambiguous')],
            [[U,A,node('Y','A',y),node('C','AY','1/8 1/3 2/3 7/8')]],family,randomized=True)
        elif family=='selected_recording':
          add(title,'A changes defined failure Y. Initial class X and later failure both influence inspection C; inspection affects whether Y is retained. A might also directly alter inspection. Defined outcomes for unretained units remain missing; the response law is unknown.',
            ['U','X','A','Y','C'],[e('A','Y','The setting affects failure.'),e('X','C','Initial class affects inspection.'),e('Y','C','Failure affects inspection.'),e('A','C','Setting may alter inspection.','ambiguous')],
            [[U,X,A,node('Y','A',y),node('C','XY','1/10 3/5 2/5 9/10')]],family,randomized=True,selection=True)
        elif family=='weak_overlap':
          add(title,'Initial class X nearly determines A, though both settings remain possible. X and A affect failure Y. Residual environment U may also affect A and Y. Rare assignment combinations could make adjusted estimation unstable.',
            ['U','X','A','Y'],[e('X','A','Class strongly changes assignment.'),e('X','Y','Class changes failure.'),e('A','Y','The actuator changes failure.'),e('U','A','Environment may change assignment.','ambiguous'),e('U','Y','Environment may change failure.','ambiguous')],
            [[U,X,node('A','X','1/200 199/200'),node('Y','XA','1/10 4/5 1/5 9/10' if positive else '4/5 1/10 9/10 1/5')]],family)
        elif family=='effect_modification':
          add(title,'A and initial class X jointly change failure Y. The direction can differ across X, with no overall sign stipulated. A hidden environment U may affect X. The lottery does not depend on X or U.',
            ['U','X','A','Y'],[e('X','Y','Material class affects failure.'),e('A','Y','The setting affects failure conditional on class.'),e('U','X','Environment may affect class.','ambiguous')],
            [[U,node('X','','2/5' if positive else '3/5'),node('A','','1/3'),node('Y','XA','1/10 9/10 9/10 1/10')]],family,randomized=True)
        elif family=='latent_proxy':
          add(title,'Unmeasured wear U may influence A, Y and the noisy measured proxy X. X does not fully measure wear. A direct actuator-to-failure route is possible. Proxy measurement alone does not establish exchangeability.',
            ['U','X','A','Y'],[e('U','X','Wear might affect the proxy.','ambiguous'),e('U','A','Wear might affect setting.','ambiguous'),e('U','Y','Wear might affect failure.','ambiguous'),e('A','Y','The actuator may affect failure.','ambiguous')],
            [[U,node('X','U','1/3 4/5'),node('A','U','1/8 7/8'),node('Y','UA','1/10 3/10 3/5 4/5' if positive else '3/10 1/10 4/5 3/5')]],family)
        elif family=='uncertain_indirect_route':
          add(title,'Initial class X changes setting A and failure Y. A changes intermediate telemetry M, but whether M physically influences Y is unverified. A direct A-to-Y wire is absent; exclusion of every indirect path has not been established.',
            ['U','X','A','M','Y'],[e('X','A','Class affects setting.'),e('X','Y','Class affects failure.'),e('A','M','Setting changes telemetry.'),e('M','Y','Telemetry might change failure.','ambiguous'),e('A','Y','There is no direct wire.','contradicted')],
            [[U,X,node('A','X','1/5 4/5'),node('M','A','1/4 3/4'),node('Y','X',y)]],family)
        else:
          add(title,'A baseline encouragement X changes A. Unmeasured U may influence both A and failure Y. A direct A-to-Y channel is plausible. The observations do not certify exclusion, independence of encouragement, monotonicity or any instrumental-variable estimand.',
            ['U','X','A','Y'],[e('X','A','Encouragement changes setting.'),e('U','A','Unmeasured input may change setting.','ambiguous'),e('U','Y','Unmeasured input may change failure.','ambiguous'),e('A','Y','A direct channel may exist.','ambiguous'),e('X','Y','Encouragement may have another channel.','ambiguous')],
            [[U,X,node('A','UX','1/10 2/5 3/5 9/10'),node('Y','UA','1/10 2/5 3/5 9/10' if positive else '2/5 1/10 9/10 3/5')]],family)
    for name,data in [('public_tasks.json',tasks),('scoring_truth.json',worlds),
                      ('families.json',dict(assignments=assignments)),
                      ('generation_plan.json',dict(schema_version='generation-plan-2.4',plans=plans))]:
        path=out/name; path.write_bytes(canonical(data)); path.chmod(0o600)
    return dict(public_cards=len(tasks),world_probes=len(worlds),declared_families=len(themes))
