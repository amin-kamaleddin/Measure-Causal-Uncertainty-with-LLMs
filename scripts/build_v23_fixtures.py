"""Build fictional exposed development examples. Does not generate unseen holdouts."""
from pathlib import Path
import csv,json
from sharecausal.preflight.contracts import make_evidence,sha,canonical
from sharecausal.preflight_v23.packets import PublicTask,task_hash
from sharecausal.preflight_v23.certificates import BinaryDAG,effect

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tests/fixtures/v2_preflight_v23'


def node(name,parents,probabilities):
    return {'name':name,'parents':parents,'probability_one':probabilities}


def build():
    tasks=[];worlds=[];assignments=[];definitions=[]
    def add(key,title,domain,family,features,description,variables,edge_specs,scms,
            assignment='assignment_unknown',selection='complete_outcomes',exclusion=False,proof='none'):
        note=make_evidence(kind='general',source='',target='',relation='supported',statement=description)
        evidence=[note];rules=[]
        for source,target,relation,statement in edge_specs:
            e=make_evidence(kind='edge',source=source,target=target,relation=relation,statement=statement)
            evidence.append(e)
            if relation=='ambiguous':
                rules.append({'evidence_id':e['evidence_id'],'basis_evidence_id':note['evidence_id'],
                    'mechanism':'common_cause' if source=='U' else 'direct_channel',
                    'rationale':'Candidate physical channel explicitly described by the public apparatus design note; existence is unresolved.',
                    'status':'candidate_stipulation_requires_PI_review'})
        if assignment=='randomized_documented':
            evidence.append(make_evidence(kind='assignment',source='A',target='Y',relation='supported',
                statement='An independent physical lottery sets the switch before the apparatus run; it is independent of all other exogenous inputs.'))
        if exclusion:
            evidence.append(make_evidence(kind='exclusion',source='A',target='Y',relation='supported',
                statement='The sealed switch has no direct or indirect causal channel to this detector, for any eligible apparatus unit.'))
        role={'U':('latent_hypothesis',0,False),'X':('baseline',0,True),'A':('treatment',1,True),
              'M':('mediator',2,True),'Y':('outcome',3,True),'C':('post_outcome',4,True)}
        card={'schema_version':'public-world-2.2','title':title,
              'variables':[dict(name=v,role=role[v][0],time=role[v][1],observed=role[v][2]) for v in variables],
              'estimand':{'treatment':'A','outcome':'Y','contrast':'E[Y(do(A=1))-Y(do(A=0))]',
                'intervention':'Set '+domain+' switch to engaged versus bypass before each run.',
                'population':'All eligible fictional '+domain+' apparatus units.',
                'observation_target':'Y_defined_for_all_eligible_units',
                'consistency':'stipulated_no_interference_and_single_treatment_version',
                'positivity':'both_interventions_possible_in_target'},
              'assignment_description':'Independent documented switch lottery.' if assignment=='randomized_documented' else 'The switch-setting procedure is not disclosed.',
              'assignment':assignment,'selection_description':('Every eligible detector result is recorded.' if selection=='complete_outcomes' else
                'A post-run inspection process may determine which defined detector results are recorded; its law is not supplied.'),
              'selection':selection,'structural_exclusion':exclusion,'evidence':evidence}
        card={'card_id':'card-'+sha(card),**card}
        data={'schema_version':'public-task-2.3','card':card,'rubric_version':'admissibility-2.3.1','credibility':rules}
        task=PublicTask.model_validate({'task_id':'task-'+sha(data),**data});tasks.append(task.model_dump());h=task_hash(task)
        assignments.append({'task_hash':h,'family_id':family,'narrative_style':domain})
        if family not in {d['family_id'] for d in definitions}:
            definitions.append({'family_id':family,'cohort':'development_exposed','mechanism_features':features,
                'narrative_domain':domain,'custody':'single_team_exposed_not_independent',
                'rationale':'Mechanistic features and temporal/measurement structure differ from other exposed example families; no independent unseen validation is claimed.'})
        ids=['world-'+key+('-'+str(i) if len(scms)>1 else '') for i in range(len(scms))]
        for wid,nodes in zip(ids,scms):
            scm=BinaryDAG.model_validate({'nodes':nodes})
            worlds.append({'schema_version':'truth-2.3','world_id':wid,'public_task_hash':h,
                'scm':scm.model_dump(),'population_effect_exact':str(effect(scm)),'family_id':family,
                'proof':{'kind':proof,'witness_ids':ids if proof=='observational_pair' else []}})
    U=node('U',[],['1/2']);A=node('A',[],['1/2'])
    add('pump','Sealed pneumatic actuator','pneumatic','ambiguous_actuation',
        ['unresolved_assignment','latent_common_drive','only_A_Y_observed'],
        'A sealed upstream pressure source could drive both actuator and detector. A direct actuator tube could also transmit force; neither wiring route is verified.',
        ['U','A','Y'],[('A','Y','ambiguous','A direct tube is physically plausible but its connection is sealed.'),
        ('U','A','ambiguous','A shared upstream source may set the actuator.'),('U','Y','ambiguous','That same source may activate the detector.')],
        [[U,A,node('Y',['A'],['1/5','4/5'])],
         [U,node('A',['U'],['1/5','4/5']),node('Y',['U'],['0','1'])]],proof='observational_pair')
    add('heater','Thermostatic chamber','thermal','randomized_direct',
        ['independent_switch_lottery','direct_channel','complete_recording'],
        'The documented switch engages a heater connected directly to the threshold detector chamber. All detector flags are recorded.',
        ['U','A','Y'],[('A','Y','supported','A verified heater lead changes the chamber detector mechanism.')],
        [[U,A,node('Y',['A'],['1/10','9/10'])]],assignment='randomized_documented',proof='randomized_complete')
    add('optics','Optical attenuation bench','optical','randomized_direct',
        ['independent_switch_lottery','direct_channel','complete_recording'],
        'A shutter switch inserts an attenuator before a binary light detector. The lottery is independent and every detector flag is recorded.',
        ['U','A','Y'],[('A','Y','supported','A verified attenuation path connects the switch to the light detector.')],
        [[U,node('A',[],['2/5']),node('Y',['A'],['4/5','1/5'])]],assignment='randomized_documented',proof='randomized_complete')
    add('isolation','Electrically isolated alarm','electrical','public_isolation',
        ['global_causal_exclusion','observed_baseline_driver','complete_recording'],
        'The switch circuit is sealed away from the alarm. A measured battery state X drives the alarm on a separate circuit.',
        ['U','X','A','Y'],[('X','Y','supported','The battery lead is connected to the alarm.'),
        ('A','Y','contradicted','No switch-to-alarm path exists, including indirect paths.')],
        [[U,node('X',[],['1/2']),A,node('Y',['X'],['1/5','4/5'])]],exclusion=True,proof='public_exclusion')
    add('relay','Two-stage mechanical relay','mechanical','randomized_mediation',
        ['independent_switch_lottery','measured_intermediate_relay','complete_recording'],
        'The switch moves an observed relay M before the detector is read. The relay linkage is verified, with no direct bypass.',
        ['U','A','M','Y'],[('A','M','supported','The switch moves the relay.'),('M','Y','supported','The relay drives the detector.')],
        [[U,A,node('M',['A'],['1/10','9/10']),node('Y',['M'],['3/20','17/20'])]],
        assignment='randomized_documented',proof='randomized_complete')
    add('feedstock','Feedstock routing valve','chemical','measured_common_cause',
        ['observed_feedstock_common_cause','unknown_assignment','possible_direct_valve_channel'],
        'Observed feedstock state X influences both valve setting and detector chemistry. A bypass valve can additionally feed the detector; unlisted influences are not ruled out.',
        ['U','X','A','Y'],[('X','A','supported','The feedstock controller sets the valve.'),
        ('X','Y','supported','The feedstock affects detector chemistry.'),('A','Y','ambiguous','The bypass connection could directly affect detector chemistry.')],
        [[U,node('X',[],['1/2']),node('A',['X'],['1/5','4/5']),node('Y',['X','A'],['1/10','2/5','3/5','9/10'])]])
    add('inspection','Post-run inspection archive','archival','outcome_selection',
        ['independent_switch_lottery','post_outcome_common_effect','unknown_recording'],
        'Switch activity and the defined detector flag can both affect a later inspection flag C. Recording may depend on this inspection; its law is not supplied.',
        ['U','A','Y','C'],[('A','Y','supported','The switch lead drives the detector.'),
        ('A','C','supported','Switch activity affects the later inspection flag.'),('Y','C','supported','The detector flag affects later inspection.')],
        [[U,A,node('Y',['A'],['1/5','4/5']),node('C',['A','Y'],['1/10','7/10','3/5','9/10'])]],
        assignment='randomized_documented',selection='selection_unknown',proof='randomized_complete')
    add('proxy','Acoustic proxy monitor','acoustic','proxy_common_drive',
        ['latent_acoustic_driver','observed_noisy_proxy','unknown_assignment'],
        'A hidden vibration source could drive switch and detector and an observed microphone X. A direct switch-to-detector route is also plausible; the source is not observed.',
        ['U','X','A','Y'],[('U','X','ambiguous','The hidden source could drive the microphone.'),
        ('U','A','ambiguous','The hidden source could set the switch.'),('U','Y','ambiguous','The hidden source could drive the detector.'),
        ('A','Y','ambiguous','A direct acoustic route from switch to detector is possible.')],
        [[U,node('X',['U'],['1/5','4/5']),node('A',['U'],['1/5','4/5']),node('Y',['U'],['0','1'])]])
    add('hydraulic','Load-dependent hydraulic relay','hydraulic','moderated_mediation',
        ['independent_switch_lottery','baseline_load_modifies_relay','measured_mediated_channel'],
        'An observed baseline load X changes how a hydraulic relay M responds to the randomized valve A. The relay then drives the detector; all results are recorded.',
        ['U','X','A','M','Y'],[('X','M','supported','Baseline load changes the relay response.'),
        ('A','M','supported','The valve controls relay actuation.'),('M','Y','supported','Relay actuation drives the detector.')],
        [[U,node('X',[],['1/2']),A,node('M',['X','A'],['1/10','1/2','2/5','9/10']),
          node('Y',['M'],['1/10','9/10'])]],assignment='randomized_documented',proof='randomized_complete')
    for fid,features,domain,rationale in [
        ('candidate_encouragement',['random_encouragement_before_noncompliant_switch','instrument_exclusion','noncompliance'],
         'allocation','A randomized encouragement precedes noncompliant treatment, with explicit exclusion and compliance assumptions requiring a new certificate.'),
        ('candidate_feedback',['two_time_indexed_switches','feedback_state_transition','longitudinal_regime'],
         'navigation','Time-indexed feedback and two interventions define a longitudinal regime, beyond a renamed baseline or a new random seed.'),
        ('candidate_interference',['network_neighbor_exposure','partial_interference','spillover_target'],
         'network','Neighbor assignment and spillovers require a different intervention target and interference assumptions; the current no-interference cards cannot instantiate it.')]:
        definitions.append({'family_id':fid,'cohort':'candidate_heldout_definition_only','mechanism_features':features,
            'narrative_domain':domain,'custody':'DECISION_REQUIRED','rationale':rationale})
    files={'public_tasks.json':tasks,'private_truth.json':worlds,
           'families.json':{'definitions':definitions,'assignments':assignments}}
    with (ROOT/'docs/v2_pi_decision_matrix.csv').open(newline='') as stream:
        files['decisions.json']=[{'id':r['id'],'status':'DECISION_REQUIRED',
            'review_recommendation_not_approval':r['independent_scientific_recommendation_not_approved'],
            'required_gate':r['required_gate']} for r in csv.DictReader(stream)]
    for name,data in files.items():(DEST/name).write_bytes(canonical(data))
    print(json.dumps({'public_cards':len(tasks),'world_probes':len(worlds),
                      'development_families':sum(d['cohort']=='development_exposed' for d in definitions),
                      'candidate_heldout_definitions':3,'heldout_cards_created':0}))


if __name__=='__main__':build()
